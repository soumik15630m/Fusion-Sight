"""Dataset: images in, augmented tensor + three scales of YOLO targets out.

Differences from the video's dataset.py, all forced by this repo's data:

* It walks image directories (the ultralytics `images/` <-> `labels/` layout
  already on disk) instead of reading a `train.csv` of image/label filename
  pairs. There is no CSV here, and generating one would be a second copy of the
  same information.
* Labels are parsed straight into (x, y, w, h, class) rather than loaded with
  `np.loadtxt` and `np.roll`-ed. `np.loadtxt` raises on an empty label file,
  and VisDrone/WiderPerson both contain images with no annotations at all.
* Cell indices are clamped to S-1. A box whose centre normalises to exactly
  1.0 -- which happens with boxes touching the right/bottom edge -- gives
  `int(S * 1.0) == S` and indexes off the end of the target tensor.
* `scale_idx` / `anchor_on_scale` are cast to int; on a tensor they stay
  0-dim tensors and silently make the later indexing much slower.
"""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from yolov3.utils import iou_width_height as iou

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def label_path_for(img_path):
    """Ultralytics convention: .../images/... -> .../labels/....txt"""
    parts = list(Path(img_path).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _read_label(path):
    """Return a list of [x, y, w, h, class]; empty if the file is missing."""
    if not os.path.isfile(path):
        return []
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) < 5:
                continue
            cls, x, y, w, h = (float(v) for v in fields[:5])
            boxes.append([x, y, w, h, cls])
    return boxes


class YOLODataset(Dataset):
    def __init__(
        self,
        img_dirs,
        anchors,
        image_size=416,
        S=(13, 26, 52),
        C=4,
        transform=None,
    ):
        if isinstance(img_dirs, (str, Path)):
            img_dirs = [img_dirs]

        self.img_files = []
        for d in img_dirs:
            d = Path(d)
            if not d.is_dir():
                raise FileNotFoundError(f"image directory not found: {d}")
            self.img_files += sorted(
                p for p in d.iterdir() if p.suffix.lower() in IMG_EXTENSIONS
            )
        if not self.img_files:
            raise FileNotFoundError(f"no images found under {img_dirs}")

        self.image_size = image_size
        self.transform = transform
        self.S = list(S)
        # All 3 scales' anchors in one flat list, so argsort ranks them together.
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        bboxes = _read_label(label_path_for(img_path))
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # One target per scale: (3 anchors, S, S, [obj, x, y, w, h, class]).
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3      # one anchor per scale, best-fitting first

            for anchor_idx in anchor_indices:
                anchor_idx = int(anchor_idx)
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                # Clamp: a centre of exactly 1.0 would index S, one past the end.
                i = min(int(S * y), S - 1)
                j = min(int(S * x), S - 1)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i          # both in [0, 1]
                    # Relative to the cell, so these can exceed 1.
                    width_cell, height_cell = width * S, height * S
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # Good enough to be ambiguous: don't punish a prediction here.
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)


if __name__ == "__main__":
    from yolov3 import config
    from yolov3.utils import cells_to_bboxes, non_max_suppression, plot_image

    dataset = YOLODataset(
        config.TRAIN_IMG_DIRS,
        anchors=config.ANCHORS,
        S=config.S,
        C=config.NUM_CLASSES,
        transform=config.train_transforms,
    )
    print(f"{len(dataset)} images, {config.NUM_CLASSES} classes: {config.CLASS_NAMES}")

    scaled_anchors = torch.tensor(config.ANCHORS) * torch.tensor(config.S).unsqueeze(
        1
    ).unsqueeze(1).repeat(1, 3, 2)

    # Decode the *targets* back into boxes: if these land on the objects, the
    # anchor assignment and the augmentation box maths agree.
    image, targets = dataset[0]
    boxes = []
    for i in range(3):
        S = targets[i].shape[1]
        boxes += cells_to_bboxes(
            targets[i].unsqueeze(0), scaled_anchors[i], S=S, is_preds=False
        )[0]
    boxes = non_max_suppression(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
    print(f"decoded {len(boxes)} target boxes from sample 0")
    plot_image(
        image,
        boxes,
        class_names=config.CLASS_NAMES,
        save_path=str(config.BASE_DIR / "runs" / "yolov3" / "target_check.png"),
    )
