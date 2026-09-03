"""The pieces the video refers to but never writes: IoU, NMS, mAP, decoding
cell-relative predictions back to image coordinates, checkpointing and the
dataloader factory.
"""
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
#  progress bar (tqdm is not installed here, so keep a minimal stand-in)
# --------------------------------------------------------------------------- #
try:
    from tqdm import tqdm  # noqa: F401
except ImportError:

    class tqdm:  # noqa: N801 - deliberately shadowing the tqdm name
        """Just enough of tqdm's surface for train.py: iteration + set_postfix."""

        def __init__(self, iterable=None, total=None, leave=True, desc=""):
            self.iterable = iterable
            self.total = total if total is not None else _safe_len(iterable)
            self.desc = desc
            self.postfix = ""
            self.n = 0

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
                self._draw()
            sys.stdout.write("\n")
            sys.stdout.flush()

        def set_postfix(self, **kwargs):
            self.postfix = " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in kwargs.items()
            )

        def set_description(self, desc):
            self.desc = desc

        def _draw(self):
            total = f"/{self.total}" if self.total else ""
            sys.stdout.write(f"\r{self.desc} {self.n}{total} {self.postfix}   ")
            sys.stdout.flush()

    def _safe_len(iterable):
        try:
            return len(iterable)
        except TypeError:
            return None


# --------------------------------------------------------------------------- #
#  IoU
# --------------------------------------------------------------------------- #
def iou_width_height(boxes1, boxes2):
    """IoU of boxes sharing a centre -- only widths and heights matter.

    Used to pick which anchor a ground-truth box belongs to, where position is
    irrelevant by construction.

    boxes1: (2,) or (N, 2) tensor of (w, h)
    boxes2: (M, 2) tensor of (w, h)
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / (union + 1e-6)


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """IoU between two equally shaped batches of boxes.

    box_format: "midpoint" for (x, y, w, h), "corners" for (x1, y1, x2, y2).
    """
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    elif box_format == "corners":
        box1_x1, box1_y1, box1_x2, box1_y2 = (boxes_preds[..., i: i + 1] for i in range(4))
        box2_x1, box2_y1, box2_x2, box2_y2 = (boxes_labels[..., i: i + 1] for i in range(4))
    else:
        raise ValueError(f"unknown box_format {box_format!r}")

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


# --------------------------------------------------------------------------- #
#  NMS and mAP
# --------------------------------------------------------------------------- #
def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """bboxes: list of [class_pred, prob_score, x, y, w, h]. Class-aware."""
    assert isinstance(bboxes, list)

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """mAP at a single IoU threshold.

    pred_boxes / true_boxes: lists of [image_idx, class_pred, prob, x, y, w, h].
    """
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = [d for d in pred_boxes if d[1] == c]
        ground_truths = [t for t in true_boxes if t[1] == c]

        # How many ground-truth boxes of this class each image holds.
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)   # one slot per gt, 1 == already matched

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold and amount_bboxes[detection[0]][best_gt_idx] == 0:
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1   # each gt counts once
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    if not average_precisions:
        return torch.tensor(0.0)
    return sum(average_precisions) / len(average_precisions)


# --------------------------------------------------------------------------- #
#  decoding predictions
# --------------------------------------------------------------------------- #
def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """Convert one scale of cell-relative output to whole-image boxes.

    predictions: (N, 3, S, S, 5 + num_classes)
    anchors:     the 3 anchors for this scale, already scaled by S
    returns:     (N, 3 * S * S, 6) of [class, score, x, y, w, h], x/y/w/h in 0-1
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5].clone()

    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:4] = torch.exp(box_predictions[..., 2:4]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(BATCH_SIZE, num_anchors, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]

    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(
        BATCH_SIZE, num_anchors * S * S, 6
    )
    return converted_bboxes.tolist()


@torch.no_grad()
def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    """Run the model over a loader and return (pred_boxes, true_boxes) for mAP."""
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []

    for batch_idx, (x, labels) in enumerate(tqdm(loader, desc="eval")):
        x = x.to(device)
        predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]], device=device).float() * S
            boxes_scale_i = cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
            for idx, box in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # Ground truth is identical across scales; the finest one is enough.
        true_bboxes = cells_to_bboxes(labels[2], anchor, S=S, is_preds=False)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


@torch.no_grad()
def check_class_accuracy(model, loader, threshold, device="cuda"):
    """Class / no-object / object accuracy over a loader. Cheap sanity metric."""
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for x, y in tqdm(loader, desc="acc"):
        x = x.to(device)
        out = model(x)

        for i in range(3):
            y[i] = y[i].to(device)
            obj = y[i][..., 0] == 1
            noobj = y[i][..., 0] == 0

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    acc = {
        "class": (correct_class / (tot_class_preds + 1e-16)).item() * 100,
        "no_obj": (correct_noobj / (tot_noobj + 1e-16)).item() * 100,
        "obj": (correct_obj / (tot_obj + 1e-16)).item() * 100,
    }
    print(
        f"Class accuracy: {acc['class']:.2f}%  "
        f"No-obj accuracy: {acc['no_obj']:.2f}%  "
        f"Obj accuracy: {acc['obj']:.2f}%"
    )
    model.train()
    return acc


# --------------------------------------------------------------------------- #
#  checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar", **extra):
    print(f"=> Saving checkpoint to {filename}")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        **extra,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer=None, lr=None, device="cuda"):
    print(f"=> Loading checkpoint {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        # The lr in the checkpoint is the one the run was saved with; a resume
        # should honour the lr the user asked for now.
        if lr is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
    return checkpoint


# --------------------------------------------------------------------------- #
#  dataloaders
# --------------------------------------------------------------------------- #
def get_loaders(train_dirs=None, val_dirs=None, test_dirs=None):
    """Build train / test / train-eval loaders from image directories.

    Defaults come from config, i.e. from the dataset YAML. The train-eval loader
    is the training split under *test* transforms, for measuring fit without
    augmentation noise.
    """
    from yolov3 import config
    from yolov3.dataset import YOLODataset

    train_dirs = train_dirs or config.TRAIN_IMG_DIRS
    val_dirs = val_dirs or config.VAL_IMG_DIRS
    test_dirs = test_dirs or config.TEST_IMG_DIRS

    common = dict(
        anchors=config.ANCHORS,
        image_size=config.IMAGE_SIZE,
        S=config.S,
        C=config.NUM_CLASSES,
    )

    train_dataset = YOLODataset(train_dirs, transform=config.train_transforms, **common)
    test_dataset = YOLODataset(val_dirs, transform=config.test_transforms, **common)
    train_eval_dataset = YOLODataset(train_dirs, transform=config.test_transforms, **common)

    loader_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    train_eval_loader = DataLoader(train_eval_dataset, shuffle=False, **loader_kwargs)

    return train_loader, test_loader, train_eval_loader


# --------------------------------------------------------------------------- #
#  visualisation
# --------------------------------------------------------------------------- #
def plot_image(image, boxes, class_names=None, save_path=None):
    """Draw [class, score, x, y, w, h] boxes (0-1 midpoint) on a CHW/HWC image."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().permute(1, 2, 0).numpy()
    im = np.array(image)
    if im.max() <= 1.0:
        im = (im * 255).astype(np.uint8)
    height, width, _ = im.shape

    cmap = plt.get_cmap("tab20b")
    n = len(class_names) if class_names else 20
    colors = [cmap(i) for i in np.linspace(0, 1, n)]

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        class_pred = int(box[0])
        score = box[1]
        x, y, w, h = box[2:6]
        upper_left_x = (x - w / 2) * width
        upper_left_y = (y - h / 2) * height
        rect = patches.Rectangle(
            (upper_left_x, upper_left_y),
            w * width,
            h * height,
            linewidth=2,
            edgecolor=colors[class_pred % n],
            facecolor="none",
        )
        ax.add_patch(rect)
        label = class_names[class_pred] if class_names else str(class_pred)
        ax.text(
            upper_left_x,
            upper_left_y,
            f"{label} {score:.2f}",
            color="white",
            verticalalignment="top",
            bbox={"color": colors[class_pred % n], "pad": 0},
        )

    plt.axis("off")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"=> wrote {save_path}")
    else:
        plt.show()


@torch.no_grad()
def plot_couple_examples(
    model, loader, thresh, iou_thresh, anchors, class_names=None, device="cuda", save_dir=None
):
    """Predict on one batch and draw the surviving boxes."""
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    out = model(x)

    bboxes = [[] for _ in range(x.shape[0])]
    for i in range(3):
        S = out[i].shape[2]
        anchor = torch.tensor([*anchors[i]], device=device).float() * S
        boxes_scale_i = cells_to_bboxes(out[i], anchor, S=S, is_preds=True)
        for idx, box in enumerate(boxes_scale_i):
            bboxes[idx] += box

    for i in range(min(x.shape[0], 4)):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint"
        )
        save_path = f"{save_dir}/example_{i}.png" if save_dir else None
        plot_image(x[i], nms_boxes, class_names=class_names, save_path=save_path)

    model.train()
