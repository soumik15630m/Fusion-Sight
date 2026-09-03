"""Hyperparameters, paths, class list and augmentation pipeline.

Two deliberate departures from the video's config.py:

1. The class list is read from the dataset YAML rather than hard-coded as the
   20 PASCAL VOC classes, so this trains on the taxonomy the rest of the repo
   already uses (personnel / two_wheeler / light_vehicle / heavy_vehicle).
2. Augmentations come from transforms.py instead of albumentations, which is
   not in requirements.txt here -- and whose current releases removed
   IAAAffine, so the pipeline as written in the video no longer runs on a fresh
   install regardless. transforms.py keeps albumentations' calling convention,
   so swapping back is a one-line change if you install it.

Override anything at the shell with the YOLOV3_* environment variables.
"""
import os
from pathlib import Path

import torch
import yaml

from yolov3 import transforms as T

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"

# --------------------------------------------------------------------------- #
#  dataset
# --------------------------------------------------------------------------- #
DATA_YAML = Path(os.getenv("YOLOV3_DATA", BASE_DIR / "data" / "battlesight.yaml"))


def _resolve(entry, root):
    """A YAML split may be one path or a list of them; always return a list."""
    entries = entry if isinstance(entry, list) else [entry]
    return [root / e if not Path(e).is_absolute() else Path(e) for e in entries]


with open(DATA_YAML, "r", encoding="utf-8") as f:
    _data = yaml.safe_load(f)

# `path:` is relative to the ultralytics datasets_dir, which is G:/fusionsight/datasets.
_root = Path(_data.get("path", "."))
DATASET_ROOT = _root if _root.is_absolute() else DATASETS_DIR / _root

TRAIN_IMG_DIRS = _resolve(_data["train"], DATASET_ROOT)
VAL_IMG_DIRS = _resolve(_data.get("val", _data["train"]), DATASET_ROOT)
TEST_IMG_DIRS = _resolve(_data.get("test", _data.get("val", _data["train"])), DATASET_ROOT)

_names = _data["names"]
CLASS_NAMES = [_names[i] for i in sorted(_names)] if isinstance(_names, dict) else list(_names)
NUM_CLASSES = len(CLASS_NAMES)

# --------------------------------------------------------------------------- #
#  training
# --------------------------------------------------------------------------- #
DEVICE = os.getenv("YOLOV3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.getenv("YOLOV3_SEED", "42"))
# Windows spawns dataloader workers instead of forking, so every worker re-imports
# this module and re-reads the YAML. 4 is fine; 0 if you are debugging a worker.
NUM_WORKERS = int(os.getenv("YOLOV3_WORKERS", "4"))
# 61.5M params at 416px: batch 8 is what fits the 8 GB 4060 alongside nothing else.
BATCH_SIZE = int(os.getenv("YOLOV3_BATCH", "8"))
IMAGE_SIZE = int(os.getenv("YOLOV3_IMGSZ", "416"))
LEARNING_RATE = float(os.getenv("YOLOV3_LR", "1e-4"))
WEIGHT_DECAY = float(os.getenv("YOLOV3_WD", "1e-4"))
NUM_EPOCHS = int(os.getenv("YOLOV3_EPOCHS", "100"))
CONF_THRESHOLD = float(os.getenv("YOLOV3_CONF", "0.05"))   # low on purpose: mAP wants recall
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
PIN_MEMORY = True

S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]   # 13, 26, 52 grid

CHECKPOINT_DIR = BASE_DIR / "weights"
CHECKPOINT_FILE = Path(os.getenv("YOLOV3_CKPT", CHECKPOINT_DIR / "yolov3_scratch.pth.tar"))
LOAD_MODEL = os.getenv("YOLOV3_LOAD", "0") not in ("0", "false", "False")
SAVE_MODEL = True

# --------------------------------------------------------------------------- #
#  anchors
# --------------------------------------------------------------------------- #
# Width/height as a fraction of the image, ordered coarse grid (13x13, large
# objects) -> fine grid (52x52, small ones), 3 anchors per scale.
#
# These are k-means anchors over the 343,204 VisDrone training boxes, not the
# PASCAL VOC ones from the video -- see anchors.py. It matters a lot here: the
# largest VOC anchor is 0.9 x 0.78 of the frame, while VisDrone's *median*
# object has a sqrt(area) of 0.0215 of the frame and even the 99th percentile
# is 0.13. On VOC anchors nearly every aerial target lands on the single
# smallest anchor and the width/height head has to regress a correction an
# order of magnitude away from its prior. Measured mean IoU of a training box
# with its best anchor: 0.653 with these, versus 0.458 with the VOC set.
#
# Regenerate for a different dataset with:
#     python -m yolov3.anchors --data data/battlesight_multi.yaml
ANCHORS = [
    [(0.1512, 0.1727), (0.0809, 0.1172), (0.0662, 0.0553)],
    [(0.0363, 0.0830), (0.0374, 0.0336), (0.0193, 0.0495)],
    [(0.0207, 0.0193), (0.0104, 0.0289), (0.0063, 0.0129)],
]

# --------------------------------------------------------------------------- #
#  augmentation
# --------------------------------------------------------------------------- #
scale = 1.2

train_transforms = T.Compose(
    [
        T.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        T.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
        ),
        T.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.1, p=0.4),
        T.OneOf(
            [
                T.ShiftScaleRotate(rotate_limit=20, p=1.0),
                T.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=1.0),
            ],
            p=0.5,
        ),
        # No vertical flip: aerial targets have a consistent "up", and the rest
        # of the repo trains with flipud=0 for the same reason.
        T.HorizontalFlip(p=0.5),
        T.Blur(p=0.1),
        T.CLAHE(p=0.1),
        T.Posterize(p=0.1),
        T.ToGray(p=0.1),
        T.ChannelShuffle(p=0.05),
        T.NormalizeToTensor(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
    ],
    min_visibility=0.4,
)

test_transforms = T.Compose(
    [
        T.LongestMaxSize(max_size=IMAGE_SIZE),
        T.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE),
        T.NormalizeToTensor(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
    ],
    min_visibility=0.4,
)
