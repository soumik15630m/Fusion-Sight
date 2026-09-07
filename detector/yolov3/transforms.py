"""A small, dependency-free stand-in for the albumentations pipeline.

The video's config.py builds its augmentations with albumentations, which is not
installed in this checkout (and whose current releases dropped IAAAffine, so that
config no longer runs as written anyway). These ops mirror the same pipeline on
top of cv2 + numpy, which the project already depends on, and keep
albumentations' calling convention:

    out = transform(image=np_hwc_uint8, bboxes=[[x, y, w, h, cls], ...])
    out["image"], out["bboxes"]

Boxes are YOLO format -- normalised centre-x, centre-y, width, height -- exactly
as dataset.py hands them over. Internally each box is carried as
(x1, y1, x2, y2, cls, ref_area) in absolute pixels; ref_area is the box's area
before any cropping, so min_visibility can be judged at the end.

If albumentations *is* installed, config.py prefers it; this module is the
fallback, and it is what runs on this machine today.
"""
import random

import cv2
import numpy as np
import torch

X1, Y1, X2, Y2, CLS, REF = range(6)


# --------------------------------------------------------------------------- #
#  box helpers
# --------------------------------------------------------------------------- #
def _to_abs(bboxes, h, w):
    """[cx, cy, bw, bh, cls] normalised  ->  (N, 6) absolute corners."""
    out = np.zeros((len(bboxes), 6), dtype=np.float32)
    for i, b in enumerate(bboxes):
        cx, cy = float(b[0]) * w, float(b[1]) * h
        bw, bh = float(b[2]) * w, float(b[3]) * h
        out[i] = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, float(b[4]), bw * bh)
    return out


def _to_yolo(boxes, h, w, min_visibility):
    """(N, 6) absolute corners -> list of [cx, cy, bw, bh, cls] normalised."""
    result = []
    for b in boxes:
        bw, bh = b[X2] - b[X1], b[Y2] - b[Y1]
        if bw <= 1 or bh <= 1:
            continue
        if b[REF] > 0 and (bw * bh) / b[REF] < min_visibility:
            continue
        cx, cy = (b[X1] + b[X2]) / 2, (b[Y1] + b[Y2]) / 2
        result.append(
            [
                float(np.clip(cx / w, 0.0, 1.0)),
                float(np.clip(cy / h, 0.0, 1.0)),
                float(np.clip(bw / w, 0.0, 1.0)),
                float(np.clip(bh / h, 0.0, 1.0)),
                float(b[CLS]),
            ]
        )
    return result


def _clip(boxes, h, w):
    if len(boxes) == 0:
        return boxes
    boxes[:, X1] = np.clip(boxes[:, X1], 0, w)
    boxes[:, X2] = np.clip(boxes[:, X2], 0, w)
    boxes[:, Y1] = np.clip(boxes[:, Y1], 0, h)
    boxes[:, Y2] = np.clip(boxes[:, Y2], 0, h)
    keep = (boxes[:, X2] - boxes[:, X1] > 1) & (boxes[:, Y2] - boxes[:, Y1] > 1)
    return boxes[keep]


# --------------------------------------------------------------------------- #
#  geometric ops
# --------------------------------------------------------------------------- #
class LongestMaxSize:
    """Resize so the longest side equals max_size, keeping the aspect ratio."""

    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, img, boxes):
        h, w = img.shape[:2]
        s = self.max_size / max(h, w)
        if s != 1.0:
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
            new_wh = (max(1, int(round(w * s))), max(1, int(round(h * s))))
            img = cv2.resize(img, new_wh, interpolation=interp)
            if len(boxes):
                boxes[:, :4] *= s
                boxes[:, REF] *= s * s
        return img, boxes


class PadIfNeeded:
    """Centre-pad up to (min_height, min_width) with a constant border."""

    def __init__(self, min_height, min_width, value=0):
        self.min_height, self.min_width, self.value = min_height, min_width, value

    def __call__(self, img, boxes):
        h, w = img.shape[:2]
        dh, dw = max(0, self.min_height - h), max(0, self.min_width - w)
        if dh or dw:
            top, left = dh // 2, dw // 2
            img = cv2.copyMakeBorder(
                img, top, dh - top, left, dw - left,
                cv2.BORDER_CONSTANT, value=(self.value,) * 3,
            )
            if len(boxes):
                boxes[:, [X1, X2]] += left
                boxes[:, [Y1, Y2]] += top
        return img, boxes


class RandomCrop:
    def __init__(self, height, width):
        self.height, self.width = height, width

    def __call__(self, img, boxes):
        h, w = img.shape[:2]
        if h < self.height or w < self.width:
            img, boxes = PadIfNeeded(max(h, self.height), max(w, self.width))(img, boxes)
            h, w = img.shape[:2]
        top = random.randint(0, h - self.height)
        left = random.randint(0, w - self.width)
        img = img[top: top + self.height, left: left + self.width]
        if len(boxes):
            boxes[:, [X1, X2]] -= left
            boxes[:, [Y1, Y2]] -= top
            boxes = _clip(boxes, self.height, self.width)
        return img, boxes


class ShiftScaleRotate:
    """Random affine: translate, scale and rotate about the image centre."""

    def __init__(self, shift_limit=0.06, scale_limit=0.1, rotate_limit=20, p=0.5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.p = p

    def __call__(self, img, boxes):
        if random.random() > self.p:
            return img, boxes
        h, w = img.shape[:2]
        angle = random.uniform(-self.rotate_limit, self.rotate_limit)
        scale = 1 + random.uniform(-self.scale_limit, self.scale_limit)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += random.uniform(-self.shift_limit, self.shift_limit) * w
        M[1, 2] += random.uniform(-self.shift_limit, self.shift_limit) * h
        img = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        if len(boxes):
            # Warp all four corners; the new axis-aligned box is their extent.
            corners = np.stack(
                [
                    boxes[:, [X1, Y1]], boxes[:, [X2, Y1]],
                    boxes[:, [X2, Y2]], boxes[:, [X1, Y2]],
                ],
                axis=1,
            )  # (N, 4, 2)
            ones = np.ones((corners.shape[0], corners.shape[1], 1), dtype=np.float32)
            warped = np.concatenate([corners, ones], axis=2) @ M.T.astype(np.float32)
            boxes[:, X1] = warped[:, :, 0].min(axis=1)
            boxes[:, X2] = warped[:, :, 0].max(axis=1)
            boxes[:, Y1] = warped[:, :, 1].min(axis=1)
            boxes[:, Y2] = warped[:, :, 1].max(axis=1)
            boxes = _clip(boxes, h, w)
        return img, boxes


class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, boxes):
        if random.random() > self.p:
            return img, boxes
        w = img.shape[1]
        img = np.ascontiguousarray(img[:, ::-1])
        if len(boxes):
            x1 = boxes[:, X1].copy()
            boxes[:, X1] = w - boxes[:, X2]
            boxes[:, X2] = w - x1
        return img, boxes


# --------------------------------------------------------------------------- #
#  photometric ops (boxes pass through untouched)
# --------------------------------------------------------------------------- #
class ColorJitter:
    def __init__(self, brightness=0.6, contrast=0.6, saturation=0.6, hue=0.1, p=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, img, boxes):
        if random.random() > self.p:
            return img, boxes
        out = img.astype(np.float32)
        if self.brightness:
            out *= 1 + random.uniform(-self.brightness, self.brightness)
        if self.contrast:
            mean = out.mean()
            out = (out - mean) * (1 + random.uniform(-self.contrast, self.contrast)) + mean
        out = np.clip(out, 0, 255).astype(np.uint8)
        if self.saturation or self.hue:
            hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
            if self.hue:
                hsv[..., 0] = (hsv[..., 0] + random.uniform(-self.hue, self.hue) * 180) % 180
            if self.saturation:
                hsv[..., 1] = np.clip(
                    hsv[..., 1] * (1 + random.uniform(-self.saturation, self.saturation)), 0, 255
                )
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return out, boxes


class Blur:
    def __init__(self, blur_limit=7, p=0.1):
        self.blur_limit, self.p = blur_limit, p

    def __call__(self, img, boxes):
        if random.random() > self.p:
            return img, boxes
        k = random.choice(list(range(3, self.blur_limit + 1, 2)))
        return cv2.blur(img, (k, k)), boxes


class CLAHE:
    def __init__(self, clip_limit=4.0, p=0.1):
        self.clip_limit, self.p = clip_limit, p

    def __call__(self, img, boxes):
        if random.random() > self.p:
            return img, boxes
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), boxes


class Posterize:
    def __init__(self, num_bits=4, p=0.1):
        self.num_bits, self.p = num_bits, p

    def __call__(self, img, boxes):
        if random.random() > self.p:
            return img, boxes
        shift = 8 - self.num_bits
        return ((img >> shift) << shift), boxes


class ToGray:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img, boxes):
        if random.random() > self.p:
            return img, boxes
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB), boxes


class ChannelShuffle:
    def __init__(self, p=0.05):
        self.p = p

    def __call__(self, img, boxes):
        if random.random() > self.p:
            return img, boxes
        order = list(range(3))
        random.shuffle(order)
        return img[..., order], boxes


class OneOf:
    def __init__(self, ops, p=1.0):
        self.ops, self.p = ops, p

    def __call__(self, img, boxes):
        if random.random() > self.p:
            return img, boxes
        return random.choice(self.ops)(img, boxes)


# --------------------------------------------------------------------------- #
#  output
# --------------------------------------------------------------------------- #
class NormalizeToTensor:
    """Scale to [0, 1] (mean 0 / std 1, as in the video's config) and go CHW."""

    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0):
        self.mean = np.array(mean, dtype=np.float32) * max_pixel_value
        self.std = np.array(std, dtype=np.float32) * max_pixel_value

    def __call__(self, img, boxes):
        out = (img.astype(np.float32) - self.mean) / self.std
        return torch.from_numpy(out.transpose(2, 0, 1)).contiguous(), boxes


class Compose:
    def __init__(self, ops, min_visibility=0.4):
        self.ops = ops
        self.min_visibility = min_visibility

    def __call__(self, image, bboxes):
        h, w = image.shape[:2]
        boxes = _to_abs(bboxes, h, w)
        for op in self.ops:
            image, boxes = op(image, boxes)
            if isinstance(image, torch.Tensor):
                h, w = image.shape[1], image.shape[2]
            else:
                h, w = image.shape[:2]
        return {"image": image, "bboxes": _to_yolo(boxes, h, w, self.min_visibility)}
