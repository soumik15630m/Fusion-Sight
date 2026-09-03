"""Inference for the from-scratch YOLOv3: image, folder, video or webcam.

    python -m yolov3.detect --weights runs/yolov3/yolov3_scratch/best.pth.tar --source tests/assets/bus.jpg
    python -m yolov3.detect --weights ... --source v1.mp4 --out v1_yolov3.mp4
    python -m yolov3.detect --weights ... --source 0            # webcam

Letterboxing is undone before boxes are drawn: the network sees a padded square,
but the boxes have to land on the original frame. Coordinates are also printed
normalised to the original frame, matching what app/routers/detect.py returns,
so this is directly comparable with the ultralytics side of the repo.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from yolov3 import config
from yolov3.model import YOLOv3
from yolov3.utils import cells_to_bboxes, non_max_suppression

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Distinct, readable BGR colours; cycled if a dataset has more classes.
PALETTE = [
    (56, 56, 255), (49, 210, 207), (10, 249, 72), (255, 157, 151),
    (255, 112, 31), (207, 84, 255), (26, 147, 52), (151, 157, 255),
]


def letterbox(frame_rgb, size):
    """Resize longest side to `size` and centre-pad. Returns (img, scale, pad)."""
    h, w = frame_rgb.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame_rgb, (nw, nh), interpolation=interp)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.zeros((size, size, 3), dtype=resized.dtype)
    out[top: top + nh, left: left + nw] = resized
    return out, s, (left, top)


@torch.no_grad()
def predict(model, frame_bgr, anchors, device, conf, iou, imgsz):
    """Run one frame. Returns [(cls, score, x1, y1, x2, y2)] in original pixels."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    padded, scale, (pad_x, pad_y) = letterbox(frame_rgb, imgsz)

    tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float().div(255.0)
    tensor = tensor.unsqueeze(0).to(device)

    outputs = model(tensor)

    boxes = []
    for i in range(3):
        # Anchors are fractions of the image; the decoder wants them in cell
        # units, so scale by this head's grid size (which follows imgsz).
        S = outputs[i].shape[2]
        anchor = torch.tensor(anchors[i], device=device).float() * S
        boxes += cells_to_bboxes(outputs[i], anchor, S=S)[0]

    boxes = non_max_suppression(boxes, iou_threshold=iou, threshold=conf, box_format="midpoint")

    # Undo the letterbox: padded-square 0-1 -> original-frame pixels.
    results = []
    for cls, score, x, y, bw, bh in boxes:
        x1 = (x - bw / 2) * imgsz - pad_x
        y1 = (y - bh / 2) * imgsz - pad_y
        x2 = (x + bw / 2) * imgsz - pad_x
        y2 = (y + bh / 2) * imgsz - pad_y
        h, w = frame_bgr.shape[:2]
        results.append(
            (
                int(cls),
                float(score),
                float(np.clip(x1 / scale, 0, w)),
                float(np.clip(y1 / scale, 0, h)),
                float(np.clip(x2 / scale, 0, w)),
                float(np.clip(y2 / scale, 0, h)),
            )
        )
    return results


def draw(frame, detections, class_names):
    for cls, score, x1, y1, x2, y2 in detections:
        colour = PALETTE[cls % len(PALETTE)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, colour, 2)
        name = class_names[cls] if cls < len(class_names) else str(cls)
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), colour, -1)
        cv2.putText(
            frame, label, (p1[0] + 2, p1[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame


def load_model(weights, device):
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(device)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="image, folder, video, or a webcam index")
    p.add_argument("--out", help="output file (video) or folder (images)")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=config.NMS_IOU_THRESH)
    p.add_argument("--imgsz", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--show", action="store_true", help="display frames as they are processed")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    model = load_model(args.weights, device)
    names = config.CLASS_NAMES

    source = args.source
    is_webcam = source.isdigit()
    src_path = Path(source)

    # ---- images ---------------------------------------------------------- #
    if not is_webcam and src_path.exists() and src_path.suffix.lower() in IMG_EXTENSIONS:
        images = [src_path]
    elif not is_webcam and src_path.is_dir():
        images = sorted(p for p in src_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    else:
        images = None

    if images is not None:
        out_dir = Path(args.out or config.BASE_DIR / "runs" / "yolov3" / "predict")
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in images:
            frame = cv2.imread(str(p))
            if frame is None:
                print(f"skipping unreadable {p}")
                continue
            t0 = time.time()
            dets = predict(model, frame, config.ANCHORS, device, args.conf, args.iou, args.imgsz)
            ms = (time.time() - t0) * 1000
            h, w = frame.shape[:2]
            print(f"{p.name}: {len(dets)} detections in {ms:.0f} ms")
            for cls, score, x1, y1, x2, y2 in dets:
                print(
                    f"  {names[cls] if cls < len(names) else cls:<14} {score:.3f}  "
                    f"[{x1 / w:.4f} {y1 / h:.4f} {x2 / w:.4f} {y2 / h:.4f}]"
                )
            cv2.imwrite(str(out_dir / p.name), draw(frame, dets, names))
        print(f"=> wrote annotated images to {out_dir}")
        return

    # ---- video / webcam --------------------------------------------------- #
    cap = cv2.VideoCapture(int(source) if is_webcam else source)
    if not cap.isOpened():
        raise SystemExit(f"could not open source {source!r}")

    writer = None
    if args.out:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames, total_ms = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.time()
            dets = predict(model, frame, config.ANCHORS, device, args.conf, args.iou, args.imgsz)
            total_ms += (time.time() - t0) * 1000
            frames += 1
            frame = draw(frame, dets, names)

            if writer:
                writer.write(frame)
            if args.show:
                cv2.imshow("yolov3", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if frames % 30 == 0:
                print(f"{frames} frames, {total_ms / frames:.1f} ms/frame")
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"=> wrote {args.out}")
        if args.show:
            cv2.destroyAllWindows()

    if frames:
        print(f"{frames} frames at {total_ms / frames:.1f} ms/frame "
              f"({1000 * frames / total_ms:.1f} fps)")


if __name__ == "__main__":
    main()
