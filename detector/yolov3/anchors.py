"""k-means anchor boxes over a dataset's own labels.

The 9 anchors in the video are PASCAL VOC's, where objects fill much of the
frame. Aerial imagery is the opposite extreme, so reusing them cripples the
width/height head: it starts every regression from a prior an order of
magnitude too large and has to learn a near-constant correction instead of a
shape.

Distance is 1 - IoU(box, centroid) rather than Euclidean, because Euclidean
distance on (w, h) is dominated by the large boxes, which are exactly the ones
this dataset barely has.

    python -m yolov3.anchors                                  # config's dataset
    python -m yolov3.anchors --data data/battlesight_multi.yaml --clusters 9
"""
import argparse
from pathlib import Path

import numpy as np


def iou_wh(boxes, centroids):
    """(N, 2) x (K, 2) -> (N, K) IoU, boxes assumed centred on each other."""
    w = np.minimum(boxes[:, None, 0], centroids[None, :, 0])
    h = np.minimum(boxes[:, None, 1], centroids[None, :, 1])
    inter = w * h
    area_b = (boxes[:, 0] * boxes[:, 1])[:, None]
    area_c = (centroids[:, 0] * centroids[:, 1])[None, :]
    return inter / (area_b + area_c - inter + 1e-12)


def kmeans(boxes, k, max_iter=300, seed=0):
    rng = np.random.default_rng(seed)
    centroids = boxes[rng.choice(len(boxes), k, replace=False)]
    last = None

    for _ in range(max_iter):
        assignment = iou_wh(boxes, centroids).argmax(axis=1)
        if last is not None and (assignment == last).all():
            break
        for j in range(k):
            member = boxes[assignment == j]
            if len(member):
                centroids[j] = member.mean(axis=0)
            else:  # a cluster can die; reseed it on a random box
                centroids[j] = boxes[rng.integers(len(boxes))]
        last = assignment

    mean_iou = iou_wh(boxes, centroids).max(axis=1).mean()
    return centroids, mean_iou


def collect_boxes(img_dirs):
    from yolov3.dataset import IMG_EXTENSIONS, _read_label, label_path_for

    wh = []
    for d in img_dirs:
        for p in sorted(Path(d).iterdir()):
            if p.suffix.lower() not in IMG_EXTENSIONS:
                continue
            for box in _read_label(label_path_for(p)):
                if box[2] > 0 and box[3] > 0:
                    wh.append((box[2], box[3]))
    return np.array(wh, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="dataset YAML (default: whatever config uses)")
    parser.add_argument("--clusters", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.data:
        import os

        os.environ["YOLOV3_DATA"] = str(Path(args.data).resolve())

    from yolov3 import config

    print(f"reading labels under {[str(d) for d in config.TRAIN_IMG_DIRS]} ...")
    boxes = collect_boxes(config.TRAIN_IMG_DIRS)
    print(f"{len(boxes):,} boxes")

    pct = np.percentile(np.sqrt(boxes[:, 0] * boxes[:, 1]), [50, 90, 99])
    print(f"sqrt(area) as a fraction of the image -- p50 {pct[0]:.4f} "
          f"p90 {pct[1]:.4f} p99 {pct[2]:.4f}")

    centroids, mean_iou = kmeans(boxes, args.clusters, seed=args.seed)
    print(f"mean IoU of a box with its best anchor: {mean_iou:.4f}")

    # Largest first: scale 0 is the 13x13 grid, which handles big objects.
    centroids = centroids[np.argsort(-(centroids[:, 0] * centroids[:, 1]))]
    per_scale = len(centroids) // 3

    print("\nANCHORS = [")
    for s in range(3):
        row = centroids[s * per_scale: (s + 1) * per_scale]
        print("    [" + ", ".join(f"({w:.4f}, {h:.4f})" for w, h in row) + "],")
    print("]")


if __name__ == "__main__":
    main()
