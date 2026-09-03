"""Training loop for the from-scratch YOLOv3.

    python -m yolov3.train --epochs 100 --batch 8
    python -m yolov3.train --data data/battlesight_multi.yaml --resume

Differences from the video's train.py, all of them things that break on this
stack rather than preferences:

* `torch.cuda.amp.autocast()` / `GradScaler()` are deprecated in torch 2.x and
  warn on every step; the `torch.amp` spellings with an explicit device are used
  instead, and AMP switches itself off on CPU rather than erroring.
* Gradients are clipped. A from-scratch YOLOv3 with lambda_noobj=10 can take a
  single bad step early on and never recover; clipping is cheap insurance.
* The best checkpoint is kept separately from the last one, chosen on val mAP,
  so a run that overfits late does not overwrite the model worth keeping.
* Everything is logged to runs/yolov3/<name>/, matching where the ultralytics
  side of this repo puts its runs.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.optim as optim

from yolov3 import config
from yolov3.loss import YoloLoss
from yolov3.model import YOLOv3
from yolov3.utils import (
    check_class_accuracy,
    get_evaluation_bboxes,
    get_loaders,
    load_checkpoint,
    mean_average_precision,
    save_checkpoint,
    seed_everything,
    tqdm,
)

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, device, epoch):
    loop = tqdm(train_loader, leave=True, desc=f"epoch {epoch}")
    losses = []
    use_amp = scaler is not None and scaler.is_enabled()

    for x, y in loop:
        x = x.to(device, non_blocking=True)
        y0, y1, y2 = (y[0].to(device), y[1].to(device), y[2].to(device))

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        loop.set_postfix(loss=sum(losses) / len(losses))

    return sum(losses) / len(losses)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", help="dataset YAML (default: data/battlesight.yaml)")
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    p.add_argument("--imgsz", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--name", default="yolov3_scratch")
    p.add_argument("--resume", action="store_true", help="continue from the run's last.pth.tar")
    p.add_argument("--eval-every", type=int, default=3, help="epochs between mAP evaluations")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # config reads these at import time, so anything overridden on the command
    # line has to be written back before the loaders are built.
    config.BATCH_SIZE = args.batch
    config.NUM_WORKERS = args.workers
    config.DEVICE = args.device
    if args.imgsz != config.IMAGE_SIZE:
        raise SystemExit(
            f"--imgsz {args.imgsz} needs the anchors and grid rebuilt; set "
            f"YOLOV3_IMGSZ={args.imgsz} in the environment instead so config "
            f"derives S and the transforms from it."
        )

    device = args.device
    seed_everything(config.SEED)

    run_dir = config.BASE_DIR / "runs" / "yolov3" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt, best_ckpt = run_dir / "last.pth.tar", run_dir / "best.pth.tar"

    print(f"device={device}  classes={config.CLASS_NAMES}  imgsz={config.IMAGE_SIZE}")
    print(f"run dir: {run_dir}")

    model = YOLOv3(num_classes=config.NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp and device != "cpu")

    train_loader, test_loader, train_eval_loader = get_loaders()
    print(f"{len(train_loader.dataset)} train / {len(test_loader.dataset)} val images")

    start_epoch, best_map = 0, 0.0
    if args.resume or config.LOAD_MODEL:
        ckpt_path = last_ckpt if last_ckpt.exists() else config.CHECKPOINT_FILE
        if not Path(ckpt_path).exists():
            raise SystemExit(f"nothing to resume from: {ckpt_path} does not exist")
        ckpt = load_checkpoint(ckpt_path, model, optimizer, args.lr, device=device)
        start_epoch = ckpt.get("epoch", -1) + 1
        best_map = ckpt.get("best_map", 0.0)
        print(f"resumed at epoch {start_epoch} (best mAP so far {best_map:.4f})")

    # Anchors are stored as a fraction of the image; the loss works in cell
    # units, so each scale's anchors are multiplied by that scale's grid size.
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        mean_loss = train_fn(
            train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, device, epoch
        )
        epoch_time = time.time() - t0
        record = {"epoch": epoch, "loss": mean_loss, "seconds": round(epoch_time, 1)}

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=last_ckpt, epoch=epoch, best_map=best_map)

        if args.eval_every and epoch % args.eval_every == 0 and epoch > 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD, device=device)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
                device=device,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            ).item()
            record["mAP50"] = mapval
            print(f"epoch {epoch}  loss {mean_loss:.4f}  mAP50 {mapval:.4f}  {epoch_time:.0f}s")

            if mapval > best_map:
                best_map = mapval
                save_checkpoint(
                    model, optimizer, filename=best_ckpt, epoch=epoch, best_map=best_map
                )
                print(f"=> new best mAP50 {best_map:.4f}")
        else:
            print(f"epoch {epoch}  loss {mean_loss:.4f}  {epoch_time:.0f}s")

        history.append(record)
        with open(run_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"done. best mAP50 {best_map:.4f}; weights in {run_dir}")


if __name__ == "__main__":
    main()
