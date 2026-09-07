# BattleSight AR / FusionSight — Detailed Project Report

**Inspection date:** 2026-08-31 (evening). All numbers below were read directly
from files in this checkout (`G:\fusionsight`) or from live process/GPU state
at inspection time — none are copied from memory of a prior session without
re-verification. Where a number could not be independently confirmed, it is
marked as such.

**Status at inspection time: a training run is actively in progress** (see
§6). This document only reads files; nothing was started, stopped, or
modified to produce it.

---

## 1. What this is

A YOLO26-based object detector for drone/helmet-cam situational-awareness
footage, served over FastAPI (`app/`). It performs single-frame detection,
multi-feed live tracking over WebSocket with a moving/static motion filter,
and can launch/monitor/cancel its own training jobs via the API. Declared
scope (per prior session notes): real-time person/vehicle detection feeding
an AR/HUD display for a human operator — situational awareness, not
automated fire-control or classification of intent.

Four trained classes: `personnel`, `two_wheeler`, `light_vehicle`,
`heavy_vehicle`.

## 2. Environment (verified live)

Read directly from the venv interpreter and `nvidia-smi` at inspection time:

| Component | Verified value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB total |
| GPU driver | 610.88 |
| torch | 2.13.0+cu126, `torch.cuda.is_available()` → **True** |
| CUDA (torch build) | 12.6 |
| Python | 3.12.10 (`.venv`) |
| ultralytics | 8.4.135 (a newer 8.4.136 is available on PyPI, not installed) |
| fastapi | 0.141.1 |
| uvicorn | 0.52.4 |
| pydantic | 2.13.5 |
| opencv-python | 5.0.0.93 |
| numpy | 2.5.2 |

Full pinned list is in `requirements.txt` (55 packages).

**GPU state at inspection:** `nvidia-smi` reports **5535 MiB used / 8188 MiB
total, 28% utilization** — consistent with an active training job (see §6),
not an idle server. 14 `python.exe` processes were running concurrently
(main trainer + dataloader workers), matching `workers=4` plus the API
process's own children.

## 3. Layout

```
fusionsight/
├─ .venv/                     torch 2.13.0+cu126
├─ app/                       config.py, detector.py, motion_filter.py,
│                             exclusion.py, trainer.py, main.py, schemas.py,
│                             routers/{detect,track,train,exclude}.py
├─ data/                      battlesight.yaml, battlesight_multi.yaml
├─ datasets/                  VisDrone/, WiderPerson/, WiderPerson_raw/, coco8/
├─ scripts/                   10 scripts (train, remap, diagnose, bench, etc.)
├─ tests/                     6 test files + tests/assets/ (4 clips + 1 image)
├─ runs/detect/<name>/        training + validation output (9 subfolders)
├─ weights/                   best.pt (deployed), best_probe_1epoch.pt, yolo26n.pt
├─ logs/                      2 training console logs
├─ README.md                  20.6 KB — primary engineering log/reference
├─ progrssreport1.txt          prior session's narrative progress note
└─ start.txt                   a queued manual command with resume instructions
```

Root also holds several demo clips and their annotated outputs (`v1.mp4`,
`v1_annotated.mp4`, `v2.mp4`, `v2_annotated.mp4`, `v3.mp4`, `v3_annotated.mp4`,
`v1_annotated_ego.mp4`, one WhatsApp-sourced clip and its annotated version) —
these are visual QA artifacts from prior sessions, not part of the served
system.

A `yolov3/` package (193 KB, 9 files: `model.py`, `loss.py`, `train.py`,
`dataset.py`, `anchors.py`, `transforms.py`, `utils.py`, `config.py`,
`detect.py`) also exists in the repo root. Per prior-session notes this was a
from-scratch YOLOv3 implementation built in response to pasted reference
code and is **not** the system in production use — the deployed detector is
the Ultralytics YOLO26 pipeline under `app/` and `weights/`. It is dead code
relative to the served product; nothing currently imports from it.

## 4. Dataset (image counts verified by direct directory listing)

`data/battlesight.yaml` (VisDrone only, 4-class remap of 10 VisDrone
classes):

| split | declared in YAML | actual files on disk |
|---|---|---|
| train | 6471 | **6474** |
| val | 548 | **551** |
| test | 1610 | **1613** |

(The +3 discrepancy in each split is consistent and small — likely non-image
files such as `.cache` companions or a handful of extra images added after
the YAML comment was written; not investigated further as it does not affect
training, which scans the actual directory.)

Class remap (from `README.md`, `scripts/remap_visdrone.py`), train-split box
counts:

| id | class | source VisDrone classes | train boxes |
|---|---|---|---|
| 0 | `personnel` | pedestrian, people | 106,396 |
| 1 | `two_wheeler` | bicycle, tricycle, awning-tricycle, motor | 48,185 |
| 2 | `light_vehicle` | car, van | 169,823 |
| 3 | `heavy_vehicle` | truck, bus | 18,801 |

`heavy_vehicle` is both the rarest class by a wide margin (18,801 vs. 169,823
for `light_vehicle`, ~9×) and — per §5 — the weakest by mAP.

`data/battlesight_multi.yaml` (VisDrone + WiderPerson, personnel-only
contribution from WiderPerson):

| dataset | split | files on disk |
|---|---|---|
| WiderPerson | train | **8003** |
| WiderPerson | val | **1003** |

Combined train/val used by the *currently running* training job (confirmed
from its own log, §6): **14,471 train / 1,548 val** images — this equals
6,471 (VisDrone declared) + 8,003 (WiderPerson actual with 3 extra files) ≈
14,474, and the log's actual scanned count of 14,471 / 1,548 is what the
dataloader used. WiderPerson contributes personnel-class boxes only;
`flipud` (vertical-flip augmentation) is conceptually invalid for its
ground-level imagery — see the discrepancy noted in §6.

Disk footprint (measured with `du`, VisDrone completed before the check was
cut off by a slow-storage warning also visible in the training log —
`Slow image access detected ... read: 21.0±15.3 MB/s ... local storage
recommended`, suggesting `G:\` may be a mounted/network volume):

| path | size |
|---|---|
| `datasets/VisDrone` | 3.8 GB |
| `datasets/WiderPerson_raw` | 1020 MB |
| `datasets/WiderPerson` | 723 MB |
| `datasets/coco8` | 489 KB |
| **repo total (`G:\fusionsight`)** | **11 GB** |

(`WiderPerson_raw` is the original download; `WiderPerson` is the converted
YOLO-format labels + copied images per `scripts/convert_widerperson.py` —
keeping both roughly doubles WiderPerson's footprint on disk.)

## 5. Detection accuracy — verified from `runs/diagnose.json`

This file (produced by `scripts/diagnose.py`, a checkpoint/resolution sweep
against `data/battlesight.yaml`'s VisDrone-only 548-image val set) contains
four full evaluation runs. Read directly, values below are exact
transcriptions:

| weights | imgsz | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|
| `weights/best.pt` (1-epoch probe, was deployed) | 640 | 0.2475 | 0.1009 | 0.3371 | 0.2922 |
| `weights/best.pt` (1-epoch probe) | 1280 | 0.2638 | 0.1115 | 0.3454 | 0.3136 |
| `battlesight_v1/weights/best.pt` (13-epoch) | 640 | 0.4331 | 0.2308 | 0.5798 | 0.4182 |
| **`battlesight_v1/weights/best.pt` (13-epoch, now deployed)** | **1280** | **0.5046** | **0.2772** | **0.5857** | **0.4855** |

Per-class breakdown for the currently deployed configuration
(`battlesight_v1` 13-epoch checkpoint at imgsz 1280):

| class | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| personnel | 0.6352 | 0.4874 | 0.5337 | 0.2249 |
| two_wheeler | 0.5225 | 0.3302 | 0.3435 | 0.1361 |
| light_vehicle | 0.6845 | 0.7958 | **0.8073** | 0.5362 |
| heavy_vehicle | 0.5005 | 0.3287 | 0.3338 | 0.2116 |

`light_vehicle` is comfortably the strongest class; `two_wheeler` and
`heavy_vehicle` are both well behind and close to each other.

**Confidence threshold sweep** (from `app/config.py` comment, itself sourced
from a val-set F1 sweep — the raw per-conf sweep file was not located
separately from the comment, so this table is taken from the documented
values in the deployed config rather than re-derived from a raw log):

| conf | mean F1 (4 classes) | precision | recall |
|---|---|---|---|
| 0.25 (deployed) | 0.499 | 0.697 | 0.412 |
| 0.35 (previous default) | 0.447 | 0.790 | 0.343 |
| 0.50 | 0.367 | 0.873 | 0.265 |

## 6. Current training run — LIVE, in progress at inspection time

**A training job is running right now.** Verified two ways: (a)
`nvidia-smi` shows 5535 MiB used / 28% utilization, far above an idle
inference server's ~1.5–2 GB; (b) 14 `python.exe` processes are alive; (c)
the newest log file (`logs/battlesight_v1_20260831_182850.log`, last write
18:35) ends mid-run with no completion banner.

**What it is, exactly, per its own log and `runs/detect/battlesight_v1/args.yaml`:**

- Resumed from `runs\detect\battlesight_v1\weights\last.pt` at **epoch 14 of
  30** (started 18:28:50, epoch 14 completed 18:35:40).
- `data = data\battlesight_multi.yaml` — **the combined VisDrone + WiderPerson
  set**, not VisDrone alone. Model: YOLO26s, 260 layers, 9,950,960 params,
  22.8 GFLOPs. `imgsz=640`, `batch=8`.
- `optimizer=auto` **overrode** the requested `lr0=0.002` and picked AdamW at
  `lr=0.00125` automatically — the requested learning rate was not actually
  used.
- **`flipud=0.5`** (default), not `0`. `start.txt` (a note left for this
  exact resume) explicitly says to pass `--flipud 0` when mixing in
  WiderPerson's ground-level imagery, since vertically flipping a standing
  person is not a realistic augmentation the way it is for top-down drone
  footage. `--resume` in Ultralytics reuses the checkpoint's saved
  hyperparameters rather than accepting new ones, so this run is
  vertically flipping ground-level pedestrian images throughout. This is a
  real, currently-active discrepancy between documented intent and what is
  executing — flagged here, not corrected, per instruction to inspect only.

**Epoch-by-epoch history so far**, from `runs/detect/battlesight_v1/results.csv`
(all 14 rows present in the file, transcribed exactly):

| epoch | precision | recall | mAP50 | mAP50-95 | box_loss | cls_loss |
|---|---|---|---|---|---|---|
| 1 | 0.4746 | 0.3669 | 0.3672 | 0.1774 | 2.1550 | 1.8834 |
| 2 | 0.5128 | 0.3836 | 0.3989 | 0.2026 | 2.0495 | 1.6447 |
| 3 | 0.5047 | 0.3751 | 0.3909 | 0.2046 | 2.0246 | 1.5891 |
| 4 | 0.5324 | 0.3992 | 0.4187 | 0.2203 | 2.0036 | 1.5467 |
| 5 | 0.5403 | 0.4099 | 0.4322 | 0.2297 | 1.9785 | 1.5015 |
| 6 | 0.5682 | 0.4071 | 0.4394 | 0.2402 | 1.9562 | 1.4717 |
| 7 | 0.5799 | 0.4130 | 0.4500 | 0.2484 | 1.9427 | 1.4436 |
| 8 | 0.5762 | 0.4258 | 0.4621 | 0.2551 | 1.9347 | 1.4239 |
| 9 | 0.5944 | 0.4278 | 0.4667 | 0.2591 | 1.9144 | 1.4035 |
| 10 | 0.5990 | 0.4367 | 0.4734 | 0.2698 | 1.9082 | 1.3928 |
| 11 | 0.6140 | 0.4438 | 0.4849 | 0.2703 | 1.8953 | 1.3665 |
| 12 | 0.6243 | 0.4492 | 0.4957 | 0.2786 | 1.8835 | 1.3601 |
| 13 | 0.6256 | 0.4517 | 0.4966 | 0.2821 | 1.8707 | 1.3483 |
| **14 (latest, in progress run)** | **0.6247** | **0.4441** | **0.4936** | **0.2804** | 1.8671 | 1.3378 |

**Important caveat on comparing these numbers to §5:** this per-epoch mAP is
measured against the **1,548-image combined VisDrone+WiderPerson val set**,
not the 548-image VisDrone-only val set that produced the 0.5046 headline
number in `runs/diagnose.json`. The two are **not directly comparable** —
this run's 0.4936 at epoch 14 (imgsz 640) is not an apples-to-apples
regression or improvement versus the deployed model's 0.5046 (imgsz 1280,
VisDrone-only val). A fair comparison requires re-running `diagnose.py` /
`eval_rubric.py` against the same val set and imgsz once this run finishes or
is checkpointed.

Training also appears to be plateauing epoch-to-epoch (13→14: mAP50 0.4966 →
0.4936, precision and recall both ticked down slightly) rather than still
climbing steeply, in contrast to the earlier note (from before this resume)
that "mAP still climbing" at epoch 13. `patience=20` means it will not
early-stop for a long while yet (16 epochs of no improvement allowed), so it
will very likely run all the way to epoch 30 unless stopped.

`max_det=902` appears in this run's training/validation args — a training-time
NMS cap, distinct from and much higher than the deployed inference-time
`MAX_DET=500` in `app/config.py`; the two are independent settings for
different phases and this is not a misconfiguration, just worth not
conflating.

## 7. Deployed inference configuration (`app/config.py`, verified by reading the file)

| setting | value | source/rationale documented in-file |
|---|---|---|
| `MODEL_PATH` | `weights/best.pt` | env override `BATTLESIGHT_MODEL` |
| `DEVICE` | `"0"` (GPU 0) | env override `BATTLESIGHT_DEVICE` |
| `CONF_THRESHOLD` | 0.25 | F1 sweep, §5 |
| `IOU_THRESHOLD` | 0.5 | — |
| `MAX_DET` | 500 | VisDrone val frames hold up to 317 objects; default 300 truncated dense frames |
| `IMGSZ` | 1280 | median object 11 px at 640; measured mAP50 0.4331→0.5046 |
| `MOTION_GATED` | **off** (`0`) | motion-gated path found 0.0 det/frame on a static-scene pan vs 26.5 ungated |
| `EGO_COMPENSATION` | on (`1`) | cancels global 2D transform for panning cameras |
| `EGO_STATIC_SHIFT` | 0.3 px | was 1.0 px; too coarse for handheld jitter |
| `EGO_MAX_RESIDUAL` | 0.02 (2%) | above this the motion pass is dropped for that frame, falls back to full-frame classification |
| `MOTION_MIN_BLOB_AREA` | 80 px² | was 30 px²; too small a floor let phantom blobs through |
| `MOTION_CLAIM_IOU` | 0.1 | scored as `max(intersection/blob_area, IoU)`, not IoU alone |
| `MAX_TRACKS_PER_SOURCE` | 512 | bounded LRU, prevents unbounded growth on long-lived feeds |

## 8. API surface (`app/routers/`, verified by file listing + README)

| endpoint | file | purpose |
|---|---|---|
| `GET /health` | — | model loaded, device, classes, training active |
| `POST /detect` | `detect.py` | single frame, stateless |
| `POST /detect/tracked` | `track.py` | frame-in-stream, track IDs + motion |
| `DELETE /detect/state/{source_id}` | `track.py` | drop a feed's tracker/history |
| `WS /ws/track/{source_id}` | `track.py` | live feed, JSON per frame |
| `POST /train`, `GET /train`, `GET /train/{id}`, `POST /train/{id}/cancel` | `train.py` | training job lifecycle |
| exclusion endpoints | `exclude.py` | reference-image exclusion store |

## 9. Fixes applied this project (from `README.md` §"Deviations", 16 items — condensed)

All verified present as committed behavior in the current code (not
re-verified line-by-line against source in this pass, since README.md is
itself the engineering log for these and was written by measurement at the
time — flagged here as a secondary source, not independently re-derived):

1. `project=` path-doubling bug in `train.py` fixed.
2. `_is_moving()` numpy.bool_ → JSON serialization fix (WebSocket-only bug).
3. Per-feed trackers — shared single tracker was contaminating track IDs
   across feeds.
4. Subprocess encoding fix — cp1252 decode crash was hanging every
   API-launched training job permanently.
5. Unbuffered subprocess output — log tail / current epoch were frozen
   without it.
6. ANSI escape stripping — epoch regex wasn't matching, `current_epoch`
   stuck at 0.
7. `_class_name()` checkpoint-aware — avoids `IndexError` on stock COCO
   checkpoints.
8. Blocking inference moved to threadpool — `/health` was blocked for the
   duration of every streamed frame; now ~1.2 ms vs ~25 ms/frame contention.
9. Detector lock-guarded — prevented tracker-swap races across feeds.
10. Motion history bounded LRU (512) — was an unbounded per-track dict, a
    slow leak on long-running feeds.
11. `trainer.start()` lock across check-and-spawn — verified 4 simultaneous
    POSTs yield exactly one 202 and three 409s.
12. Cancel kills the whole process tree — verified 16 processes mid-training
    drop to 2 after cancel, 4212→1551 MiB GPU.
13. `best_weights` scraped from real `save_dir` — was reporting stale paths
    on reused run names.
14. `is_training()` checks the process is actually alive.
15. WebSocket hardening — model-loaded check + `finally`-block cleanup.
16. Clear error on missing model file, in both serving and `--resume`.

## 10. Latency / throughput (from README, sourced from `scripts/bench_imgsz.py` and live WS tests — not re-run in this inspection pass)

| scenario | value |
|---|---|
| training, `yolo26s`, batch 8, imgsz 640 | 149–162 s/epoch, peak 5539 MiB VRAM |
| inference, 1080p frame, imgsz 1280 | 24–57 ms |
| tracked path (motion ungated), imgsz 1280 | ~85–96.7 ms/frame, ~10 fps |
| tracked path (old, motion gated) | 28.3 ms/frame but 0.0 detections/frame on a static-camera pan |
| `/health` under an active feed | ~1.2 ms |
| WS round trip steady-state, loopback | ~100 ms (≈10 fps); first ~2 s up to ~200 ms (GPU clock ramp-up) |
| glass-to-overlay, 1080p q85 @ 50 Mbps | ~190 ms total, 100% detections kept |
| glass-to-overlay, 720p q70 @ 50 Mbps | ~125 ms total, 74% detections kept |

**Note:** these were measured in prior sessions per README.md and were not
re-run live in this inspection (a training job currently holds ~4 GB of the
8 GB GPU, so a fresh benchmark right now would not reflect steady-state
serving-only performance).

## 11. Test coverage (verified by file listing, not by running the suite — running tests would compete with the live training job for the same GPU)

| file | what it covers |
|---|---|
| `tests/make_clips.py` | builds synthetic static/moving clips |
| `tests/test_history_cap.py` | bounded LRU motion history |
| `tests/test_feed_isolation.py` | two feeds get independent track IDs (needs API running) |
| `tests/test_exclusion.py` | reference-image exclusion store |
| `tests/test_motion_coherence.py` | motion straightness/coherence gate |
| `tests/test_motion_gating.py` | motion-gated vs ungated detection path |
| `tests/assets/` | `bus.jpg`, `drone_pan.mp4`, `moving.mp4`, `static.mp4` — the only real test fixtures on disk |

No automated tests were executed as part of producing this report.

## 12. Outstanding / known-incomplete work (cross-checked against current state)

- **Training incomplete, and currently running.** As of this inspection it
  is at epoch 14/30 of a resumed run (§6), on the combined
  VisDrone+WiderPerson set, with mAP50 essentially flat epoch-13-to-14
  (0.4966→0.4936) rather than clearly still climbing. Whether it clears the
  deployed model's bar is not yet known and cannot be, until it finishes and
  is evaluated on a matching val set/imgsz.
- **The active run is flipping ground-level WiderPerson images vertically**
  (`flipud=0.5`), contrary to the explicit `--flipud 0` guidance left in
  `start.txt` for this exact scenario — a side effect of Ultralytics
  `--resume` reusing the checkpoint's original hyperparameters rather than
  accepting the new ones passed on the command line.
- **Drones cannot be detected at all.** VisDrone is footage taken *from*
  drones, not *of* them, and WiderPerson has no aerial imagery either — no
  threshold or resolution change fixes this; it needs UAV-labelled data and
  a 5th class. Not started.
- **Buses are diluted into `heavy_vehicle`** together with trucks — the
  rarest (18,801 train boxes) and per §5 one of the two weakest classes
  (mAP50 0.334). A split would require re-running `scripts/remap_visdrone.py`
  and full retraining; not done.
- **Confident off-distribution false positives** — documented example: a
  desk/highlighter scene drew a `light_vehicle` box at 0.32–0.42 confidence.
  Attributed to zero indoor/close-range negative imagery in training data;
  not a threshold problem.
- **Motion filter is validated only on synthetic clips** (`tests/make_clips.py`
  outputs) plus a handful of real handheld/drone demo clips in the repo
  root, not on a broad real-world corpus. `MOTION_THRESHOLD` is explicitly
  flagged in README as needing tuning against real feeds before being
  trusted operationally.
- **`allow_origins=["*"]`** in the API CORS config — flagged in README as
  needing tightening before any deployment off the local machine. Not
  independently re-checked against `app/main.py` in this pass.
- **Job/training state is in-memory** in the FastAPI process — restarting
  the API loses job history (a live training subprocess would survive but
  become unmonitored/orphaned from the API's perspective).

## 13. Summary judgment (for someone reading only this document)

The served system (4-class YOLO26 detector + FastAPI tracking service) is
functional and its accuracy figures are independently verified from
`runs/diagnose.json`: **mAP50 0.5046 / mAP50-95 0.2772 / recall 0.4855** on a
548-image VisDrone-only validation set, at the currently deployed
configuration (13-epoch `battlesight_v1` checkpoint, imgsz 1280, conf 0.25).
`light_vehicle` is strong (mAP50 0.807); `two_wheeler` and `heavy_vehicle`
are the weak classes (mAP50 0.34 / 0.33). Sixteen concrete bugs across the
serving/tracking/training-orchestration stack were found and fixed, each
backed by a measured before/after in README.md.

The single largest lever on the headline number — finishing training past
epoch 13 — **is in progress right now** as this document is written, but
early evidence (epoch 14 essentially flat vs. epoch 13) does not yet show
the further gain the project is counting on, and the run's val set differs
from the one the deployed model was scored against, so its result will need
a like-for-like re-evaluation before being trusted. Drone detection and
bus-specific detection remain entirely unimplemented (data problems, not
tuning problems). Nothing in this report was altered by inspecting it — no
files, weights, or the running training job were touched.
