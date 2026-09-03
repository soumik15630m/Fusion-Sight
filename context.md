# FusionSight / BattleSight AR — session context

**Written 2026-09-01, end of session. For a new agent/chat picking this up cold.**

Read this file first, then `README.md` §16 for the measurements behind it.
`start.txt` is the operator's runbook (how to launch and resume training).

> **This session corrected two factual errors in the project's own notes.**
> `summary.txt` and `latest.txt` describe `v10.mp4` and `v11.mp4` incorrectly,
> and README §§14–15 were tuned against those wrong descriptions. §16 supersedes
> them. Do not re-derive conclusions from §§14–15 without reading §16 first.

---

## 0. 2026-09-02 update — the fpv run finished and IS PROMOTED

> **Later the same day**, two further changes landed on top of the promotion:
> a personnel recall change and a motion-filter discriminator. Both are below
> (§0a, §0b) and both are measured. README §18 has the full tables.

### 0a. Personnel floor 0.20 → 0.10 (ground view)

The old note in `config.py` said dropping this floor "buys only ~2 points of
recall". **That was an artefact of the val set, not the model.** The sweep
behind it ran on the BLENDED val, where VisDrone's aerial personnel dominate by
instance count and their P/R curve genuinely is flat — which masked the
ground-level headroom underneath. Split by domain:

| WiderPerson val (ground-level, 1,000 imgs) | P | R | F1 | F2 |
|---|---|---|---|---|
| 0.10 (**adopted**) | 0.657 | **0.725** | 0.689 | 0.710 |
| 0.20 (was) | 0.814 | 0.636 | 0.714 | 0.665 |

Ground-level personnel mAP50 is **0.757**, better than the blended 0.706 and
than aerial's 0.605 — the model is stronger at ground-level people than the
headline number suggests.

Chosen on **F2, not F1**: F1 weights precision and recall equally, which
contradicts §9 ("a suppressed real contact is the failure this system must never
have"). F2 peaks at 0.08; 0.10 is within 0.004 of that while recovering 5.7
points of precision, so 0.10 is the knee. Drone view stays at 0.25 — aerial
precision collapses to 0.431 at 0.10.

Effect on the clip sweep: personnel totals up 15–42% on every clip, with
`moving_object` and vehicle counts unchanged. Also feeds tracking, since
`_model_conf_floor()` hands the lowest floor to the model and ByteTrack's second
association pass is what keeps a briefly-visible person on a track.

### 0b. Lights rejected, split-second targets reported

The motion channel could not separate a light (whose bright region drifts and
reads as coherent travel) from a person, and could not report anyone visible for
less than `MOTION_COHERENCE_MIN_POINTS` = 4 frames **at all** — not a tuning
problem, the required evidence outlasted the event.

`MotionDetector._structure_score` now decides from one frame: align the previous
frame, then test whether the change is a uniform brightness shift or preserves
structure under z-scoring. Illumination does both; a real object does neither.
**Fails open** on anything it cannot judge. A **fast path** then reports 2-point
blobs that clear a stricter structure bar and real displacement — deliberately
NOT using `_straightness`, which is 1.0 by construction at 2 points. And the
**ego-residual degraded band** stops the all-or-nothing drop that was blinding
the channel on 535/1068 frames of v6.

Headline: **v10 449 → 34** `moving_object` (−92%), **v11 2707 → 2105** with
bursts 65 → 43, **v6 blind frames 535 → 70** with its verified-real count held
(344 → 326). **v5 16 → 0 was verified visually, not assumed** — sodium-lit night
courtyard, every rejected blob on empty pavement or a shadow edge, never on a
person. 9.5–31% of motion detections now come via the fast path.

Still blind: v2/v9 (residual far above 2× the threshold, degraded band never
reaches them) and the static-camera `mog2_fraction` drop, left alone on purpose.


`battlesight_fpv` completed all 8 epochs (interrupted after epoch 3, resumed
cleanly with `--resume` exactly as §1 describes — the resume path is now proven
on a real interruption, not just a test).

`eval_rubric.py` vs the then-deployed `weights/best.pt`: **PASS on all four
criteria**, so it was promoted.

| | old `best.pt` | new (deployed) |
|---|---|---|
| overall mAP50 | 0.5910 | **0.6274** |
| personnel mAP50 | 0.6785 | **0.7064** |
| recall | 0.540 | **0.570** |
| precision | 0.669 | **0.716** |

Per-class mAP50: personnel 0.706, two_wheeler 0.494, light_vehicle 0.860,
heavy_vehicle 0.449. **No class collapsed** — that was the specific risk the
AerialPerson vehicle pseudo-labelling (§5) existed to prevent, and it held.

**Deployed state now:**

| file | what |
|---|---|
| `weights/best.pt` | the fpv checkpoint |
| `weights/best.engine` | rebuilt FP16, 256 MB |
| `weights/best_pre_fpv.pt` | rollback checkpoint |
| `weights/best_pre_fpv.engine` | rollback engine — keeps rollback a file copy, not a 3-min rebuild |
| `weights/drone_best.*` | **also the fpv checkpoint** (see below) |
| `weights/drone_best_pre_fpv.pt` / `.engine` | drone rollback pair |

The served FP16 engine was validated against the checkpoint, because the rubric
only ever measures the `.pt` while production serves the engine:
mAP50 0.6252 vs 0.6274, mAP50-95 identical at 0.384, every class within 0.002.
That is FP16 rounding, not degradation (contrast INT8, §9). The engine is also
**~2x faster than the `.pt`** — 14.8 ms vs 28.8 ms inference.

`weights/best_int8.engine` is now **stale** (built from the old checkpoint). It
is not loaded by default, so nothing breaks, but rebuild or delete it before
anyone experiments with INT8 again.

**Drone view was promoted too, on its own measurement.** `eval_rubric.py` with
`--baseline weights/drone_best.pt` on the same val: **PASS**, and by a far wider
margin than the ground comparison — overall mAP50 0.5036 -> 0.6274, and
**personnel 0.3162 -> 0.7064**, more than double, on VisDrone val, which is drone
view's own home domain. Honest caveat: `drone_best.pt` was trained at imgsz 640
and judged here at 1280, so part of that gap is the train/serve mismatch this run
existed to close — but 1280 is what production serves at, so this is the
deployment-realistic comparison.

So `weights/best.pt` and `weights/drone_best.pt` are now **byte-identical**
(md5 62398e6f15ede0dbf51e1882b5c0accc), with separately-built engines. The two
views still behave differently, but now only through config — `MOTION_COHERENCE_
THRESHOLD_DRONE` 0.85 vs `_GROUND` 0.5, and `CONF_THRESHOLD_PERSONNEL_GROUND`
0.20 — not through different weights. **Worth knowing:** both engines are loaded
simultaneously by `Detector.load()`, so an identical model is held in GPU memory
twice (~380 MB of avoidable overhead on an 8 GB card). Deleting
`weights/drone_best.*` would make `_model_for` fall back to the default model and
give identical detections for less memory, at the cost of the ability to let the
two views diverge again later. Not done — flagged as a judgement call.

**The v10 question is now answered: NO, and the answer is more interesting than
a failure.** Measured with stateless `Detector.detect(view="drone")` over all 374
frames, old drone checkpoint vs new:

| v10.mp4 | old `drone_best` | new |
|---|---|---|
| `personnel` | 245 (in 132 frames, 35.3%) | **0** |
| `light_vehicle` | 385 | **58** (−85%) |
| `heavy_vehicle` | 20 | **0** |

Those 245 old `personnel` boxes were **all false positives on vegetation**.
Rendered and inspected directly: frame 186 has a person plainly visible, prone in
the open, and the old model put 8 small `personnel` boxes (conf 0.26–0.53)
scattered across the magenta foliage with **not one on the person**. The new
model returns zero detections on that frame — it stopped hallucinating people in
the grass, but it still does not find the person.

So the run **removed ~592 false positives on this clip and gained no true
positives**. That is a real improvement (a phantom contact is exactly what §9
says must not be normalised), but it is not what the run was for.

This is precisely what §7 predicted before the run: *"nothing in AerialPerson
shows a prone or crawling person, which is the specific v10 failure. Expect
partial help."* The prediction was right, and the remaining gap is now narrower
and better characterised — it is **pose**, not palette, viewpoint or resolution.
The new model is trained on false-colour EO/IR-like augmentation and on aerial
personnel, and it still misses a prone body from above. Closing this needs
training data containing prone/crawling people seen from a UAV; no dataset
currently in the mix has it.

---

---

### 0c. The 40 ms budget: profile before tuning

Asked to improve accuracy at distance under a 40 ms/frame ceiling. Measuring
found the ceiling was **already blown before any of today's work** — and that
the cause was structural, not a threshold anywhere.

**Recall collapses with target size.** Ground-level personnel, imgsz 1280:
`<16 px = 0.187`, `16-32 = 0.622`, `32-48 = 0.805`, `>96 = 0.940`. 41% of all
people are under 32 px and fewer than half are found. All the loss is far-field.

**Two obvious levers are dead — do not retry them:**
- **Resolution does not help.** 1280→1536 buys +0.4% detections for +46% cost,
  and the larger buckets get *worse*: above the trained size the model is
  off-distribution for its own scale priors.
- **NMS tuning is impossible.** YOLO26 reports `end2end: True` — it is NMS-free,
  and `iou=` is silently ignored. Sweeping it gave byte-identical results at
  every value. `config.IOU_THRESHOLD` never affected model inference in this
  project; it governs only this repo's own IoU arithmetic. Crowd/occlusion
  behaviour is **learned**, so only training changes it.

**The real finding — `predict()` between `track()` calls corrupts the
predictor**, costing ~60% of detections silently. Bisected by running the
far-field pass with every box discarded (v5 personnel 1560 → 562, i.e. the
`predict()` call alone did it). Restoring `predictor.trackers` did NOT fix it;
the interference is broader. `_gated_detections` and `_detect_on_crops` both
call `predict()` and would both hit this if `MOTION_GATED` were ever enabled.

**The win — the CPU motion stage now runs beside the GPU pass.** It was 23.6 ms
of a 52.2 ms frame, running in sequence while the GPU idled, despite nothing on
the model path reading `blobs` until the very end. Overlapping them:
**52 → ~25 ms median, p90 under 30**, comfortably inside the budget.

Verified as a **pure latency change**: detection counts byte-identical
(v2 914/914, v5 1560/1560) and the whole 10-clip burst sweep byte-identical in
every column. `BATTLESIGHT_MOTION_PARALLEL=0` reverts it.

**Far-field second pass** (`FARFIELD_*`, `_far_field_pass`) — the cheap pass
picks the region where the small boxes already are and spends one extra
inference there: +4-8% more people on video, +4.4% on val. **OFF by default**,
not for cost but for jitter (p90 48.5 ms vs 26.9 median — an AR overlay would
stutter). A smaller tile would fix that, but the `--static` engine has a fixed
1280 input and anything else crashes; it needs a second engine built at the
smaller size. Good for OFFLINE use (`annotate_video.py`) where latency is free.

**Health warning:** absolute latency on this machine varied ~3x on unchanged
code today (25 to 74 ms for the same path). An ablation even measured the
structure test as *slower when disabled* — impossible, and proof the noise
exceeded the effect. Trust warmed, back-to-back, within-run comparisons only.

## 1. The two commands

Run from `G:\fusionsight` with `.\.venv\Scripts\Activate.ps1` active.
**Stop uvicorn first** — batch 4 at imgsz 1280 peaks at 7.51 GB of 8 GB.

### Start

```powershell
python scripts\train.py `
    --model weights\best.pt `
    --data data\battlesight_fpv.yaml `
    --aug-profile fpv `
    --imgsz 1280 --batch 4 --epochs 8 `
    --lr0 0.002 --flipud 0 `
    --name battlesight_fpv
```

`python scripts\prepare_training.py` prints this exact command and re-checks the
data first. ~51 min/epoch × 8 ≈ **7 hours**.

### Resume after any interruption

```powershell
python scripts\train.py --resume --name battlesight_fpv
```

That is the whole command. Do **not** re-pass `--data`, `--imgsz`, `--epochs` or
`--aug-profile` — ultralytics restores all of them from the checkpoint.

Verified end-to-end on this machine (killed a run mid-epoch, resumed it):
it printed `Resuming training ... from epoch 2 to 3 total epochs`, kept the
optimizer state and `best_fitness`, and restored all 10 fpv augmentation
transforms with exact parameters despite `--aug-profile` not being passed.

Three caveats:
- **Epoch 1 is unprotected.** `last.pt` is written at the END of each epoch, so
  nothing exists to resume from during the first ~51 minutes. Just restart.
- **If it died of CUDA OOM, lower the batch on resume:** `--resume --name
  battlesight_fpv --batch 2`. `batch` is one of the few resume-overridable keys.
- **`--flipud` is NOT resume-overridable.** It comes back from the checkpoint.
  Harmless here (the run starts at 0), but this is the exact quirk that gave the
  project its upside-down-people bug — never rely on it to fix a bad run.

### After training — do not promote blindly

```powershell
python scripts\eval_rubric.py runs\detect\battlesight_fpv\weights\best.pt `
    --baseline weights\best.pt --data data\battlesight_fpv.yaml --imgsz 1280
```

Only promote on PASS, and keep a way back:

```powershell
copy weights\best.pt weights\best_pre_fpv.pt
copy runs\detect\battlesight_fpv\weights\best.pt weights\best.pt
$env:PYTHONPATH="."
python scripts\export_engine.py --model weights\best.pt --static   # or serving uses the OLD weights
```

The TensorRT engine step is not optional: `Detector.load()` prefers
`weights/best.engine` over the `.pt`, so skipping it silently serves the old model.

---

## 2. What this project is

Ultralytics YOLO26 detector served over FastAPI (`app/`) for a drone/helmet-cam
AR situational-awareness overlay — **human operator support, explicitly not
fire-control or automated targeting**.

Four classes: `personnel`, `two_wheeler`, `light_vehicle`, `heavy_vehicle`, plus
a class-agnostic `moving_object` (class_id −1) from the motion filter.

Two checkpoints, both loaded at once, selected per request by `view`:
- `weights/best.pt` — "ground" (default). As of 2026-09-02 this is the
  `battlesight_fpv` checkpoint: VisDrone + VisDrone test-dev + WiderPerson +
  AerialPerson, fine-tuned at imgsz 1280 with the `fpv` augmentation profile (§0).
- `weights/drone_best.pt` — "drone", VisDrone only

Both are actually served as **TensorRT FP16 engines** (`weights/*.engine`).
`MOTION_GATED` is **off** and must stay off (it goes blind to stationary targets).

Detection coordinates are **normalised 0–1** in the API — a detail that will
silently produce invisible boxes if you forget it while drawing overlays.

---

## 3. The central finding of this session

**`v10.mp4` and `v11.mp4` are UAV/FPV combat footage, not the generic YouTube
content the older notes claim.**

- `v10` — a drone watching **personnel moving through vegetation**, false-colour
  EO/IR palette (grass renders magenta), HUD plus burned-in subtitles.
- `v11` — an **FPV drone** low over open terrain, 564×480 analog, fisheye, heavy
  compression, HUD with dotted reticle columns and a telemetry string.

### 3a. v11's 54,658 phantom boxes were the HUD

31.3 `light_vehicle` per frame on a clip with no vehicles. A spatial heatmap of
those boxes reproduced the overlay exactly — one box per reticle dash, one per
telemetry character, median box 9×9 px. §14's diagnosis ("no training coverage
of this camera angle") and its fix ("add BDD100K/KITTI") were both wrong: the
boxes were never on the scene.

Note **rolling-median burst detection reports zero bursts** on that series,
because it is a constant per-frame error rather than a spike. Do not rely on
burst counts alone to find this class of bug.

Fixed by `app/overlay_mask.py` (new). A HUD glyph is painted on the sensor and
holds image position while the world slides past; a real object is attached to
the world. So it learns which cells keep producing **small** detections while
the camera is **established to be moving** (reusing the motion pass's ego
estimate). Never masks a large box; **fails open** past `OVERLAY_MAX_FRACTION`.

| | before | after |
|---|---|---|
| v11 `light_vehicle` | 54,658 (31.3/frame) | **6,713 (3.84/frame)** −87.7% |
| v10 every class | — | **byte-identical** |

### 3b. Personnel detection genuinely fails on the real domain

On `v10` the system boxes vegetation and misses every visible person — frame 300
has two people plainly visible and returns **zero** detections. Ruled out by
measurement, not assumption:

- **Not resolution** — the people are 50–60 px, not tiny.
- **Not our fine-tuning** — a stock COCO `yolo26s.pt` finds 1 of ~5.
- **Not recoverable from motion** — at *either* drone coherence threshold (0.85
  or 0.5) the motion filter tags only the burned-in subtitles, never the people;
  they crawl too slowly to clear `MOTION_COHERENCE_MIN_PATH`. **§15's
  drone/ground coherence reasoning does not affect v10 personnel recall at all.**

What remains is pose, viewpoint and palette. That is a training-data gap, which
is what the prepared run addresses.

---

## 4. What changed on disk

**New files**
| path | what |
|---|---|
| `app/overlay_mask.py` | HUD/OSD rejection (§3a) |
| `scripts/diagnose_bursts.py` | regression harness — **was missing**, recreated |
| `scripts/fpv_augment.py` | `fpv` augmentation profile |
| `scripts/convert_aerialperson.py` | AerialPerson → YOLO layout |
| `scripts/pseudo_label_vehicles.py` | vehicle pseudo-labels (§5, important) |
| `scripts/prepare_training.py` | data check + writes yaml + prints command |
| `tests/test_overlay_mask.py` | 10 assertions, all passing |
| `data/battlesight_fpv.yaml` | generated; do not hand-edit |
| `context.md`, `start.txt` | this file, and the runbook |

**Modified**: `app/config.py` (`OVERLAY_*`, `MOTION_CHRONIC_COOLDOWN`),
`app/detector.py` (overlay stage + reset), `app/motion_filter.py` (cooldown),
`scripts/train.py` (`--aug-profile`, adaptive `warmup_epochs`/`close_mosaic`,
`--patience`), `README.md` (§16, §16g, "Still outstanding"), `requirements` env
(albumentations 2.0.8 installed).

**`v7.mp4` frame 261 is FIXED** — `MOTION_CHRONIC_COOLDOWN=3`. This was the open
item that EMA smoothing made worse. The distinction that matters: smoothing made
the decision *slower*; this gate needed the decision to *last* longer. Both v7
bursts gone; v2/v3/v4/v5/v6/v9 identical; v1 improved 10→4.

---

## 5. Training data — read before touching the dataset

Final prepared set:

| | images | boxes |
|---|---|---|
| train | 18,694 | 925,650 |
| val | 1,548 | 66,112 |

Train sources: VisDrone train (6,471) + **VisDrone test-dev (1,610 — fully
labelled and previously unused by both train and val, free data)** + WiderPerson
train (8,000) + AerialPerson train (2,613).

### The trap, already fixed — do not undo it

**AerialPerson (Zenodo 7740081, CC-BY-4.0) labels PEOPLE ONLY**, but its
top-down aerial imagery is full of parking lots — the *same viewpoint VisDrone
teaches vehicles from*. Measured with `drone_best.pt` over 40 train images:
**97.3 `light_vehicle` per image, none labelled** → ~258,000 unlabelled vehicles
presented to the trainer as confirmed negatives. That would have destroyed the
vehicle classes, surfacing only after a 7-hour run as an unexplained
`light_vehicle` collapse.

`scripts/pseudo_label_vehicles.py` fixed it: **263,700 vehicle pseudo-labels**
merged into train, all **61,708 human person boxes preserved**, 86 dropped for
overlapping a real person, conf floor 0.5, originals backed up to
`labels/<split>.orig/` (`--restore` undoes it). Visually spot-checked.

**This is a class of bug, not a one-off:** any single-class dataset merged into a
multi-class taxonomy turns its unlabelled objects into confident negatives.
**WiderPerson has the same shape** (ground-level street scenes, personnel-only
labels, unlabelled traffic) and has been in the mix since
`battlesight_multi.yaml`. Weaker there (ground cars ≠ aerial cars), but it is a
prime suspect if vehicle metrics ever look inexplicably poor.

### Why AerialPerson's val split is excluded

It is train-only on purpose. Its val labels were **restored to clean person-only
ground truth** and left out of the yaml, because scoring vehicles against
person-only labels penalises every correct car, and pseudo-labelling val would be
circular (those labels come from `drone_best.pt`, the rubric's own baseline).

Happy consequence: **val is now identical to `battlesight_multi.yaml`'s**, so
`eval_rubric.py` numbers stay directly comparable to every figure in README's
history. The 523 held-out on-domain images remain on disk with clean labels —
evaluate personnel on them explicitly with `classes=0`.

---

## 6. How to verify anything you change

```powershell
python scripts\diagnose_bursts.py --view ground          # all clips, served path
python scripts\diagnose_bursts.py v10.mp4 v11.mp4 --view drone
python scripts\diagnose_bursts.py --motion-only          # CPU only, safe during training
```

Run it after **any** motion-filter or threshold change. Two caveats: `--motion-only`
counts run higher than the served path (`_claim_motion_blobs` removes
YOLO-covered blobs), and burst *counts* can rise when a fix lowers totals,
because the rolling-median baseline falls with them — read totals and max/frame
alongside.

Tests (no pytest installed; run directly):
```powershell
python tests\test_overlay_mask.py
python tests\test_motion_structure.py
python tests\test_motion_coherence.py
$env:PYTHONPATH="."; python tests\test_history_cap.py
python tests\test_exclusion.py
```
`test_feed_isolation.py` needs a live uvicorn.

**Success for the training run is not mAP50.** It is whether
`python scripts\annotate_video.py v10.mp4 --view drone` puts boxes on the
*people*. The val set does not contain that case — which is exactly how this
stayed invisible for so long.

---

## 7. Still open

- **Personnel in vegetation from UAV — STILL OPEN, and now precisely located.**
  The fpv run (§0) did NOT fix it. It removed every vegetation false positive on
  v10 (245 phantom `personnel` → 0, `light_vehicle` 385 → 58) but gained zero true
  positives: frame 186 shows a prone person in the open and the model returns
  nothing. Palette, viewpoint, resolution and augmentation are now all ruled out
  by this run — what is left is **pose**. Needs UAV imagery of prone/crawling
  people; nothing currently in the mix has it.
- **Burned-in subtitles tagged as `moving_object`** on v10 — genuinely changing
  pixels fixed in image space. The overlay mask exempts `moving_object` by
  design. The same attachment argument would apply if it matters.
- **No drone/UAV-as-target class.** VisDrone is footage *from* drones, not *of*
  them. Needs UAV-labelled data and a 5th class.
- **Buses diluted** into `heavy_vehicle` with trucks.
- **`two_wheeler` confused for `light_vehicle`** under blur/low light (v6 frames
  517/735 — visible as 3 residual bursts). The fpv profile targets this.
- **v9's ego-residual check drops 224/225 frames** as parallax-unreliable, so
  class-agnostic motion tagging is blind on it. Under-detection, not a burst.
- **Tracked throughput ~10 fps** at imgsz 1280. Lowering imgsz does not help —
  fixed per-frame overhead dominates.

---

## 8. Environment gotchas that cost real time

- **8 GB GPU, and 1280 does not truly fit.** At batch 4 ultralytics reports
  ~10.2 G *reserved* — it only works by spilling into shared system RAM under
  Windows WDDM, which is why it runs at 1.6 it/s. Do not raise the batch.
- **Windows multiprocessing deadlock**: any one-off `model.val()`/`predict()`
  script needs `if __name__ == "__main__":` **and** `workers=0`, or it hangs
  forever. The tell is process memory staying perfectly static.
- **`annotate_video.py` / `annotate_screen.py` exit 139 (segfault)** during CUDA
  teardown *after* printing "Wrote N frames". Output is complete. Check for that
  line, not the exit code.
- **Both need `PYTHONPATH=.`** or to be run as `python scripts\x.py` from the repo root.
- **`onnxruntime-gpu` must be < 1.29** here (pin 1.20.2); newer needs CUDA 13 and
  silently falls back to CPU, then dies deep inside inference.
- **TensorRT export: use `--static`.** Dynamic shapes size the memory pool for
  batch 16 and OOM on this GPU.
- Piping a long-running command through `grep`/`tail` in this harness buffers all
  output until it exits — it looks hung when it is fine. Redirect to a log file
  and poll that instead.
- **`export_engine.py` takes `--model` as a FLAG, not a positional**, and needs
  `PYTHONPATH=.` like the annotate scripts. The form written in these notes
  before 2026-09-02 failed twice in a row: first `ModuleNotFoundError: No module
  named 'app'`, then `unrecognized arguments`. Correct form is in §1.
- **Never let `opencv-python-headless` into this venv.** It and `opencv-python`
  unpack into the SAME `cv2/` directory, so whichever installs last wins. The
  headless build has no highgui, and every preview window
  (`annotate_screen.py`, `annotate_video.py --show`) dies with
  `The function is not implemented`. **albumentations depends on headless**, so
  installing it for the `fpv` profile silently broke live capture. Fix:
  `pip uninstall -y opencv-python opencv-python-headless` then
  `pip install opencv-python==5.0.0.93` — uninstall BOTH first, they share files.
  Verify: `python -c "import cv2; print(cv2.getBuildInformation())" | findstr GUI`
  must say `WIN32UI`. Both packages are now pinned and commented in requirements.txt.
- `G:\` is intermittently slow; "Slow image access" warnings in training logs are
  usually not a real problem.

---

## 9. Standing judgement calls — do not silently reverse

- `MOTION_GATED` stays **off** (blind to stationary targets).
- FP16 is standard; **INT8 measured and rejected** (~4 pt mAP50 / 5 pt recall loss).
- The overlay mask **fails open** by design. A phantom contact is a nuisance; a
  suppressed real one is the failure this system must never have.
- **Detections-per-frame is not a quality proxy.** This model has put confident
  `light_vehicle` boxes on a pencil case. Look at annotated frames.
- Warm the GPU before timing — it idles at 270 MHz and short benchmarks report
  roughly double the true latency.
