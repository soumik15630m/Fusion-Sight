BattleSight AR — Progress Report 1
Session date: 2026-08-31

===============================================================================
1. ENVIRONMENT / MODEL VERIFICATION
===============================================================================
- Set Claude Code default model to Sonnet 5 (~/.claude/settings.json).
- Reviewed project layout (README.md) for the BattleSight AR detection service:
  YOLO26 fine-tuned on VisDrone (4 tactical classes: personnel, two_wheeler,
  light_vehicle, heavy_vehicle), served via FastAPI with single-frame
  detection, WebSocket live tracking, and a moving/static motion filter.
- Verified GPU/environment: torch 2.13.0+cu126, CUDA available, RTX 4060
  Laptop GPU (8 GB VRAM) — scripts\check_gpu.py passes.
- Confirmed the API server was already running and healthy:
  GET /health -> model_loaded: true, device 0, 4 classes loaded correctly,
  two feeds (drone-01, helmet-A) had prior tracking state.

===============================================================================
2. INTENDED USE CLARIFICATION
===============================================================================
- Discussed the system's purpose. Agreed scope: real-time person/vehicle
  detection feeding an AR/HUD display for a human operator (situational
  awareness), not an automated weapons/fire-control or enemy-classification
  system. Work in this session was scoped to the detection/tracking
  engineering only.

===============================================================================
3. LATENCY MEASUREMENT
===============================================================================
- Ran scripts\test_stream.py against tests\assets\drone_pan.mp4 over the live
  WebSocket endpoint (ws/track/drone-01).
- Result: ~48-63 ms round-trip per frame (JPEG encode + send + server
  decode/infer/track + JSON reply), i.e. ~16-20 fps for a single feed.
- Noted: GPU work serializes across multiple simultaneous feeds (one GPU),
  so N feeds share that throughput rather than each getting their own.

===============================================================================
4. DATASET RESEARCH
===============================================================================
- Discussed public datasets for generalizing person detection beyond
  VisDrone's sparse aerial traffic scenes: CrowdHuman, COCO (person class),
  MOT17/20, WiderPerson.
- Selected WiderPerson (dense, ground-level pedestrian scenes) to
  complement VisDrone's aerial-only imagery, for better generalization to
  helmet-cam / static-camera sources.
- WiderPerson dataset was manually downloaded and extracted to
  datasets\WiderPerson_raw\ (8000 train / 999 val images, per official
  train.txt / val.txt).

===============================================================================
5. NEW / MODIFIED FILES
===============================================================================
- scripts\convert_widerperson.py (NEW)
    Converts WiderPerson's raw annotations (Images/ + Annotations/) into
    YOLO-format labels, mapping pedestrian/rider/partially-visible classes
    into BattleSight class 0 (personnel). Ignore-regions and crowd boxes are
    dropped (no reliable per-instance box). Output written to
    datasets\WiderPerson\images|labels\{train,val}\.
    Already run successfully: 8000 train / 1000 val images converted.

- data\battlesight_multi.yaml (NEW)
    Merged dataset config: VisDrone (all 4 classes) + WiderPerson
    (personnel only), same class taxonomy as data\battlesight.yaml.
    BUG FOUND & FIXED: initial `path: .` resolved relative to the current
    working directory instead of the Ultralytics datasets_dir setting
    (Ultralytics only falls back to datasets_dir when the relative path
    does not already exist on its own, and "." always exists). Fixed to an
    absolute path: `path: G:/fusionsight/datasets`.

- scripts\train.py (MODIFIED)
    Added two CLI flags:
      --lr0     (default 0.01) — initial learning rate; use a lower value
                 (e.g. 0.002) when fine-tuning from an existing checkpoint
                 rather than training from scratch.
      --flipud  (default 0.5) — vertical flip augmentation probability;
                 was hardcoded, tuned for top-down drone imagery only. Set
                 to 0 when mixing in ground-level imagery (e.g. WiderPerson)
                 since flipping people upside-down doesn't make sense there.

- scripts\annotate_video.py (NEW)
    Batch equivalent of test_stream.py: runs a video file through the same
    Detector class the live API uses (identical tracking/motion logic,
    no server required), draws boxes/class/confidence/track ID/"MOVING"
    tag per frame, and writes an annotated output video. Useful for visual
    demonstration of detection quality.

- scripts\run_pipeline.py (NEW)
    One-command pipeline chaining: convert WiderPerson -> fine-tune on the
    merged dataset -> validate new run vs. current weights\best.pt side by
    side -> optionally promote (--promote flag; off by default) ->
    annotate a demo video with the result. Each stage shells out to the
    existing standalone scripts.

===============================================================================
6. TRAINING RUN
===============================================================================
- Started a fine-tuning run:
    python scripts\train.py --model weights\best.pt
      --data data\battlesight_multi.yaml --epochs 30 --batch 8 --imgsz 640
      --lr0 0.002 --flipud 0 --name battlesight_v2
  (fine-tunes the current weights\best.pt on VisDrone + WiderPerson)
- First attempt failed immediately due to the path bug in section 5 above
  (fixed, then restarted).
- Second attempt started successfully (model loaded, AMP checks passed,
  began training) but was STOPPED EARLY at the user's request before any
  epochs completed. No results/weights were produced by this run.
- Verified cleanup after stopping: GPU memory dropped back to ~1.9 GB
  (only the running API server's inference model remains loaded); no
  orphaned training process left holding the GPU.
- Landed run folder: runs\detect\battlesight_v2-2\ (auto-incremented
  because an earlier failed attempt had already created an empty
  runs\detect\battlesight_v2\ folder). This folder has no usable weights
  and can be deleted.

===============================================================================
7. CURRENT STATE / NEXT STEPS
===============================================================================
- weights\best.pt is UNCHANGED — still the original VisDrone-only model.
  No merged-dataset training has completed yet.
- To train: run the command in section 6 yourself (see chat for full
  manual walkthrough), or ask to have it started again in the background.
- After training completes: validate with `yolo val` (new run vs. current
  weights\best.pt on data\battlesight_multi.yaml), inspect
  runs\detect\battlesight_v2\{results.png,confusion_matrix.png,
  val_batch0_pred.jpg}, then promote manually:
    copy runs\detect\battlesight_v2\weights\best.pt weights\best.pt
  and restart uvicorn to pick up the new weights (model loads once at
  startup, no hot-reload).
- Optionally run scripts\annotate_video.py on a real clip afterward to
  visually confirm detection quality on both aerial and ground-level
  footage.
