BattleSight AR / FusionSight — Session Summary
Session date: 2026-09-01
Written for: a new agent/session picking this project up cold.
Full detail for everything below lives in README.md ("Detection accuracy
work", sections numbered 1-12) — this file is an orientation map, not a
replacement for it. When in doubt, read README.md section-by-section rather
than trusting only this summary.

===============================================================================
1. WHAT THIS PROJECT IS
===============================================================================
A YOLO26-based object detector for drone/helmet-cam situational-awareness
footage, served over FastAPI (app/). Declared scope: real-time person/vehicle
detection feeding an AR/HUD display for a human operator — situational
awareness, NOT automated fire-control or enemy classification.

Four trained classes: personnel, two_wheeler, light_vehicle, heavy_vehicle.
Plus a class-agnostic "moving_object" tag (class_id -1) for anything that
moves coherently but isn't one of the four — e.g. a UAV, which the trained
classes cannot recognize as an object type (see section 4 below).

Core pieces:
  app/config.py      - every path/threshold/flag, all env-overridable, all
                        commented with the measured numbers that justify them
  app/detector.py     - Detector class: loads model(s), runs the 4-stage
                        detect/track cascade, motion-blob claiming
  app/motion_filter.py - MotionDetector: background subtraction + trajectory
                        coherence, independent of the trained classes
  app/main.py, app/routers/ - FastAPI app, /detect, /detect/tracked,
                        /ws/track/{source_id}, /train, /exclude
  scripts/train.py, eval_rubric.py, run_pipeline.py - training + promotion
  scripts/annotate_video.py  - run a file/camera/stream through Detector,
                        write an annotated output. No server needed.
  scripts/annotate_screen.py - NEW this session. Same as annotate_video.py
                        but source is a live screen/window capture (mss +
                        pygetwindow), for testing against whatever's playing
                        on screen (e.g. a YouTube video) without downloading
                        it first. See section 5 below.
  scripts/export_engine.py - export a .pt checkpoint to a TensorRT .engine

===============================================================================
2. CURRENT DEPLOYED STATE (what's actually live right now)
===============================================================================
Two model "views", both loaded simultaneously by Detector.load(), selected
per-request via a `view` param ("ground" default, or "drone"):

  GROUND VIEW (config.MODEL_PATH, weights/best.pt)
    Trained on VisDrone + WiderPerson (battlesight_multi.yaml). General-
    purpose: handheld, CCTV-angle, elevated-balcony footage.
  DRONE VIEW (config.DRONE_MODEL_PATH, weights/drone_best.pt)
    NEW this session. Trained on VisDrone ONLY (no WiderPerson dilution).
    Specialised for genuine top-down aerial footage. Beats the ground model
    on every metric when evaluated on VisDrone-only val (mAP50 0.563->0.582,
    recall 0.526->0.545) — see README section 10.

Both are actually SERVED via TensorRT FP16 engines, not raw .pt:
  weights/best.engine, weights/drone_best.engine  (FP16, what's loaded)
  weights/best_int8.engine, drone_best_int8.engine (INT8, built, NOT loaded
                                                      by default — see below)
Detector.load() auto-prefers a matching .engine over .pt when
BATTLESIGHT_USE_TENSORRT != "0" (default on) and falls back to .pt if the
engine fails to load or DEVICE=cpu. No code path change needed to pick this
up — building the engine files was enough.

Inference precision: config.QUANTIZE = 16 (FP16) whenever DEVICE != "cpu",
else None. This is now the project standard (see section 4). Threaded
through every model.predict()/.track() call in app/detector.py, including
warmup.

MOTION_GATED is OFF (config.py default "0") and confirmed to STAY off —
explicitly asked and answered this session (see section 4). Do not flip it
on without re-reading README section 5: the gated path only sees things that
moved, so it goes blind to parked vehicles / stationary people.

===============================================================================
3. WHAT CHANGED THIS SESSION, IN ORDER
===============================================================================

A) Fixed the "explosion" of periodic moving_object detections in the
   annotated videos (README section 9). Two NEW causes found beyond the
   already-fixed v3.mp4 case from a prior session:
   - v6.mp4 (fast handheld pan + motion blur): the ego-compensation residual
     fraction hovered right AT the EGO_MAX_RESIDUAL threshold instead of
     resolving clearly, so the reliable/unreliable flag flickered frame to
     frame. Fixed with an EMA smoother (config.EGO_RESIDUAL_EMA_ALPHA=0.25)
     in app/motion_filter.py.
   - v7.mp4 (drone hovering over brick paving): fine texture fragmented into
     30-58 tiny blobs/frame for 30+ consecutive frames while the FOREGROUND
     FRACTION stayed under the existing 0.12 threshold (that threshold was
     tuned only against v3's tree-line case, a different failure shape).
     Fixed with a new blob-COUNT gate, config.MOTION_CHRONIC_BLOB_COUNT=20.
   Verified via a new diagnostic, scripts/diagnose_bursts.py (rolling-median
   outlier detection across all vN.mp4 clips) — not present before this
   session, useful for any future motion-filter regression check.
   One residual, NOT fixed (and not the same bug): v6.mp4 has two isolated
   frames (517, 735) with a cluster of low-confidence light_vehicle boxes
   from the CLASSIFIER during a blurry night pass over parked motorbikes —
   this is the known-weak two_wheeler class being confused under blur/low
   light, unrelated to the motion filter. Documented as an outstanding item,
   not chased further.

B) Added the drone-view specialised model (README section 10). See section 2
   above for the deployed state. Mechanics: fine-tuned
   weights/best_e12_visdrone.pt (a pre-existing 13-epoch VisDrone-only
   checkpoint) for 25 more epochs on data/battlesight.yaml. Wired in as a
   `view` query param on POST /detect, POST /detect/tracked,
   ws://.../ws/track/{source_id}?view=drone, and --view on
   annotate_video.py/annotate_screen.py. Per-feed tracker state is keyed by
   (view, source_id) internally since each loaded YOLO model owns its own
   ByteTrack predictor.

C) Made FP16 the standard inference precision, after MEASURING it rather
   than assuming (README section 11). Also measured INT8 for real and
   REJECTED it as the default (real ~4-point mAP50 / ~5-point recall loss on
   both models, not worth the extra ~1.6x speed over FP16 by default) — the
   INT8 engines exist (built, calibrated on VisDrone val) but nothing in
   app/ loads them automatically. Confirmed true FP8 does not exist in this
   ultralytics version (checked the exporter source directly: only 32/16/8/
   w8a16/w8a32 quantize levels exist) — do not try to add an FP8 path
   without re-verifying this against whatever ultralytics version is
   installed at the time, it may have changed.
   Also installed and got working in this venv: tensorrt (cu12-matched),
   onnx, onnxslim, onnxruntime-gpu (pinned to 1.20.2, NOT latest — see
   section 6 gotchas), nvidia-modelopt (auto-installed by the TensorRT
   export path). requirements.txt documents the exact install commands that
   worked here, including the CUDA-version pitfalls.

D) Added scripts/annotate_screen.py (README section 12) — live screen/window
   capture through the same Detector pipeline, for testing against content
   already playing on screen (e.g. a YouTube video in a browser) without
   downloading it first. `--window TEXT` matches by window title substring
   and re-reads live position every frame; `--region x,y,w,h` for a fixed
   area. Verified for real: decoded an actual output frame and confirmed it
   captured real screen content (a browser window), not just a clean exit.

E) All 7 vN_annotated.mp4 demo files regenerated at the end, reflecting (A)
   the explosion fix, (B) v5/v7 using the drone-view model, and (C) the
   FP16/TensorRT backend.

===============================================================================
4. DECISIONS MADE WITH THE USER (don't silently reverse these)
===============================================================================
- Motion gating: asked directly whether to enable it for throughput; user
  said leave it OFF. Reason stands: goes blind to stationary targets.
- FP16 vs INT8: FP16 adopted as standard (negligible accuracy cost, ~45%
  latency win). INT8 measured and explicitly NOT made default (real accuracy
  cost). Do not flip config.QUANTIZE to 8 or point Detector at the *_int8.
  engine files without raising this tradeoff again.
- Screen capture: user explicitly wants a REAL live-capture capability (not
  just "download the video first"), which is why annotate_screen.py exists
  as a new script rather than just recommending yt-dlp.

===============================================================================
5. ENVIRONMENT GOTCHAS (would waste time rediscovering)
===============================================================================
- G:\ appears to be a mounted/slow-ish drive at times ("Slow image access
  detected" warnings show up in training/val logs intermittently) — not
  always slow, but don't be surprised by it.
- This network's connection to pypi.nvidia.com is UNSTABLE. Installing
  tensorrt-cu12-libs (~2.25 GB) took several retries with
  `--retries 10 --timeout 60`, resuming from pip's cache each time. Budget
  real time for this if you ever need to reinstall.
- onnxruntime-gpu MUST be <1.29 on this machine (pin 1.20.2). 1.29+ requires
  CUDA 13 (cublasLt64_13.dll) and silently falls back to CPU instead of
  failing to import — it then crashes on a device-mismatch error deep inside
  inference, which is a confusing place to debug this from. Symptom if you
  hit it again: "No registered plugin EP device found for
  'CUDAExecutionProvider'" followed by a device-mismatch RuntimeError.
- The bare `pip install tensorrt` resolves to a cu13 build on this box,
  mismatched with torch's cu126 build. Use the explicit
  tensorrt-cu12/-libs/-bindings packages (see requirements.txt).
- Building a TensorRT engine here: use --static (fixed batch=1) not the
  dynamic-shape default. Dynamic sizes its memory pool for max batch (16 by
  default) and OOM'd on this 8 GB GPU. Static matches how this service
  actually calls the model anyway (batch=1, MOTION_GATED off means
  MOTION_CROP_IMGSZ is never invoked).
- scripts/annotate_video.py and annotate_screen.py now RELIABLY SEGFAULT
  (exit 139) a moment AFTER printing their final "Wrote N frames" summary,
  because two YOLO/torch models (ground + drone) are loaded and something in
  CUDA/Python interpreter teardown doesn't like that combination. Output
  files are complete and correct every time this was observed. Treat a
  nonzero exit from either script as informational — check for the "Wrote N
  frames" line, not the exit code. Not root-caused; not worth chasing
  further unless it starts corrupting output.
- Both annotate_*.py scripts need PYTHONPATH=. OR to be run as
  `python scripts\whatever.py` from the repo root (not `python whatever.py`
  from inside scripts\) — they insert the repo root onto sys.path
  themselves when run the normal way, but a raw `-c` one-liner or a script
  run from elsewhere needs PYTHONPATH set explicitly.

===============================================================================
6. STILL OUTSTANDING (unchanged from before this session, or newly noted)
===============================================================================
- Drones-AS-TARGETS still cannot be detected (no 5th UAV class — VisDrone is
  footage FROM drones, not OF them). Do not confuse this with the drone-VIEW
  model added this session, which is a different, already-solved problem
  (detecting the same 4 classes better on aerial footage).
- Buses are still diluted into heavy_vehicle with trucks.
- two_wheeler is confused for light_vehicle under blur/low light (see 3A
  above, v6.mp4 frames 517/735) — same root cause as the pre-existing
  "confident off-distribution false positives" item, needs training data,
  not a threshold change.
- flipud=0.5 was inherited into the ground-model's WiderPerson training by
  an Ultralytics --resume quirk (wrong for ground-level imagery) — never
  retrained without it, still open.

===============================================================================
7. WHERE TO GO FOR MORE DETAIL
===============================================================================
README.md, "Detection accuracy work" section — every fix in this project's
history (including everyone else's, before this session) is numbered
sequentially with the measured before/after numbers, not just described.
Sections 9-12 are this session's work. Read the specific numbered section
before touching anything it covers, rather than re-deriving from scratch.
