BattleSight AR / FusionSight — start training when you have time
================================================================
Written 2026-09-01. Supersedes the previous start.txt (that run,
`battlesight_v2`, was superseded by battlesight_v1's completed 30-epoch run —
see README "Detection accuracy work" §"Still outstanding").

*** 2026-09-02: THIS RUN IS DONE AND PROMOTED. ***
Everything below is kept as the record of how it was run (and as the recipe for
the next run) -- but you do not need to start it. Status:

  * battlesight_fpv completed all 8 epochs (interrupted after epoch 3, resumed
    with `--resume` exactly as section 2b describes; the resume path is now
    proven on a real interruption).
  * eval_rubric.py vs the then-deployed weights/best.pt: PASS on all four
    criteria.  mAP50 0.5910 -> 0.6274, personnel 0.6785 -> 0.7064,
    recall 0.540 -> 0.570, no collapsed class.
  * PROMOTED. weights/best.pt is the fpv checkpoint, weights/best.engine was
    rebuilt from it (FP16, validated: mAP50 0.6252 vs 0.6274 for the .pt, and
    ~2x faster -- 14.8 ms vs 28.8 ms). Rollback pair kept as
    weights/best_pre_fpv.pt and .engine, so rolling back is a file copy.
  * drone view PROMOTED TOO, on its own rubric run (--baseline drone_best.pt):
    PASS, mAP50 0.5036 -> 0.6274 and personnel 0.3162 -> 0.7064 on VisDrone val.
    weights/drone_best.pt is now byte-identical to weights/best.pt, engine
    rebuilt. Rollback pair: weights/drone_best_pre_fpv.pt and .engine.
    So `--view drone` (what every v10/v11 command below uses) DOES now run the
    new model. The two views differ only by config thresholds, not weights.
  * weights/best_int8.engine is now STALE (built from the old checkpoint).
  * v10 ANSWERED, and the answer is no. The run removed every vegetation false
    positive (personnel 245 -> 0, light_vehicle 385 -> 58, heavy_vehicle 20 -> 0
    across 374 frames) but gained no true positives -- frame 186 has a prone
    person plainly visible and the model returns nothing. Strictly better on
    this clip, but the original failure stands. Remaining gap is POSE: needs
    UAV imagery of prone/crawling people. See context.md section 0.

See context.md section 0 for the same summary with full tables.

Full detail for everything below is in README.md §16. Read §16e before
deciding whether this run is worth the GPU time — it says what the run is
actually expected to fix and what it is not.


-------------------------------------------------------------------------------
0. WHAT IS ALREADY LIVE — no training needed, no action from you
-------------------------------------------------------------------------------
These landed this session and are in effect right now (restart uvicorn to pick
them up if it is running):

  * HUD/OSD overlay rejection (app/overlay_mask.py, new).
    v11.mp4 phantom `light_vehicle` boxes: 54,658 -> 6,713  (-87.7%)
                              per frame:      31.3 -> 3.84
    v10.mp4: byte-identical before and after — nothing real was lost.
    This was the single largest error source in the whole system and it was
    NOT what the previous session diagnosed. See §16a/§16b.

  * v7.mp4 frame 261 burst FIXED (MOTION_CHRONIC_COOLDOWN=3). Regression
    swept across every clip: v2/v3/v4/v5/v6/v9 identical, v1 improved.

  * scripts/diagnose_bursts.py recreated (it was missing). This is the
    regression harness — run it after ANY motion-filter or threshold change:

        python scripts/diagnose_bursts.py --view ground
        python scripts/diagnose_bursts.py v10.mp4 v11.mp4 --view drone
        python scripts/diagnose_bursts.py --motion-only     # CPU, no GPU

    --motion-only needs no GPU, so it is safe to run WHILE training.

  * tests/test_overlay_mask.py — 10 assertions, all passing.


-------------------------------------------------------------------------------
1. BEFORE YOU START: one command
-------------------------------------------------------------------------------
    cd G:\fusionsight
    .\.venv\Scripts\Activate.ps1
    python scripts\prepare_training.py

That checks what data is on disk, converts the AerialPerson archive if its
download has finished, writes data\battlesight_fpv.yaml naming ONLY paths that
exist (ultralytics fails hard on a missing path), and prints the exact training
command. It never starts training itself.

The AerialPerson download (Zenodo 7740081, ~3.7 GB) was still in flight when
this was written — Zenodo throttles this network to ~125 KB/s, so budget ~8 h.
Training does NOT need it; prepare_training.py just omits it and you can rerun
the training later with it included. It is, however, the only dataset here that
contains the case the model actually fails on, so it is worth waiting for if
you can.


-------------------------------------------------------------------------------
1b. DOWNLOADING AerialPerson YOURSELF
-------------------------------------------------------------------------------
Source: Zenodo record 7740081, "Small Object Aerial Person Detection Dataset",
CC-BY-4.0. 3,136 UAV images over a university campus and Civil Defense
exercises, already in YOLO format, single class `people` which is already class
id 0 == `personnel` here, so no remapping is needed. Median box ~11 px at
imgsz 1280.

*** CHECK NOTHING IS ALREADY DOWNLOADING IT FIRST ***
Two curl processes writing the same file will corrupt it:

    tasklist | findstr curl

If that prints nothing, run this from G:\fusionsight (PowerShell). Use
curl.exe, NOT curl — in Windows PowerShell 5.1 `curl` is an alias for
Invoke-WebRequest, which takes different flags and cannot resume:

    mkdir datasets\AerialPerson_raw -Force

    curl.exe -L --retry 20 --retry-delay 10 --retry-all-errors -C - `
        -o datasets\AerialPerson_raw\Images.zip `
        "https://zenodo.org/api/records/7740081/files/Images.zip/content"

    curl.exe -L --retry 10 -o datasets\AerialPerson_raw\Annotations.zip `
        "https://zenodo.org/api/records/7740081/files/Annotations.zip/content"

`-C -` means RESUME. If it dies or you Ctrl-C it, rerun the exact same command
and it picks up where it stopped rather than starting over. Zenodo throttles
this network to roughly 125 KB/s, so the 3.7 GB image archive takes ~8 h.

Then convert it (prepare_training.py also does this for you automatically):

    python scripts\convert_aerialperson.py

It verifies every box and refuses to invent labels: an image whose annotation
rows were all malformed is SKIPPED rather than written out as a background
image, because a silently-empty label would teach the model to miss real people.


-------------------------------------------------------------------------------
1c. IMPORTANT: AerialPerson labels PEOPLE ONLY — vehicles must be pseudo-labelled
-------------------------------------------------------------------------------
This nearly wasted a whole training run, so do not skip it.

AerialPerson is a SINGLE-CLASS dataset: it annotates people and nothing else.
But its imagery is top-down aerial over a university campus, which means large
parking lots — and none of those cars carry a label. That is the SAME viewpoint
VisDrone teaches vehicles from, so mixing the two raw presents every one of
those cars to the trainer as a confirmed NEGATIVE for `light_vehicle`.

Measured by running weights/drone_best.pt over 40 random AerialPerson train
images at conf 0.35:

    personnel        11.6 / image
    light_vehicle    97.3 / image     <-- none of these are labelled
    two_wheeler       0.8 / image
    heavy_vehicle     0.5 / image

Across 2,613 train images that is roughly 258,000 unlabelled vehicles — more
negative vehicle evidence than VisDrone supplies positive. It would not dilute
the vehicle classes, it would destroy them.

Fix (prepare_training.py warns loudly if you have not done this):

    python scripts\pseudo_label_vehicles.py

That labels the vehicles with the existing drone-view checkpoint and merges
them in. Human-annotated person boxes are never touched, and a pseudo-box
overlapping a real person box is dropped — the human label always wins.
Confidence floor is 0.5, well above serving's 0.25, because a false positive
here becomes a permanent wrong label. Originals are backed up first:

    python scripts\pseudo_label_vehicles.py --restore     # undo, any time
    python scripts\pseudo_label_vehicles.py --dry-run     # report only

RESULT of that run on this dataset: 61,708 human person boxes preserved
exactly, 263,700 vehicle pseudo-labels added, 86 dropped for overlapping a
real person. Spot-checked visually — boxes are tight on the cars, the person
labels are untouched, and a bus came through correctly as heavy_vehicle.

AerialPerson contributes to TRAIN ONLY. Its val split was deliberately left
out of the metric and RESTORED to clean, person-only ground truth:
  * scoring vehicles against person-only labels would count every correctly
    detected car as a false positive, and
  * pseudo-labelling it instead would be circular — those labels come from
    drone_best.pt, which is the baseline eval_rubric.py compares against.
Happy side effect: the val set is now identical to battlesight_multi.yaml's,
so eval_rubric.py numbers ARE directly comparable to the figures in README's
history after all. The 523 held-out on-domain images stay on disk with clean
ground truth; evaluate personnel on them explicitly when you want to
(`classes=0`).

NOTE the same issue applies in principle to WiderPerson, which has been in this
project's training mix since battlesight_multi.yaml: ground-level street scenes,
personnel-only labels, unlabelled traffic. It is weaker there because
ground-level cars look different from VisDrone's aerial ones, so the
contradiction is less direct — but if vehicle metrics ever look inexplicably
poor, this is a prime suspect and it predates this session.


-------------------------------------------------------------------------------
2. THE TRAINING RUN
-------------------------------------------------------------------------------
    python scripts\train.py `
        --model weights\best.pt `
        --data data\battlesight_fpv.yaml `
        --aug-profile fpv `
        --imgsz 1280 --batch 4 --epochs 8 `
        --lr0 0.002 --flipud 0 `
        --name battlesight_fpv

Every flag, and why:

  --imgsz 1280   The deployed checkpoints were trained at 640 but are SERVED
                 at 1280 (config.IMGSZ). That train/serve mismatch is on
                 exactly the tiny targets this system exists to find.
  --batch 4      MEASURED, not guessed. See section 3 — do not raise it.
  --epochs 8     Sized to your 6-8 h budget at this resolution, see section 3.
                 (Was 10 before AerialPerson landed and grew the train set to
                 18,694 images.)
                 This is a fine-tune from an already-trained checkpoint, not a
                 run from scratch, so 10 epochs is a reasonable adaptation.
  --aug-profile fpv
                 NEW. albumentations was not installed in this venv AT ALL, so
                 every previous checkpoint had ZERO blur/noise/compression/
                 colour augmentation — ultralytics silently skips that whole
                 stage when the import fails. The profile models the real
                 capture chain in order (blur -> downscale -> noise ->
                 compression -> exposure -> palette), including a wide hue
                 rotation for the false-colour EO/IR case in §16e.
  --flipud 0     Fixes a known-wrong inherited setting: the deployed ground
                 model was trained with flipud=0.5 (upside-down people) via an
                 ultralytics --resume quirk. Long-standing open item.
  --lr0 0.002    Fine-tuning from an existing checkpoint, not training fresh.

train.py now also adapts `warmup_epochs` and `close_mosaic` to the run length.
The old hardcoded close_mosaic=10 would have disabled mosaic for the ENTIRE
run at epochs=10, and a fixed 3-epoch warmup would have been 30% of it.

Logging is on by default -> logs\battlesight_fpv_<timestamp>.log.


-------------------------------------------------------------------------------
2b. IF THE RUN STOPS MIDWAY — yes, you can resume it
-------------------------------------------------------------------------------
    python scripts\train.py --resume --name battlesight_fpv

That is the whole command. Do NOT re-pass --imgsz/--epochs/--aug-profile: on
resume ultralytics restores every argument from the checkpoint itself, so
passing them again is at best ignored and at worst confusing.

What is preserved: ultralytics writes last.pt at the end of EVERY epoch, so you
lose at most the epoch in progress. The optimizer state, LR schedule position,
epoch counter and best-so-far all come back.

The fpv augmentation survives resume — verified, not assumed. The trainer
serializes the transform list with `A.to_dict()` into the checkpoint args and
restores it with `A.from_dict()`, and `augmentations` is in ultralytics'
resume-overridable key list (ultralytics/engine/trainer.py, check_resume).

CAVEAT, the same one that bit this project before (README "Still outstanding"):
`flipud` is NOT in that overridable list. On resume it comes back from the
checkpoint, and passing `--flipud 0` on the resume command line will NOT
override it. That is fine here — the interrupted run already started with
flipud=0, so it resumes with flipud=0. It only matters if you resume a run that
was STARTED with the wrong value, in which case restart from scratch instead.

If it stopped because of CUDA out-of-memory rather than a clean interrupt,
resuming at the same batch size will just OOM again. Lower it on the resume
command (--batch IS overridable, unlike flipud):

    python scripts\train.py --resume --name battlesight_fpv --batch 2


-------------------------------------------------------------------------------
3. TIME AND MEMORY — measured on this machine, do not guess
-------------------------------------------------------------------------------
RTX 4060 Laptop, 8 GB, imgsz 1280, AMP on, 16,081 train images:

    batch 4  ->  ~1.6 it/s, peak torch alloc 7.51 GB, 4,020 it/epoch
             ->  ~42 min/epoch + ~4 min val  =  ~46 min/epoch
             ->  10 epochs ~= 7-8 h          <- sized to your budget

IMPORTANT: at batch 4 ultralytics reports ~10.2 G RESERVED on an 8 GB card. It
only fits by spilling into shared system RAM under Windows WDDM, which is why
it is that slow. It works, but it is right at the edge:

  * Stop uvicorn before starting. The inference model holds ~1.5-2 GB and
    there is not room for both.
  * Do not raise --batch. Do not run anything else on the GPU.
  * scripts/diagnose_bursts.py --motion-only is CPU-only and safe to run
    alongside.

If you want more epochs instead of more resolution, --imgsz 960 --batch 6 is
roughly half the per-epoch cost, at the price of leaving part of the
train/serve mismatch in place.


-------------------------------------------------------------------------------
4. AFTER IT FINISHES — do not promote blindly
-------------------------------------------------------------------------------
    python scripts\eval_rubric.py runs\detect\battlesight_fpv\weights\best.pt `
        --baseline weights\best.pt --data data\battlesight_fpv.yaml --imgsz 1280

Four criteria, PASS/FAIL: no overall mAP50 regression, no personnel regression,
no collapsed class, recall floor. Only promote on PASS:

    copy weights\best.pt weights\best_pre_fpv.pt          # keep a way back
    copy runs\detect\battlesight_fpv\weights\best.pt weights\best.pt

NOTE the rubric's absolute numbers are NOT comparable to the ones recorded in
README against battlesight_multi.yaml, because the val set gained AerialPerson.
The verdict is still valid (it compares baseline and candidate on the same
data). For a like-for-like historical check add:
    --data data\battlesight_multi.yaml

Then the thing that actually matters — look at the real footage, not the
val numbers, because the val set does not represent it (§16e):

    python scripts\annotate_video.py v10.mp4 --view drone
    python scripts\annotate_video.py v11.mp4 --view drone

Success for this run means personnel boxes ON THE PEOPLE in v10, not a higher
mAP50. Right now it puts boxes on vegetation and misses them entirely.

If you promote, rebuild the TensorRT engine or serving will keep using the old
weights compiled into weights\best.engine:

    $env:PYTHONPATH="."
    python scripts\export_engine.py --model weights\best.pt --static

NOTE: `--model` is a FLAG, not a positional argument. The form written here
before 2026-09-02 (`export_engine.py weights\best.pt --static`) fails with
"unrecognized arguments". It also needs PYTHONPATH=. like annotate_video.py,
or it dies with ModuleNotFoundError: No module named 'app'.

Reminder from README §10: annotate_video.py reliably exits 139 (segfault)
during CUDA teardown AFTER printing "Wrote N frames". Check for that line, not
the exit code.
