BattleSight AR / FusionSight
CONCLUSION OF 2026-09-02, AND THE PLAN FOR THE CROWDHUMAN RUN
=============================================================
Written 2026-09-02, end of session. For the agent/session picking this up next.

Read order for a cold start:
  1. this file
  2. context.md  (sections 0, 0a, 0b, 0c are today's work)
  3. README.md   (sections 17, 18, 19 are today's measurements in full)
  4. start.txt   (the operator runbook; its fpv run is DONE)


###############################################################################
PART 1 -- WHAT HAPPENED TODAY (conclusion)
###############################################################################

-------------------------------------------------------------------------------
1.1 The fpv training run finished, was measured, and is PROMOTED
-------------------------------------------------------------------------------
battlesight_fpv completed all 8 epochs (interrupted after epoch 3, resumed with
`--resume` -- that path is now proven against a real interruption).

eval_rubric.py vs the then-deployed weights/best.pt:  PASS on all four criteria.
    overall mAP50    0.5910 -> 0.6274
    personnel mAP50  0.6785 -> 0.7064
    recall           0.540  -> 0.570
    precision        0.669  -> 0.716
    no collapsed class -- the direct check on the AerialPerson pseudo-label trap

Drone view was promoted TOO, on its own separate rubric run
(--baseline weights/drone_best.pt), and by a far wider margin:
    overall mAP50    0.5036 -> 0.6274
    personnel mAP50  0.3162 -> 0.7064      (more than double)

So weights/best.pt and weights/drone_best.pt are now BYTE-IDENTICAL
(md5 62398e6f15ede0dbf51e1882b5c0accc), with separately built engines. The
drone specialisation of README section 10 is retired. The two views still differ,
but only through config thresholds now, not weights.

DEPLOYED STATE:
    weights/best.pt              fpv checkpoint          (16:55)
    weights/best.engine          rebuilt FP16, 256 MB    (16:59)
    weights/drone_best.pt        same checkpoint         (17:44)
    weights/drone_best.engine    rebuilt FP16            (18:02)
    weights/best_pre_fpv.pt/.engine          rollback pair
    weights/drone_best_pre_fpv.pt/.engine    rollback pair
Rollback is a file copy, not a 3-minute rebuild, because the OLD ENGINES were
kept as well as the old checkpoints.

STALE, not loaded by default, rebuild or delete before touching INT8 again:
    weights/best_int8.engine, weights/drone_best_int8.engine

FP16 engine parity was verified separately, because the rubric only ever
measures the .pt while production serves the engine:
    mAP50 0.6274 (.pt) vs 0.6252 (.engine); mAP50-95 identical at 0.384;
    every class within 0.002.  The engine is also ~2x FASTER: 14.8 vs 28.8 ms.

-------------------------------------------------------------------------------
1.2 The v10 question was answered: NO, and the answer was informative
-------------------------------------------------------------------------------
The run did NOT fix the v10 personnel failure. Measured over all 374 frames,
stateless detect(view="drone"), old drone checkpoint vs new:

    personnel        245 (in 35.3% of frames)  ->  0
    light_vehicle    385                       ->  58   (-85%)
    heavy_vehicle    20                        ->  0

Those 245 personnel boxes were ALL false positives on vegetation. Rendered and
inspected: frame 186 has a person plainly visible, PRONE in the open, and the
old model put 8 small personnel boxes across the magenta foliage with NOT ONE on
the person. The new model returns zero detections there -- it stopped
hallucinating people in the grass, but still does not find the person.

So the run removed ~592 false positives on that clip and gained no true
positives. Strictly better, but not what it was for.

This narrows the gap usefully. Palette, viewpoint, resolution and augmentation
are now all ruled out by this run. What is left is POSE. Nothing in the current
mix contains a prone or crawling person seen from a UAV.

-------------------------------------------------------------------------------
1.3 Inference changes that shipped (all measured, all env-overridable)
-------------------------------------------------------------------------------
a) PERSONNEL FLOOR 0.20 -> 0.10 (ground view only)
   The old note claiming "0.10 buys only ~2 points of recall" was an artefact of
   the BLENDED val set: VisDrone's aerial personnel dominate it by instance
   count and their P/R curve genuinely is flat, masking the ground-level
   headroom underneath. Split by domain, on WiderPerson val (1,000 ground-level
   images, the bodycam-like case):

     conf     P       R      F1      F2
     0.08   0.600   0.750   0.667   0.714   <- F2 optimum
     0.10   0.657   0.725   0.689   0.710   <- ADOPTED (the knee)
     0.20   0.814   0.636   0.714   0.665   <- previous floor, F1 optimum

   Chosen on F2, not F1: F1 weights precision and recall equally, which
   contradicts the standing judgement that a suppressed real contact is the
   failure this system must never have. Ground-level personnel mAP50 is 0.757,
   BETTER than the blended 0.706 and than aerial's 0.605.
   Effect: personnel totals up 15-42% on every clip, no vehicle/motion
   regression, ~1 ms latency cost.
   Drone view stays at 0.25 (at 0.10 aerial precision collapses to 0.431).

b) ILLUMINATION-VS-STRUCTURE DISCRIMINATOR (app/motion_filter.py)
   The motion channel could not separate a light (whose bright region drifts and
   reads as coherent travel) from a person, and could not report anyone visible
   for under MOTION_COHERENCE_MIN_POINTS=4 frames AT ALL -- not a tuning
   problem; the required evidence outlasted the event.
   _structure_score() decides from ONE frame: align the previous frame, then
   test whether the change is a uniform brightness shift or preserves structure
   under z-scoring. Illumination does both; a real object does neither. FAILS
   OPEN on anything it cannot judge.
   A FAST PATH then reports 2-point blobs that clear a stricter structure bar
   AND real displacement -- deliberately NOT using _straightness, which is 1.0
   by construction at two points.
   The EGO-RESIDUAL DEGRADED BAND stops the all-or-nothing drop that was
   blinding the channel on 535/1068 frames of v6.

     clip   moving_object      bursts    frames blind
     v5     16 -> 0            0 -> 0    0 -> 0
     v6     344 -> 326         0 -> 1    535 -> 70
     v10    449 -> 34          6 -> 1    97 -> 69
     v11    2707 -> 2105       65 -> 43  508 -> 445

   v5's 16 -> 0 was VERIFIED VISUALLY, not assumed: sodium-lit night courtyard,
   every rejected blob on empty pavement or a shadow edge, never on a person.
   9.5-31% of motion detections now come via the fast path.

c) PARALLEL MOTION STAGE -- the latency win (app/detector.py)
   Profiling found the CPU motion filter was 23.6 ms of a 52.2 ms frame, running
   IN SEQUENCE with the GPU pass while the GPU idled -- despite nothing on the
   model path reading `blobs` until _claim_motion_blobs at the very end. Both
   sides release the GIL (OpenCV; TensorRT), so overlapping them costs max()
   instead of sum.
       sequential   46.4 ms median / 50.4 p90
       parallel     24.6 ms median / 28.2 p90     <- ~2x, and more STABLE
   Verified as a PURE latency change: detection counts byte-identical
   (v2 914/914, v5 1560/1560) and the full 10-clip burst sweep byte-identical in
   every column. BATTLESIGHT_MOTION_PARALLEL=0 reverts it.

d) FAR-FIELD SECOND PASS -- built, measured, OFF BY DEFAULT
   Recall collapses with target size: <16px 0.187, 16-32 0.622, >96 0.940.
   41% of all people are under 32 px and fewer than half are found.
   The cheap full-frame pass CHOOSES where to spend one extra pass (the smallest
   boxes ARE the distant ones). +4-8% more people on video, +4.4% on val.
   OFF not for cost but for JITTER: p90 48.5 ms against a 26.9 ms median, so an
   AR overlay would stutter. Good for OFFLINE use (annotate_video.py).
   BATTLESIGHT_FARFIELD=1 enables it.

-------------------------------------------------------------------------------
1.4 THREE TRAPS FOUND -- these cost real time, do not rediscover them
-------------------------------------------------------------------------------
* predict() BETWEEN track() CALLS CORRUPTS THE PREDICTOR.
  Costs ~60% of detections, silently. Bisected by running the far-field pass
  with every box DISCARDED, leaving only the predict() call: v5 personnel fell
  1560 -> 562. Saving/restoring predictor.trackers did NOT fix it; the
  interference is broader. _gated_detections and _detect_on_crops both call
  predict() and would BOTH hit this the moment MOTION_GATED is switched on.
  Fix used: a dedicated second model instance for the far-field pass.

* config.IOU_THRESHOLD IS A NO-OP for model inference, and always has been.
  YOLO26's head reports end2end: True -- it is an NMS-free one-to-one detector
  and the iou= argument is silently ignored. Sweeping 0.5/0.6/0.7/0.8 gave
  BYTE-IDENTICAL results at every value. Do not reach for it to fix
  crowd/occlusion recall; there is no NMS to loosen. It still governs this
  repo's OWN IoU arithmetic (_claim_motion_blobs, overlay_mask, motion merging).

* opencv-python-headless SILENTLY BREAKS ALL PREVIEW WINDOWS.
  It and opencv-python unpack into the same cv2/ directory; last install wins.
  albumentations depends on headless, so installing it for the fpv profile
  broke annotate_screen.py. Both are now pinned/commented in requirements.txt
  (albumentations was NOT in there at all -- a rebuilt env would have trained
  with NO augmentation, the exact bug that left every pre-2026-09-02 checkpoint
  augmentation-free).

Also corrected: the export command recorded in start.txt/context.md
(`export_engine.py weights\best.pt --static`) FAILS TWICE -- --model is a flag,
not a positional, and the script needs PYTHONPATH=. like the annotate scripts.
Working form is now in both files.

-------------------------------------------------------------------------------
1.5 HEALTH WARNING on every latency figure in this project
-------------------------------------------------------------------------------
Absolute timings on this machine varied ~3x on UNCHANGED code in one session
(the same path measured 25, 46, 48.9, 52.2, 67 and 74 ms). An env-var ablation
even measured the structure test as SLOWER when disabled -- physically
impossible, and proof the noise floor exceeded the effect being measured.
Trust warmed, back-to-back, WITHIN-RUN comparisons only. Warm 40 frames first.

-------------------------------------------------------------------------------
1.6 Where accuracy stands, and why the next move is training
-------------------------------------------------------------------------------
Three independent measurements now point the same way:
  * resolution does not help (1280->1536 is +0.4% detections for +46% cost, and
    the larger size buckets get WORSE -- above the trained size the model is
    off-distribution for its own scale priors);
  * NMS cannot be loosened (there is none; suppression is LEARNED);
  * the v10 person is missed because of POSE.
All three are properties of the weights, not the pipeline. With latency now at
~25 ms median there is also ~15 ms of headroom that did not exist this morning.


###############################################################################
PART 2 -- NEXT SESSION: THE CROWDHUMAN RUN
###############################################################################

GOAL: raise personnel recall on DENSE, OCCLUDED, GROUND-LEVEL people -- the
bodycam / COD-style case the user actually cares about, and the case where
<32 px recall is 0.187-0.622 and NMS offers no help because suppression is
learned.

WHY CROWDHUMAN: 15,000 train / 4,370 val images, ~340k person instances,
~22.6 persons per image with heavy mutual occlusion. It is the closest public
match to a crowded bodycam frame. WiderPerson (already in the mix) is dense but
its scenes are static street photography; CrowdHuman is far more varied in pose,
occlusion and viewpoint.

-------------------------------------------------------------------------------
2.0 READ THIS FIRST -- CrowdHuman has the SAME TRAP as AerialPerson
-------------------------------------------------------------------------------
CrowdHuman is a PERSON-ONLY dataset. Its imagery is full of street scenes with
CARS AND BUSES, and NONE of them carry a label. Merged raw into this 4-class
taxonomy, every one of those vehicles becomes a confirmed NEGATIVE for
light_vehicle / heavy_vehicle.

This is the exact failure scripts/pseudo_label_vehicles.py exists to prevent
(README 16g; it nearly destroyed the vehicle classes on AerialPerson, where
97.3 unlabelled vehicles per image were measured). CrowdHuman is GROUND-LEVEL,
so the contradiction is with WiderPerson-style and VisDrone-style vehicles
rather than aerial ones -- weaker than AerialPerson's case but far from zero,
because 15,000 images is a lot of negative evidence.

=> MEASURE IT FIRST (step 2.4), then pseudo-label if the count justifies it.
   Do NOT skip straight to training.

-------------------------------------------------------------------------------
2.1 DOWNLOAD (manual -- do not guess URLs)
-------------------------------------------------------------------------------
CrowdHuman is hosted at  http://www.crowdhuman.org/  behind Google Drive links
that rotate. There is no stable direct curl URL, so DO NOT fabricate one --
open the site and download by hand into:

    G:\fusionsight\datasets\CrowdHuman_raw\

Files needed (train + val only; the test split ships without ground truth):
    CrowdHuman_train01.zip, _train02.zip, _train03.zip   (~15 GB total)
    CrowdHuman_val.zip                                    (~2.5 GB)
    annotation_train.odgt
    annotation_val.odgt

Disk check before starting -- 267 GB free as of today, and the converter copies
images, so budget ~35 GB for raw + converted:

    df -h /g          (or: Get-PSDrive G)

Extract so that the layout is:
    datasets/CrowdHuman_raw/Images/<id>.jpg          (all splits flattened here;
                                                      the odgt says which is which)
    datasets/CrowdHuman_raw/annotation_train.odgt
    datasets/CrowdHuman_raw/annotation_val.odgt

*** CHECK NOTHING IS ALREADY DOWNLOADING FIRST ***
    tasklist | findstr curl

-------------------------------------------------------------------------------
2.2 WRITE scripts/convert_crowdhuman.py
-------------------------------------------------------------------------------
Does not exist yet. Model it on scripts/convert_widerperson.py, which is the
house style for this: reads the raw layout, writes YOLO labels, refuses to
invent anything, prints counts.

CrowdHuman's .odgt is ONE JSON OBJECT PER LINE:
    {"ID": "<image id>",
     "gtboxes": [{"tag": "person" | "mask",
                  "fbox": [x, y, w, h],        # FULL body box
                  "vbox": [x, y, w, h],        # VISIBLE part only
                  "hbox": [x, y, w, h],        # head
                  "extra":      {"ignore": 0|1, ...},
                  "head_attr":  {...}}, ...]}

VERIFY THAT SHAPE against the actual file before trusting it:
    head -1 datasets/CrowdHuman_raw/annotation_val.odgt | python -m json.tool | head -40

Decisions to encode, and why:
  * Use "fbox" (full body), NOT "vbox". This project's other person labels
    (WiderPerson classes 1-3, AerialPerson, VisDrone pedestrian/people) are all
    whole-person boxes. Mixing visible-only boxes in would teach two different
    box conventions for one class and blur the regression head.
  * tag == "mask"  -> DROP. That is CrowdHuman's ignore region, not a person.
  * extra.ignore == 1 -> DROP. Same reasoning as WiderPerson's classes 4/5:
    no reliable per-instance box.
  * class id 0 (personnel), matching data/battlesight.yaml.
  * CLAMP boxes to the image and drop degenerate ones after clamping --
    CrowdHuman fboxes routinely extend outside the frame for occluded people,
    and a negative-width box silently corrupts a YOLO label file.
  * An image whose every box was dropped must be SKIPPED, not written as an
    empty label. convert_aerialperson.py already established this rule: a
    silently-empty label teaches the model to MISS real people.
  * Split: train from annotation_train.odgt, val from annotation_val.odgt.

Then run and read the printed counts:

    cd G:\fusionsight
    .\.venv\Scripts\Activate.ps1
    python scripts\convert_crowdhuman.py

Sanity gates before continuing:
  * ~15,000 train images written, ~4,370 val
  * ~339,000 train person boxes (roughly 22 per image). An order of magnitude
    off means fbox/vbox or the ignore filter is wrong.
  * Spot-check 3 images with boxes drawn before spending 9 hours on them.

-------------------------------------------------------------------------------
2.3 SMALL-OBJECT COPY-PASTE AUGMENTATION (optional, do it in the same run)
-------------------------------------------------------------------------------
The other half of the <16 px problem, and it costs NOTHING at inference.
Add a third profile to scripts/fpv_augment.py (PROFILES is currently
("none", "fpv")) -- e.g. "fpv_small" -- that keeps the whole fpv chain and adds
scale-down copy-paste: take existing person crops from the same image, rescale
to 0.3-0.6x, paste at plausible positions, and append the matching labels.

Keep it honest: paste only onto plausible ground/background regions, never
overlapping an existing box, and cap the number pasted per image. If this turns
into a fight, DROP IT and run CrowdHuman alone -- one variable at a time is
worth more than two half-measured ones.

-------------------------------------------------------------------------------
2.4 MEASURE THE UNLABELLED-VEHICLE CONTRADICTION (do not skip)
-------------------------------------------------------------------------------
Same measurement that caught AerialPerson. Dry run first -- it only reports:

    python scripts\pseudo_label_vehicles.py `
        --dataset datasets\CrowdHuman `
        --weights weights\best.pt `
        --splits train `
        --dry-run

NOTE --weights weights/best.pt, NOT drone_best.pt. The script's default is the
drone checkpoint because AerialPerson is aerial; CrowdHuman is ground-level.
(As of today both files are byte-identical anyway, but be explicit -- that will
stop being true the moment either is retrained.)

Decision rule:
  * more than ~3 unlabelled vehicles per image  -> pseudo-label it (drop
    --dry-run, keep --splits train). Originals are backed up to
    labels/<split>.orig/ and `--restore` undoes it.
  * under ~1 per image                          -> skip; not worth the false
    labels a 0.5-confidence pass would introduce.
  * in between                                  -> pseudo-label, and say so in
    the README entry, because it is a judgement call not a measurement.

CrowdHuman VAL is TRAIN-ONLY here -- it does NOT go into the yaml's val list.
Same reasoning as AerialPerson (context.md section 5): scoring vehicles against
person-only labels penalises every correctly detected car, and pseudo-labelling
val would be circular. Keeping val unchanged also keeps eval_rubric.py numbers
directly comparable to every figure in README's history. Its 4,370 images stay
on disk with clean person-only ground truth as a held-out on-domain set:
    yolo val model=weights/best.pt data=<a val-only yaml> classes=0

-------------------------------------------------------------------------------
2.5 REGISTER THE DATASET
-------------------------------------------------------------------------------
Add ONE line to SOURCES in scripts/prepare_training.py (around line 28),
alongside the existing entries:

    ("CrowdHuman train", "CrowdHuman/images/train", "train",
     "15000, dense occluded ground-level personnel"),

Do NOT add a val entry (see 2.4). Do NOT hand-edit data/battlesight_fpv.yaml --
it is generated. Then:

    python scripts\prepare_training.py

That re-checks what is on disk, writes a yaml naming only paths that exist
(ultralytics fails hard on a missing path), warns if pseudo-labelling was
skipped, and prints the exact training command.

-------------------------------------------------------------------------------
2.6 THE TRAINING RUN
-------------------------------------------------------------------------------
*** STOP UVICORN FIRST *** -- batch 4 at imgsz 1280 peaks at 7.51 GB of 8 GB,
and the inference model holds ~1.5-2 GB. There is not room for both.

    cd G:\fusionsight
    .\.venv\Scripts\Activate.ps1

    python scripts\train.py `
        --model weights\best.pt `
        --data data\battlesight_fpv.yaml `
        --aug-profile fpv `
        --imgsz 1280 --batch 4 --epochs 6 `
        --lr0 0.002 --flipud 0 `
        --name battlesight_crowd

(use --aug-profile fpv_small instead if 2.3 was done)

TIME BUDGET, extrapolated from today's measured run -- 18,694 images took
34-99 min/epoch at 4,674 it/epoch, so ~50 min/epoch typical:
    train set grows 18,694 -> ~33,694 images  =  ~8,424 it/epoch
    ~90 min/epoch  x 6 epochs  ~=  9-10 HOURS
Drop to --epochs 4 (~6 h) if that does not fit. Today's run was still climbing
at epoch 8, so more epochs genuinely help -- this is a budget call, not a
convergence one.

Every flag, and why (unchanged reasoning from start.txt section 2):
  --imgsz 1280   the deployed checkpoints are SERVED at 1280 (config.IMGSZ);
                 training below that reopens the train/serve mismatch on
                 exactly the tiny targets this system exists to find.
  --batch 4      MEASURED. At batch 4 ultralytics reports ~10.2 G reserved on
                 an 8 GB card and only fits by spilling into shared system RAM
                 under Windows WDDM. DO NOT RAISE IT.
  --flipud 0     the deployed ground model was once trained with flipud=0.5
                 (upside-down people) via a --resume quirk. Never inherit it.
  --lr0 0.002    fine-tuning from an existing checkpoint, not training fresh.

IF IT STOPS MIDWAY:
    python scripts\train.py --resume --name battlesight_crowd
That is the WHOLE command -- ultralytics restores data/imgsz/epochs/augmentation
from the checkpoint. Re-passing them is at best ignored. Two caveats: epoch 1 is
unprotected (last.pt is written at the END of each epoch), and if it died of
CUDA OOM add --batch 2 (batch IS resume-overridable; flipud is NOT).

Safe to run alongside training (CPU only, no GPU):
    python scripts\diagnose_bursts.py --motion-only

-------------------------------------------------------------------------------
2.7 EVALUATE -- do not promote blindly
-------------------------------------------------------------------------------
    python scripts\eval_rubric.py runs\detect\battlesight_crowd\weights\best.pt `
        --baseline weights\best.pt --data data\battlesight_fpv.yaml --imgsz 1280

Four PASS/FAIL criteria: no overall mAP50 regression, no personnel regression,
no collapsed class, recall floor. `no_collapsed_class` is the one that matters
most here -- it is the direct check on the section 2.0 vehicle trap.

THEN THE MEASUREMENT THIS RUN IS ACTUALLY FOR. mAP50 will not tell you whether
dense-crowd recall improved. Re-run today's size-stratified recall on
WiderPerson val and compare against these numbers:

    BASELINE TO BEAT (2026-09-02, weights/best.pt, 300 images, conf 0.10):
        size(px)     GT    recall
           <16     1427    0.187
          16-32    2214    0.622
          32-48    1426    0.805
          48-64    1128    0.855
          64-96    1481    0.937
           >96     1207    0.940
        overall: 6279 found, precision 0.662

The script that produced it now lives in the repo (it was a scratchpad
throwaway; moved and given a CLI on 2026-09-02, because the scratchpad does not
survive the session). It is the only tool here that measures the thing which
actually limits this system:

    $env:PYTHONPATH="."
    python scripts\eval_size_recall.py `
        --weights runs\detectattlesight_crowd\weightsest.pt

    # and against the OLD model, same command with --weights weightsest.pt
    # it prints the 2026-09-02 baseline at the bottom of every run for comparison

    # also worth running on CrowdHuman's own held-out val (person-only labels,
    # so class 0 only -- which is what --cls already defaults to):
    python scripts\eval_size_recall.py `
        --images datasets\CrowdHuman\imagesal --limit 500

Use --limit 300 to compare against the recorded baseline; a different --limit
changes the absolute counts (the per-bucket RECALL stays comparable).

SUCCESS = the <16 and 16-32 buckets move. Overall mAP50 barely will, because
those buckets are a minority of the instance-weighted metric even though they
are 41% of the people.

-------------------------------------------------------------------------------
2.8 PROMOTE (only on PASS)
-------------------------------------------------------------------------------
    copy weights\best.pt weights\best_pre_crowd.pt
    copy weights\best.engine weights\best_pre_crowd.engine
    copy runs\detect\battlesight_crowd\weights\best.pt weights\best.pt

    $env:PYTHONPATH="."
    python scripts\export_engine.py --model weights\best.pt --static

    # decide separately whether drone view should follow -- it is currently the
    # same file, but CrowdHuman is ground-level data and may not help aerial.
    # Run the rubric with --baseline weights\drone_best.pt before copying.

KEEP THE OLD ENGINE, not just the old .pt -- that is what makes rollback a file
copy instead of a 3-minute rebuild.

THE ENGINE STEP IS NOT OPTIONAL: Detector.load() prefers weights/best.engine
over the .pt, so skipping it silently keeps serving the OLD model.

-------------------------------------------------------------------------------
2.9 REGRESSION -- required after any promotion or threshold change
-------------------------------------------------------------------------------
    python scripts\diagnose_bursts.py --view ground
    python scripts\diagnose_bursts.py v10.mp4 v11.mp4 --view drone

    python tests\test_overlay_mask.py
    python tests\test_motion_structure.py
    python tests\test_motion_coherence.py
    python tests\test_exclusion.py
    $env:PYTHONPATH="."; python tests\test_history_cap.py

BASELINE TO COMPARE AGAINST (2026-09-02, --view ground, served path):
    clip   frames  moving_object  max  bursts  dropped
    v1        77         3          1     0       47
    v2       225         0          0     0      221
    v3       368        44          3     0      167
    v4       311         6          1     0        3
    v5       175         0          0     0        0
    v6      1068       326          6     1       70
    v7       370        29          3     0       29
    v9       225         0          0     0      223
    v10      374        34          7     1       69
    v11     1746      2105         12    43      445
    Total burst frames across all classes: 121

Caveat that will bite you: personnel "bursts" in this harness are mostly REAL
CROWDS, not hallucinations -- v2 frame 176 was rendered and all 10 boxes were on
actual people. The rolling-median baseline cannot tell an explosion from a busy
frame. Read totals and max/frame alongside the burst count, and LOOK AT FRAMES.
CrowdHuman will make this worse: denser scenes, more legitimate bursts.

Also re-check the far-field path if you touch it -- it is OFF by default and
therefore UNEXERCISED, which is exactly how a latent break survives.

-------------------------------------------------------------------------------
2.10 STILL OPEN AFTER ALL OF THIS
-------------------------------------------------------------------------------
* PRONE / CRAWLING PERSONNEL FROM UAV (the v10 failure). CrowdHuman will NOT
  fix this -- it is ground-level. Needs UAV imagery of prone people; nothing
  public in the mix has it. This is the oldest open item and the hardest.
* v2 and v9 still go blind (ego residual 221/223 of 225 frames). Their residual
  sits far above 2x EGO_MAX_RESIDUAL so the degraded band never reaches them.
  Raising EGO_RESIDUAL_DEGRADED_FACTOR would reach them at the cost of admitting
  genuinely parallax-broken frames -- UNMEASURED.
* The static-camera mog2_fraction drop (122 frames on v3) has the same
  all-or-nothing shape as the ego guard and could take the same degraded
  treatment. Deliberately left alone today to keep the blast radius small.
* Far-field tile cannot run below 1280 while the engine is --static. A second
  engine at 640 would make its p90 fit the 40 ms budget and let it ship ON.
* No drone/UAV-as-target class. Buses still diluted into heavy_vehicle.
* two_wheeler is the weakest class (0.494 mAP50) and CrowdHuman does nothing
  for it.
* Nothing in this model has ever seen a rendered game character. If COD-style
  footage is a real target, that needs its own data, and no amount of
  photographic person data substitutes for it.
