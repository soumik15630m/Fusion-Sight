BattleSight AR — Inference Summary (this session)
Session date: 2026-09-01 (continuation of the session behind summary.txt)
Written for: a new agent/session picking up inference work cold.
Full detail lives in README.md, "Detection accuracy work" sections 13-15 —
this file is an orientation map, not a replacement for it.

===============================================================================
1. WHY THIS SESSION HAPPENED
===============================================================================
User reported recurring "explosions" of boxes during inference (phantom
`moving_object` swarms and, later, classifier false-positive floods) on
clips beyond the ones already fixed in prior sessions (README sections 7-9).
This session re-audited those fixes against the LIVE production path
(TensorRT/FP16 engines, not the .pt files they were originally measured
against), found and fixed a new cause, then diagnosed two more clips that
turned out to be a different problem entirely, then tuned the system's
sensitivity per view at the user's explicit direction.

===============================================================================
2. WHAT WAS FOUND AND CHANGED, IN ORDER
===============================================================================

A) Re-audit of fixes 7-9 (README section 13)
   `scripts/diagnose_bursts.py` -- cited throughout the README as what
   verified every prior explosion fix -- DOES NOT EXIST in this checkout.
   No file matching `*burst*` exists anywhere outside .venv. Every number
   attributed to it should be treated as unverified until it's recreated.
   Re-ran the fixes directly against Detector.track() instead: they hold
   (v3.mp4 moving_object max 16->3, confirmed live) -- but turned up a 4th,
   undocumented cause:

   `MotionDetector.detect()` (app/motion_filter.py) picked the ego-
   compensated-vs-plain-MOG2 branch from the RAW per-frame estimated camera
   shift, no smoothing. A slow genuine pan whose shift hovers near
   EGO_STATIC_SHIFT (0.3px) flickered the branch choice itself frame to
   frame -- misrouted frames ran plain MOG2 on real pan-smeared structure,
   which is coherent by construction and sailed past every existing gate.
   Measured on v6.mp4 frame 962: 0 -> 2 -> 5 -> 7 phantom moving_object boxes.

   FIX: EGO_SHIFT_EMA_ALPHA (0.25, config.py) smooths the shift before the
   threshold check, same shape as the pre-existing EGO_RESIDUAL_EMA_ALPHA.
   Verified: frame-962 burst gone, no regression on v3/v9.

   Tried and REVERTED: the same EMA trick on MOTION_CHRONIC_BLOB_COUNT for
   an identical-looking flicker on v7.mp4 frame 261. Made it WORSE (7->13
   boxes) -- that gate needs to react fast to a spike; averaging delays the
   trip instead of preventing a false one. v7.mp4 frame 261 is still open.

B) v10.mp4 / v11.mp4 diagnosis (README section 14)
   User supplied two more clips, NOT tactical/aerial/handheld content --
   v10 is a wildlife/nature clip with wind-blown grass, v11 is a forward-
   facing highway dashcam. Two unrelated causes:

   v11: PURE CLASSIFIER hallucination. Confirmed by running Detector.detect()
   (stateless, motion filter entirely out of the loop) on a burst frame: 45
   of 55 boxes were already there -- dozens of overlapping `light_vehicle`
   boxes (conf 0.25-0.83) on empty road/sky. Same failure class as the
   pre-existing "confident false positives off-distribution" item, much
   worse: no camera angle like this exists in VisDrone (aerial) or
   WiderPerson (ground pedestrians). NOT fixable by threshold/filter tuning
   -- needs training-data coverage (a driving dataset like BDD100K/KITTI/
   Mapillary remapped to the 4 classes, or at minimum hard-negative
   empty-road frames). Not implemented, recommendation only.

   v10: a genuinely new motion-filter gap. All boxes were `moving_object`
   (zero classifier detections) -- wind-blown grass scored 0.55-0.97 on the
   trajectory-straightness check over just 4 judged frames, while sitting
   far under both existing chronic-noise gates (fraction ~0.04, blob count
   ~9-10, both way under their 0.12/20 cutoffs). Measured a full
   MOTION_COHERENCE_THRESHOLD sweep (0.5/0.65/0.75/0.85/0.92) against v10
   plus the already-fixed v3/v6 as regression checks. 0.85 was the first
   value that fully cleared v10's bursts, at a real cost: cut v3's/v6's
   already-verified-real moving_object counts by ~47%/~63%. Applied
   globally at first, at the user's request.

C) Per-view split + personnel recall floor (README section 15)
   User clarified: drone view should keep the strict 0.85 (terrain/grass
   noise is the problem there); ground view (handheld/bodycam-style, whose
   whole point is catching a person visible only briefly) should stay
   permissive AND get more personnel recall specifically. Global change
   from (B) was wrong for ground view -- reverted and split:

   - MOTION_COHERENCE_THRESHOLD_DRONE = 0.85 (kept)
   - MOTION_COHERENCE_THRESHOLD_GROUND = 0.5 (reverted to original, what
     fixes 7-9/13 were tuned against)
   - MotionDetector.detect() now takes `view`; Detector.track() passes it
     through. v10.mp4 grass bursts return under ground view (deliberate,
     accepted -- v10 was never the target of the "short span personnel"
     ask) but stay clean under --view drone.

   - NEW: CONF_THRESHOLD_PERSONNEL_GROUND = 0.20 (config.py), personnel-
     class-only, ground-view-only. Measured with a class-specific P/R/F1
     sweep first, not guessed:

       conf   personnel P   R       F1
       0.10      0.777    0.568   0.656
       0.15      0.777    0.568   0.656
       0.20      0.762    0.579   0.658   <- adopted, best measured balance
       0.25      0.803    0.549   0.652   (previous floor, all classes)

     Flatter than expected -- little headroom in this lever on the standard
     val set. Deliberately NOT a blanket confidence drop (kept vehicles at
     0.25) to avoid reopening the v11-style hallucination risk, which was
     specifically about `light_vehicle` at low confidence.

   Mechanically: Detector._model_conf_floor(view) passes the lowest
   threshold any class/view needs to model.predict()/.track() itself (so
   ultralytics never strips a box early); Detector._to_detections() applies
   the real per-class/view floor afterward via _conf_threshold_for().
   Threaded through detect(), _full_frame_track(), _gated_detections()'s
   fallback, and _detect_on_crops() (the last two unreachable while
   MOTION_GATED stays off, updated for consistency anyway).

   CAVEAT, not yet resolved: the val set (VisDrone+WiderPerson) is mostly
   clean, static, well-lit imagery, not motion-blurred video -- it likely
   UNDERSTATES how much confidence drops on genuine fast-handheld/bodycam
   footage. If short-span personnel recall still isn't good enough in
   practice, the real fix is motion-blurred/fast-handheld training data,
   not further threshold tuning (this sweep shows little room left there).

===============================================================================
3. CURRENT INFERENCE-RELEVANT CONFIG (app/config.py)
===============================================================================
  CONF_THRESHOLD                    = 0.25   (all classes, all views, default floor)
  CONF_THRESHOLD_PERSONNEL_GROUND   = 0.20   (personnel class, ground view only)
  MOTION_COHERENCE_THRESHOLD_DRONE  = 0.85
  MOTION_COHERENCE_THRESHOLD_GROUND = 0.5
  EGO_SHIFT_EMA_ALPHA               = 0.25   (new, fix 13)
  MOTION_GATED                      = off    (confirmed to stay off -- goes
                                               blind to stationary targets)
  QUANTIZE                          = 16 (FP16), served via weights/*.engine
                                        (TensorRT) with automatic .pt fallback

===============================================================================
4. VERIFIED / NOT VERIFIED
===============================================================================
Verified directly against Detector.track() this session:
  v1, v3, v6, v9 (ground) -- no regressions, bursts stay fixed
  v10 (drone view) -- clean; (ground view) -- bursts return, expected/accepted
  v7 -- frame 261 burst confirmed still open (not fixed)

NOT verified (flagged, not fixed):
  - scripts/diagnose_bursts.py is missing; recreate it for an automated
    regression sweep instead of manual spot checks like this session's.
  - v11.mp4's classifier hallucination -- needs training data, not config.
  - Personnel short-span recall on REAL blurry/fast footage -- only measured
    against clean static imagery so far (see caveat in 2C above).
  - v9.mp4's ego-residual check drops 224/225 frames as unreliable
    (parallax) -- class-agnostic motion tagging is blind on it, a different,
    non-urgent gap, not a burst.

===============================================================================
5. WHERE TO GO FOR MORE DETAIL
===============================================================================
README.md, "Detection accuracy work" -- sections 13, 14, 15 are this
session's work, each with full measured before/after tables. Read the
specific section before touching anything it covers. The
`fusionsight-detection-baseline` memory (persistent across sessions) also
carries this history plus one non-obvious gotcha: a bare model.val() sweep
script with default `workers` hung indefinitely on Windows (missing
`if __name__ == "__main__":` guard, classic multiprocessing deadlock) --
looked like slow disk I/O at first; the tell was process memory staying
perfectly static across checks. Guard with __main__ and pass workers=0 for
any one-off val()/predict() script on this machine.
