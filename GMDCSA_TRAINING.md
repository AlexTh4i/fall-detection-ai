# GMDCSA24 temporal experiment

Use the existing GPU environment from this project directory:

```powershell
.\venv\Scripts\python.exe gmdcsa_pipeline.py prepare
.\venv\Scripts\python.exe gmdcsa_pipeline.py extract
.\venv\Scripts\python.exe gmdcsa_pipeline.py train
.\venv\Scripts\python.exe gmdcsa_pipeline.py predict --video path-to-video.mp4
```

Outputs: `data/gmdcsa24_experiment/manifest.json`, `splits.json`, `keypoints/*.npz`, `temporal_fall.pt`, `evaluation.json`.

This is a small temporal CNN baseline, not ST-GCN. It uses 3-second windows at 10 Hz, 17 YOLO26 COCO joints, confidence masks, local pose motion and global box motion. A common isotropic coordinate scale preserves image geometry. Single-person selection follows bounding-box overlap and falls back to the largest person; this assumption must be revisited for multi-person data.

Train: Subjects 1 and 2. Validation: Subject 3. Test: Subject 4. The probability decision threshold is chosen on validation only. No OBS recording is included in training or threshold selection.

CSV labels describe whole videos, not exact fall intervals. Multiple-instance learning pools window logits within each video rather than labelling every frame as falling. Window scores are experimental and not calibrated event probabilities. Event timing annotations are still needed for event-level evaluation and reliable realtime integration. Short clips are left-padded with a missing-data mask; windows with less than half observed samples are excluded from video scoring.

CSV inconsistencies are preserved and flagged in manifest.json. Videos include falls onto beds as well as floors. This experiment follows the supplied labels and does not equate all falls with lying on a floor.

Metrics are video-level. Four subjects and a few rooms do not establish generalization to arbitrary camera angles. Keep the existing realtime FSM until an event-level comparison supports replacing it. See the dataset's local README and LICENSE for attribution and terms.

## Completed experiment

See GMDCSA_RESULTS.md for the measured results and remaining errors. The model still misses the initial overhead OBS fall segment. The original rules are preserved; the original entry point now defaults to an experimental temporal mode. `event_annotations.csv` is a review template only; its timestamps are not yet consumed by this weak-label training pipeline.

Replay the original rules for an exploratory comparison:

```powershell
.\venv\Scripts\python.exe gmdcsa_evaluate_rules.py
```

## Realtime in the original entry point

```powershell
.\venv\Scripts\python.exe realtime_yolo_fall_detection.py
.\venv\Scripts\python.exe realtime_yolo_fall_detection.py 1
.\venv\Scripts\python.exe realtime_yolo_fall_detection.py --mode rules
```

The default temporal mode loads `data/gmdcsa24_experiment/temporal_fall.pt` and checks that the YOLO weights match training. Each tracked person has an independent 30-sample window at 10 Hz and one prior sample for velocity context. Allow roughly 3 seconds to collect initial observations. The score threshold comes from the saved validation checkpoint (0.43); it is not a per-camera angle threshold. A 1-second persistence filter controls the experimental alert and score decrease. This filter is not event-validated, and low scores do not prove physical recovery. Missing tracks/poses show UNKNOWN rather than recovery. Multi-person realtime behavior has not been validated by the four-subject training dataset.

Press Q/ESC to exit, R to clear history, F to mirror and clear history, S to save a snapshot. Temporal mode starts without mirroring, consistent with training input. Rules mode preserves the original FSM. The trained mode still has the errors described in GMDCSA_RESULTS.md and does not guarantee arbitrary camera-angle accuracy.

For a camera-free smoke check:

```powershell
.\venv\Scripts\python.exe realtime_yolo_fall_detection.py --video path-to-video.mp4 --headless --max-frames 150
```
