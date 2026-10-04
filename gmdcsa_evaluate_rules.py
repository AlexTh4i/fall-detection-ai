"""Replay the original FSM on the same sampled keypoints; exploratory comparison only."""
import contextlib
import io
import json
from pathlib import Path
import numpy as np
from realtime_yolo_fall_detection import PersonTracker, STATE_SUSPECTED, STATE_CONFIRMED
from gmdcsa_pipeline import metrics, OUT, save_json

root=OUT
manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
pred=[]
for record in manifest['records']:
    if record['subject'] != 'Subject 4':
        continue
    tracker=PersonTracker(1)
    suspected=False
    confirmed=False
    with np.load(root/'keypoints'/(record['id']+'.npz'),allow_pickle=False) as data:
        for pose,box,t in zip(data['pose'],data['box'],data['time']):
            pose=pose.copy()
            scale=max(record['width'],record['height'])
            pose[:,0] *= scale/record['width']
            pose[:,1] *= scale/record['height']
            with contextlib.redirect_stdout(io.StringIO()):
                tracker.update(pose,box*scale,float(t)+1000)
            suspected |= tracker.state in [STATE_SUSPECTED,STATE_CONFIRMED]
            confirmed |= tracker.state == STATE_CONFIRMED
    pred.append(dict(id=record['id'],label=record['label'],suspected=suspected,confirmed=confirmed))
report=dict(warning='Original FSM replayed at 10 Hz on cached single-person poses. This differs from full-rate live tracking. Temporal model candidate scores and FSM confirmed alerts have different confirmation requirements; do not treat as equal event-level metrics.',videos=pred)
for stage in ['suspected','confirmed']:
    report[stage]=metrics([r['label'] for r in pred],[int(r[stage]) for r in pred],0.5)
save_json(root/'original_rules_evaluation.json',report)
print(json.dumps({k:report[k] for k in ['suspected','confirmed']},indent=2))
