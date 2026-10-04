"""GMDCSA24 experiment: weak video labels, temporal windows, subject-disjoint evaluation."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'gmdcsa24_experiment'
HZ = 10
STEPS = 30
FEATURES = 93


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def prepare(dataset, out):
    roots = list(dataset.rglob('Subject 1'))
    if len(roots) != 1:
        raise ValueError('Expected one Subject 1 directory')
    base = roots[0].parent
    records = []
    for subject in sorted(base.glob('Subject *')):
        for activity in ['ADL', 'Fall']:
            table = subject / (activity + '.csv')
            with table.open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.reader(f))
            for row in rows[1:]:
                if not row:
                    continue
                filename = row[0].strip()
                video = subject / activity / filename
                if not video.is_file():
                    raise FileNotFoundError(video)
                cap = cv2.VideoCapture(str(video))
                ok, frame = cap.read()
                fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                if not ok or fps <= 0:
                    raise ValueError('Unreadable video: ' + str(video))
                desc_start = 2 if len(rows[0]) == 3 else 4
                records.append(dict(id=f'{subject.name.replace(" ", "")}_{activity}_{video.stem}',
                    subject=subject.name, activity=activity, label=int(activity == 'Fall'),
                    path=str(video.resolve()), description=','.join(row[desc_start:]).strip(),
                    csv_row=row, csv_columns=rows[0], csv_needs_review=len(row) != len(rows[0]),
                    fps=fps, frames=int(frames), duration=frames / fps,
                    width=frame.shape[1], height=frame.shape[0],
                    label_scope='video', event_start=None, event_end=None))
    actual = {str(p.resolve()) for p in base.rglob('*.mp4')}
    if actual != {r['path'] for r in records}:
        raise ValueError('CSV/video inventory mismatch')
    manifest = dict(dataset_root=str(base), sample_hz=HZ, window_seconds=STEPS / HZ,
        warning='Video labels only. Event start/end unknown. Fall clips include normal frames.', records=records)
    save_json(out / 'manifest.json', manifest)
    splits = dict(train=['Subject 1', 'Subject 2'], validation=['Subject 3'], test=['Subject 4'])
    save_json(out / 'splits.json', splits)
    print('Prepared', len(records), 'videos;', sum(r['csv_needs_review'] for r in records), 'CSV rows require review', flush=True)


def sample_video(path, crop=None, start=0, end=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise ValueError('Cannot open ' + str(path))
    n = 0
    next_sample = start
    try:
        while cap.grab():
            t = n / fps
            n += 1
            if end is not None and t > end:
                break
            if t + 1e-8 < next_sample:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            if crop:
                x, y, w, h = crop
                if min(x, y) < 0 or w <= 0 or h <= 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                    raise ValueError('Crop outside video')
                frame = frame[y:y+h, x:x+w]
            yield t, frame
            next_sample += 1 / HZ
    finally:
        cap.release()


def iou(a, b):
    lo = np.maximum(a[:2], b[:2]); hi = np.minimum(a[2:], b[2:])
    inter = np.prod(np.maximum(hi-lo, 0))
    area_a = np.prod(np.maximum(a[2:]-a[:2], 0)); area_b = np.prod(np.maximum(b[2:]-b[:2], 0))
    return float(inter / max(area_a+area_b-inter, 1e-8))


def extract_video(model, path, device, batch_size=8, crop=None, start=0, end=None):
    poses, boxes, timestamps = [], [], []
    batch, times = [], []
    previous = None
    def process():
        nonlocal previous
        results = model.predict(batch, imgsz=640, conf=0.25, classes=[0], device=device, verbose=False)
        for t, frame, result in zip(times, batch, results):
            scale = max(frame.shape[:2])
            if result.keypoints is None or len(result.boxes) == 0:
                poses.append(np.zeros((17, 3), np.float32)); boxes.append(np.zeros(4, np.float32)); previous = None
            else:
                bb = result.boxes.xyxy.cpu().numpy()
                areas = np.prod(bb[:, 2:]-bb[:, :2], axis=1)
                overlaps = np.array([iou(previous, b) for b in bb]) if previous is not None else np.zeros(len(bb))
                index = int(overlaps.argmax()) if overlaps.max() >= 0.1 else int(areas.argmax())
                previous = bb[index].copy()
                kp = result.keypoints.data[index].cpu().numpy().copy()
                kp[:, :2] /= scale
                poses.append(kp.astype(np.float32)); boxes.append((previous / scale).astype(np.float32))
            timestamps.append(t)
        batch.clear(); times.clear()
    for t, frame in sample_video(path, crop, start, end):
        times.append(t); batch.append(frame)
        if len(batch) >= batch_size:
            process()
    if batch:
        process()
    if not timestamps:
        raise ValueError('No frames sampled')
    return dict(pose=np.stack(poses), box=np.stack(boxes), time=np.array(timestamps, np.float64))


def extract(out, weights, device, batch_size):
    from ultralytics import YOLO
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    model = YOLO(str(weights))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    target = out / 'keypoints'; target.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(manifest['records'], 1):
        dest = target / (record['id']+'.npz')
        signature = json.dumps(dict(model_sha256=digest, hz=HZ, imgsz=640,
            source_size=Path(record['path']).stat().st_size,
            source_mtime_ns=Path(record['path']).stat().st_mtime_ns), sort_keys=True)
        if dest.exists():
            with np.load(dest, allow_pickle=False) as cached:
                if str(cached['signature']) == signature:
                    print(f'[{i}/{len(manifest["records"])}] cached {record["id"]}', flush=True)
                    continue
        t = time.time()
        data = extract_video(model, record['path'], device, batch_size)
        temp = dest.with_suffix('.partial.npz')
        np.savez_compressed(temp, **data, signature=np.array(signature))
        temp.replace(dest)
        print(f'[{i}/{len(manifest["records"])}] {record["id"]}: {len(data["time"])} samples, {time.time()-t:.1f}s', flush=True)


def features(data):
    pose = data['pose'].astype(np.float32).copy(); box = data['box'].astype(np.float32)
    good = pose[:, :, 2] >= 0.2
    valid = (box[:, 2] > box[:, 0]) & (good.sum(axis=1) >= 4)
    center = (box[:, :2] + box[:, 2:]) / 2
    hip = pose[:, [11,12], :2].mean(axis=1)
    reliable_hip = good[:, 11] & good[:, 12]
    origin = np.where(reliable_hip[:, None], hip, center)
    size = np.maximum(np.linalg.norm(box[:, 2:]-box[:, :2], axis=1), 0.05)
    local = (pose[:, :, :2]-origin[:, None]) / size[:, None, None]
    local[~good] = 0
    confidence = pose[:, :, 2:3] * good[:, :, None]
    velocity = np.zeros_like(local)
    dt = np.maximum(np.diff(data['time']).astype(np.float32), 1e-3)
    velocity[1:] = (local[1:]-local[:-1]) / dt[:, None, None]
    velocity[1:] *= (good[1:] & good[:-1])[:, :, None]
    global_pos = np.concatenate([center, box[:, 2:]-box[:, :2]], axis=1)
    global_velocity = np.zeros_like(global_pos)
    global_velocity[1:] = (global_pos[1:]-global_pos[:-1]) / dt[:, None]
    global_velocity[1:] *= (valid[1:] & valid[:-1])[:, None]
    x = np.concatenate([local.reshape(-1,34), confidence.reshape(-1,17),
        velocity.reshape(-1,34), global_pos, global_velocity], axis=1)
    x[~valid] = 0
    return np.nan_to_num(x), valid.astype(np.float32)


def windows(data):
    x, mask = features(data)
    starts = list(range(0, max(1, len(x)-STEPS+1), HZ))
    last = max(0, len(x)-STEPS)
    if starts[-1] != last: starts.append(last)
    bags, masks, ends = [], [], []
    for s in starts:
        xx, mm = x[s:s+STEPS], mask[s:s+STEPS]
        if len(xx) < STEPS:
            # Short clips are left-padded; missing time is explicitly masked.
            pad = STEPS-len(xx); xx=np.pad(xx,((pad,0),(0,0))); mm=np.pad(mm,(pad,0))
        bags.append(xx); masks.append(mm); ends.append(float(data['time'][min(s+STEPS-1,len(x)-1)]))
    return np.stack(bags), np.stack(masks), ends


class TemporalNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]; incoming=FEATURES
        for dilation in [1,2,4]:
            layers += [nn.Conv1d(incoming,64,3,padding=dilation,dilation=dilation), nn.ReLU(), nn.Dropout(0.2)]
            incoming=64
        self.temporal=nn.Sequential(*layers); self.head=nn.Linear(128,1)
    def forward(self,x,mask):
        z=self.temporal(x.transpose(1,2)); m=mask[:,None,:]
        avg=(z*m).sum(2)/m.sum(2).clamp_min(1)
        peak=z.masked_fill(m==0,-1e4).max(2).values
        peak=torch.where(m.sum(2)>0,peak,torch.zeros_like(peak))
        return self.head(torch.cat([avg,peak],1)).squeeze(1)


def metrics(labels, probabilities, threshold):
    y=np.array(labels,dtype=bool); pred=np.array(probabilities)>=threshold
    tp=int((pred & y).sum()); tn=int((~pred & ~y).sum()); fp=int((pred & ~y).sum()); fn=int((~pred & y).sum())
    recall=tp/max(tp+fn,1); specificity=tn/max(tn+fp,1); precision=tp/max(tp+fp,1)
    return dict(tp=tp,tn=tn,fp=fp,fn=fn,recall=recall,specificity=specificity,
        precision=precision,f1=2*precision*recall/max(precision+recall,1e-8),balanced_accuracy=(recall+specificity)/2)


def load_bags(out, manifest, device):
    bags=[]
    for record in manifest['records']:
        with np.load(out/'keypoints'/(record['id']+'.npz'),allow_pickle=False) as data:
            x,m,ends=windows(data)
        bags.append((record,torch.tensor(x,device=device),torch.tensor(m,device=device),ends))
    return bags


def score(model,bags):
    model.eval(); results=[]
    with torch.no_grad():
        for r,x,m,ends in bags:
            logits=model(x,m); usable=m.mean(1)>=0.5
            probability=float(torch.sigmoid(logits[usable].max()).item()) if usable.any() else 0.0
            results.append(dict(id=r['id'],subject=r['subject'],label=r['label'],probability=probability,
                usable_windows=int(usable.sum().item()),window_probabilities=torch.sigmoid(logits).cpu().tolist(),window_end_seconds=ends))
    return results


def train(out,device,epochs):
    torch.manual_seed(42); np.random.seed(42)
    if device == '0': device='cuda:0'
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    splits=json.loads((out/'splits.json').read_text(encoding='utf-8'))
    signatures=[]
    for record in manifest['records']:
        with np.load(out/'keypoints'/(record['id']+'.npz'),allow_pickle=False) as data:
            signatures.append(json.loads(str(data['signature'])))
    if any(s['hz'] != HZ or s['imgsz'] != 640 for s in signatures) or len({s['model_sha256'] for s in signatures}) != 1:
        raise ValueError('Mixed/incompatible keypoint extraction settings')
    pose_model_sha256=signatures[0]['model_sha256']
    bags=load_bags(out,manifest,device)
    groups={k:[b for b in bags if b[0]['subject'] in subjects] for k,subjects in splits.items()}
    if any(not b for b in groups.values()): raise ValueError('Empty split')
    model=TemporalNet().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
    best=float('inf'); patience=0; history=[]; checkpoint=out/'temporal_fall.pt'
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for i in np.random.permutation(len(groups['train'])):
            r,x,m,_=groups['train'][i]; usable=m.mean(1)>=0.5
            if not usable.any(): continue
            logits=model(x,m)[usable]
            # Multiple-instance learning: only the video bag is labelled.
            bag_logit=logits.max()
            loss=nn.functional.binary_cross_entropy_with_logits(bag_logit,torch.tensor(float(r['label']),device=device))
            if r['label']==0:
                loss=(loss+nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits)))/2
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); losses.append(float(loss.item()))
        val=score(model,groups['validation'])
        yy=np.array([r['label'] for r in val]); pp=np.clip([r['probability'] for r in val],1e-6,1-1e-6)
        val_loss=float(-(yy*np.log(pp)+(1-yy)*np.log(1-pp)).mean())
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=val_loss); history.append(row)
        print(row,flush=True)
        if val_loss<best-1e-4:
            best=val_loss; patience=0
            torch.save(dict(state_dict=model.state_dict(),hz=HZ,steps=STEPS,features=FEATURES,epoch=epoch,
                splits=splits,pose_model_sha256=pose_model_sha256,method='weak video labels, MIL max pooling; experimental'),checkpoint)
        else: patience+=1
        if patience>=12: break
    saved=torch.load(checkpoint,map_location=device,weights_only=True); model.load_state_dict(saved['state_dict'])
    val=score(model,groups['validation'])
    thresholds=np.linspace(0.05,0.95,91)
    threshold=float(max(thresholds,key=lambda t:metrics([r['label'] for r in val],[r['probability'] for r in val],t)['balanced_accuracy']))
    saved['threshold']=threshold; torch.save(saved,checkpoint)
    report=dict(best_epoch=saved['epoch'],threshold=threshold,threshold_selected_on='Subject 3 only',
        limitation='Video-level classification metrics, not event timing or camera-independent accuracy. Window scores are not calibrated event probabilities.',history=history,splits=splits)
    for split in groups:
        scores=score(model,groups[split]); report[split]=dict(metrics=metrics([r['label'] for r in scores],[r['probability'] for r in scores],threshold),videos=scores)
    save_json(out/'evaluation.json',report)
    print('FINAL',json.dumps({k:report[k]['metrics'] for k in groups}),flush=True)


def predict(args):
    from ultralytics import YOLO
    device='cuda:0' if args.device=='0' else args.device
    saved=torch.load(args.checkpoint,map_location=device,weights_only=True)
    if (saved['hz'],saved['steps'],saved['features']) != (HZ,STEPS,FEATURES): raise ValueError('Incompatible preprocessing')
    if hashlib.sha256(args.weights.read_bytes()).hexdigest() != saved['pose_model_sha256']:
        raise ValueError('Use the same YOLO pose weights as training')
    model=TemporalNet().to(device);model.load_state_dict(saved['state_dict']);model.eval()
    data=extract_video(YOLO(str(args.weights)),args.video,args.device,args.batch,args.crop,args.start,args.end)
    x,m,ends=windows(data)
    with torch.no_grad(): probs=torch.sigmoid(model(torch.tensor(x,device=device),torch.tensor(m,device=device))).cpu().numpy()
    records=[dict(window_end_seconds=end,fall_score=float(p),observed_fraction=float(mm.mean()),
        candidate=bool(p>=saved['threshold'] and mm.mean()>=0.5)) for p,mm,end in zip(probs,m,ends)]
    save_json(args.output,dict(threshold=saved['threshold'],video=str(args.video),crop=args.crop,
        warning='Experimental window scores; no event-level accuracy claim. Screen recordings differ from raw camera input.',windows=records))
    print('Prediction saved:',args.output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--dataset',type=Path,default=ROOT/'GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.0');p.add_argument('--out',type=Path,default=OUT)
    p=sub.add_parser('extract');p.add_argument('--out',type=Path,default=OUT);p.add_argument('--weights',type=Path,default=ROOT/'yolo26n-pose.pt');p.add_argument('--device',default='0' if torch.cuda.is_available() else 'cpu');p.add_argument('--batch',type=int,default=8)
    p=sub.add_parser('train');p.add_argument('--out',type=Path,default=OUT);p.add_argument('--device',default='0' if torch.cuda.is_available() else 'cpu');p.add_argument('--epochs',type=int,default=60)
    p=sub.add_parser('predict');p.add_argument('--video',type=Path,required=True);p.add_argument('--checkpoint',type=Path,default=OUT/'temporal_fall.pt');p.add_argument('--weights',type=Path,default=ROOT/'yolo26n-pose.pt');p.add_argument('--device',default='0' if torch.cuda.is_available() else 'cpu');p.add_argument('--batch',type=int,default=8);p.add_argument('--crop',type=int,nargs=4);p.add_argument('--start',type=float,default=0);p.add_argument('--end',type=float);p.add_argument('--output',type=Path,default=OUT/'prediction.json')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.dataset,args.out)
    elif args.command=='extract':extract(args.out,args.weights,args.device,args.batch)
    elif args.command=='train':train(args.out,args.device,args.epochs)
    else:predict(args)

if __name__=='__main__': main()
