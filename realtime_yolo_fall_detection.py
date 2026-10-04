"""
Real-time Fall Detection AI - YOLO26-Pose + Temporal CNN / Legacy Rules
Chay truc tiep qua WEBCAM / Camera Real-time.

Toi uu hoa toc do (High FPS Optimization):
1. Su dung backend DirectShow (cv2.CAP_DSHOW) tren Windows + MJPG Codec de dat 30-60 FPS tu webcam.
2. Tang toc suy luan tren GPU NVIDIA CUDA (GTX 1070) voi Warm-up khoi tao.
3. Bo loc chong bao dong gia: Ket hop Body Angle (>50 deg), Aspect Ratio W/H (>=0.88), Hip Drop va Timeout (2.5s).
"""

import cv2
import math
import argparse
import hashlib
import time
import os
import torch
import numpy as np
from collections import deque
from pathlib import Path
from ultralytics import YOLO

# ==================== CAU HINH THAM SO ====================
WEBCAM_INDEX = 0               # Index webcam (0 la webcam mac dinh)
MODEL_NAME = "yolo26n-pose.pt" # yolov8n-pose.pt (Nhe, 30-60 FPS tren GPU) hoac yolov8m-pose.pt
INFERENCE_SIZE = 640           # 640 cho realtime muot ma
TRACK_CONFIDENCE = 0.25        # Do tin cay toi thieu de track nguoi

# Nguong logic nhan dien te nga chuan xac
ANGLE_THRESHOLD = 50           # Goc nghieng than nguoi (>50 do)
SPEED_THRESHOLD = 0.20         # Van toc thay doi trong tam dot ngot (ty le man hinh / giay)
CONFIRM_DURATION = 2.5         # Thoi gian duy tri bat dong de xac nhan te nga (2.5 giay)
MOVEMENT_TOLERANCE = 0.03      # Nguong dich chuyen toi da de coi la bat dong
ASPECT_RATIO_THRESHOLD = 0.88  # Ti le Width/Height cua Bounding Box: Nguoi nam san thuong W/H >= 0.90

STATE_NORMAL = "NORMAL"
STATE_SUSPECTED = "FALL_SUSPECTED"
STATE_CONFIRMED = "FALL_CONFIRMED"
STATE_WARMING = "WARMING_UP"
STATE_UNKNOWN = "UNKNOWN"
PROJECT_ROOT = Path(__file__).resolve().parent
TEMPORAL_CHECKPOINT = PROJECT_ROOT / "data/gmdcsa24_experiment/temporal_fall.pt"
TEMPORAL_CONFIRM_SECONDS = 1.0  # Experimental persistence filter; not event-validated.

# YOLO Pose Keypoint Connections (COCO 17 keypoints)
KEYPOINT_CONNECTIONS = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


class PersonTracker:
    def __init__(self, track_id):
        self.track_id = track_id
        self.state = STATE_NORMAL
        self.suspected_start_time = None
        self.suspected_pos = None
        self.confirmed_time = None

        self.position_history = deque(maxlen=5)
        self.time_history = deque(maxlen=5)
        self.speed_history = deque(maxlen=15)
        self.angle_history = deque(maxlen=5)
        self.last_seen = time.time()
        self.aspect_ratio = 0.0

    def reset_to_normal(self):
        self.state = STATE_NORMAL
        self.suspected_start_time = None
        self.suspected_pos = None
        self.confirmed_time = None
        self.position_history.clear()
        self.time_history.clear()
        self.speed_history.clear()

    def update(self, keypoints, box, current_time):
        self.last_seen = current_time
        ls, rs = keypoints[5], keypoints[6]    # Vai trai, vai phai
        lh, rh = keypoints[11], keypoints[12]  # Hong trai, hong phai
        la, ra = keypoints[15], keypoints[16]  # Co chan trai, phai

        if ls[2] < 0.20 or rs[2] < 0.20 or lh[2] < 0.20 or rh[2] < 0.20:
            return None, 0.0, self.aspect_ratio

        bw = abs(box[2] - box[0])
        bh = abs(box[3] - box[1])
        self.aspect_ratio = bw / (bh + 1e-6)

        mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        mid_hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

        dx = mid_hip[0] - mid_shoulder[0]
        dy = mid_hip[1] - mid_shoulder[1]
        body_angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

        has_ankles = (la[2] > 0.20 and ra[2] > 0.20)
        mid_ankle_y = (la[1] + ra[1]) / 2 if has_ankles else None
        hip_ankle_drop = (mid_ankle_y - mid_hip[1]) if mid_ankle_y is not None else 0.0

        centroid = ((ls[0] + rs[0] + lh[0] + rh[0]) / 4,
                    (ls[1] + rs[1] + lh[1] + rh[1]) / 4)

        self.position_history.append(centroid)
        self.time_history.append(current_time)
        self.angle_history.append(body_angle)

        speed = 0.0
        if len(self.position_history) >= 2:
            total_dist = 0.0
            for i in range(1, len(self.position_history)):
                p1 = self.position_history[i - 1]
                p2 = self.position_history[i]
                total_dist += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_time = self.time_history[-1] - self.time_history[0]
            if total_time > 0:
                speed = total_dist / total_time

        self.speed_history.append(speed)
        recent_max_speed = max(self.speed_history) if self.speed_history else 0.0

        # Dieu kien tu the nam san that su
        is_bounding_box_horizontal = self.aspect_ratio >= ASPECT_RATIO_THRESHOLD
        is_standing_bent = (self.aspect_ratio < 0.75) or (has_ankles and hip_ankle_drop > 0.35)
        is_true_lying = (body_angle > ANGLE_THRESHOLD) and (is_bounding_box_horizontal or body_angle > 75) and (not is_standing_bent)

        # State Machine Logic
        if self.state == STATE_NORMAL:
            had_fast_motion = recent_max_speed > SPEED_THRESHOLD
            if is_true_lying and (had_fast_motion or (body_angle > 70 and is_bounding_box_horizontal)):
                self.state = STATE_SUSPECTED
                self.suspected_start_time = current_time
                self.suspected_pos = centroid

        elif self.state == STATE_SUSPECTED:
            elapsed = current_time - self.suspected_start_time
            dx_m = centroid[0] - self.suspected_pos[0]
            dy_m = centroid[1] - self.suspected_pos[1]
            moved_dist = math.sqrt(dx_m**2 + dy_m**2)

            if body_angle < 38 or self.aspect_ratio < 0.70 or moved_dist > MOVEMENT_TOLERANCE * 3:
                self.reset_to_normal()
            elif elapsed >= CONFIRM_DURATION:
                if is_true_lying:
                    self.state = STATE_CONFIRMED
                    if self.confirmed_time is None:
                        self.confirmed_time = current_time
                        print(f"\n[CANH BAO] Phat hien nguoi ID {self.track_id} TE NGA luc {time.strftime('%H:%M:%S')}! (Angle: {body_angle:.0f}deg, W/H: {self.aspect_ratio:.2f})")
                else:
                    self.reset_to_normal()

        elif self.state == STATE_CONFIRMED:
            if body_angle < 38 or self.aspect_ratio < 0.70:
                print(f"[THONG TIN] Nguoi ID {self.track_id} da dung day tro lai binh thuong.")
                self.reset_to_normal()

        return body_angle, speed, self.aspect_ratio


class TemporalBackend:
    """Share trained weights and the exact training preprocessing across people."""
    def __init__(self, checkpoint, pose_weights, device):
        from gmdcsa_pipeline import TemporalNet, features, HZ, STEPS, FEATURES
        self.device = "cuda:0" if device == 0 else "cpu"
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if (saved["hz"], saved["steps"], saved["features"]) != (HZ, STEPS, FEATURES):
            raise ValueError("Checkpoint preprocessing does not match gmdcsa_pipeline.py")
        if hashlib.sha256(Path(pose_weights).read_bytes()).hexdigest() != saved["pose_model_sha256"]:
            raise ValueError("Temporal mode requires the same YOLO weights used for training")
        self.hz, self.steps = HZ, STEPS
        self.threshold = float(saved["threshold"])
        self.features = features
        self.model = TemporalNet().to(self.device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    def predict(self, samples):
        data = {name: np.stack([s[i] for s in samples])
                for i, name in enumerate(["pose", "box", "time"])}
        x, mask = self.features(data)
        # Keep one extra sample so the first window velocity has prior context.
        x, mask = x[-self.steps:], mask[-self.steps:]
        observed = float(mask.mean())
        if observed < 0.5 or mask[-1] == 0:
            return None, observed
        with torch.inference_mode():
            score = torch.sigmoid(self.model(
                torch.as_tensor(x[None], device=self.device),
                torch.as_tensor(mask[None], device=self.device))).item()
        return float(score), observed


class TemporalPersonTracker:
    def __init__(self, track_id, backend):
        self.track_id, self.backend = track_id, backend
        self.samples = deque(maxlen=backend.steps + 1)
        self.last_seen = 0.0
        self.reset_to_normal()

    def reset_to_normal(self):
        self.samples.clear()
        self.next_sample = None
        self.state = STATE_WARMING
        self.score = None
        self.observed = 0.0
        self.suspected_start_time = None
        self.low_start_time = None

    def update(self, pixel_keypoints, box, frame_shape, current_time):
        self.last_seen = current_time
        scale = max(frame_shape[:2])
        pose = np.asarray(pixel_keypoints, dtype=np.float32).copy()
        pose[:, :2] /= scale
        normalized_box = np.asarray(box, dtype=np.float32) / scale
        self._sample(pose, normalized_box, current_time)

    def mark_missing(self, current_time):
        # Missing observations are not recovery or fabricated stationary poses.
        self._sample(np.zeros((17, 3), np.float32), np.zeros(4, np.float32), current_time)
        self.state, self.score = STATE_UNKNOWN, None
        self.suspected_start_time = self.low_start_time = None

    def _sample(self, pose, box, now):
        period = 1.0 / self.backend.hz
        if self.samples and now - self.samples[-1][2] > self.backend.steps * period:
            self.reset_to_normal()
        if self.next_sample is None:
            self.next_sample = now
        if now + 1e-8 < self.next_sample:
            return
        while self.next_sample < now - period * 0.5:
            self.samples.append((np.zeros((17, 3), np.float32), np.zeros(4, np.float32), self.next_sample))
            self.next_sample += period
        self.samples.append((pose, box, now))
        self.next_sample += period
        if len(self.samples) < self.backend.steps:
            self.state, self.score = STATE_WARMING, None
            return
        self.score, self.observed = self.backend.predict(list(self.samples))
        if self.score is None:
            self.state = STATE_UNKNOWN
            self.suspected_start_time = self.low_start_time = None
            return
        if self.score >= self.backend.threshold:
            self.low_start_time = None
            if self.state != STATE_CONFIRMED:
                if self.suspected_start_time is None:
                    self.suspected_start_time = now
                self.state = STATE_SUSPECTED
                if now - self.suspected_start_time >= TEMPORAL_CONFIRM_SECONDS:
                    self.state = STATE_CONFIRMED
                    print(f"[TEMPORAL EXPERIMENTAL] ID {self.track_id}: high fall score {self.score:.3f}")
        else:
            self.suspected_start_time = None
            if self.state == STATE_CONFIRMED:
                if self.low_start_time is None:
                    self.low_start_time = now
                if now - self.low_start_time >= TEMPORAL_CONFIRM_SECONDS:
                    self.state = STATE_NORMAL
                    self.low_start_time = None
                    print(f"[TEMPORAL] ID {self.track_id}: fall score decreased; not a verified recovery")
            else:
                self.state = STATE_NORMAL


def run_realtime_fall_detection(cam_index=0, model_path=MODEL_NAME, mode="temporal",
                                checkpoint=TEMPORAL_CHECKPOINT, video=None,
                                headless=False, max_frames=None):
    # Kiem tra GPU CUDA
    has_cuda = torch.cuda.is_available()
    device = 0 if has_cuda else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "CPU Only"

    print("=" * 65)
    print("Khoi tao Fall Detection Real-time (GPU Accelerated)")
    print(f"- Model: {model_path}")
    print(f"- Thiet bi tinh toan (Device): {gpu_name}")
    print(f"- Camera Index: {cam_index}")
    model_path = Path(model_path)
    if not model_path.is_absolute() and (PROJECT_ROOT / model_path).exists():
        model_path = PROJECT_ROOT / model_path
    temporal = TemporalBackend(checkpoint, model_path, device) if mode == "temporal" else None
    if temporal:
        print(f"- Mode: TEMPORAL EXPERIMENTAL | window {temporal.steps / temporal.hz:.1f}s | score >= {temporal.threshold:.2f}")
        print("- Model con bao nham/bo sot; khong bao dam moi goc camera.")
    else:
        print(f"- Mode: RULES | Angle > {ANGLE_THRESHOLD}deg | W/H >= {ASPECT_RATIO_THRESHOLD} | Time >= {CONFIRM_DURATION}s")
    print("=" * 65)

    print("Dang tai model YOLO-Pose...")
    model = YOLO(str(model_path))

    # Warm-up GPU truoc khi vao camera
    if has_cuda:
        dummy_img = np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8)
        for _ in range(3):
            model.track(dummy_img, persist=True, verbose=False, device=device, imgsz=INFERENCE_SIZE)
        torch.cuda.synchronize()

    # Mo camera voi backend DirectShow de dat FPS toi da tren Windows
    if video is not None:
        cap = cv2.VideoCapture(str(video))
    elif os.name == 'nt':
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened() and video is None:
        cap.release()
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Khong the mo nguon hinh anh: {video if video is not None else cam_index}")

    # Ep Webcam dung MJPG de khong bi nghen co chai bang thong USB
    if video is None:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    trackers = {}
    is_flipped = mode == "rules"  # Temporal training videos were not mirrored.
    prev_time = time.monotonic() if temporal else time.time()
    fps_smooth = 30.0
    frame_index = 0
    video_fps = cap.get(cv2.CAP_PROP_FPS) if video is not None else None
    if video is not None and video_fps <= 0:
        cap.release()
        raise ValueError("Video has invalid FPS")
    if video is not None:
        prev_time = -1.0 / video_fps

    print("\n>>> DANG CHAY REALTIME. Nhan 'q' de thoat, 'r' de reset, 'f' de doi mirror, 's' de chup anh <<<\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[CANH BAO] Khong doc duoc frame tu camera!")
                break

            if is_flipped:
                frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape
            now = frame_index / video_fps if video is not None else (time.monotonic() if temporal else time.time())
            frame_index += 1

            # Tinh toan FPS
            dt = now - prev_time
            prev_time = now
            current_fps = (1.0 / dt) if dt > 0 else 30.0
            fps_smooth = 0.85 * fps_smooth + 0.15 * current_fps

            # Chay YOLO Tracking tren GPU
            results = model.track(
                frame,
                persist=True,
                verbose=False,
                conf=TRACK_CONFIDENCE,
                imgsz=INFERENCE_SIZE,
                device=device,
                classes=[0]
            )

            any_confirmed_fall = False
            active_track_ids = set()

            if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None and results[0].keypoints is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                kpts_xyn = results[0].keypoints.xyn.cpu().numpy()
                kpts_xy = results[0].keypoints.xy.cpu().numpy()
                kpts_conf = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else np.ones((len(boxes), 17))

                for box, track_id, xy, pixel_xy, conf in zip(boxes, track_ids, kpts_xyn, kpts_xy, kpts_conf):
                    active_track_ids.add(track_id)
                    kpts = np.hstack([xy, conf[:, None]])

                    if track_id not in trackers:
                        trackers[track_id] = TemporalPersonTracker(track_id, temporal) if temporal else PersonTracker(track_id)

                    tracker = trackers[track_id]
                    if temporal:
                        tracker.update(np.hstack([pixel_xy, conf[:, None]]), box, frame.shape, now)
                        angle = None
                    else:
                        angle, speed, ar = tracker.update(kpts, box, now)

                    state = tracker.state
                    if state == STATE_CONFIRMED:
                        box_color = (0, 0, 255)       # Do
                        any_confirmed_fall = True
                    elif state == STATE_SUSPECTED:
                        box_color = (0, 165, 255)     # Cam
                    elif state in (STATE_WARMING, STATE_UNKNOWN):
                        box_color = (160, 160, 160)
                    else:
                        box_color = (0, 255, 0)       # Xanh la

                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    tag = f"ID:{track_id} | {state}"
                    if temporal:
                        if tracker.score is not None:
                            tag += f" | score:{tracker.score:.2f}"
                        elif state == STATE_WARMING:
                            tag += f" | {min(len(tracker.samples), temporal.steps)}/{temporal.steps}"
                    elif state == STATE_SUSPECTED and tracker.suspected_start_time is not None:
                        remain = max(0.0, CONFIRM_DURATION - (now - tracker.suspected_start_time))
                        tag += f" ({remain:.1f}s)"
                    elif angle is not None:
                        tag += f" | {angle:.0f}deg | W/H:{ar:.2f}"

                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
                    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), box_color, -1)
                    cv2.putText(frame, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

                    # Ve Skeleton
                    for p1_idx, p2_idx in KEYPOINT_CONNECTIONS:
                        if kpts[p1_idx][2] > 0.20 and kpts[p2_idx][2] > 0.20:
                            pt1 = (int(kpts[p1_idx][0] * w), int(kpts[p1_idx][1] * h))
                            pt2 = (int(kpts[p2_idx][0] * w), int(kpts[p2_idx][1] * h))
                            cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

                    for kpt in kpts:
                        if kpt[2] > 0.20:
                            cv2.circle(frame, (int(kpt[0] * w), int(kpt[1] * h)), 3, (0, 255, 255), -1)

            if temporal:
                for tid, tracker in trackers.items():
                    if tid not in active_track_ids:
                        tracker.mark_missing(now)

            # Xoa tracker cu khong con xuat hien qua 10 giay
            expired_ids = [tid for tid, trk in trackers.items() if now - trk.last_seen > 10.0]
            for tid in expired_ids:
                del trackers[tid]

            # Top HUD Dashboard
            cv2.rectangle(frame, (0, 0), (w, 45), (30, 30, 30), -1)
            dev_label = gpu_name.split(" ")[-1] if "GeForce" in gpu_name or "NVIDIA" in gpu_name else gpu_name
            mode_label = "Temporal EXPERIMENTAL" if temporal else "Rules"
            cv2.putText(frame, f"{model_path.stem} | {mode_label} | GPU:{dev_label} | FPS:{fps_smooth:.1f}",
                        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Tracking: {len(active_track_ids)} person(s)",
                        (w - 220, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Canh bao toan man hinh khi co TE NGA
            if any_confirmed_fall:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
                cv2.rectangle(frame, (w // 2 - 270, 55), (w // 2 + 270, 105), (0, 0, 255), -1)
                message = "!!! NGHI NGO TE NGA - THU NGHIEM !!!" if temporal else "!!! PHAT HIEN TE NGA (FALL DETECTED) !!!"
                cv2.putText(frame, message,
                            (w // 2 - 255, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # Bottom Guide Bar
            cv2.putText(frame, "[Q]: Thoat | [R]: Reset | [F]: Lat guong | [S]: Chup anh",
                        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if not headless:
                cv2.imshow("Fall Detection AI - Real-time Webcam (YOLO-Pose)", frame)
            key = (cv2.waitKey(1) & 0xFF) if not headless else -1
            if max_frames is not None and frame_index >= max_frames:
                break
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                for trk in trackers.values():
                    trk.reset_to_normal()
                print("[RESET] Dang thu lai chuoi tu the." if temporal else "[RESET] Da reset tat ca trang thai ve NORMAL.")
            elif key == ord('f'):
                is_flipped = not is_flipped
                for tracker in trackers.values():
                    tracker.reset_to_normal()
                print(f"[MIRROR] Che do lat guong: {'BAT' if is_flipped else 'TAT'}")
            elif key == ord('s'):
                snap_name = f"snapshot_fall_{int(time.time())}.jpg"
                cv2.imwrite(snap_name, frame)
                print(f"[SNAPSHOT] Da luu anh chup man hinh tai: {snap_name}")

    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
    print("\nDa dong chuong trinh Real-time Fall Detection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 + trained temporal fall detection (experimental)")
    parser.add_argument("camera", nargs="?", type=int, default=WEBCAM_INDEX)
    parser.add_argument("model", nargs="?", default=MODEL_NAME)
    parser.add_argument("--mode", choices=["temporal", "rules"], default="temporal")
    parser.add_argument("--checkpoint", type=Path, default=TEMPORAL_CHECKPOINT)
    parser.add_argument("--video", type=Path, help="Replay a video using its original timestamps")
    parser.add_argument("--headless", action="store_true", help="Run without an OpenCV window")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames (smoke testing)")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    run_realtime_fall_detection(args.camera, args.model, args.mode, args.checkpoint,
                                args.video, args.headless, args.max_frames)
