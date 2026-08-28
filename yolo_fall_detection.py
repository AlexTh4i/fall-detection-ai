"""
Fall Detection AI - YOLOv8-Pose + Multi-Person Tracking + State Machine
Giai quyet triet de:
1. Chi nhan dien nguoi (Person Class), khong bao gio bat nham o to / vat the.
2. Ho tro tracking da nguoi (Multi-person) trong khung hinh.
3. State Machine nhan dien te nga chuan xac cho tung nguoi rieng biet.
"""

import cv2
import math
import sys
import numpy as np
from collections import deque
from ultralytics import YOLO

# ==================== CAU HINH THAM SO ====================
ANGLE_THRESHOLD = 50       # Goc nghieng than nguoi (>50 do)
SPEED_THRESHOLD = 0.20     # Toc do dich chuyen trong tam
CONFIRM_DURATION = 2.0     # Thoi gian xac nhan bat dong (giay)
TRACK_CONFIDENCE = 0.15    # Nguong nhay cao de bat duoc nguoi o xa va bi che khuat
ASPECT_RATIO_THRESHOLD = 0.88 # Ti le Width / Height
MODEL_NAME = "yolov8m-pose.pt" # Model manh hon nhieu (Medium) de bat vat the nho

STATE_NORMAL = "NORMAL"
STATE_SUSPECTED = "FALL_SUSPECTED"
STATE_CONFIRMED = "FALL_CONFIRMED"

# YOLOv8-Pose Keypoints indices:
# 5: Left Shoulder, 6: Right Shoulder
# 11: Left Hip, 12: Right Hip
# 13: Left Knee, 14: Right Knee
# 15: Left Ankle, 16: Right Ankle
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
        self.confirmed_frame = None
        
        self.position_history = deque(maxlen=5)
        self.time_history = deque(maxlen=5)
        self.speed_history = deque(maxlen=15)
        self.angle_history = deque(maxlen=5)
        self.aspect_ratio = 0.0

    def update(self, keypoints, box, video_time, frame_idx, fps):
        # keypoints: [17, 3] (x, y, conf)
        ls, rs = keypoints[5], keypoints[6]
        lh, rh = keypoints[11], keypoints[12]
        la, ra = keypoints[15], keypoints[16]

        # Kiem tra do tin cay cua cac khop than chinh
        if ls[2] < 0.25 or rs[2] < 0.25 or lh[2] < 0.25 or rh[2] < 0.25:
            return None, 0.0, self.aspect_ratio

        bw = abs(box[2] - box[0])
        bh = abs(box[3] - box[1])
        self.aspect_ratio = bw / (bh + 1e-6)

        mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        mid_hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

        # Tinh goc nghieng than nguoi so voi truc dung
        dx = mid_hip[0] - mid_shoulder[0]
        dy = mid_hip[1] - mid_shoulder[1]
        body_angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

        has_ankles = (la[2] > 0.20 and ra[2] > 0.20)
        mid_ankle_y = (la[1] + ra[1]) / 2 if has_ankles else None
        hip_ankle_drop = (mid_ankle_y - mid_hip[1]) if mid_ankle_y is not None else 0.0

        centroid = ((ls[0] + rs[0] + lh[0] + rh[0]) / 4,
                    (ls[1] + rs[1] + lh[1] + rh[1]) / 4)

        self.position_history.append(centroid)
        self.time_history.append(video_time)
        self.angle_history.append(body_angle)

        # Tinh van toc
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

        is_bounding_box_horizontal = self.aspect_ratio >= ASPECT_RATIO_THRESHOLD
        is_standing_bent = (self.aspect_ratio < 0.75) or (has_ankles and hip_ankle_drop > 0.35)
        is_true_lying = (body_angle > ANGLE_THRESHOLD) and (is_bounding_box_horizontal or body_angle > 75) and (not is_standing_bent)

        # ==================== STATE MACHINE ====================
        if self.state == STATE_NORMAL:
            had_motion = recent_max_speed > SPEED_THRESHOLD

            if is_true_lying and (had_motion or (body_angle > 70 and is_bounding_box_horizontal)):
                self.state = STATE_SUSPECTED
                self.suspected_start_time = video_time
                self.suspected_pos = centroid

        elif self.state == STATE_SUSPECTED:
            elapsed = video_time - self.suspected_start_time
            if body_angle < 38 or self.aspect_ratio < 0.70:
                self.state = STATE_NORMAL
            elif elapsed >= CONFIRM_DURATION:
                if is_true_lying:
                    self.state = STATE_CONFIRMED
                    if self.confirmed_frame is None:
                        self.confirmed_frame = frame_idx
                else:
                    self.state = STATE_NORMAL

        elif self.state == STATE_CONFIRMED:
            if body_angle < 38 or self.aspect_ratio < 0.70:
                self.state = STATE_NORMAL

        return body_angle, speed, self.aspect_ratio


def process_video_yolo(input_path, output_path):
    print(f"Loading YOLOv8-Pose model ({MODEL_NAME})...")
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"LOI: Khong the mo video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    trackers = {}
    frame_idx = 0
    fall_events = []

    print(f"Processing: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        video_time = frame_idx / fps

        # Track person keypoints voi do phan giai cao imgsz=1280
        results = model.track(
            frame,
            persist=True,
            verbose=False,
            conf=TRACK_CONFIDENCE,
            imgsz=1280,
            classes=[0]
        )

        any_confirmed_fall = False

        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            kpts_xyn = results[0].keypoints.xyn.cpu().numpy() # [N, 17, 2]
            kpts_conf = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else np.ones((len(boxes), 17))

            for box, track_id, xy, conf in zip(boxes, track_ids, kpts_xyn, kpts_conf):
                # Gop lai thanh format [17, 3] (x, y, conf)
                kpts = np.hstack([xy, conf[:, None]])

                if track_id not in trackers:
                    trackers[track_id] = PersonTracker(track_id)

                tracker = trackers[track_id]
                angle, speed, ar = tracker.update(kpts, box, video_time, frame_idx, fps)

                # Ve Bounding Box & Skeleton
                x1, y1, x2, y2 = map(int, box)
                state = tracker.state

                if state == STATE_CONFIRMED:
                    box_color = (0, 0, 255) # Do
                    any_confirmed_fall = True
                    if track_id not in [e['id'] for e in fall_events]:
                        fall_events.append({'id': track_id, 'frame': frame_idx, 'time': video_time})
                elif state == STATE_SUSPECTED:
                    box_color = (0, 165, 255) # Cam
                else:
                    box_color = (0, 255, 0) # Xanh

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                tag = f"ID:{track_id} | {state}"
                if angle is not None:
                    tag += f" | {angle:.0f}deg | W/H:{ar:.2f}"
                cv2.putText(frame, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # Ve cac duong noi Skeleton
                for p1_idx, p2_idx in KEYPOINT_CONNECTIONS:
                    if kpts[p1_idx][2] > 0.25 and kpts[p2_idx][2] > 0.25:
                        pt1 = (int(kpts[p1_idx][0] * width), int(kpts[p1_idx][1] * height))
                        pt2 = (int(kpts[p2_idx][0] * width), int(kpts[p2_idx][1] * height))
                        cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

                for kpt in kpts:
                    if kpt[2] > 0.25:
                        cv2.circle(frame, (int(kpt[0] * width), int(kpt[1] * height)), 3, (0, 255, 255), -1)

        if any_confirmed_fall:
            cv2.putText(frame, "!!! PHAT HIEN TE NGA (FALL DETECTED) !!!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 6)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Progress: {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"\nHOAN TAT! Video saved to: {output_path}")
    if fall_events:
        for ev in fall_events:
            print(f"=> Person ID {ev['id']} NGA tai frame {ev['frame']} (t={ev['time']:.1f}s)")
    else:
        print("=> Khong phat hien te nga.")


if __name__ == "__main__":
    in_video = sys.argv[1] if len(sys.argv) > 1 else "test/video_6.mp4"
    out_video = sys.argv[2] if len(sys.argv) > 2 else "test/video_6_yolov8m_output.mp4"
    process_video_yolo(in_video, out_video)
