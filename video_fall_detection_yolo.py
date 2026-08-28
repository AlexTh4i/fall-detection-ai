"""
Fall Detection - YOLOv11-Pose (ho tro MULTI-PERSON) + State Machine rieng cho tung nguoi
Input: video file (vd 15s)
Output: video co annotation + trang thai tung nguoi

Khac biet lon nhat so voi ban MediaPipe: track duoc NHIEU NGUOI cung luc,
moi nguoi co state machine (NORMAL/FALL_SUSPECTED/FALL_CONFIRMED) DOC LAP.

Cach chay:
    python video_fall_detection_yolo.py
    (hoac: python video_fall_detection_yolo.py input.mp4 output.mp4)
"""

import cv2
import math
import sys
from collections import deque
from ultralytics import YOLO

# ==================== CAU HINH NGUONG (giong ban MediaPipe, de so sanh cong bang) ====================
ANGLE_THRESHOLD = 50
SPEED_THRESHOLD = 0.35
CONFIRM_DURATION = 3.0
MOVEMENT_TOLERANCE = 0.03
HISTORY_SIZE = 5

STATE_NORMAL = "NORMAL"
STATE_SUSPECTED = "FALL_SUSPECTED"
STATE_CONFIRMED = "FALL_CONFIRMED"

# YOLOv11-Pose keypoint index (chuan COCO 17 diem)
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12


class PersonTracker:
    """
    Moi nguoi duoc track se co 1 object rieng, giu state machine + lich su vi tri RIENG
    -> tranh nham lan giua nguoi nay voi nguoi khac
    """
    def __init__(self, track_id):
        self.track_id = track_id
        self.state = STATE_NORMAL
        self.suspected_start_frame = None
        self.suspected_position = None
        self.position_history = deque(maxlen=HISTORY_SIZE)
        self.time_history = deque(maxlen=HISTORY_SIZE)
        self.fall_confirmed_frame = None
        self.last_seen_frame = 0

    def reset_to_normal(self):
        self.state = STATE_NORMAL
        self.suspected_start_frame = None
        self.suspected_position = None

    def update(self, keypoints_xy, frame_idx, fps):
        self.last_seen_frame = frame_idx
        video_time = frame_idx / fps

        left_shoulder = keypoints_xy[KP_LEFT_SHOULDER]
        right_shoulder = keypoints_xy[KP_RIGHT_SHOULDER]
        left_hip = keypoints_xy[KP_LEFT_HIP]
        right_hip = keypoints_xy[KP_RIGHT_HIP]

        if (left_shoulder[0] == 0 and left_shoulder[1] == 0) or \
           (left_hip[0] == 0 and left_hip[1] == 0):
            return None, 0.0

        mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
        mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        mid_hip_x = (left_hip[0] + right_hip[0]) / 2
        mid_hip_y = (left_hip[1] + right_hip[1]) / 2

        body_angle = self._calculate_angle(mid_shoulder_x, mid_shoulder_y, mid_hip_x, mid_hip_y)
        centroid = ((mid_shoulder_x + mid_hip_x) / 2, (mid_shoulder_y + mid_hip_y) / 2)

        self.position_history.append(centroid)
        self.time_history.append(video_time)
        movement_speed = self._calculate_speed()

        self._update_state_machine(body_angle, movement_speed, centroid, frame_idx, fps)

        return body_angle, movement_speed

    @staticmethod
    def _calculate_angle(sx, sy, hx, hy):
        dx = hx - sx
        dy = hy - sy
        angle_rad = math.atan2(abs(dx), abs(dy) + 1e-6)
        return math.degrees(angle_rad)

    def _calculate_speed(self):
        if len(self.position_history) < 2:
            return 0.0
        total_distance = 0.0
        for i in range(1, len(self.position_history)):
            x1, y1 = self.position_history[i - 1]
            x2, y2 = self.position_history[i]
            total_distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_time = self.time_history[-1] - self.time_history[0]
        if total_time <= 0:
            return 0.0
        return total_distance / total_time

    def _update_state_machine(self, body_angle, movement_speed, centroid, frame_idx, fps):
        if self.state == STATE_NORMAL:
            if body_angle > ANGLE_THRESHOLD and movement_speed > SPEED_THRESHOLD:
                self.state = STATE_SUSPECTED
                self.suspected_start_frame = frame_idx
                self.suspected_position = centroid

        elif self.state == STATE_SUSPECTED:
            elapsed = (frame_idx - self.suspected_start_frame) / fps
            dx = centroid[0] - self.suspected_position[0]
            dy = centroid[1] - self.suspected_position[1]
            moved_distance = math.sqrt(dx ** 2 + dy ** 2)

            if body_angle < ANGLE_THRESHOLD - 10 or moved_distance > MOVEMENT_TOLERANCE * 3:
                self.reset_to_normal()
            elif elapsed >= CONFIRM_DURATION:
                self.state = STATE_CONFIRMED
                if self.fall_confirmed_frame is None:
                    self.fall_confirmed_frame = frame_idx

        elif self.state == STATE_CONFIRMED:
            if body_angle < ANGLE_THRESHOLD - 10:
                self.reset_to_normal()


def get_state_color(state):
    if state == STATE_NORMAL:
        return (0, 255, 0)
    elif state == STATE_SUSPECTED:
        return (0, 165, 255)
    else:
        return (0, 0, 255)


def process_video(input_path, output_path):
    model = YOLO('yolo11n-pose.pt')

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"LOI: Khong mo duoc video: {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video input: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trackers = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False, device=0)

        if results[0].keypoints is not None and results[0].boxes is not None and \
           results[0].boxes.id is not None:

            keypoints_all = results[0].keypoints.xy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for person_idx, track_id in enumerate(track_ids):
                if track_id not in trackers:
                    trackers[track_id] = PersonTracker(track_id)

                tracker = trackers[track_id]
                kp_normalized = keypoints_all[person_idx].copy()
                kp_normalized[:, 0] /= width
                kp_normalized[:, 1] /= height

                body_angle, movement_speed = tracker.update(kp_normalized, frame_idx, fps)

                color = get_state_color(tracker.state)
                x1, y1, x2, y2 = boxes[person_idx].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                kp_raw = keypoints_all[person_idx]
                skeleton_pairs = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
                                   (11,12),(11,13),(13,15),(12,14),(14,16)]
                for (a, b) in skeleton_pairs:
                    xa, ya = kp_raw[a]
                    xb, yb = kp_raw[b]
                    if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                        cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)
                for (px, py) in kp_raw:
                    if px > 0 and py > 0:
                        cv2.circle(frame, (int(px), int(py)), 3, (255, 0, 0), -1)

                label = f"ID{track_id}: {tracker.state}"
                if tracker.state == STATE_SUSPECTED:
                    elapsed = (frame_idx - tracker.suspected_start_frame) / fps
                    remaining = max(0, CONFIRM_DURATION - elapsed)
                    label += f" ({remaining:.1f}s)"

                cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} | So nguoi dang track: {len(trackers)}",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"Da xu ly {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()

    print(f"\nHOAN TAT. Video output luu tai: {output_path}")
    print(f"Tong so nguoi da track duoc trong video: {len(trackers)}")
    for track_id, tracker in trackers.items():
        if tracker.fall_confirmed_frame is not None:
            print(f"=> Nguoi ID {track_id}: XAC NHAN TE NGA tai frame {tracker.fall_confirmed_frame} "
                  f"(khoang giay thu {tracker.fall_confirmed_frame / fps:.1f}s)")
        else:
            print(f"=> Nguoi ID {track_id}: khong phat hien te nga")


if __name__ == "__main__":
    INPUT_VIDEO = "test/sample_2.mp4"
    OUTPUT_VIDEO = "test/sample_2_yolo11n_output.mp4"

    if len(sys.argv) == 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    elif len(sys.argv) == 1:
        input_path = INPUT_VIDEO
        output_path = OUTPUT_VIDEO
    else:
        print("Cach dung:")
        print("  Cach 1: python video_fall_detection_yolo.py")
        print("  Cach 2: python video_fall_detection_yolo.py input.mp4 output.mp4")
        sys.exit(1)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    process_video(input_path, output_path)
