"""
Fall Detection - Xu ly tu VIDEO FILE (khong can webcam/realtime)
Input: video ngan (vd 15s) co san
Output: video moi voi khung xuong + trang thai NORMAL/FALL_SUSPECTED/FALL_CONFIRMED duoc ve len tung frame

Cach chay:
    python video_fall_detection.py input.mp4 output.mp4
"""

import cv2
import mediapipe as mp
import math
import sys
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ==================== CAU HINH NGUONG (giong ban realtime, co the chinh lai) ====================
ANGLE_THRESHOLD = 50
SPEED_THRESHOLD = 0.25
CONFIRM_DURATION = 1.0          # giay
MOVEMENT_TOLERANCE = 0.03
HISTORY_SIZE = 5

STATE_NORMAL = "NORMAL"
STATE_SUSPECTED = "FALL_SUSPECTED"
STATE_CONFIRMED = "FALL_CONFIRMED"


def calculate_body_angle(shoulder, hip):
    dx = hip.x - shoulder.x
    dy = hip.y - shoulder.y
    angle_rad = math.atan2(abs(dx), abs(dy) + 1e-6)
    return math.degrees(angle_rad)


def get_centroid(landmarks):
    key_indices = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP
    ]
    xs = [landmarks[i].x for i in key_indices]
    ys = [landmarks[i].y for i in key_indices]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def calculate_movement_speed(position_history, time_history):
    if len(position_history) < 2:
        return 0.0
    total_distance = 0.0
    for i in range(1, len(position_history)):
        x1, y1 = position_history[i - 1]
        x2, y2 = position_history[i]
        total_distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    total_time = time_history[-1] - time_history[0]
    if total_time <= 0:
        return 0.0
    return total_distance / total_time


def process_video(input_path, output_path):
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,           # Model nang cao, nhan dien chinh xac hon
        min_detection_confidence=0.65, # Tang nguong phat hien de tranh nhan nham vat the
        min_tracking_confidence=0.65
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"LOI: Khong mo duoc video: {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # fallback neu video khong doc duoc fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video input: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # State machine
    current_state = STATE_NORMAL
    suspected_start_frame = None
    suspected_position = None

    position_history = deque(maxlen=HISTORY_SIZE)
    time_history = deque(maxlen=HISTORY_SIZE)  # dung "thoi gian video" (frame_idx / fps) thay vi time.time()

    frame_idx = 0
    fall_confirmed_frame = None   # ghi lai frame nao xac nhan te, de bao cao

    def reset_to_normal():
        nonlocal current_state, suspected_start_frame, suspected_position
        current_state = STATE_NORMAL
        suspected_start_frame = None
        suspected_position = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        video_time = frame_idx / fps  # "thoi gian" tinh theo video, khong phai thoi gian thuc te xu ly

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        body_angle = None
        movement_speed = 0.0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

            # Kiem tra do tin cay (visibility) cac khop cot loi de loc vat the / oto
            core_visibilities = [
                left_shoulder.visibility,
                right_shoulder.visibility,
                left_hip.visibility,
                right_hip.visibility
            ]
            avg_vis = sum(core_visibilities) / len(core_visibilities)

            # Chi xu ly neu thuc su la con nguoi ro rang (visibility du cao)
            if avg_vis >= 0.5:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                class Point:
                    def __init__(self, x, y):
                        self.x = x
                        self.y = y

                mid_shoulder = Point((left_shoulder.x + right_shoulder.x) / 2,
                                      (left_shoulder.y + right_shoulder.y) / 2)
                mid_hip = Point((left_hip.x + right_hip.x) / 2,
                                 (left_hip.y + right_hip.y) / 2)

                body_angle = calculate_body_angle(mid_shoulder, mid_hip)
                centroid = get_centroid(landmarks)

                position_history.append(centroid)
                time_history.append(video_time)
                movement_speed = calculate_movement_speed(position_history, time_history)

                # ==================== STATE MACHINE ====================
                if current_state == STATE_NORMAL:
                    is_lying_posture = body_angle > ANGLE_THRESHOLD
                    is_sudden_movement = movement_speed > SPEED_THRESHOLD
                    if is_lying_posture and is_sudden_movement:
                        current_state = STATE_SUSPECTED
                        suspected_start_frame = frame_idx
                        suspected_position = centroid

                elif current_state == STATE_SUSPECTED:
                    elapsed = (frame_idx - suspected_start_frame) / fps
                    dx = centroid[0] - suspected_position[0]
                    dy = centroid[1] - suspected_position[1]
                    moved_distance = math.sqrt(dx ** 2 + dy ** 2)

                    if body_angle < ANGLE_THRESHOLD - 10 or moved_distance > MOVEMENT_TOLERANCE * 3:
                        reset_to_normal()
                    elif elapsed >= CONFIRM_DURATION:
                        current_state = STATE_CONFIRMED
                        if fall_confirmed_frame is None:
                            fall_confirmed_frame = frame_idx

                elif current_state == STATE_CONFIRMED:
                    if body_angle < ANGLE_THRESHOLD - 10:
                        reset_to_normal()

        # ==================== VE THONG TIN LEN FRAME ====================
        if current_state == STATE_NORMAL:
            color = (0, 255, 0)
            text = "BINH THUONG"
        elif current_state == STATE_SUSPECTED:
            color = (0, 165, 255)
            elapsed = (frame_idx - suspected_start_frame) / fps
            remaining = max(0, CONFIRM_DURATION - elapsed)
            text = f"NGHI NGO TE NGA - xac nhan sau {remaining:.1f}s"
        else:
            color = (0, 0, 255)
            text = "!!! DA XAC NHAN TE NGA !!!"

        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if body_angle is not None:
            cv2.putText(frame, f"Goc than: {body_angle:.1f} do | Toc do: {movement_speed:.3f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} | t={video_time:.1f}s",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if current_state == STATE_CONFIRMED:
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 8)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"Da xu ly {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    pose.close()

    print(f"\nHOAN TAT. Video output luu tai: {output_path}")
    if fall_confirmed_frame is not None:
        print(f"=> Phat hien TE NGA XAC NHAN tai frame {fall_confirmed_frame} "
              f"(khoang giay thu {fall_confirmed_frame / fps:.1f}s trong video)")
    else:
        print("=> Khong phat hien te nga xac nhan trong video nay.")


if __name__ == "__main__":
    # ==================== SUA 2 DONG DUOI DAY MOI LAN MUON DOI VIDEO TEST ====================
    INPUT_VIDEO = "test/video_6.mp4"          # video dau vao, dat trong folder test/
    OUTPUT_VIDEO = "test/video_6_mediapipe_output.mp4"  # video ket qua se duoc luu, cung trong folder test/
    # ============================================================================

    # Van cho phep chay theo cach cu: python video_fall_detection.py input.mp4 output.mp4
    # Neu khong truyen tham so, se dung 2 duong dan cau hinh o tren
    if len(sys.argv) == 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    elif len(sys.argv) == 1:
        input_path = INPUT_VIDEO
        output_path = OUTPUT_VIDEO
    else:
        print("Cach dung:")
        print("  Cach 1 (dung cau hinh san trong code): python video_fall_detection.py")
        print("  Cach 2 (tu nhap duong dan):             python video_fall_detection.py input.mp4 output.mp4")
        sys.exit(1)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    process_video(input_path, output_path)