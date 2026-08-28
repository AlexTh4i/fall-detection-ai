"""
Fall Detection - MediaPipe Pose + State Machine
Khop voi state machine trong file IoT spec: NORMAL -> FALL_SUSPECTED -> FALL_CONFIRMED/CANCELLED

Logic:
1. Tinh goc than nguoi + vi tri trong tam moi frame
2. So sanh voi frame truoc do -> tinh TOC DO thay doi (khong chi tu the tinh)
3. Neu thay doi dot ngot (toc do cao) + tu the nam -> chuyen FALL_SUSPECTED
4. Dem thoi gian xac nhan (vd 3 giay) -> kiem tra co tiep tuc BAT DONG khong
5. Bat dong du lau -> FALL_CONFIRMED
   Cu dong tro lai (dung day) -> CANCELLED, quay ve NORMAL
"""

import cv2
import mediapipe as mp
import math
import time
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# ==================== CAU HINH NGUONG (can hieu chinh sau khi test thuc te) ====================
ANGLE_THRESHOLD = 50           # do - goc than nguoi vuot qua nay coi la tu the nam
SPEED_THRESHOLD = 0.35         # toc do thay doi trong tam (don vi: ty le man hinh / giay) - phat hien "dot ngot"
CONFIRM_DURATION = 3.0         # giay - thoi gian cho xac nhan bat dong sau khi nghi ngo te
MOVEMENT_TOLERANCE = 0.03      # ty le man hinh - neu di chuyen it hon nay coi la "bat dong"
HISTORY_SIZE = 5               # so frame gan nhat de tinh toc do trung binh (giam nhieu)

# ==================== STATE MACHINE ====================
STATE_NORMAL = "NORMAL"
STATE_SUSPECTED = "FALL_SUSPECTED"
STATE_CONFIRMED = "FALL_CONFIRMED"

current_state = STATE_NORMAL
suspected_start_time = None
suspected_position = None  # vi tri trong tam luc bat dau nghi ngo, de so sanh bat dong

# Luu lich su vi tri trong tam va thoi gian de tinh toc do
position_history = deque(maxlen=HISTORY_SIZE)
time_history = deque(maxlen=HISTORY_SIZE)

prev_time = 0


def calculate_body_angle(shoulder, hip):
    dx = hip.x - shoulder.x
    dy = hip.y - shoulder.y
    angle_rad = math.atan2(abs(dx), abs(dy) + 1e-6)
    return math.degrees(angle_rad)


def get_centroid(landmarks):
    """Trong tam co the = trung binh toa do tat ca keypoints chinh (vai, hong)"""
    key_indices = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP
    ]
    xs = [landmarks[i].x for i in key_indices]
    ys = [landmarks[i].y for i in key_indices]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def calculate_movement_speed(position_history, time_history):
    """
    Tinh toc do di chuyen trung binh cua trong tam trong HISTORY_SIZE frame gan nhat
    Tra ve: khoang cach di chuyen / thoi gian (don vi ty le man hinh / giay)
    """
    if len(position_history) < 2:
        return 0.0

    total_distance = 0.0
    for i in range(1, len(position_history)):
        x1, y1 = position_history[i - 1]
        x2, y2 = position_history[i]
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += dist

    total_time = time_history[-1] - time_history[0]
    if total_time <= 0:
        return 0.0

    return total_distance / total_time


def reset_to_normal():
    global current_state, suspected_start_time, suspected_position
    current_state = STATE_NORMAL
    suspected_start_time = None
    suspected_position = None


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    now = time.time()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    body_angle = None
    movement_speed = 0.0

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

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
        time_history.append(now)
        movement_speed = calculate_movement_speed(position_history, time_history)

        # ==================== STATE MACHINE LOGIC ====================
        if current_state == STATE_NORMAL:
            # Dieu kien chuyen sang NGHI NGO: tu the nam + toc do thay doi dot ngot
            is_lying_posture = body_angle > ANGLE_THRESHOLD
            is_sudden_movement = movement_speed > SPEED_THRESHOLD

            if is_lying_posture and is_sudden_movement:
                current_state = STATE_SUSPECTED
                suspected_start_time = now
                suspected_position = centroid

        elif current_state == STATE_SUSPECTED:
            elapsed = now - suspected_start_time

            # Kiem tra da di chuyen ra khoi vi tri nghi ngo chua (dung day lai)
            dx = centroid[0] - suspected_position[0]
            dy = centroid[1] - suspected_position[1]
            moved_distance = math.sqrt(dx ** 2 + dy ** 2)

            if body_angle < ANGLE_THRESHOLD - 10 or moved_distance > MOVEMENT_TOLERANCE * 3:
                # Nguoi da dung day / thay doi tu the ro ret -> HUY canh bao
                reset_to_normal()
            elif elapsed >= CONFIRM_DURATION:
                # Het thoi gian xac nhan ma van nam bat dong -> XAC NHAN TE
                current_state = STATE_CONFIRMED

        elif current_state == STATE_CONFIRMED:
            # Da xac nhan te - trong thuc te se goi API canh bao o day (POST /events/fall)
            # Cho phep reset ve NORMAL neu nguoi dung day sau do (demo/test)
            if body_angle < ANGLE_THRESHOLD - 10:
                reset_to_normal()

    else:
        # Khong phat hien nguoi trong khung hinh -> khong doi trang thai (tranh nhieu)
        pass

    # ==================== HIEN THI ====================
    if current_state == STATE_NORMAL:
        color = (0, 255, 0)
        text = "BINH THUONG"
    elif current_state == STATE_SUSPECTED:
        color = (0, 165, 255)
        remaining = max(0, CONFIRM_DURATION - (now - suspected_start_time))
        text = f"NGHI NGO TE NGA - xac nhan sau {remaining:.1f}s"
    else:  # CONFIRMED
        color = (0, 0, 255)
        text = "!!! DA XAC NHAN TE NGA !!!"

    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if body_angle is not None:
        cv2.putText(frame, f"Goc than: {body_angle:.1f} do | Toc do: {movement_speed:.3f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 1)
    cv2.putText(frame, "Nhan 'q' de thoat | 'r' de reset trang thai", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Vien mau do neu da confirmed, de de nhan biet
    if current_state == STATE_CONFIRMED:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

    cv2.imshow('Fall Detection - State Machine Test', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        reset_to_normal()
        position_history.clear()
        time_history.clear()

cap.release()
cv2.destroyAllWindows()
pose.close()