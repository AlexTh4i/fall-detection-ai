# Ket qua ban thu nghiem GMDCSA24

Da trich YOLO26-Pose cho 160 video: 81 ADL, 79 Fall; 12,891 mau tai 10 Hz. Model la temporal CNN nho, hoc bang nhan theo video va multiple-instance learning, khong phai ST-GCN. Trich xuat va huan luyen da chay tren GTX 1070 qua venv.

Train: Subject 1-2 (80 video). Validation: Subject 3 (43 video). Test: Subject 4 (37 video). Khong dung video OBS de train hoac chon nguong. Checkpoint tot nhat o epoch 10, nguong chon tren validation: 0.43.

Test Subject 4: TP=15/17, FN=2/17, TN=16/20, FP=4/20. Recall=88.24%, precision=78.95%, F1=83.33%. Day la phan loai video, khong phai do chinh xac theo su kien hoac bao dam moi goc camera.

Bon bao nham: Subject4_ADL_05, 06, 07 (chong day); ADL_10 (di roi chu dong nam tren giuong). Hai bo sot: Subject4_Fall_08 va Fall_17 (nga tu ngoi tren giuong). Validation cung nham Subject3_ADL_08 (chu dong nam tren san).

OBS: crop x=190,y=250,w=1280,h=688. Goc tren cao 00:30-00:50: diem thap den khoang 00:43, vuot nguong lan dau o cua so ket thuc 00:43.9. Van chua bat duoc doan dau khi nguoi da nam. Goc ngang 01:25-01:40: vuot nguong tu cua so ket thuc 01:28.9, giam sau khi ngoi day. OBS la quay man hinh co skeleton/HUD, khac video camera goc; cac diem nay chi la kiem tra tham khao, chua co moc nga ground truth.

FSM cu replay tren cung keypoint 10 Hz: SUSPECTED TP=15,FP=7,FN=2,TN=13; CONFIRMED TP=4,FP=2,FN=13,TN=18. Khong so sanh ngang model candidate voi FSM CONFIRMED vi FSM can thoi gian xac nhan 2.5 giay va replay 10 Hz khac camera full FPS. Mot so clip ngan khong du thoi gian xac nhan.

Da tich hop temporal CNN vao realtime_yolo_fall_detection.py (mac dinh --mode temporal; FSM cu van co qua --mode rules). Che do nay la thu nghiem, chua duoc danh gia theo su kien realtime. Can gan moc bat dau/ket thuc nga trong event_annotations.csv (cot reviewed=yes sau khi xem), them ADL kho phan biet va du lieu goc tren cao, roi danh gia theo su kien tren tap kiem thu doc lap. Pipeline hien tai chua tu su dung cac moc trong event_annotations.csv; day la mau thu thap nhan cho vong huan luyen tiep theo.

Files: manifest.json, splits.json, keypoints/*.npz, temporal_fall.pt, evaluation.json, original_rules_evaluation.json, obs_overhead.json, obs_side.json, event_annotations.csv.
