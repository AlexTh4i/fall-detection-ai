# Nhận diện té ngã realtime

Webcam → YOLO26-Pose và tracking → chuỗi tư thế từng người → temporal CNN → cảnh báo trên màn hình.

`realtime_yolo_fall_detection.py` mặc định dùng mô hình đã train. Repository có cả `yolo26n-pose.pt` và `data/gmdcsa24_experiment/temporal_fall.pt`; không cần tải dataset hoặc train lại để chạy webcam.

## Cài đặt trên Windows

Dùng Python 3.11 và Git. Mở PowerShell:

```powershell
git clone https://github.com/AlexTh4i/fall-detection-ai.git
cd fall-detection-ai
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
```

Chọn **một** cách cài PyTorch trước khi cài các thư viện còn lại.

CPU, dành cho máy không có NVIDIA GPU:

```powershell
.\venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

NVIDIA GPU, cấu hình CUDA 11.8 đã chạy thử với GTX 1070:

```powershell
.\venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

Lệnh PyTorch theo [hướng dẫn phiên bản chính thức](https://pytorch.org/get-started/previous-versions/#v271). Máy dùng GPU cần driver NVIDIA tương thích. Sau đó:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe realtime_yolo_fall_detection.py
```

Nếu webcam mặc định không đúng, chọn camera 1:

```powershell
.\venv\Scripts\python.exe realtime_yolo_fall_detection.py 1
```

Cho phép ứng dụng desktop truy cập camera trong Windows Settings nếu không mở được webcam. CPU có thể chạy chậm hơn cấu hình GPU đã thử.

## Chạy và so sánh

```powershell
# Temporal CNN đã train — mặc định
.\venv\Scripts\python.exe realtime_yolo_fall_detection.py --mode temporal

# Luật góc/W-H và FSM cũ
.\venv\Scripts\python.exe realtime_yolo_fall_detection.py --mode rules
```

Temporal cần khoảng 3 giây thu lịch sử khi bắt đầu track. HUD hiển thị `WARMING_UP`, điểm nghi ngờ và trạng thái; cảnh báo thử nghiệm cần điểm cao duy trì 1 giây. Mất người/điểm khớp được đánh dấu `UNKNOWN`. Q hoặc ESC: thoát; R: reset lịch sử; F: đổi mirror và reset; S: lưu snapshot. Temporal khởi động không mirror để phù hợp dữ liệu train.

## Kiểm thử không cần dataset

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests dùng bộ xương mẫu tự tạo và trọng số có trong repository. Để replay video của bạn mà không mở cửa sổ:

```powershell
.\venv\Scripts\python.exe realtime_yolo_fall_detection.py --video "video.mp4" --headless --max-frames 150
```

## Kết quả và giới hạn

Bản thử nghiệm temporal CNN học bằng nhãn theo video từ GMDCSA24: train Subject 1–2 (80 video), validation Subject 3 (43), test Subject 4 (37). Trên test: nhận ra 15/17 video ngã, báo nhầm 4/20 video bình thường, F1 83,33%. Đây là kết quả phân loại video; bộ lọc cảnh báo realtime chưa được đánh giá theo sự kiện. Vẫn có lỗi ở góc camera trên cao, chống đẩy và chủ động nằm; không bảo đảm mọi góc camera hoặc nhiều người.

Xem [kết quả chi tiết](GMDCSA_RESULTS.md), [pipeline huấn luyện](GMDCSA_TRAINING.md) và [context realtime](REALTIME_CONTEXT.md). Dataset và video cá nhân không nằm trong repository. Huấn luyện hoặc replay báo cáo FSM trên dataset cần tải dữ liệu riêng theo hướng dẫn; chạy webcam và tests không cần dataset.

Dữ liệu nghiên cứu: [GMDCSA24](https://doi.org/10.5281/zenodo.12921216), Ekram Alam và cộng sự (2024). Chỉ số của checkpoint hiện tại dựa trên bản dữ liệu đã dùng trong thí nghiệm; cần đánh giá thêm trên dữ liệu độc lập để xác định khả năng tổng quát.
