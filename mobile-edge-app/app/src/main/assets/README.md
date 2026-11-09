# Hướng dẫn nạp Model AI vào App Android (ZTE Nubia Red Magic 7)

Sau khi chạy xong Notebook trên Kaggle (hoặc chạy lệnh `python export_mobile_models.py` trên PC), bạn chỉ cần copy các file sau vào thư mục này:

1. `yolo26n-pose.onnx` (Model YOLO26n bắt 17 khớp xương)
2. `temporal_fall.onnx` (Model TemporalNet phân loại chuỗi ngã)
3. `model_metadata.json` (File cấu hình ngưỡng threshold và kích thước ảnh)

Thư mục đích:
`fall-detection-ai/mobile-edge-app/app/src/main/assets/`

