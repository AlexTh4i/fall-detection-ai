"""
Xuất 2 mô hình (yolo26n-pose.pt & temporal_fall.pt) sang định dạng .onnx chuẩn cho App Android (Red Magic 7).
Có thể chạy ngay trên máy cá nhân hoặc sau khi tải temporal_fall.pt mới từ Kaggle về.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gmdcsa_pipeline import FEATURES, HZ, STEPS, TemporalNet

ROOT = Path(__file__).resolve().parent


class TemporalNetONNXWrapper(nn.Module):
    """Wrapper gộp Sigmoid để trả trực tiếp xác suất té ngã [0.0, 1.0] trên Android."""

    def __init__(self, base_model: TemporalNet):
        super().__init__()
        self.base = base_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.base(x, mask))


def export_models(
    yolo_weights: Path,
    temporal_ckpt: Path,
    out_dir: Path,
    imgsz: int = 640,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Xuất TemporalNet -> temporal_fall.onnx
    saved = torch.load(temporal_ckpt, map_location="cpu", weights_only=True)
    model = TemporalNet().cpu().eval()
    model.load_state_dict(saved["state_dict"])
    wrapper = TemporalNetONNXWrapper(model).eval()

    dummy_x = torch.randn(1, STEPS, FEATURES, dtype=torch.float32)
    dummy_mask = torch.ones(1, STEPS, dtype=torch.float32)
    temporal_onnx = out_dir / "temporal_fall.onnx"

    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_mask),
        str(temporal_onnx),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["features", "mask"],
        output_names=["fall_probability"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "mask": {0: "batch_size"},
            "fall_probability": {0: "batch_size"},
        },
    )
    print(f"[OK] Đã xuất TemporalNet ONNX: {temporal_onnx} ({temporal_onnx.stat().st_size / 1024:.1f} KB)")

    # Kiểm tra khớp đầu ra nếu đã cài onnxruntime
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(temporal_onnx), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            pt_out = wrapper(dummy_x, dummy_mask).numpy()
        ort_out = sess.run(None, {"features": dummy_x.numpy(), "mask": dummy_mask.numpy()})[0]
        diff = float(np.max(np.abs(pt_out - ort_out)))
        print(f"[OK] Kiểm chứng PyTorch vs ONNX Runtime max diff: {diff:.2e}")
    except ImportError:
        print("[INFO] Bỏ qua bước kiểm chứng onnxruntime (chưa cài onnxruntime trong môi trường hiện tại).")

    # 2. Xuất YOLO26n-Pose -> yolo26n-pose.onnx
    from ultralytics import YOLO

    yolo_model = YOLO(str(yolo_weights))
    exported_path = Path(yolo_model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True))
    target_yolo_onnx = out_dir / "yolo26n-pose.onnx"
    if exported_path.exists() and exported_path.resolve() != target_yolo_onnx.resolve():
        shutil.move(str(exported_path), str(target_yolo_onnx))
    print(f"[OK] Đã xuất YOLO26n-Pose ONNX: {target_yolo_onnx} ({target_yolo_onnx.stat().st_size / 1024:.1f} KB)")

    # 3. Lưu metadata cho App Android
    meta = {
        "hz": int(saved.get("hz", HZ)),
        "steps": int(saved.get("steps", STEPS)),
        "features": int(saved.get("features", FEATURES)),
        "threshold": round(float(saved.get("threshold", 0.43)), 4),
        "pose_model_sha256": saved.get("pose_model_sha256", ""),
        "input_size": imgsz,
    }
    meta_path = out_dir / "model_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] Đã lưu cấu hình metadata: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo", type=Path, default=ROOT / "yolo26n-pose.pt")
    parser.add_argument("--temporal", type=Path, default=ROOT / "data/gmdcsa24_experiment/temporal_fall.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "export_models")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    export_models(args.yolo, args.temporal, args.out, args.imgsz)

