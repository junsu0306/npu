#!/usr/bin/env python3
"""
ResNet50 sanity-check compile script.

EfficientViT 예측이 계속 틀리는 원인이 pipeline 문제인지
모델 아키텍처 문제인지 구분하기 위한 테스트.

Inside container:
  pip install timm
  python3 /home/airlab_compression/npu/compile/mxq/compile_resnet50.py
"""

import glob
import os
import torch
import timm
from qbcompiler import mxq_compile

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
NPU_ROOT   = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
ASSETS_DIR = os.path.join(NPU_ROOT, "assets")

CALIB_HWC_DIR = os.path.join(ASSETS_DIR, "calib_hwc")
hwc_files = sorted(glob.glob(os.path.join(CALIB_HWC_DIR, "*.npy")))[:200]
if not hwc_files:
    raise FileNotFoundError(f"No HWC calibration .npy files found in {CALIB_HWC_DIR}")

CALIB_TXT = os.path.join(BASE_DIR, "calib_hwc.txt")
with open(CALIB_TXT, "w") as f:
    f.write("\n".join(hwc_files) + "\n")
print(f"Calibration list: {len(hwc_files)} files")

MXQ_DIR = os.path.join(ASSETS_DIR, "mxq")
os.makedirs(MXQ_DIR, exist_ok=True)

mxq_path = os.path.join(MXQ_DIR, "resnet50_timm.mxq")
if os.path.exists(mxq_path):
    print(f"[SKIP] 이미 존재: {mxq_path}")
else:
    model = timm.create_model("resnet50.a1_in1k", pretrained=True)
    model.eval().cpu()

    mxq_compile(
        model=model,
        save_path=mxq_path,
        calib_data_path=CALIB_TXT,
        backend="torch",
        feed_dict={"x": torch.randn(1, 3, 224, 224).cpu()},
        quantization_method=0,
        quantization_mode=0,   # max — 가장 단순
        quantization_output=0,
    )
    print(f"Saved: {mxq_path}")

print(f"""
테스트:
  python3 benchmark/runtime/test_npu.py --model assets/mxq/resnet50_timm.mxq --labels imagenet_classes.txt --n 10
  python3 benchmark/runtime/test_npu.py --model assets/mxq/resnet50_timm.mxq --labels imagenet_classes.txt --n 10 --chw
""")
