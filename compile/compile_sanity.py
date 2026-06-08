#!/usr/bin/env python3
"""
양자화 문제 진단용 컴파일 스크립트.

현재 EfficientViT MXQ 모델이 틀린 예측을 내놓는 원인을 찾기 위해
다양한 양자화 설정을 시도한다.

시도 순서:
  1. quantization_mode=0 (max) — 가장 단순, 안정적
  2. quantization_mode=1 (maxPercentile, percentile=0.9999)
  3. quantization_method=1 (WChAMulti: per-channel activation) + histogram/KL

PDF (KR_SDK_qb_Compiler_v1.0.0_User_Manual) 근거:
  - quantization_method: 0=WChALayer, 1=WChAMulti, 2=WChALayerZeropoint, 3=WChAMultiZeropoint
  - quantization_mode:   0=max, 1=maxPercentile, 2=histogram
  - hist_search_type:    0=percentile, 1=mse, 2=kl
  - quantization_output: 0=Layer, 1=Ch, 2=Sigmoid

Docker run:
  docker run -it --ipc=host --name qbcompiler \\
    -v /home/airlab_compression/npu:/home/airlab_compression/npu \\
    mobilint/qbcompiler:v0.9.0.2 /bin/bash

Inside container:
  pip install timm
  python3 /home/airlab_compression/npu/compile/compile_sanity.py
"""

import glob
import os
import torch
import timm
from qbcompiler import mxq_compile

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))

CALIB_HWC_DIR = os.path.join(ASSETS_DIR, "calib_hwc")
hwc_files = sorted(glob.glob(os.path.join(CALIB_HWC_DIR, "*.npy")))
if not hwc_files:
    raise FileNotFoundError(f"No HWC calibration .npy files found in {CALIB_HWC_DIR}")

# 빠른 테스트를 위해 200개만 사용
hwc_files = hwc_files[:200]

CALIB_TXT = os.path.join(BASE_DIR, "calib_hwc.txt")
with open(CALIB_TXT, "w") as f:
    f.write("\n".join(hwc_files) + "\n")
print(f"Calibration list: {len(hwc_files)} HWC files -> {CALIB_TXT}")

MXQ_DIR = os.path.join(ASSETS_DIR, "mxq")
os.makedirs(MXQ_DIR, exist_ok=True)

# B0만 테스트 (빠름)
MODEL_NAME = "efficientvit_b0.r224_in1k"
feed_dict  = {"x": torch.randn(1, 3, 224, 224).cpu()}

VARIANTS = [
    # (suffix, compile_kwargs)
    (
        "max",
        dict(
            quantization_method=0,   # WChALayer
            quantization_mode=0,     # max — 가장 단순
            quantization_output=0,   # Layer
        ),
    ),
    (
        "maxpercentile",
        dict(
            quantization_method=0,   # WChALayer
            quantization_mode=1,     # maxPercentile
            percentile=0.9999,
            quantization_output=0,
        ),
    ),
    (
        "hist_kl_wchAmulti",
        dict(
            quantization_method=1,   # WChAMulti: per-channel activation
            quantization_mode=2,     # histogram
            hist_search_type=2,      # kl
            quantization_output=0,
        ),
    ),
]

model = timm.create_model(MODEL_NAME, pretrained=True)
model.eval().cpu()
print(f"\nModel loaded: {MODEL_NAME}")

for suffix, extra_kwargs in VARIANTS:
    mxq_path = os.path.join(MXQ_DIR, f"efficientvit_b0_r224_timm_{suffix}.mxq")
    if os.path.exists(mxq_path):
        print(f"\n[SKIP] 이미 존재: {mxq_path}")
        continue

    print(f"\n{'='*60}")
    print(f"Compiling variant: {suffix}")
    print(f"  kwargs: {extra_kwargs}")
    print(f"  output: {mxq_path}")
    print(f"{'='*60}")

    mxq_compile(
        model=model,
        save_path=mxq_path,
        calib_data_path=CALIB_TXT,
        backend="torch",
        feed_dict=feed_dict,
        **extra_kwargs,
    )
    print(f"Saved: {mxq_path}")

print("\nAll done. NPU에서 test_npu.py 로 각 mxq 파일을 테스트하세요:")
for suffix, _ in VARIANTS:
    mxq_path = f"assets/mxq/efficientvit_b0_r224_timm_{suffix}.mxq"
    print(f"  python3 test_npu.py --model {mxq_path} --labels imagenet_classes.txt --n 10")
