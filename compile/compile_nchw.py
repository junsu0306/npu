#!/usr/bin/env python3
"""
EfficientViT-B0/B1 NCHW ONNX → MXQ compile script
Runs inside the Mobilint qbcompiler Docker container.

Key settings (confirmed working in QUANTIZE_ISSUE_SUMMARY_KR.md):
  - backend="onnx"
  - in_dformats={"input": "NCHW"}  : standard NCHW ONNX, no NHWC conversion
  - cpu_offload=True               : required for EfficientViT unsupported ops

Directory layout (relative to npu/):
  compile/compile_nchw.py   ← this file
  compile/calib_nchw.txt    ← calibration file list (paths inside Docker)
  assets/onnx/*.onnx        ← input models
  assets/*.mxq              ← output compiled models

Docker run (mount host npu/ so that calib_nchw.txt absolute paths resolve):
  docker run -it --ipc=host --name qbcompiler \
    -v /home/airlab_compression/npu:/home/airlab_compression/npu \
    mobilint/qbcompiler:v0.9.0.2 /bin/bash

Then inside container:
  python3 /home/airlab_compression/npu/compile/compile_nchw.py
"""

import glob
import os
from qbcompiler import mxq_compile

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))          # .../npu/compile
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))
CALIB_DIR  = os.path.join(ASSETS_DIR, "calib_nchw")

# Regenerate the calibration file list with paths resolved relative to this
# script. The checked-in calib_nchw.txt holds hardcoded absolute paths from a
# different host/mount (/home/airlab_compression/npu/...); on another server the
# Docker mount is /workspace/npu/..., so those paths fail to load (0 tensors).
# Building the list from the actual .npy files keeps it mount-independent.
CALIB_TXT  = os.path.join(BASE_DIR, "calib_nchw.txt")
calib_files = sorted(glob.glob(os.path.join(CALIB_DIR, "*.npy")))
if not calib_files:
    raise FileNotFoundError(f"No calibration .npy files found in {CALIB_DIR}")
with open(CALIB_TXT, "w") as f:
    f.write("\n".join(calib_files) + "\n")
print(f"Calibration list: {len(calib_files)} files -> {CALIB_TXT}")

MXQ_DIR = os.path.join(ASSETS_DIR, "mxq")
os.makedirs(MXQ_DIR, exist_ok=True)

MODELS = [
    (
        os.path.join(ASSETS_DIR, "onnx", "efficientvit_b0_r224_timm_nchw.onnx"),
        os.path.join(MXQ_DIR, "efficientvit_b0_r224_timm_nchw.mxq"),
    ),
    (
        os.path.join(ASSETS_DIR, "onnx", "efficientvit_b1_r224_timm_nchw.onnx"),
        os.path.join(MXQ_DIR, "efficientvit_b1_r224_timm_nchw.mxq"),
    ),
    # 원본(mit-han-lab) repo 가중치 → export_original_to_onnx.py 로 생성한 ONNX
    (
        os.path.join(ASSETS_DIR, "onnx", "efficientvit_b0_original_r224_nchw.onnx"),
        os.path.join(MXQ_DIR, "efficientvit_b0_original_r224_nchw.mxq"),
    ),
    (
        os.path.join(ASSETS_DIR, "onnx", "efficientvit_b1_original_r224_nchw.onnx"),
        os.path.join(MXQ_DIR, "efficientvit_b1_original_r224_nchw.mxq"),
    ),
]

for onnx_path, mxq_path in MODELS:
    if not os.path.exists(onnx_path):
        print(f"\n[SKIP] ONNX 없음: {onnx_path}")
        continue

    if os.path.exists(mxq_path):
        print(f"\n[SKIP] 이미 존재: {mxq_path} (재컴파일하려면 파일 삭제)")
        continue

    print(f"\n{'='*60}")
    print(f"Compiling: {os.path.basename(onnx_path)}")
    print(f"  calib : {CALIB_TXT}")
    print(f"  output: {mxq_path}")
    print(f"{'='*60}")

    mxq_compile(
        model=onnx_path,
        calib_data_path=CALIB_TXT,
        save_path=mxq_path,
        backend="onnx",
        in_dformats={"input": "NCHW"},
        cpu_offload=True,
        quantize_method="Percentile",
        quantize_percentile=0.99999,  # was 0.99995 — too aggressive for GELU/attention
        is_quant_ch=True,
    )

    print(f"Saved: {mxq_path}")

print("\nAll done.")
