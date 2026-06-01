#!/usr/bin/env python3
"""
EfficientViT-B0/B1 ONNX → MXQ compile script
Runs inside the Mobilint qbcompiler Docker container.

Key settings:
  - backend="onnx"
  - in_dformats omitted : NPU works natively in HWC (NHWC); omitting lets the
      compiler insert correct NCHW↔NHWC transposes at every CPU/NPU boundary.
      Setting in_dformats="NCHW" skips those boundary transposes, causing
      catastrophic accuracy loss when cpu_offload=True splits the graph.
  - cpu_offload=True    : required for EfficientViT unsupported ops (attention/GELU)
  - calib_nhwc/         : HWC-ordered calibration data (transposed from calib_nchw/)

Directory layout (relative to npu/):
  compile/compile_nchw.py   ← this file
  compile/calib_nhwc.txt    ← calibration file list (paths inside Docker)
  assets/calib_nchw/        ← CHW source calibration tensors
  assets/calib_nhwc/        ← HWC calibration tensors (auto-generated below)
  assets/onnx/*.onnx        ← input models
  assets/mxq/*.mxq          ← output compiled models

Docker run (mount host npu/ so that calib_nhwc.txt absolute paths resolve):
  docker run -it --ipc=host --name qbcompiler \
    -v /home/airlab_compression/npu:/home/airlab_compression/npu \
    mobilint/qbcompiler:v0.9.0.2 /bin/bash

Then inside container:
  python3 /home/airlab_compression/npu/compile/compile_nchw.py
"""

import glob
import os
import numpy as np
from qbcompiler import mxq_compile

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))          # .../npu/compile
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))

# NPU expects HWC (NHWC) input — transpose the existing CHW calibration files.
CALIB_CHW_DIR = os.path.join(ASSETS_DIR, "calib_nchw")
CALIB_HWC_DIR = os.path.join(ASSETS_DIR, "calib_nhwc")
os.makedirs(CALIB_HWC_DIR, exist_ok=True)

chw_files = sorted(glob.glob(os.path.join(CALIB_CHW_DIR, "*.npy")))
if not chw_files:
    raise FileNotFoundError(f"No CHW calibration .npy files found in {CALIB_CHW_DIR}")

hwc_files = sorted(glob.glob(os.path.join(CALIB_HWC_DIR, "*.npy")))
if len(hwc_files) < len(chw_files):
    print(f"Transposing {len(chw_files)} CHW→HWC calibration files...")
    for chw_path in chw_files:
        chw = np.load(chw_path)                        # (3, 224, 224)
        hwc = chw.transpose(1, 2, 0)                   # (224, 224, 3)
        dst = os.path.join(CALIB_HWC_DIR, os.path.basename(chw_path))
        np.save(dst, hwc)
    hwc_files = sorted(glob.glob(os.path.join(CALIB_HWC_DIR, "*.npy")))

# Regenerate the calibration file list with paths resolved relative to this
# script so it stays mount-independent across Docker environments.
CALIB_TXT = os.path.join(BASE_DIR, "calib_nhwc.txt")
with open(CALIB_TXT, "w") as f:
    f.write("\n".join(hwc_files) + "\n")
print(f"Calibration list: {len(hwc_files)} HWC files -> {CALIB_TXT}")

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
        # in_dformats intentionally omitted: NPU default is HWC (NHWC).
        # Specifying NCHW prevents the compiler from inserting the required
        # format-transpose ops at CPU/NPU boundaries (cpu_offload splits the
        # graph), causing catastrophic accuracy loss.
        cpu_offload=True,
        quantize_method="Percentile",
        quantize_percentile=0.99999,
        is_quant_ch=True,
    )

    print(f"Saved: {mxq_path}")

print("\nAll done.")
