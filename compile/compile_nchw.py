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

import os
from qbcompiler import mxq_compile

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))          # .../npu/compile
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))
CALIB_TXT  = os.path.join(BASE_DIR, "calib_nchw.txt")

MODELS = [
    (
        os.path.join(ASSETS_DIR, "onnx", "efficientvit_b0_r224_nchw.onnx"),
        os.path.join(ASSETS_DIR, "efficientvit_b0_r224_nchw.mxq"),
    ),
    (
        os.path.join(ASSETS_DIR, "onnx", "efficientvit_b1_r224_nchw.onnx"),
        os.path.join(ASSETS_DIR, "efficientvit_b1_r224_nchw.mxq"),
    ),
]

for onnx_path, mxq_path in MODELS:
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
