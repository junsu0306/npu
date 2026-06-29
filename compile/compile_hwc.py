#!/usr/bin/env python3
"""
ONNX → MXQ 범용 컴파일 스크립트.
Mobilint qbcompiler Docker 컨테이너 안에서 실행.

assets/onnx/*.onnx 를 자동 스캔해서 assets/mxq/*.mxq 로 컴파일.
이미 .mxq 가 존재하는 모델은 건너뜀 (재컴파일하려면 해당 .mxq 삭제).

설정:
  - backend="onnx"
  - HWC 캘리브레이션 (calib_hwc/, shape: [224, 224, 3])
  - qbcompiler 기본값: HWC 입력

디렉토리 구조:
  compile/compile_hwc.py   ← this file
  assets/calib_hwc/        ← HWC 캘리브레이션 텐서 (gen_calib_hwc.py 로 생성)
  assets/onnx/*.onnx       ← 입력 모델
  assets/mxq/*.mxq         ← 출력 컴파일 모델

실행:
  python3 /workspace/npu/compile/compile_hwc.py
"""

import glob
import os
from qbcompiler import mxq_compile

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))

# ── 캘리브레이션 파일 목록 ──────────────────────────────────────────────────────
CALIB_HWC_DIR = os.path.join(ASSETS_DIR, "calib_hwc")
hwc_files = sorted(glob.glob(os.path.join(CALIB_HWC_DIR, "*.npy")))
if not hwc_files:
    raise FileNotFoundError(f"No HWC calibration .npy files found in {CALIB_HWC_DIR}")

CALIB_TXT = os.path.join(BASE_DIR, "calib_hwc.txt")
with open(CALIB_TXT, "w") as f:
    f.write("\n".join(hwc_files) + "\n")
print(f"Calibration list: {len(hwc_files)} HWC files -> {CALIB_TXT}")

# ── ONNX 파일 자동 스캔 ────────────────────────────────────────────────────────
ONNX_DIR = os.path.join(ASSETS_DIR, "onnx")
MXQ_DIR  = os.path.join(ASSETS_DIR, "mxq")
os.makedirs(MXQ_DIR, exist_ok=True)

onnx_files = sorted(glob.glob(os.path.join(ONNX_DIR, "*.onnx")))
if not onnx_files:
    raise FileNotFoundError(f"No .onnx files found in {ONNX_DIR}")

print(f"Found {len(onnx_files)} ONNX model(s) in {ONNX_DIR}")

# ── 컴파일 ───────────────────────────────────────────────────────────────────
for onnx_path in onnx_files:
    stem    = os.path.splitext(os.path.basename(onnx_path))[0]
    mxq_path = os.path.join(MXQ_DIR, stem + ".mxq")

    if os.path.exists(mxq_path):
        print(f"\n[SKIP] 이미 존재: {os.path.basename(mxq_path)}")
        continue

    print(f"\n{'='*60}")
    print(f"Compiling: {os.path.basename(onnx_path)}")
    print(f"  output : {mxq_path}")
    print(f"{'='*60}")

    mxq_compile(
        model=onnx_path,
        calib_data_path=CALIB_TXT,
        save_path=mxq_path,
        backend="onnx",
        quantization_method=0,   # WChALayer: per-channel weight, per-layer activation
        quantization_mode=2,     # histogram
        hist_search_type=2,      # kl-divergence
        quantization_output=0,   # Layer: per-layer output quantization
    )

    print(f"Saved: {mxq_path}")

print("\nAll done.")
