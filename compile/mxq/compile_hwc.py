#!/usr/bin/env python3
"""
ONNX → MXQ 범용 컴파일 스크립트.
Mobilint qbcompiler Docker 컨테이너 안에서 실행.

기본적으로 assets/onnx/*.onnx 를 스캔해서 assets/mxq/*.mxq 로 컴파일.
NPU_ONNX_DIR/NPU_MXQ_DIR 환경변수로 진단·실험 폴더를 별도로 지정할 수 있음.
이미 .mxq 가 존재하는 모델은 건너뜀 (재컴파일하려면 해당 .mxq 삭제).

설정:
  - backend="onnx"
  - HWC 캘리브레이션 (calib_hwc/, shape: [224, 224, 3])
  - inference_scheme: "single" | "multi" | "global" | "global4" | "global8"

디렉토리 구조:
  compile/mxq/compile_hwc.py   ← this file
  assets/calib_hwc/        ← HWC calibration tensor (compile/calibration/generate_hwc.py)
  assets/onnx/*.onnx       ← 입력 모델
  assets/mxq/*.mxq         ← 출력 컴파일 모델

실행:
  python3 /workspace/npu/compile/mxq/compile_hwc.py
"""

import glob
import json
import os
from qbcompiler import mxq_compile

# ── 컴파일 설정 ────────────────────────────────────────────────────────────────
# "single" | "multi" | "global" | "global4" | "global8"
INFERENCE_SCHEME = os.environ.get("NPU_INFERENCE_SCHEME", "global8")

# 파일명에 이 문자열이 포함된 .onnx 만 컴파일. None 이면 전체 스캔.
FILE_FILTER = os.environ.get("NPU_ONNX_FILTER", "_npusafe") or None

# 미지원 op(dynamic gather 등)을 CPU 서브그래프로 분리
CPU_OFFLOAD = True

# qbcompiler 프리셋. None 이면 아래 MANUAL_QUANT_ARGS 사용.
#   "vision_transformer" → calibration(method=1, mode=0) + transformer 활성값 16bit
#   "classification"     → calibration(mode=1, output=0)
CONFIG_PRESET = "vision_transformer"

# CONFIG_PRESET=None 일 때만 적용. 프리셋보다 우선순위가 높아서
# 프리셋과 같이 넘기면 프리셋 값을 덮어써 버린다.
MANUAL_QUANT_ARGS = {
    "quantization_method": 0,   # WChALayer: per-channel weight, per-layer activation
    "quantization_mode":   2,   # histogram
    "hist_search_type":    2,   # kl-divergence
    "quantization_output": 0,   # Layer: per-layer output quantization
}

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
NPU_ROOT   = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
ASSETS_DIR = os.path.join(NPU_ROOT, "assets")

# ── 캘리브레이션 파일 목록 ──────────────────────────────────────────────────────
CALIB_HWC_DIR = os.environ.get(
    "NPU_CALIB_DIR", os.path.join(ASSETS_DIR, "calib_hwc")
)
CALIB_HWC_DIR = os.path.abspath(CALIB_HWC_DIR)
hwc_files = sorted(glob.glob(os.path.join(CALIB_HWC_DIR, "*.npy")))
if not hwc_files:
    raise FileNotFoundError(f"No HWC calibration .npy files found in {CALIB_HWC_DIR}")

profile_path = os.path.join(CALIB_HWC_DIR, "preprocess_profile.json")
if os.path.isfile(profile_path):
    with open(profile_path) as handle:
        profile_metadata = json.load(handle)
    print("Calibration preprocess profile: %s" % profile_metadata.get("profile", "unknown"))
else:
    print(
        "[WARN] preprocess_profile.json is missing in %s; "
        "verify calibration/runtime preprocessing manually" % CALIB_HWC_DIR
    )

CALIB_TXT = os.path.join(BASE_DIR, "calib_hwc.txt")
with open(CALIB_TXT, "w") as f:
    f.write("\n".join(hwc_files) + "\n")
print(f"Calibration list: {len(hwc_files)} HWC files -> {CALIB_TXT}")

# ── ONNX 파일 자동 스캔 ────────────────────────────────────────────────────────
ONNX_DIR = os.path.abspath(
    os.environ.get("NPU_ONNX_DIR", os.path.join(ASSETS_DIR, "onnx"))
)
MXQ_DIR = os.path.abspath(
    os.environ.get("NPU_MXQ_DIR", os.path.join(ASSETS_DIR, "mxq"))
)
os.makedirs(MXQ_DIR, exist_ok=True)

onnx_files = sorted(glob.glob(os.path.join(ONNX_DIR, "*.onnx")))
if FILE_FILTER:
    onnx_files = [f for f in onnx_files if FILE_FILTER in os.path.basename(f)]
if not onnx_files:
    raise FileNotFoundError(f"No .onnx files matching '{FILE_FILTER}' found in {ONNX_DIR}")

print(f"Found {len(onnx_files)} ONNX model(s) (filter='{FILE_FILTER}')")

# ── 컴파일 ───────────────────────────────────────────────────────────────────
# 프리셋 사용 시 quantization_* kwargs 를 넘기지 않는다 (우선순위가 더 높아 프리셋을 덮어씀).
if CONFIG_PRESET:
    quant_args = {"config_preset": CONFIG_PRESET}
    tag        = f"_{CONFIG_PRESET.replace('vision_transformer', 'vt')}"
else:
    quant_args = dict(MANUAL_QUANT_ARGS)
    tag        = ""

for onnx_path in onnx_files:
    stem     = os.path.splitext(os.path.basename(onnx_path))[0]
    mxq_path = os.path.join(MXQ_DIR, f"{stem}{tag}_{INFERENCE_SCHEME}.mxq")

    if os.path.exists(mxq_path):
        print(f"\n[SKIP] 이미 존재: {os.path.basename(mxq_path)}")
        continue

    print(f"\n{'='*60}")
    print(f"Compiling: {os.path.basename(onnx_path)}")
    print(f"  output      : {mxq_path}")
    print(f"  scheme      : {INFERENCE_SCHEME}")
    print(f"  cpu_offload : {CPU_OFFLOAD}")
    print(f"  quant       : {CONFIG_PRESET or MANUAL_QUANT_ARGS}")
    print(f"{'='*60}")

    mxq_compile(
        model=onnx_path,
        calib_data_path=CALIB_TXT,
        save_path=mxq_path,
        backend="onnx",
        inference_scheme=INFERENCE_SCHEME,
        cpu_offload=CPU_OFFLOAD,
        **quant_args,
    )

    print(f"Saved: {mxq_path}")

print("\nAll done.")
