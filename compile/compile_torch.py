#!/usr/bin/env python3
"""
EfficientViT-B0/B1 PyTorch → MXQ compile script (backend="torch")
Runs inside the Mobilint qbcompiler Docker container.

ONNX 변환 없이 timm / 원본 PyTorch 모델을 직접 컴파일.

CPU_OFFLOAD 플래그:
  True  → cpu_offload=True: 캘리브레이션 포워드를 CPU에서 실행
           NPU가 지원 못 하는 op(INT64, scalar constant 등)도 통과 가능.
           출력 파일명: *_cpuoffload.mxq
  False → 기본 동작 (이전과 동일)
           출력 파일명: *.mxq

사전 요건 (Docker 안에서):
  pip install timm

Directory layout (relative to npu/):
  compile/compile_torch.py   ← this file
  compile/calib_hwc.txt      ← calibration file list (자동 생성)
  assets/calib_hwc/          ← HWC calibration tensors (.npy)
  assets/torch/              ← 원본 .pt 모델 파일
  assets/mxq/                ← 출력 .mxq 파일

실행:
  python3 /workspace/npu/compile/compile_torch.py
"""

import glob
import os
import sys
import torch
from qbcompiler import mxq_compile

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))

# ── 캘리브레이션 파일 목록 ────────────────────────────────────────────────────
CALIB_HWC_DIR = os.path.join(ASSETS_DIR, "calib_hwc")
hwc_files = sorted(glob.glob(os.path.join(CALIB_HWC_DIR, "*.npy")))
if not hwc_files:
    raise FileNotFoundError(f"No HWC calibration .npy files found in {CALIB_HWC_DIR}")

CALIB_TXT = os.path.join(BASE_DIR, "calib_hwc.txt")
with open(CALIB_TXT, "w") as f:
    f.write("\n".join(hwc_files) + "\n")
print(f"Calibration list: {len(hwc_files)} HWC files -> {CALIB_TXT}")

MXQ_DIR = os.path.join(ASSETS_DIR, "mxq")
PT_DIR  = os.path.join(ASSETS_DIR, "torch")
os.makedirs(MXQ_DIR, exist_ok=True)

# ── 모드 설정 ─────────────────────────────────────────────────────────────────
# cpu_offload=True: 캘리브레이션을 CPU에서 실행 → INT64/scalar constant 문제 우회
CPU_OFFLOAD = True

suffix = "_cpuoffload" if CPU_OFFLOAD else ""

feed_dict = {"x": torch.randn(1, 3, 224, 224).cpu()}

COMPILE_ARGS = dict(
    calib_data_path=CALIB_TXT,
    backend="torch",
    feed_dict=feed_dict,
    quantization_method=0,   # WChALayer: per-channel weight, per-layer activation
    quantization_mode=2,     # histogram
    hist_search_type=2,      # kl-divergence
    quantization_output=0,   # Layer: per-layer output quantization
)

if CPU_OFFLOAD:
    COMPILE_ARGS["cpu_offload"] = True

print(f"Mode: cpu_offload={CPU_OFFLOAD}  suffix='{suffix}'")


def compile_model(model, mxq_path, label):
    if os.path.exists(mxq_path):
        print(f"\n[SKIP] 이미 존재: {mxq_path}  (재컴파일하려면 파일 삭제)")
        return
    print(f"\n{'='*60}")
    print(f"Compiling: {label}  [cpu_offload={CPU_OFFLOAD}]")
    print(f"  output: {mxq_path}")
    print(f"{'='*60}")
    mxq_compile(model=model, save_path=mxq_path, **COMPILE_ARGS)
    print(f"Saved: {mxq_path}")


# ── 1. timm pretrained 모델 ───────────────────────────────────────────────────
try:
    import timm
    TIMM_MODELS = [
        ("efficientvit_b0.r224_in1k",
         os.path.join(MXQ_DIR, f"efficientvit_b0_r224_timm{suffix}.mxq")),
        ("efficientvit_b1.r224_in1k",
         os.path.join(MXQ_DIR, f"efficientvit_b1_r224_timm{suffix}.mxq")),
    ]
    for model_name, mxq_path in TIMM_MODELS:
        model = timm.create_model(model_name, pretrained=True)
        model.eval().cpu()
        compile_model(model, mxq_path, model_name)
except ImportError:
    print("\n[SKIP] timm 미설치 — 원본 모델만 컴파일합니다.")


# ── 2. 원본(mit-han-lab) .pt 모델 ────────────────────────────────────────────
sys.path.insert(0, "/workspace/efficientvit2026")

ORIGINAL_MODELS = [
    (os.path.join(PT_DIR, "efficientvit_b0_original.pt"),
     os.path.join(MXQ_DIR, f"efficientvit_b0_original{suffix}.mxq")),
    (os.path.join(PT_DIR, "efficientvit_b1_original.pt"),
     os.path.join(MXQ_DIR, f"efficientvit_b1_original{suffix}.mxq")),
]

for pt_path, mxq_path in ORIGINAL_MODELS:
    if not os.path.exists(pt_path):
        print(f"\n[SKIP] .pt 없음: {pt_path}")
        continue
    model = torch.load(pt_path, map_location="cpu", weights_only=False)
    model.eval().cpu()
    compile_model(model, mxq_path, os.path.basename(pt_path))


print("\nAll done.")
