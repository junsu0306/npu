#!/usr/bin/env python3
"""
EfficientViT-B0/B1 cpu_offload=True 컴파일 스크립트.

EfficientViT의 attention 연산이 INT8 양자화와 충돌하여
cpu_offload=True 로 CPU에서 처리하도록 설정.

cpu_offload=True 시 주의사항 (PDF p.61):
  - calibration 데이터: CHW (3, 224, 224)  ← calib_chw/ 사용
  - runtime 입력:       CHW (3, 224, 224)  ← test_npu.py --chw 사용

사전 준비 (Docker 밖):
  python3 compile/gen_calib_chw.py

Inside Docker:
  pip install timm
  python3 /home/airlab_compression/npu/compile/compile_cpuoffload.py
"""

import glob
import os
import sys
import torch
import timm
from qbcompiler import mxq_compile

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))

# CHW calibration 데이터 (cpu_offload=True 전용)
CALIB_CHW_DIR = os.path.join(ASSETS_DIR, "calib_chw")
chw_files = sorted(glob.glob(os.path.join(CALIB_CHW_DIR, "*.npy")))
if not chw_files:
    raise FileNotFoundError(
        f"No CHW calibration .npy files found in {CALIB_CHW_DIR}\n"
        f"먼저 실행: python3 compile/gen_calib_chw.py"
    )

chw_files = chw_files[:200]
CALIB_TXT = os.path.join(BASE_DIR, "calib_chw.txt")
with open(CALIB_TXT, "w") as f:
    f.write("\n".join(chw_files) + "\n")
print(f"Calibration list: {len(chw_files)} CHW files -> {CALIB_TXT}")

MXQ_DIR = os.path.join(ASSETS_DIR, "mxq")
PT_DIR  = os.path.join(ASSETS_DIR, "torch")
os.makedirs(MXQ_DIR, exist_ok=True)

feed_dict = {"x": torch.randn(1, 3, 224, 224).cpu()}

COMPILE_ARGS = dict(
    calib_data_path=CALIB_TXT,
    backend="torch",
    feed_dict=feed_dict,
    cpu_offload=True,
    quantization_method=0,   # WChALayer
    quantization_mode=0,     # max
    quantization_output=0,   # Layer
)


def compile_model(model, mxq_path, label):
    if os.path.exists(mxq_path):
        print(f"\n[SKIP] 이미 존재: {mxq_path}")
        return
    print(f"\n{'='*60}")
    print(f"Compiling: {label}")
    print(f"  output: {mxq_path}")
    print(f"{'='*60}")
    mxq_compile(model=model, save_path=mxq_path, **COMPILE_ARGS)
    print(f"Saved: {mxq_path}")


# ── 1. timm pretrained 모델 ───────────────────────────────────────────────────
TIMM_MODELS = [
    ("efficientvit_b0.r224_in1k", os.path.join(MXQ_DIR, "efficientvit_b0_r224_timm_cpuoffload.mxq")),
    ("efficientvit_b1.r224_in1k", os.path.join(MXQ_DIR, "efficientvit_b1_r224_timm_cpuoffload.mxq")),
]

for model_name, mxq_path in TIMM_MODELS:
    model = timm.create_model(model_name, pretrained=True)
    model.eval().cpu()
    compile_model(model, mxq_path, model_name)


# ── 2. 원본(mit-han-lab) .pt 모델 ────────────────────────────────────────────
sys.path.insert(0, "/workspace/efficientvit2026")

ORIGINAL_MODELS = [
    (os.path.join(PT_DIR, "efficientvit_b0_original.pt"),
     os.path.join(MXQ_DIR, "efficientvit_b0_original_cpuoffload.mxq")),
    (os.path.join(PT_DIR, "efficientvit_b1_original.pt"),
     os.path.join(MXQ_DIR, "efficientvit_b1_original_cpuoffload.mxq")),
]

for pt_path, mxq_path in ORIGINAL_MODELS:
    if not os.path.exists(pt_path):
        print(f"\n[SKIP] .pt 없음: {pt_path}")
        continue
    model = torch.load(pt_path, map_location="cpu", weights_only=False)
    model.eval().cpu()
    compile_model(model, mxq_path, os.path.basename(pt_path))


print("""
All done. NPU에서 테스트 (--chw 필수):
  python3 test_npu.py --model assets/mxq/efficientvit_b0_r224_timm_cpuoffload.mxq --labels imagenet_classes.txt --n 10 --chw
""")
