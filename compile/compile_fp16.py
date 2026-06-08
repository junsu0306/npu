#!/usr/bin/env python3
"""
EfficientViT-B0/B1 최대한 많은 레이어를 FP16으로 컴파일.

EfficientViT INT8 양자화가 attention 연산에서 실패하므로
BitConfig로 모든 attention bit widths를 16bit로 설정하고
activation_16bits / weight_16bits 에 전체 레이어명을 넣어 최대한 FP16으로 유지.

PDF 근거:
  - QuantizationConfig(calibration=CalibrationConfig, bit=BitConfig)
  - mxq_compile(..., quantization_config=QuantizationConfig)
  - BitConfig.activation_16bits: List[str] — 해당 레이어 activation을 16bit로 강제

Inside Docker:
  pip install timm
  python3 /home/airlab_compression/npu/compile/compile_fp16.py
"""

import glob
import os
import sys
import torch
import timm
from qbcompiler import mxq_compile
from qbcompiler.configs.quantization_config import (
    get_calibration_config,
    get_bit_config,
    get_quantization_config,
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))

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

feed_dict = {"x": torch.randn(1, 3, 224, 224).cpu()}


def make_quant_config(model):
    """모델 전체 레이어명을 activation_16bits / weight_16bits 에 등록."""
    layer_names = [name for name, _ in model.named_modules() if name]
    print(f"  FP16 대상 레이어 수: {len(layer_names)}")

    calib_cfg = get_calibration_config(
        quantization_method=0,   # WChALayer
        quantization_mode=0,     # max
        quantization_output=0,   # Layer
    )

    bit_cfg = get_bit_config(
        query_act_bits=16,
        key_act_bits=16,
        value_act_bits=16,
        output_act_bits=16,
        ffn_act_bits=16,
        head_act_bits=16,
        query_weight_bits=16,
        key_weight_bits=16,
        value_weight_bits=16,
        output_weight_bits=16,
        ffn_weight_bits=16,
        head_weight_bits=16,
        mixed_precision_apply=True,
        activation_16bits=layer_names,
        weight_16bits=layer_names,
    )

    return get_quantization_config(calibration=calib_cfg, bit=bit_cfg)


def compile_model(model, mxq_path, label):
    if os.path.exists(mxq_path):
        print(f"\n[SKIP] 이미 존재: {mxq_path}")
        return
    print(f"\n{'='*60}")
    print(f"Compiling: {label}")
    print(f"  output: {mxq_path}")
    print(f"{'='*60}")

    quant_cfg = make_quant_config(model)

    mxq_compile(
        model=model,
        save_path=mxq_path,
        calib_data_path=CALIB_TXT,
        backend="torch",
        feed_dict=feed_dict,
        quantization_config=quant_cfg,
    )
    print(f"Saved: {mxq_path}")


# ── 1. timm pretrained 모델 ───────────────────────────────────────────────────
TIMM_MODELS = [
    ("efficientvit_b0.r224_in1k", os.path.join(MXQ_DIR, "efficientvit_b0_r224_timm_fp16.mxq")),
    ("efficientvit_b1.r224_in1k", os.path.join(MXQ_DIR, "efficientvit_b1_r224_timm_fp16.mxq")),
]

for model_name, mxq_path in TIMM_MODELS:
    model = timm.create_model(model_name, pretrained=True)
    model.eval().cpu()
    compile_model(model, mxq_path, model_name)


# ── 2. 원본(mit-han-lab) .pt 모델 ────────────────────────────────────────────
sys.path.insert(0, "/workspace/efficientvit2026")

ORIGINAL_MODELS = [
    (os.path.join(PT_DIR, "efficientvit_b0_original.pt"),
     os.path.join(MXQ_DIR, "efficientvit_b0_original_fp16.mxq")),
    (os.path.join(PT_DIR, "efficientvit_b1_original.pt"),
     os.path.join(MXQ_DIR, "efficientvit_b1_original_fp16.mxq")),
]

for pt_path, mxq_path in ORIGINAL_MODELS:
    if not os.path.exists(pt_path):
        print(f"\n[SKIP] .pt 없음: {pt_path}")
        continue
    model = torch.load(pt_path, map_location="cpu", weights_only=False)
    model.eval().cpu()
    compile_model(model, mxq_path, os.path.basename(pt_path))


print("""
All done. NPU에서 테스트:
  python3 test_npu.py --model assets/mxq/efficientvit_b0_r224_timm_fp16.mxq --labels imagenet_classes.txt --n 10
""")
