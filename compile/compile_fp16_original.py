#!/usr/bin/env python3
"""
EfficientViT original(.pt → ONNX) 모델만 FP16 컴파일.

context_module 노드명을 각 ONNX 파일에서 동적으로 추출해 activation_16bits에 적용.
(timm과 original의 노드 구조가 달라 하드코딩 불가)

사전 준비 (Docker):
  pip install onnx
  pip install /workspace/npu/assets/qbcompiler-1.1.0+aries2-py3-none-any.whl
  python3 /workspace/npu/compile/compile_fp16_original.py
"""

import glob
import os
import onnx
from qbcompiler import mxq_compile
from qbcompiler.configs import BitConfig, CalibrationConfig, CompileConfig

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))

CALIB_HWC_DIR = os.path.join(ASSETS_DIR, "calib_hwc")
hwc_files = sorted(glob.glob(os.path.join(CALIB_HWC_DIR, "*.npy")))
if not hwc_files:
    raise FileNotFoundError(f"No HWC calibration .npy files in {CALIB_HWC_DIR}")

CALIB_TXT = os.path.join(BASE_DIR, "calib_hwc.txt")
with open(CALIB_TXT, "w") as f:
    f.write("\n".join(hwc_files) + "\n")
print(f"Calibration list: {len(hwc_files)} HWC files -> {CALIB_TXT}")

MXQ_DIR  = os.path.join(ASSETS_DIR, "mxq")
ONNX_DIR = os.path.join(ASSETS_DIR, "onnx")
os.makedirs(MXQ_DIR, exist_ok=True)

FLOAT_OPS = {"Conv", "MatMul", "Relu", "Div", "Add", "Mul",
             "Slice", "Reshape", "Transpose", "Pad", "Concat"}


def extract_ctx_layer_names(onnx_path: str) -> list:
    """각 ONNX 파일의 context_module float-op 노드명 추출."""
    model = onnx.load(onnx_path)
    names = [
        n.name.lstrip("/")
        for n in model.graph.node
        if "context_module" in n.name and n.op_type in FLOAT_OPS
    ]
    return names


def make_compile_config(layer_names: list) -> CompileConfig:
    print(f"  activation_16bits: {len(layer_names)} context_module 노드")
    bit_cfg = BitConfig(
        transformer=BitConfig.Transformer(
            activation=BitConfig.Transformer.Activation(
                query=16, key=16, value=16, output=16, ffn=16, head=16,
            ),
            weight=BitConfig.Transformer.Weight(
                query=16, key=16, value=16, output=16, ffn=16, head=16,
            ),
            mixed_precision=BitConfig.Transformer.MixedPrecision(apply=True),
        ),
        layer_overrides=BitConfig.LayerOverrides(
            activation_16bits=layer_names,
            weight_16bits=layer_names,
        ),
    )
    calib_cfg = CalibrationConfig(method=0, mode=0, output=0)
    return CompileConfig(calibration=calib_cfg, bit=bit_cfg)


def compile_model(onnx_path: str, mxq_path: str):
    if os.path.exists(mxq_path):
        print(f"\n[SKIP] 이미 존재: {mxq_path}")
        return
    if not os.path.exists(onnx_path):
        print(f"\n[SKIP] ONNX 없음: {onnx_path}")
        return

    print(f"\n{'='*60}")
    print(f"Compiling: {os.path.basename(onnx_path)}")
    print(f"  output:  {mxq_path}")
    print(f"{'='*60}")

    layer_names = extract_ctx_layer_names(onnx_path)
    compile_cfg = make_compile_config(layer_names)

    mxq_compile(
        model=onnx_path,
        save_path=mxq_path,
        calib_data_path=CALIB_TXT,
        backend="onnx",
        compile_config=compile_cfg,
    )
    print(f"Saved: {mxq_path}")


MODELS = [
    ("efficientvit_b0_original_r224_nchw.onnx", "efficientvit_b0_original_fp16.mxq"),
    ("efficientvit_b1_original_r224_nchw.onnx", "efficientvit_b1_original_fp16.mxq"),
]

for onnx_name, mxq_name in MODELS:
    compile_model(
        onnx_path=os.path.join(ONNX_DIR, onnx_name),
        mxq_path=os.path.join(MXQ_DIR, mxq_name),
    )

print("""
All done. NPU에서 테스트:
  python3 test_npu.py --model assets/mxq/efficientvit_b0_original_fp16.mxq --labels imagenet_classes.txt --n 10
""")
