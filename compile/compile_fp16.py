#!/usr/bin/env python3
"""
EfficientViT-B0/B1 context_module activation을 FP16으로 강제 컴파일.

전략:
  backend="onnx" + activation_16bits 에 context_module 내 ONNX 노드명 직접 지정.
  backend="torch" 시 내부적으로 다른 이름 체계를 쓸 수 있으므로 onnx 백엔드 사용.

  ONNX 노드명은 efficientvit_b0_r224_timm_nchw.onnx 에서 추출한 실제 경로.
  context_module 내 float-op (Conv/MatMul/Relu/Div/Add/Mul/Slice/Reshape/Transpose/Pad/Concat)
  총 141개를 activation_16bits/weight_16bits 에 등록.

사전 준비 (Docker):
  pip install /workspace/npu/assets/qbcompiler-1.1.0+aries2-py3-none-any.whl
  python3 /workspace/npu/compile/compile_fp16.py
"""

import glob
import os
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

# ── context_module ONNX 노드명 (앞의 / 제거) ─────────────────────────────────
# efficientvit_b0_r224_timm_nchw.onnx 에서 추출한 float-op 노드명 (141개)
# stages.2 (2개 블록) + stages.3 (2개 블록) 각각의 context_module
CTX_LAYER_NAMES = [
    # ── stages.2 / blocks.1 ───────────────────────────────────────────────────
    "stages/stages.2/blocks/blocks.1/context_module/main/qkv/conv/Conv",
    "stages/stages.2/blocks/blocks.1/context_module/main/aggreg.0/aggreg.0.0/Conv",
    "stages/stages.2/blocks/blocks.1/context_module/main/aggreg.0/aggreg.0.1/Conv",
    "stages/stages.2/blocks/blocks.1/context_module/main/Concat",
    "stages/stages.2/blocks/blocks.1/context_module/main/Reshape",
    "stages/stages.2/blocks/blocks.1/context_module/main/Transpose",
    "stages/stages.2/blocks/blocks.1/context_module/main/Add",
    "stages/stages.2/blocks/blocks.1/context_module/main/Div",
    "stages/stages.2/blocks/blocks.1/context_module/main/Mul",
    "stages/stages.2/blocks/blocks.1/context_module/main/Slice",
    "stages/stages.2/blocks/blocks.1/context_module/main/Mul_1",
    "stages/stages.2/blocks/blocks.1/context_module/main/Slice_1",
    "stages/stages.2/blocks/blocks.1/context_module/main/Mul_2",
    "stages/stages.2/blocks/blocks.1/context_module/main/Slice_2",
    "stages/stages.2/blocks/blocks.1/context_module/main/kernel_func/Relu",
    "stages/stages.2/blocks/blocks.1/context_module/main/kernel_func_1/Relu",
    "stages/stages.2/blocks/blocks.1/context_module/main/Concat_1",
    "stages/stages.2/blocks/blocks.1/context_module/main/Reshape_1",
    "stages/stages.2/blocks/blocks.1/context_module/main/Slice_3",
    "stages/stages.2/blocks/blocks.1/context_module/main/Transpose_1",
    "stages/stages.2/blocks/blocks.1/context_module/main/Reshape_2",
    "stages/stages.2/blocks/blocks.1/context_module/main/Pad",
    "stages/stages.2/blocks/blocks.1/context_module/main/Transpose_2",
    "stages/stages.2/blocks/blocks.1/context_module/main/MatMul",
    "stages/stages.2/blocks/blocks.1/context_module/main/MatMul_1",
    "stages/stages.2/blocks/blocks.1/context_module/main/Slice_4",
    "stages/stages.2/blocks/blocks.1/context_module/main/Slice_5",
    "stages/stages.2/blocks/blocks.1/context_module/main/Add_1",
    "stages/stages.2/blocks/blocks.1/context_module/main/Div_1",
    "stages/stages.2/blocks/blocks.1/context_module/main/Transpose_3",
    "stages/stages.2/blocks/blocks.1/context_module/main/Reshape_3",
    "stages/stages.2/blocks/blocks.1/context_module/main/proj/conv/Conv",
    "stages/stages.2/blocks/blocks.1/context_module/Add",
    # ── stages.2 / blocks.2 ───────────────────────────────────────────────────
    "stages/stages.2/blocks/blocks.2/context_module/main/qkv/conv/Conv",
    "stages/stages.2/blocks/blocks.2/context_module/main/aggreg.0/aggreg.0.0/Conv",
    "stages/stages.2/blocks/blocks.2/context_module/main/aggreg.0/aggreg.0.1/Conv",
    "stages/stages.2/blocks/blocks.2/context_module/main/Concat",
    "stages/stages.2/blocks/blocks.2/context_module/main/Mul",
    "stages/stages.2/blocks/blocks.2/context_module/main/Concat_1",
    "stages/stages.2/blocks/blocks.2/context_module/main/Reshape",
    "stages/stages.2/blocks/blocks.2/context_module/main/Transpose",
    "stages/stages.2/blocks/blocks.2/context_module/main/Add",
    "stages/stages.2/blocks/blocks.2/context_module/main/Div",
    "stages/stages.2/blocks/blocks.2/context_module/main/Mul_1",
    "stages/stages.2/blocks/blocks.2/context_module/main/Slice",
    "stages/stages.2/blocks/blocks.2/context_module/main/Mul_2",
    "stages/stages.2/blocks/blocks.2/context_module/main/Slice_1",
    "stages/stages.2/blocks/blocks.2/context_module/main/Mul_3",
    "stages/stages.2/blocks/blocks.2/context_module/main/Slice_2",
    "stages/stages.2/blocks/blocks.2/context_module/main/kernel_func/Relu",
    "stages/stages.2/blocks/blocks.2/context_module/main/kernel_func_1/Relu",
    "stages/stages.2/blocks/blocks.2/context_module/main/Concat_2",
    "stages/stages.2/blocks/blocks.2/context_module/main/Reshape_1",
    "stages/stages.2/blocks/blocks.2/context_module/main/Slice_3",
    "stages/stages.2/blocks/blocks.2/context_module/main/Transpose_1",
    "stages/stages.2/blocks/blocks.2/context_module/main/Reshape_2",
    "stages/stages.2/blocks/blocks.2/context_module/main/Pad",
    "stages/stages.2/blocks/blocks.2/context_module/main/Transpose_2",
    "stages/stages.2/blocks/blocks.2/context_module/main/MatMul",
    "stages/stages.2/blocks/blocks.2/context_module/main/MatMul_1",
    "stages/stages.2/blocks/blocks.2/context_module/main/Slice_4",
    "stages/stages.2/blocks/blocks.2/context_module/main/Slice_5",
    "stages/stages.2/blocks/blocks.2/context_module/main/Add_1",
    "stages/stages.2/blocks/blocks.2/context_module/main/Div_1",
    "stages/stages.2/blocks/blocks.2/context_module/main/Transpose_3",
    "stages/stages.2/blocks/blocks.2/context_module/main/Concat_3",
    "stages/stages.2/blocks/blocks.2/context_module/main/Reshape_3",
    "stages/stages.2/blocks/blocks.2/context_module/main/proj/conv/Conv",
    "stages/stages.2/blocks/blocks.2/context_module/Add",
    # ── stages.3 / blocks.1 ───────────────────────────────────────────────────
    "stages/stages.3/blocks/blocks.1/context_module/main/qkv/conv/Conv",
    "stages/stages.3/blocks/blocks.1/context_module/main/aggreg.0/aggreg.0.0/Conv",
    "stages/stages.3/blocks/blocks.1/context_module/main/aggreg.0/aggreg.0.1/Conv",
    "stages/stages.3/blocks/blocks.1/context_module/main/Concat",
    "stages/stages.3/blocks/blocks.1/context_module/main/Mul",
    "stages/stages.3/blocks/blocks.1/context_module/main/Concat_1",
    "stages/stages.3/blocks/blocks.1/context_module/main/Reshape",
    "stages/stages.3/blocks/blocks.1/context_module/main/Transpose",
    "stages/stages.3/blocks/blocks.1/context_module/main/Add",
    "stages/stages.3/blocks/blocks.1/context_module/main/Div",
    "stages/stages.3/blocks/blocks.1/context_module/main/Mul_1",
    "stages/stages.3/blocks/blocks.1/context_module/main/Slice",
    "stages/stages.3/blocks/blocks.1/context_module/main/Mul_2",
    "stages/stages.3/blocks/blocks.1/context_module/main/Slice_1",
    "stages/stages.3/blocks/blocks.1/context_module/main/Mul_3",
    "stages/stages.3/blocks/blocks.1/context_module/main/Slice_2",
    "stages/stages.3/blocks/blocks.1/context_module/main/kernel_func/Relu",
    "stages/stages.3/blocks/blocks.1/context_module/main/kernel_func_1/Relu",
    "stages/stages.3/blocks/blocks.1/context_module/main/Concat_2",
    "stages/stages.3/blocks/blocks.1/context_module/main/Reshape_1",
    "stages/stages.3/blocks/blocks.1/context_module/main/Slice_3",
    "stages/stages.3/blocks/blocks.1/context_module/main/Transpose_1",
    "stages/stages.3/blocks/blocks.1/context_module/main/Reshape_2",
    "stages/stages.3/blocks/blocks.1/context_module/main/Pad",
    "stages/stages.3/blocks/blocks.1/context_module/main/Transpose_2",
    "stages/stages.3/blocks/blocks.1/context_module/main/MatMul",
    "stages/stages.3/blocks/blocks.1/context_module/main/MatMul_1",
    "stages/stages.3/blocks/blocks.1/context_module/main/Slice_4",
    "stages/stages.3/blocks/blocks.1/context_module/main/Slice_5",
    "stages/stages.3/blocks/blocks.1/context_module/main/Add_1",
    "stages/stages.3/blocks/blocks.1/context_module/main/Div_1",
    "stages/stages.3/blocks/blocks.1/context_module/main/Transpose_3",
    "stages/stages.3/blocks/blocks.1/context_module/main/Concat_3",
    "stages/stages.3/blocks/blocks.1/context_module/main/Reshape_3",
    "stages/stages.3/blocks/blocks.1/context_module/main/proj/conv/Conv",
    "stages/stages.3/blocks/blocks.1/context_module/Add",
    # ── stages.3 / blocks.2 ───────────────────────────────────────────────────
    "stages/stages.3/blocks/blocks.2/context_module/main/qkv/conv/Conv",
    "stages/stages.3/blocks/blocks.2/context_module/main/aggreg.0/aggreg.0.0/Conv",
    "stages/stages.3/blocks/blocks.2/context_module/main/aggreg.0/aggreg.0.1/Conv",
    "stages/stages.3/blocks/blocks.2/context_module/main/Concat",
    "stages/stages.3/blocks/blocks.2/context_module/main/Mul",
    "stages/stages.3/blocks/blocks.2/context_module/main/Concat_1",
    "stages/stages.3/blocks/blocks.2/context_module/main/Reshape",
    "stages/stages.3/blocks/blocks.2/context_module/main/Transpose",
    "stages/stages.3/blocks/blocks.2/context_module/main/Add",
    "stages/stages.3/blocks/blocks.2/context_module/main/Div",
    "stages/stages.3/blocks/blocks.2/context_module/main/Mul_1",
    "stages/stages.3/blocks/blocks.2/context_module/main/Slice",
    "stages/stages.3/blocks/blocks.2/context_module/main/Mul_2",
    "stages/stages.3/blocks/blocks.2/context_module/main/Slice_1",
    "stages/stages.3/blocks/blocks.2/context_module/main/Mul_3",
    "stages/stages.3/blocks/blocks.2/context_module/main/Slice_2",
    "stages/stages.3/blocks/blocks.2/context_module/main/kernel_func/Relu",
    "stages/stages.3/blocks/blocks.2/context_module/main/kernel_func_1/Relu",
    "stages/stages.3/blocks/blocks.2/context_module/main/Concat_2",
    "stages/stages.3/blocks/blocks.2/context_module/main/Reshape_1",
    "stages/stages.3/blocks/blocks.2/context_module/main/Slice_3",
    "stages/stages.3/blocks/blocks.2/context_module/main/Transpose_1",
    "stages/stages.3/blocks/blocks.2/context_module/main/Reshape_2",
    "stages/stages.3/blocks/blocks.2/context_module/main/Pad",
    "stages/stages.3/blocks/blocks.2/context_module/main/Transpose_2",
    "stages/stages.3/blocks/blocks.2/context_module/main/MatMul",
    "stages/stages.3/blocks/blocks.2/context_module/main/MatMul_1",
    "stages/stages.3/blocks/blocks.2/context_module/main/Slice_4",
    "stages/stages.3/blocks/blocks.2/context_module/main/Slice_5",
    "stages/stages.3/blocks/blocks.2/context_module/main/Add_1",
    "stages/stages.3/blocks/blocks.2/context_module/main/Div_1",
    "stages/stages.3/blocks/blocks.2/context_module/main/Transpose_3",
    "stages/stages.3/blocks/blocks.2/context_module/main/Concat_3",
    "stages/stages.3/blocks/blocks.2/context_module/main/Reshape_3",
    "stages/stages.3/blocks/blocks.2/context_module/main/proj/conv/Conv",
    "stages/stages.3/blocks/blocks.2/context_module/Add",
]


def make_compile_config() -> CompileConfig:
    print(f"  activation_16bits: {len(CTX_LAYER_NAMES)} context_module 노드")
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
            activation_16bits=CTX_LAYER_NAMES,
            weight_16bits=CTX_LAYER_NAMES,
        ),
    )
    calib_cfg = CalibrationConfig(method=0, mode=0, output=0)
    return CompileConfig(calibration=calib_cfg, bit=bit_cfg)


def compile_model(onnx_path: str, mxq_path: str, label: str):
    if os.path.exists(mxq_path):
        print(f"\n[SKIP] 이미 존재: {mxq_path}")
        return
    if not os.path.exists(onnx_path):
        print(f"\n[SKIP] ONNX 없음: {onnx_path}")
        return
    print(f"\n{'='*60}")
    print(f"Compiling: {label}")
    print(f"  onnx:   {onnx_path}")
    print(f"  output: {mxq_path}")
    print(f"{'='*60}")

    compile_cfg = make_compile_config()
    mxq_compile(
        model=onnx_path,
        save_path=mxq_path,
        calib_data_path=CALIB_TXT,
        backend="onnx",
        compile_config=compile_cfg,
    )
    print(f"Saved: {mxq_path}")


# ── NCHW ONNX → MXQ ──────────────────────────────────────────────────────────
MODELS = [
    ("efficientvit_b0_r224_timm_nchw.onnx",     "efficientvit_b0_r224_timm_fp16.mxq"),
    ("efficientvit_b1_r224_timm_nchw.onnx",     "efficientvit_b1_r224_timm_fp16.mxq"),
    ("efficientvit_b0_original_r224_nchw.onnx", "efficientvit_b0_original_fp16.mxq"),
    ("efficientvit_b1_original_r224_nchw.onnx", "efficientvit_b1_original_fp16.mxq"),
]

for onnx_name, mxq_name in MODELS:
    compile_model(
        onnx_path=os.path.join(ONNX_DIR, onnx_name),
        mxq_path=os.path.join(MXQ_DIR, mxq_name),
        label=onnx_name,
    )

print("""
All done. NPU에서 테스트:
  python3 test_npu.py --model assets/mxq/efficientvit_b0_r224_timm_fp16.mxq --labels imagenet_classes.txt --n 10
""")
