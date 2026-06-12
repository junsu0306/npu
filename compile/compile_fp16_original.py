#!/usr/bin/env python3
"""
EfficientViT original ONNX → MXQ 컴파일 (INT16 attention 버전).

Mobilint 포럼 답변 반영 (2026-06):
  - ONNX→mblt 변환 중 레이어명이 바뀜 → ONNX에서 추출한 이름은 무효.
  - BitConfig.SaveInfo.save_path 로 실제 내부 레이어명을 먼저 확인해야 함.
  - 그 이름으로 activation_16bits / weight_16bits 재컴파일.

워크플로우:
  1. DIAG_MODE=0  → 최소 컴파일으로 quantization 크래시 여부 확인
  2. DIAG_MODE=2  → SaveInfo + MixedPrecision (8-bit 기본, 레이어명 저장)
                   assets/bitconfig_saved/<model>_bitconfig.json 확인
  3. DIAG_MODE=3  → 저장된 파일에서 context_module 레이어명 자동 추출 → 16-bit 적용
                   (파싱이 안 맞으면 DIAG_MODE=3 실행 후 JSON 직접 확인하고
                    load_int16_layer_names() 내 필터 로직을 조정)

사전 준비 (Docker):
  pip install onnx
  pip install /workspace/npu/assets/qbcompiler-1.1.0+aries2-py3-none-any.whl
  python3 /workspace/npu/compile/compile_fp16_original.py
"""

import glob
import json
import os
from qbcompiler import mxq_compile
from qbcompiler.configs import BitConfig, CalibrationConfig, CompileConfig

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))

# ── 캘리브레이션 파일 ─────────────────────────────────────────────────────────
CALIB_HWC_DIR = os.path.join(ASSETS_DIR, "calib_hwc")
hwc_files = sorted(glob.glob(os.path.join(CALIB_HWC_DIR, "*.npy")))
if not hwc_files:
    raise FileNotFoundError(f"No HWC calibration .npy files in {CALIB_HWC_DIR}")

CALIB_TXT = os.path.join(BASE_DIR, "calib_hwc.txt")
with open(CALIB_TXT, "w") as f:
    f.write("\n".join(hwc_files) + "\n")
print(f"Calibration list: {len(hwc_files)} files -> {CALIB_TXT}")

MXQ_DIR        = os.path.join(ASSETS_DIR, "mxq")
ONNX_DIR       = os.path.join(ASSETS_DIR, "onnx")
BITCFG_SAVE_DIR = os.path.join(ASSETS_DIR, "bitconfig_saved")
os.makedirs(MXQ_DIR, exist_ok=True)
os.makedirs(BITCFG_SAVE_DIR, exist_ok=True)

# ── 진단 모드 ─────────────────────────────────────────────────────────────────
# 0 = 최소 컴파일 (BitConfig 없음) ← 먼저 이걸로 quantization 크래시 해결 확인
# 1 = CalibrationConfig만
# 2 = BitConfig + SaveInfo (8-bit 기본, 실제 레이어명 저장)
# 3 = BitConfig + SaveInfo + 저장된 레이어명으로 activation_16bits 적용
DIAG_MODE = 2   # ← 0이 통과되면 2, 그 다음 3


# ── SaveInfo 파싱: 저장된 JSON에서 16-bit 적용할 레이어명 추출 ──────────────────
def load_int16_layer_names(saved_path: str) -> list:
    """
    BitConfig.SaveInfo.save_path 에 저장된 파일을 파싱해
    context_module 관련 레이어명을 반환한다.

    파일 형식을 모를 경우 DIAG_MODE=2 로 먼저 컴파일 후
    assets/bitconfig_saved/*.json 을 직접 열어 확인할 것.
    """
    if not os.path.exists(saved_path):
        print(f"  [WARN] save_path 파일 없음: {saved_path}")
        print("         DIAG_MODE=2 로 먼저 컴파일해서 파일을 생성하세요.")
        return []

    with open(saved_path) as f:
        raw = f.read()

    # JSON 파싱 시도
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # JSON이 아닌 포맷(텍스트 등)이면 라인별로 처리
        print(f"  [INFO] JSON 파싱 실패 → 텍스트로 처리: {saved_path}")
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        ctx = [l for l in lines if "context_module" in l]
        print(f"  context_module 관련 항목: {len(ctx)}개")
        return ctx

    # dict 형식: {"layer_name": {"bits": 8, ...}, ...}
    int16_targets = []
    if isinstance(data, dict):
        for name in data.keys():
            if _is_context_layer(name):
                int16_targets.append(name)

    # list 형식: [{"name": "...", "bits": 8}, ...]
    elif isinstance(data, list):
        for item in data:
            name = item.get("name", "") if isinstance(item, dict) else str(item)
            if _is_context_layer(name):
                int16_targets.append(name)

    print(f"  context_module 16-bit 대상: {len(int16_targets)}개")
    for n in int16_targets[:5]:
        print(f"    {n}")
    if len(int16_targets) > 5:
        print(f"    ... ({len(int16_targets) - 5}개 더)")
    return int16_targets


def _is_context_layer(name: str) -> bool:
    """context_module 또는 attention 관련 레이어 여부."""
    name_lower = name.lower()
    return any(kw in name_lower for kw in
               ("context_module", "context", "kernel_func", "qkv", "attn", "div"))


# ── CompileConfig 생성 ────────────────────────────────────────────────────────
def make_compile_config(saved_path: str) -> CompileConfig:
    calib_cfg = CalibrationConfig(method=0, mode=0, output=0)

    if DIAG_MODE == 2:
        # SaveInfo만 추가 → 실제 레이어명 파일로 저장, bit는 기본 8-bit
        print(f"  [Mode2] BitConfig + SaveInfo → {saved_path}")
        bit_cfg = BitConfig(
            transformer=BitConfig.Transformer(
                mixed_precision=BitConfig.Transformer.MixedPrecision(apply=True),
            ),
            save_info=BitConfig.SaveInfo(save_path=saved_path),
        )

    elif DIAG_MODE == 3:
        # 저장된 파일에서 레이어명 읽어 16-bit 적용
        layer_names = load_int16_layer_names(saved_path)
        print(f"  [Mode3] 16-bit 레이어: {len(layer_names)}개 적용")

        overrides_kwargs = {}
        if layer_names:
            overrides_kwargs["layer_overrides"] = BitConfig.LayerOverrides(
                activation_16bits=layer_names,
                weight_16bits=layer_names,
            )

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
            save_info=BitConfig.SaveInfo(save_path=saved_path),
            **overrides_kwargs,
        )

    return CompileConfig(calibration=calib_cfg, bit=bit_cfg)


# ── 컴파일 ───────────────────────────────────────────────────────────────────
def compile_model(onnx_path: str, mxq_path: str):
    model_stem = os.path.basename(onnx_path).replace(".onnx", "")

    if os.path.exists(mxq_path):
        print(f"\n[SKIP] 이미 존재: {mxq_path}  (삭제 후 재실행)")
        return
    if not os.path.exists(onnx_path):
        print(f"\n[SKIP] ONNX 없음: {onnx_path}")
        return

    print(f"\n{'='*60}")
    print(f"Compiling: {os.path.basename(onnx_path)}  [DIAG_MODE={DIAG_MODE}]")
    print(f"  output:  {mxq_path}")
    print(f"{'='*60}")

    kwargs = dict(
        model=onnx_path,
        save_path=mxq_path,
        calib_data_path=CALIB_TXT,
        backend="onnx",
    )

    if DIAG_MODE == 0:
        print("  → 옵션 없음 (순수 기본값)")

    elif DIAG_MODE == 1:
        print("  → CalibrationConfig만")
        kwargs["compile_config"] = CompileConfig(
            calibration=CalibrationConfig(method=0, mode=0, output=0)
        )

    elif DIAG_MODE in (2, 3):
        saved_path = os.path.join(BITCFG_SAVE_DIR, f"{model_stem}_bitconfig.json")
        kwargs["compile_config"] = make_compile_config(saved_path)

    mxq_compile(**kwargs)
    print(f"Saved: {mxq_path}")

    if DIAG_MODE == 2:
        saved_path = os.path.join(BITCFG_SAVE_DIR, f"{model_stem}_bitconfig.json")
        if os.path.exists(saved_path):
            print(f"\n  [SaveInfo] 레이어명 저장됨: {saved_path}")
            print("  내용 확인 후 DIAG_MODE=3 으로 재컴파일하세요.")
        else:
            print(f"\n  [WARN] save_path 파일이 생성되지 않았습니다: {saved_path}")
            print("  BitConfig.SaveInfo API 이름을 확인하세요.")


MODELS = [
    ("efficientvit_b0_original_r224_nchw_fixed.onnx",
     f"efficientvit_b0_original_mode{DIAG_MODE}.mxq"),
    ("efficientvit_b1_original_r224_nchw_fixed.onnx",
     f"efficientvit_b1_original_mode{DIAG_MODE}.mxq"),
]

for onnx_name, mxq_name in MODELS:
    compile_model(
        onnx_path=os.path.join(ONNX_DIR, onnx_name),
        mxq_path=os.path.join(MXQ_DIR, mxq_name),
    )

print(f"""
완료.
  DIAG_MODE=2 인 경우:
    cat {BITCFG_SAVE_DIR}/*.json   # 실제 레이어명 확인
    → DIAG_MODE=3 으로 변경 후 재실행

  DIAG_MODE=3 인 경우:
    python3 test_npu.py --model assets/mxq/efficientvit_b0_original_mode3.mxq
""")
