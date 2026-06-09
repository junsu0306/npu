#!/usr/bin/env python3
"""
EfficientViT 양자화 진단 스크립트.

float32 ONNX vs 다양한 양자화 방법 비교.
목적: qbcompiler 문제인지 양자화 자체 문제인지 판별.

사용법:
  pip install onnxruntime
  python3 diagnose_quant.py
"""

import glob
import os
import numpy as np
import cv2
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    CalibrationDataReader,
    CalibrationMethod,
    QuantType,
    QuantFormat,
)
try:
    from onnxruntime.quantization import quant_pre_process
except ImportError:
    from onnxruntime.quantization.preprocess import quant_pre_process

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
ONNX_DIR   = os.path.join(ASSETS_DIR, "onnx")
CALIB_DIR  = os.path.join(ASSETS_DIR, "calibration_data")
LABEL_FILE = os.path.join(BASE_DIR, "imagenet_classes.txt")

N_CALIB = 200
N_TEST  = 10

MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])


def find_models():
    """assets/onnx/ 에서 _prep / quant_diag 제외한 모든 ONNX 반환."""
    return sorted(
        p for p in glob.glob(os.path.join(ONNX_DIR, "*.onnx"))
        if "_prep" not in os.path.basename(p)
        and "quant_diag" not in p
    )


# ── 전처리 ───────────────────────────────────────────────────────────────────

def preprocess(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = 256 / min(h, w)
    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))
    h, w = img.shape[:2]
    img = img[(h-224)//2:(h-224)//2+224, (w-224)//2:(w-224)//2+224]
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[np.newaxis]   # (1,3,224,224) NCHW


# ── Calibration Data Reader ───────────────────────────────────────────────────

class CalibReader(CalibrationDataReader):
    def __init__(self, input_name: str, n: int = N_CALIB):
        jpgs = sorted(glob.glob(os.path.join(CALIB_DIR, "**", "*.jpg"), recursive=True))
        self.images = jpgs[:n]
        self.input_name = input_name
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self.images):
            return None
        img = preprocess(self.images[self._idx])
        self._idx += 1
        return {self.input_name: img}

    def rewind(self):
        self._idx = 0


# ── 레이블 ────────────────────────────────────────────────────────────────────

def load_labels():
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE) as f:
            return [l.strip() for l in f]
    return [str(i) for i in range(1000)]


# ── 추론 ─────────────────────────────────────────────────────────────────────

def infer(session: ort.InferenceSession, img: np.ndarray) -> int:
    name = session.get_inputs()[0].name
    logits = session.run(None, {name: img})[0].flatten()
    return int(np.argmax(logits))


# ── 양자화 변형 정의 ──────────────────────────────────────────────────────────

def get_variants(input_name: str, onnx_fp32: str, onnx_prep: str, out_dir: str, use_prep: bool = False):
    """(label, output_path, build_fn) 리스트 반환."""
    src = onnx_prep if use_prep else onnx_fp32
    tag = "_prep" if use_prep else ""
    variants = []

    # ── Dynamic INT8 (weight only, calibration 불필요) ────────────────────
    dyn_path = os.path.join(out_dir, f"dynamic_int8{tag}.onnx")
    def build_dynamic(out=dyn_path, s=src):
        if not os.path.exists(out):
            quantize_dynamic(s, out, weight_type=QuantType.QInt8)
    variants.append((f"Dynamic INT8{tag}", dyn_path, build_dynamic))

    # ── Static INT8 변형들 ────────────────────────────────────────────────
    static_cases = [
        ("static_int8_pertensor_minmax_qdq",
         False, CalibrationMethod.MinMax,       QuantFormat.QDQ,       QuantType.QInt8,  QuantType.QInt8),
        ("static_int8_perchannel_minmax_qdq",
         True,  CalibrationMethod.MinMax,       QuantFormat.QDQ,       QuantType.QInt8,  QuantType.QInt8),
        ("static_int8_perchannel_entropy_qdq",
         True,  CalibrationMethod.Entropy,      QuantFormat.QDQ,       QuantType.QInt8,  QuantType.QInt8),
        ("static_int8_perchannel_percentile_qdq",
         True,  CalibrationMethod.Percentile,   QuantFormat.QDQ,       QuantType.QInt8,  QuantType.QInt8),
        ("static_int8_perchannel_minmax_qop",
         True,  CalibrationMethod.MinMax,       QuantFormat.QOperator, QuantType.QInt8,  QuantType.QInt8),
        ("static_uint8_perchannel_minmax_qdq",
         True,  CalibrationMethod.MinMax,       QuantFormat.QDQ,       QuantType.QUInt8, QuantType.QUInt8),
    ]

    for (suffix, per_ch, calib_m, fmt, act_t, wt_t) in static_cases:
        out_path = os.path.join(out_dir, f"{suffix}{tag}.onnx")
        def build_static(out=out_path, pc=per_ch, cm=calib_m, f=fmt, at=act_t, wt=wt_t, s=src):
            if not os.path.exists(out):
                reader = CalibReader(input_name)
                quantize_static(
                    model_input=s,
                    model_output=out,
                    calibration_data_reader=reader,
                    calibrate_method=cm,
                    quant_format=f,
                    activation_type=at,
                    weight_type=wt,
                    per_channel=pc,
                    reduce_range=False,
                )
        variants.append((f"{suffix}{tag}", out_path, build_static))

    # ── UINT16 activation (onnxruntime >= 1.16) ───────────────────────────
    try:
        _ = QuantType.QUInt16
        int16_path = os.path.join(out_dir, f"static_uint16_perchannel_minmax{tag}.onnx")
        def build_int16(out=int16_path, s=src):
            if not os.path.exists(out):
                reader = CalibReader(input_name)
                quantize_static(
                    model_input=s,
                    model_output=out,
                    calibration_data_reader=reader,
                    calibrate_method=CalibrationMethod.MinMax,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt16,
                    weight_type=QuantType.QInt8,
                    per_channel=True,
                    reduce_range=False,
                )
        variants.append((f"UINT16 act + INT8 wt{tag}", int16_path, build_int16))
    except AttributeError:
        print("[INFO] onnxruntime < 1.16, INT16 quantization 스킵")

    return variants


# ── 결과 출력 ─────────────────────────────────────────────────────────────────

def run_variants(variants, fp32_preds, test_imgs, n_test):
    W = 52
    print(f"{'방법':<{W}} {'일치':>5}  {'match rate':>10}")
    print(f"{'─'*70}")
    print(f"{'FP32 (baseline)':<{W}} {n_test:>2}/{n_test}  {'100.0%':>10}")
    print(f"{'─'*70}")
    for label, out_path, build_fn in variants:
        try:
            build_fn()
        except Exception as e:
            print(f"{label:<{W}} [양자화 실패: {e}]")
            continue
        if not os.path.exists(out_path):
            print(f"{label:<{W}} [파일 없음]")
            continue
        try:
            sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
            preds = [infer(sess, preprocess(p)) for p in test_imgs]
            match = sum(p == q for p, q in zip(fp32_preds, preds))
            rate = match / n_test * 100
            mark = "✓" if rate >= 90 else ("△" if rate >= 70 else "✗")
            print(f"{label:<{W}} {match:>2}/{n_test}  {rate:>9.1f}%  {mark}")
        except Exception as e:
            print(f"{label:<{W}} [추론 실패: {e}]")
    print(f"{'─'*70}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def diagnose_model(onnx_fp32: str, test_imgs: list, fp32_preds: list, labels: list):
    """단일 모델에 대해 Round1(원본) + Round2(prep) 양자화 비교."""
    stem    = os.path.splitext(os.path.basename(onnx_fp32))[0]
    out_dir = os.path.join(ONNX_DIR, "quant_diag", stem)
    os.makedirs(out_dir, exist_ok=True)
    onnx_prep = os.path.join(ONNX_DIR, f"{stem}_prep.onnx")

    sess_fp32  = ort.InferenceSession(onnx_fp32, providers=["CPUExecutionProvider"])
    input_name = sess_fp32.get_inputs()[0].name

    print(f"\n{'#'*70}")
    print(f"  모델: {os.path.basename(onnx_fp32)}")
    print(f"{'#'*70}")

    # Round 1 — 원본
    print("\n[Round 1] 원본 ONNX 양자화")
    run_variants(
        get_variants(input_name, onnx_fp32, onnx_prep, out_dir, use_prep=False),
        fp32_preds, test_imgs, N_TEST,
    )

    # pre-processing
    print("\n[pre-processing] shape inference + op fusion + graph optimization...")
    if not os.path.exists(onnx_prep):
        quant_pre_process(onnx_fp32, onnx_prep, skip_optimization=False)
        print(f"  저장: {onnx_prep}")
    else:
        print(f"  [SKIP] 이미 존재")

    # Round 2 — prep
    print("\n[Round 2] pre-processed ONNX 양자화")
    run_variants(
        get_variants(input_name, onnx_fp32, onnx_prep, out_dir, use_prep=True),
        fp32_preds, test_imgs, N_TEST,
    )


def main():
    models    = find_models()
    if not models:
        print(f"[ERROR] {ONNX_DIR} 에 ONNX 파일 없음")
        return

    labels    = load_labels()
    test_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, "**", "*.jpg"), recursive=True))[:N_TEST]

    # FP32 예측은 첫 번째 모델 기준 (모델마다 다르므로 각 모델에서 재계산)
    print(f"발견된 모델 {len(models)}개: {[os.path.basename(m) for m in models]}")
    print(f"테스트 이미지: {N_TEST}장  |  calibration: {N_CALIB}장\n")

    for onnx_fp32 in models:
        sess   = ort.InferenceSession(onnx_fp32, providers=["CPUExecutionProvider"])
        fp32_preds = [infer(sess, preprocess(p)) for p in test_imgs]
        diagnose_model(onnx_fp32, test_imgs, fp32_preds, labels)

    print(f"\n{'='*70}")
    print("판단 기준:")
    print("  ✓ 90%+  : 해당 양자화 정상 → qbcompiler 문제")
    print("  △ 70%+  : 경계 케이스")
    print("  ✗ 70%-  : EfficientViT 자체가 INT8 PTQ에 부적합")


if __name__ == "__main__":
    main()
