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

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

ONNX_FP32 = os.path.join(ASSETS_DIR, "onnx", "efficientvit_b0_r224_timm_nchw.onnx")
OUT_DIR   = os.path.join(ASSETS_DIR, "onnx", "quant_diag")
CALIB_DIR = os.path.join(ASSETS_DIR, "calibration_data")
LABEL_FILE = os.path.join(BASE_DIR, "imagenet_classes.txt")

N_CALIB = 200
N_TEST  = 10

MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])

os.makedirs(OUT_DIR, exist_ok=True)


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

def get_variants(input_name: str):
    """(label, output_path, build_fn) 리스트 반환."""

    variants = []

    # ── Dynamic INT8 (weight only, no calibration needed) ─────────────────
    dyn_path = os.path.join(OUT_DIR, "dynamic_int8.onnx")
    def build_dynamic(out=dyn_path):
        if not os.path.exists(out):
            quantize_dynamic(ONNX_FP32, out, weight_type=QuantType.QInt8)
    variants.append(("Dynamic INT8 (weight-only)", dyn_path, build_dynamic))

    # ── Static INT8 변형들 ─────────────────────────────────────────────────
    static_cases = [
        # (suffix, per_channel, calib_method, quant_format, act_type, wt_type)
        ("static_int8_pertensor_minmax_qdq",
         False, CalibrationMethod.MinMax, QuantFormat.QDQ, QuantType.QInt8, QuantType.QInt8),

        ("static_int8_perchannel_minmax_qdq",
         True,  CalibrationMethod.MinMax, QuantFormat.QDQ, QuantType.QInt8, QuantType.QInt8),

        ("static_int8_perchannel_entropy_qdq",
         True,  CalibrationMethod.Entropy, QuantFormat.QDQ, QuantType.QInt8, QuantType.QInt8),

        ("static_int8_perchannel_percentile_qdq",
         True,  CalibrationMethod.Percentile, QuantFormat.QDQ, QuantType.QInt8, QuantType.QInt8),

        ("static_int8_perchannel_minmax_qop",
         True,  CalibrationMethod.MinMax, QuantFormat.QOperator, QuantType.QInt8, QuantType.QInt8),

        ("static_uint8_perchannel_minmax_qdq",
         True,  CalibrationMethod.MinMax, QuantFormat.QDQ, QuantType.QUInt8, QuantType.QUInt8),
    ]

    for (suffix, per_ch, calib_m, fmt, act_t, wt_t) in static_cases:
        out_path = os.path.join(OUT_DIR, f"{suffix}.onnx")
        def build_static(out=out_path, pc=per_ch, cm=calib_m, f=fmt, at=act_t, wt=wt_t):
            if not os.path.exists(out):
                reader = CalibReader(input_name)
                quantize_static(
                    model_input=ONNX_FP32,
                    model_output=out,
                    calibration_data_reader=reader,
                    calibrate_method=cm,
                    quant_format=f,
                    activation_type=at,
                    weight_type=wt,
                    per_channel=pc,
                    reduce_range=False,
                )
        variants.append((suffix, out_path, build_static))

    # ── INT16 (onnxruntime >= 1.16 필요) ───────────────────────────────────
    try:
        _ = QuantType.QUInt16
        int16_path = os.path.join(OUT_DIR, "static_uint16_perchannel_minmax.onnx")
        def build_int16(out=int16_path):
            if not os.path.exists(out):
                reader = CalibReader(input_name)
                quantize_static(
                    model_input=ONNX_FP32,
                    model_output=out,
                    calibration_data_reader=reader,
                    calibrate_method=CalibrationMethod.MinMax,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt16,
                    weight_type=QuantType.QInt8,
                    per_channel=True,
                    reduce_range=False,
                )
        variants.append(("Static UINT16 act + INT8 weight", int16_path, build_int16))
    except AttributeError:
        print("[INFO] onnxruntime < 1.16, INT16 quantization 스킵")

    return variants


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(ONNX_FP32):
        print(f"[ERROR] {ONNX_FP32} 없음")
        return

    labels = load_labels()

    sess_fp32 = ort.InferenceSession(ONNX_FP32, providers=["CPUExecutionProvider"])
    input_name = sess_fp32.get_inputs()[0].name

    test_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, "**", "*.jpg"), recursive=True))[:N_TEST]

    # float32 예측 미리 계산
    fp32_preds = [infer(sess_fp32, preprocess(p)) for p in test_imgs]

    variants = get_variants(input_name)

    print(f"\n{'='*70}")
    print(f"{'방법':<45} {'일치':>5}  {'Top-1 match rate':>16}")
    print(f"{'='*70}")

    # float32 baseline
    print(f"{'FP32 (baseline)':<45} {'10/10':>5}  {'100.0%':>16}")
    print(f"{'-'*70}")

    for label, out_path, build_fn in variants:
        # 컴파일
        try:
            build_fn()
        except Exception as e:
            print(f"{label:<45} [컴파일 실패: {e}]")
            continue

        if not os.path.exists(out_path):
            print(f"{label:<45} [파일 없음]")
            continue

        # 추론
        try:
            sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
            preds = [infer(sess, preprocess(p)) for p in test_imgs]
            match = sum(p == q for p, q in zip(fp32_preds, preds))
            rate = match / N_TEST * 100
            bar = "✓" if match == N_TEST else ("△" if match >= N_TEST * 0.7 else "✗")
            print(f"{label:<45} {match:>2}/{N_TEST}  {rate:>14.1f}%  {bar}")
        except Exception as e:
            print(f"{label:<45} [추론 실패: {e}]")

    print(f"{'='*70}")
    print("\n판단 기준:")
    print("  ✓ (90%+) : 해당 양자화는 정상 → qbcompiler 문제")
    print("  △ (70%+) : 부분 오차 → 경계 케이스")
    print("  ✗ (70%-) : 양자화 자체가 EfficientViT에 부적합")

    # 상세 출력: fp32 vs 각 방법 top-1 label
    print(f"\n{'─'*70}")
    print("이미지별 상세 (FP32 예측):")
    for i, (p, img_path) in enumerate(zip(fp32_preds, test_imgs)):
        print(f"  [{i+1:2d}] {os.path.basename(img_path):<30s} → [{p:4d}] {labels[p]}")


if __name__ == "__main__":
    main()
