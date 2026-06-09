#!/usr/bin/env python3
"""
EfficientViT 양자화 진단 스크립트.

float32 ONNX vs ONNX Runtime INT8 양자화 결과를 비교해서
양자화 자체가 문제인지 컴파일러가 문제인지 확인.

사용법:
  source aries_env/bin/activate
  pip install onnxruntime onnxruntime-tools  # 없으면 설치
  python3 diagnose_quant.py
"""

import glob
import os
import numpy as np
import cv2
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

ONNX_FP32  = os.path.join(ASSETS_DIR, "onnx", "efficientvit_b0_r224_timm_nchw.onnx")
ONNX_INT8  = os.path.join(ASSETS_DIR, "onnx", "efficientvit_b0_r224_timm_int8_ort.onnx")
CALIB_DIR  = os.path.join(ASSETS_DIR, "calibration_data")
LABEL_FILE = os.path.join(BASE_DIR, "imagenet_classes.txt")
N_CALIB    = 200
N_TEST     = 10

MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])


# ── 전처리 (NCHW — ONNX 모델 포맷) ──────────────────────────────────────────

def preprocess_nchw(image_path: str) -> np.ndarray:
    """이미지 → NCHW float32 (1,3,224,224)"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = 256 / min(h, w)
    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))
    h, w = img.shape[:2]
    img = img[(h-224)//2:(h-224)//2+224, (w-224)//2:(w-224)//2+224]
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[np.newaxis]  # (1,3,224,224)


# ── Calibration Data Reader ──────────────────────────────────────────────────

class ImageNetCalibReader(CalibrationDataReader):
    def __init__(self, calib_dir: str, input_name: str, n: int = 200):
        jpg_files = sorted(glob.glob(os.path.join(calib_dir, "**", "*.jpg"), recursive=True))
        self.images = jpg_files[:n]
        self.input_name = input_name
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self.images):
            return None
        img = preprocess_nchw(self.images[self._idx])
        self._idx += 1
        return {self.input_name: img}


# ── 레이블 로드 ──────────────────────────────────────────────────────────────

def load_labels():
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE) as f:
            return [l.strip() for l in f]
    return [str(i) for i in range(1000)]


# ── 추론 + Top-5 출력 ────────────────────────────────────────────────────────

def run_inference(session: ort.InferenceSession, img_nchw: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: img_nchw})[0].flatten()
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return probs


def print_top5(probs, labels, prefix=""):
    top5 = np.argsort(probs)[::-1][:5]
    for rank, idx in enumerate(top5, 1):
        print(f"  {prefix}{rank}. [{idx:4d}] {labels[idx]:<45s} {probs[idx]*100:.2f}%")


# ── INT8 양자화 ──────────────────────────────────────────────────────────────

def quantize_model():
    if os.path.exists(ONNX_INT8):
        print(f"[SKIP] INT8 모델 이미 존재: {ONNX_INT8}")
        return

    # input name 확인
    sess_fp32 = ort.InferenceSession(ONNX_FP32, providers=["CPUExecutionProvider"])
    input_name = sess_fp32.get_inputs()[0].name
    print(f"ONNX input name: {input_name}")

    print(f"\nINT8 static quantization 시작 (calibration {N_CALIB}장)...")

    # per-channel weight quantization (QInt8)
    calib_reader = ImageNetCalibReader(CALIB_DIR, input_name, n=N_CALIB)
    quantize_static(
        model_input=ONNX_FP32,
        model_output=ONNX_INT8,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=True,        # per-channel weight quantization
        reduce_range=False,
        optimize_model=True,
    )
    print(f"INT8 모델 저장: {ONNX_INT8}")


# ── 메인 비교 ────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(ONNX_FP32):
        print(f"[ERROR] float32 ONNX 없음: {ONNX_FP32}")
        return

    quantize_model()

    labels = load_labels()

    sess_fp32 = ort.InferenceSession(ONNX_FP32, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(ONNX_INT8, providers=["CPUExecutionProvider"])

    test_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, "**", "*.jpg"), recursive=True))[:N_TEST]
    print(f"\n{'='*60}")
    print(f"float32 ONNX  vs  ONNX Runtime INT8  비교 ({N_TEST}장)")
    print(f"{'='*60}\n")

    match = 0
    for img_path in test_imgs:
        img = preprocess_nchw(img_path)
        probs_fp32 = run_inference(sess_fp32, img)
        probs_int8 = run_inference(sess_int8, img)

        pred_fp32 = np.argmax(probs_fp32)
        pred_int8 = np.argmax(probs_int8)
        same = "✓ MATCH" if pred_fp32 == pred_int8 else "✗ DIFFER"
        if pred_fp32 == pred_int8:
            match += 1

        print(f"[{os.path.basename(img_path)}]  {same}")
        print(f"  FP32 → [{pred_fp32:4d}] {labels[pred_fp32]:<40s} {probs_fp32[pred_fp32]*100:.1f}%")
        print(f"  INT8 → [{pred_int8:4d}] {labels[pred_int8]:<40s} {probs_int8[pred_int8]*100:.1f}%")
        print()

    print(f"결과: {match}/{N_TEST} 일치")
    if match == N_TEST:
        print("→ ONNX Runtime INT8 양자화는 정상. 문제는 qbcompiler 쪽.")
    elif match >= N_TEST * 0.7:
        print("→ 대부분 일치. 일부 경계 케이스에서 양자화 오차 발생.")
    else:
        print("→ ONNX Runtime INT8도 틀림. EfficientViT 자체가 INT8에 취약.")


if __name__ == "__main__":
    main()
