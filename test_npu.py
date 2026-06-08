#!/usr/bin/env python3
"""
calibration_data 의 첫 10개 이미지로 NPU inference 테스트.

사용법:
  python3 test_npu.py --model assets/mxq/efficientvit_b0_r224_timm.mxq
  python3 test_npu.py --model assets/mxq/efficientvit_b0_r224_timm.mxq --labels imagenet_classes.txt
"""

import argparse
import glob
import os
import cv2
import numpy as np
import qbruntime

MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR  = os.path.join(BASE_DIR, "assets", "calibration_data")


def preprocess(image_path: str, chw: bool = False) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = 256 / min(h, w)
    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))
    h, w = img.shape[:2]
    img = img[(h-224)//2:(h-224)//2+224, (w-224)//2:(w-224)//2+224]
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD              # HWC (224,224,3)
    if chw:
        img = img.transpose(2, 0, 1)      # CHW (3,224,224)
    return img


def load_labels(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return [l.strip() for l in f]
    return [str(i) for i in range(1000)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True)
    parser.add_argument("--labels",  default=None)
    parser.add_argument("--n",       type=int, default=10, help="테스트할 이미지 수")
    parser.add_argument("--chw",     action="store_true", help="입력을 CHW (3,224,224)로 변환 (torch 백엔드 모델용)")
    args = parser.parse_args()

    labels = load_labels(args.labels)

    # calibration_data 에서 이미지 수집 (서브폴더 포함)
    all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, "**", "*.jpg"), recursive=True))
    if not all_imgs:
        all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, "**", "*.JPEG"), recursive=True))
    test_imgs = all_imgs[:args.n]
    print(f"테스트 이미지: {len(test_imgs)}개\n")

    # NPU 모델 로드
    acc = qbruntime.Accelerator(0)
    config = qbruntime.ModelConfig()
    config.set_single_core_mode(core_ids=[
        qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)
    ])
    model = qbruntime.Model(args.model, config)
    model.launch(acc)

    fmt = "CHW" if args.chw else "HWC"
    print(f"입력 포맷: {fmt}\n")

    for img_path in test_imgs:
        img = preprocess(img_path, chw=args.chw)
        logits = model.infer([img])[0].flatten()
        probs  = np.exp(logits - logits.max())
        probs /= probs.sum()
        top5   = np.argsort(probs)[::-1][:5]

        print(f"[{os.path.relpath(img_path, BASE_DIR)}]")
        for rank, idx in enumerate(top5, 1):
            print(f"  {rank}. [{idx:4d}] {labels[idx]:<45s} {probs[idx]*100:.2f}%")
        print()

    model.dispose()


if __name__ == "__main__":
    main()
