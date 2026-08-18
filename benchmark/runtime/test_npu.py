#!/usr/bin/env python3
"""
calibration_data 의 첫 N개 이미지로 NPU inference 테스트.

사용법:
  python3 benchmark/runtime/test_npu.py --model assets/mxq/model.mxq --labels imagenet_classes.txt --n 10
  python3 benchmark/runtime/test_npu.py --model assets/mxq/resnet50_timm.mxq --labels imagenet_classes.txt --n 10
"""

import argparse
import glob
import os
import numpy as np
import qbruntime

from run_npu import build_config, preprocess

NPU_ROOT  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
CALIB_DIR = os.path.join(NPU_ROOT, "assets", "calibration_data")


def load_labels(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return [l.strip() for l in f]
    return [str(i) for i in range(1000)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--n",      type=int, default=10, help="테스트할 이미지 수")
    parser.add_argument("--chw",    action="store_true", help="입력을 CHW (3,224,224)로 변환")
    parser.add_argument("--preprocess", choices=["timm", "legacy"], default="timm")
    parser.add_argument(
        "--core-mode", choices=["single", "multi", "global", "global4", "global8"],
        default="global8",
    )
    args = parser.parse_args()

    labels = load_labels(args.labels)

    all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, "**", "*.jpg"),  recursive=True))
    if not all_imgs:
        all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, "**", "*.JPEG"), recursive=True))
    test_imgs = all_imgs[:args.n]

    fmt = "CHW" if args.chw else "HWC"
    print(f"입력 포맷: {fmt}")
    print(f"테스트 이미지: {len(test_imgs)}개\n")

    acc = qbruntime.Accelerator(0)
    config = build_config(args.core_mode)
    model = qbruntime.Model(args.model, config)
    model.launch(acc)

    for img_path in test_imgs:
        img = preprocess(img_path, args.preprocess)
        if args.chw:
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
        logits = model.infer([img])[0].flatten()
        probs  = np.exp(logits - logits.max())
        probs /= probs.sum()
        top5   = np.argsort(probs)[::-1][:5]

        print(f"[{os.path.relpath(img_path, NPU_ROOT)}]")
        for rank, idx in enumerate(top5, 1):
            print(f"  {rank}. [{idx:4d}] {labels[idx]:<45s} {probs[idx]*100:.2f}%")
        print()

    model.dispose()


if __name__ == "__main__":
    main()
