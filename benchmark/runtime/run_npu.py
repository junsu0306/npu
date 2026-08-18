#!/usr/bin/env python3
"""
커스텀 MXQ 모델 NPU 추론 스크립트.
latency / FPS 측정 + Top-5 예측 출력.

실행:
  python3 benchmark/runtime/run_npu.py --mxq assets/mxq/model.mxq --image val_example.JPEG
  python3 benchmark/runtime/run_npu.py --mxq assets/mxq/model.mxq --runs 50 --warmups 5
"""

import argparse
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
import qbruntime

NPU_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])


def preprocess(image_path: str) -> np.ndarray:
    """ImageNet 전처리 → HWC float32 (224, 224, 3)."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없음: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = 256 / min(h, w)
    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))
    h, w = img.shape[:2]
    img = img[(h - 224) // 2:(h - 224) // 2 + 224,
              (w - 224) // 2:(w - 224) // 2 + 224]
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img  # (224, 224, 3)


def build_config(core_mode: str) -> qbruntime.ModelConfig:
    config = qbruntime.ModelConfig()
    C, Co = qbruntime.Cluster, qbruntime.Core

    if core_mode == "single":
        config.set_single_core_mode(core_ids=[
            qbruntime.CoreId(C.Cluster0, Co.Core0),
        ])
    elif core_mode in ("multi", "global", "global4", "global8"):
        # global 모드: 가용 코어 수에 맞게 CoreId 목록 전달
        core_map = {
            "multi":   [(C.Cluster0, Co.Core0), (C.Cluster0, Co.Core1)],
            "global":  [(C.Cluster0, Co.Core0), (C.Cluster0, Co.Core1),
                        (C.Cluster1, Co.Core0), (C.Cluster1, Co.Core1)],
            "global4": [(C.Cluster0, Co.Core0), (C.Cluster0, Co.Core1),
                        (C.Cluster1, Co.Core0), (C.Cluster1, Co.Core1)],
            "global8": [(C.Cluster0, Co.Core0), (C.Cluster0, Co.Core1),
                        (C.Cluster0, Co.Core2), (C.Cluster0, Co.Core3),
                        (C.Cluster1, Co.Core0), (C.Cluster1, Co.Core1),
                        (C.Cluster1, Co.Core2), (C.Cluster1, Co.Core3)],
        }
        ids = [qbruntime.CoreId(cl, co) for cl, co in core_map[core_mode]]
        config.set_single_core_mode(core_ids=ids)
    else:
        raise ValueError(f"알 수 없는 core-mode: {core_mode}")

    return config


def load_labels(path: str) -> list:
    if path and os.path.exists(path):
        with open(path) as f:
            return [l.strip() for l in f]
    return [str(i) for i in range(1000)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mxq",        required=True,            help="MXQ 파일 경로")
    parser.add_argument("--image",      default=os.path.join(NPU_ROOT, "val_example.JPEG"), help="추론할 이미지")
    parser.add_argument("--labels",     default=os.path.join(NPU_ROOT, "imagenet_classes.txt"))
    parser.add_argument("--core-mode",  default="global8",
                        choices=["single", "multi", "global", "global4", "global8"],
                        help="NPU 코어 모드 (컴파일된 inference_scheme과 일치해야 함)")
    parser.add_argument("--runs",       type=int, default=20,  help="측정 반복 횟수")
    parser.add_argument("--warmups",    type=int, default=5,   help="워밍업 횟수")
    parser.add_argument("--no-save",    action="store_true",   help="결과 저장 안 함")
    args = parser.parse_args()

    if not os.path.exists(args.mxq):
        raise FileNotFoundError(f"MXQ 파일 없음: {args.mxq}")

    labels = load_labels(args.labels)
    img    = preprocess(args.image)

    print(f"모델  : {args.mxq}")
    print(f"이미지: {args.image}  → shape={img.shape}")
    print(f"코어  : {args.core_mode}, warmup={args.warmups}, runs={args.runs}\n")

    acc    = qbruntime.Accelerator(0)
    config = build_config(args.core_mode)
    model  = qbruntime.Model(args.mxq, config)
    model.launch(acc)

    # 워밍업
    for _ in range(args.warmups):
        model.infer([img])

    # 측정
    t0 = time.perf_counter()
    for _ in range(args.runs):
        output = model.infer([img])
    t1 = time.perf_counter()

    model.dispose()

    total_s = t1 - t0
    avg_ms  = total_s / args.runs * 1000
    fps     = args.runs / total_s

    print(f"총 시간 : {total_s:.4f}s")
    print(f"평균 레이턴시: {avg_ms:.2f}ms")
    print(f"FPS     : {fps:.2f}\n")

    # Top-5 출력
    logits = output[0].flatten()
    probs  = np.exp(logits - logits.max())
    probs /= probs.sum()
    top5   = np.argsort(probs)[::-1][:5]
    print("Top-5 예측:")
    for rank, idx in enumerate(top5, 1):
        print(f"  {rank}. [{idx:4d}] {labels[idx]:<45s} {probs[idx]*100:.2f}%")

    if not args.no_save:
        result = {
            "mxq": args.mxq,
            "image": args.image,
            "core_mode": args.core_mode,
            "runs": args.runs,
            "warmups": args.warmups,
            "total_s": round(total_s, 4),
            "avg_ms": round(avg_ms, 2),
            "fps": round(fps, 2),
            "top5": [{"rank": i+1, "idx": int(top5[i]), "label": labels[top5[i]],
                      "prob": round(float(probs[top5[i]]*100), 2)} for i in range(5)],
            "timestamp": datetime.now().isoformat(),
        }
        results_dir = os.path.join(NPU_ROOT, "results", "runtime")
        os.makedirs(results_dir, exist_ok=True)
        out = os.path.join(results_dir, f"npu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
