#!/usr/bin/env python3
"""
EfficientViT NPU Inference (qbruntime)

infer_resnet50.py 예시 기반으로 작성.
compile_nchw.py로 생성한 .mxq 파일 사용.

전처리 차이 (resnet50 예시 vs 이 스크립트):
  resnet50: HWC (H,W,C) — cpu_offload=False 컴파일
  EfficientViT: CHW (C,H,W) — cpu_offload=True + in_dformats="NCHW" 컴파일

사용법:
  cd /home/airlab_compression
  source aries_env/bin/activate
  cd npu/
  python3 inference_npu.py --model assets/efficientvit_b0_r224_nchw.mxq --image val_example.JPEG --labels imagenet_classes.txt
  python3 inference_npu.py --model assets/efficientvit_b0_r224_nchw.mxq --benchmark
  python3 inference_npu.py --model assets/efficientvit_b0_r224_nchw.mxq --val_dir /data/imagenet/val
"""
import argparse
import time
import os
import cv2
import numpy as np
import qbruntime

# ─── 전처리 ──────────────────────────────────────────────────────────────────

MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])


def preprocess(image_path: str) -> np.ndarray:
    """
    이미지 → CHW float32 배열 (compile_nchw.py 캘리브레이션 전처리와 동일)

    resnet50 예시와의 차이:
      resnet50: HWC (224,224,3) — 비cpu_offload 컴파일
      EfficientViT: CHW (3,224,224) — NCHW cpu_offload 컴파일
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize: 짧은 변 기준 256
    h, w = img.shape[:2]
    scale = 256 / min(h, w)
    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))

    # CenterCrop 224x224
    h, w = img.shape[:2]
    top  = (h - 224) // 2
    left = (w - 224) // 2
    img = img[top:top+224, left:left+224]

    # float32 정규화
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD              # (224, 224, 3) HWC

    # HWC → CHW (NCHW 컴파일 모델에 맞춰 채널 먼저)
    img = img.transpose(2, 0, 1)          # (3, 224, 224)
    return img


# ─── NPU 모델 래퍼 ───────────────────────────────────────────────────────────

class EfficientViTNPU:
    def __init__(self, mxq_path: str, device_id: int = 0):
        self.acc = qbruntime.Accelerator(device_id)
        config = qbruntime.ModelConfig()
        config.set_single_core_mode(core_ids=[
            qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0),
        ])
        self.model = qbruntime.Model(mxq_path, config)
        self.model.launch(self.acc)
        print(f"[NPU] 로드 완료: {mxq_path}")

    def infer(self, img: np.ndarray) -> np.ndarray:
        """img: CHW (3,224,224) float32 → logits (1000,)"""
        result = self.model.infer([img])
        return result[0].flatten()

    def dispose(self):
        self.model.dispose()


# ─── 유틸 ────────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def load_labels(label_file: str = None) -> list:
    if label_file and os.path.exists(label_file):
        with open(label_file) as f:
            return [l.strip() for l in f]
    return [str(i) for i in range(1000)]


# ─── 실행 모드 ───────────────────────────────────────────────────────────────

def run_single(model: EfficientViTNPU, image_path: str, labels: list, top_k: int = 5):
    img = preprocess(image_path)
    t0 = time.perf_counter()
    logits = model.infer(img)
    ms = (time.perf_counter() - t0) * 1000

    probs = softmax(logits)
    top_idx = np.argsort(probs)[::-1][:top_k]

    print(f"\n이미지: {image_path}  ({ms:.1f} ms)")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank}. [{idx:4d}] {labels[idx]:<50s} {probs[idx]*100:.2f}%")


def run_benchmark(model: EfficientViTNPU, n_iter: int = 200):
    dummy = np.random.randn(3, 224, 224).astype(np.float32)

    # 워밍업
    for _ in range(10):
        model.infer(dummy)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        model.infer(dummy)
        times.append((time.perf_counter() - t0) * 1000)

    t = np.array(times)
    print(f"\n[벤치마크] {n_iter}회")
    print(f"  평균:  {t.mean():.2f} ms  →  {1000/t.mean():.1f} FPS")
    print(f"  최소:  {t.min():.2f} ms")
    print(f"  최대:  {t.max():.2f} ms")
    print(f"  P95:   {np.percentile(t, 95):.2f} ms")


def run_accuracy(model: EfficientViTNPU, val_dir: str, labels: list,
                 top_k: int = 5, max_images: int = None):
    """
    ImageNet val 폴더 구조 기준 Top-1 / Top-k 정확도 측정.
    폴더 구조: val/n01440764/*.JPEG
    """
    import glob

    class_dirs = sorted(os.listdir(val_dir))
    class_to_idx = {cls: i for i, cls in enumerate(class_dirs)}

    top1 = topk = total = 0

    for cls_name in class_dirs:
        cls_dir = os.path.join(val_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        gt = class_to_idx[cls_name]

        for img_path in glob.glob(os.path.join(cls_dir, "*.JPEG")):
            if max_images and total >= max_images:
                break
            try:
                img = preprocess(img_path)
                logits = model.infer(img)
                preds = np.argsort(softmax(logits))[::-1][:top_k]
                if preds[0] == gt:
                    top1 += 1
                if gt in preds:
                    topk += 1
                total += 1
                if total % 500 == 0:
                    print(f"  {total}장  Top-1: {top1/total*100:.2f}%  "
                          f"Top-{top_k}: {topk/total*100:.2f}%")
            except Exception as e:
                print(f"  [skip] {img_path}: {e}")

        if max_images and total >= max_images:
            break

    print(f"\n[정확도] {total}장")
    print(f"  Top-1:   {top1/total*100:.2f}%")
    print(f"  Top-{top_k}: {topk/total*100:.2f}%")


# ─── 진입점 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True)
    parser.add_argument("--image",      default=None,         help="단일 이미지 경로")
    parser.add_argument("--val_dir",    default=None,         help="ImageNet val 폴더")
    parser.add_argument("--labels",     default=None,         help="클래스 레이블 파일")
    parser.add_argument("--top_k",      type=int, default=5)
    parser.add_argument("--benchmark",  action="store_true")
    parser.add_argument("--n_iter",     type=int, default=200)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device_id",  type=int, default=0)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    model  = EfficientViTNPU(args.model, device_id=args.device_id)

    try:
        if args.benchmark:
            run_benchmark(model, n_iter=args.n_iter)
        elif args.image:
            run_single(model, args.image, labels, top_k=args.top_k)
        elif args.val_dir:
            run_accuracy(model, args.val_dir, labels,
                         top_k=args.top_k, max_images=args.max_images)
        else:
            parser.print_help()
    finally:
        model.dispose()


if __name__ == "__main__":
    main()