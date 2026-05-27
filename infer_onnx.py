#!/usr/bin/env python3
# /usr/bin/python3 infer_onnx.py --model efficientvit_b0_r224_nchw.onnx --image val_example.JPEG

import argparse
import cv2
import numpy as np
import onnxruntime as ort
import urllib.request

MEAN = np.float32([0.485, 0.456, 0.406]).reshape(1, 1, 3)
STD  = np.float32([0.229, 0.224, 0.225]).reshape(1, 1, 3)

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = 256 / min(h, w)
    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))
    h, w = img.shape[:2]
    img = img[(h - 224) // 2:(h - 224) // 2 + 224, (w - 224) // 2:(w - 224) // 2 + 224]
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[None]  # NCHW


def load_labels():
    with urllib.request.urlopen(LABELS_URL) as r:
        return r.read().decode().strip().split("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .onnx file")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    labels = load_labels()
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp = preprocess(args.image)
    logits = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0]

    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    top_idx = np.argsort(probs)[::-1][:args.top_k]

    print(f"\nImage: {args.image}")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank}. [{idx:4d}] {labels[idx]:<45s} {probs[idx]*100:.2f}%")


if __name__ == "__main__":
    main()
