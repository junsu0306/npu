#!/usr/bin/env python3
# CPU reference inference for an ONNX model. Paths are provided explicitly.

import argparse
import cv2
import numpy as np
import onnxruntime as ort
import os
import urllib.request

PROFILES = {
    "timm": {
        "resize_short": 249,
        "interpolation": cv2.INTER_CUBIC,
        "mean": np.float32([0.5, 0.5, 0.5]),
        "std": np.float32([0.5, 0.5, 0.5]),
    },
    "legacy": {
        "resize_short": 256,
        "interpolation": cv2.INTER_LINEAR,
        "mean": np.float32([0.485, 0.456, 0.406]),
        "std": np.float32([0.229, 0.224, 0.225]),
    },
}

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def preprocess(image_path, profile, layout):
    cfg = PROFILES[profile]
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot read image: %s" % image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = cfg["resize_short"] / min(h, w)
    img = cv2.resize(
        img, (int(round(w * scale)), int(round(h * scale))),
        interpolation=cfg["interpolation"],
    )
    h, w = img.shape[:2]
    img = img[(h - 224) // 2:(h - 224) // 2 + 224, (w - 224) // 2:(w - 224) // 2 + 224]
    img = img.astype(np.float32) / 255.0
    img = (img - cfg["mean"]) / cfg["std"]
    if layout == "nchw":
        img = img.transpose(2, 0, 1)
    return np.ascontiguousarray(img[None], dtype=np.float32)


def resolve_layout(input_shape, requested):
    if requested != "auto":
        return requested
    if len(input_shape) != 4:
        raise ValueError("cannot infer layout from input shape: %s" % input_shape)
    if input_shape[-1] == 3:
        return "nhwc"
    if input_shape[1] == 3:
        return "nchw"
    raise ValueError("cannot infer layout from input shape: %s" % input_shape)


def load_labels(path):
    if path and os.path.isfile(path):
        with open(path) as handle:
            return [line.strip() for line in handle]
    with urllib.request.urlopen(LABELS_URL) as r:
        return r.read().decode().strip().split("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .onnx file")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--preprocess", choices=sorted(PROFILES), default="timm")
    parser.add_argument("--layout", choices=["auto", "nhwc", "nchw"], default="auto")
    parser.add_argument("--labels", default="imagenet_classes.txt")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_meta = sess.get_inputs()[0]
    layout = resolve_layout(input_meta.shape, args.layout)
    inp = preprocess(args.image, args.preprocess, layout)
    print("Input: name=%s shape=%s layout=%s preprocess=%s" % (
        input_meta.name, tuple(inp.shape), layout, args.preprocess
    ))
    logits = sess.run(None, {input_meta.name: inp})[0][0]

    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    top_idx = np.argsort(probs)[::-1][:args.top_k]

    print(f"\nImage: {args.image}")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank}. [{idx:4d}] {labels[idx]:<45s} {probs[idx]*100:.2f}%")


if __name__ == "__main__":
    main()
