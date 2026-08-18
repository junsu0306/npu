#!/usr/bin/env python3
"""Generate HWC calibration tensors from ImageNet images.

The preprocessing used for calibration must be identical to the preprocessing
used for ONNX/MXQ inference.  ``timm`` is the correct profile for
``timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k``.  ``legacy`` is retained
only to reproduce older repository results.
"""

import argparse
import glob
import json
import os
import cv2
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
NPU_ROOT   = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
ASSETS_DIR = os.path.join(NPU_ROOT, "assets")

PROFILES = {
    "timm": {
        "resize_short": 249,
        "interpolation": cv2.INTER_CUBIC,
        "interpolation_name": "bicubic",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
    "legacy": {
        "resize_short": 256,
        "interpolation": cv2.INTER_LINEAR,
        "interpolation_name": "bilinear",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=sorted(PROFILES), default="timm",
        help="timm is required for the AugReg ViT-Tiny; legacy reproduces old data",
    )
    parser.add_argument(
        "--src-dir", default=os.path.join(ASSETS_DIR, "calibration_data")
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(ASSETS_DIR, "calib_hwc")
    )
    parser.add_argument("--n-images", type=int, default=1000)
    return parser.parse_args()


def preprocess(image_path, profile):
    cfg = PROFILES[profile]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot read image: %s" % image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    scale = float(cfg["resize_short"]) / min(height, width)
    image = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cfg["interpolation"],
    )
    height, width = image.shape[:2]
    top, left = (height - 224) // 2, (width - 224) // 2
    image = image[top:top + 224, left:left + 224]
    image = image.astype(np.float32) / 255.0
    mean = np.float32(cfg["mean"])
    std = np.float32(cfg["std"])
    return np.ascontiguousarray((image - mean) / std, dtype=np.float32)


def main():
    args = parse_args()
    if args.n_images < 1:
        raise ValueError("--n-images must be >= 1")
    os.makedirs(args.output_dir, exist_ok=True)

    image_files = []
    for pattern in ("*.jpg", "*.jpeg", "*.JPEG", "*.png"):
        image_files.extend(glob.glob(os.path.join(args.src_dir, "**", pattern), recursive=True))
    image_files = sorted(set(
        path for path in image_files if not path.endswith(".Zone.Identifier")
    ))
    if not image_files:
        raise FileNotFoundError("No images found in %s" % args.src_dir)

    requested = min(args.n_images, len(image_files))
    print("Generating %d calibration tensors from %s -> %s" % (
        requested, args.src_dir, args.output_dir
    ))
    print("Preprocess profile: %s" % args.profile)

    saved = 0
    last_tensor = None
    for image_path in image_files:
        try:
            tensor = preprocess(image_path, args.profile)
        except ValueError as exc:
            print("  [skip] %s" % exc)
            continue
        np.save(os.path.join(args.output_dir, "%04d.npy" % saved), tensor)
        last_tensor = tensor
        saved += 1
        if saved >= requested:
            break

    metadata = dict(PROFILES[args.profile])
    metadata.pop("interpolation")
    metadata.update({
        "profile": args.profile,
        "layout": "HWC",
        "shape": [224, 224, 3],
        "dtype": "float32",
        "source": os.path.abspath(args.src_dir),
        "num_tensors": saved,
    })
    with open(os.path.join(args.output_dir, "preprocess_profile.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    print("Done: %d files saved to %s" % (saved, args.output_dir))
    if last_tensor is not None:
        print("Sample stats: min=%.3f max=%.3f mean=%.3f std=%.3f" % (
            last_tensor.min(), last_tensor.max(),
            last_tensor.mean(), last_tensor.std(),
        ))


if __name__ == "__main__":
    main()
