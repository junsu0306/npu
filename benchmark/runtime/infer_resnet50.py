#!/usr/bin/env python3
"""Minimal direct-qbruntime ResNet50 inference demo."""

import argparse
import os

import cv2
import numpy as np
import qbruntime


NPU_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mxq", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected", type=int, default=None)
    args = parser.parse_args()

    for path in (args.mxq, args.image):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot read image: %s" % args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    scale = 256.0 / min(height, width)
    image = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))))
    height, width = image.shape[:2]
    image = image[(height - 224) // 2:(height - 224) // 2 + 224,
                  (width - 224) // 2:(width - 224) // 2 + 224]
    image = image.astype(np.float32) / 255.0
    image = (image - np.float32([0.485, 0.456, 0.406])) / np.float32(
        [0.229, 0.224, 0.225]
    )

    accelerator = qbruntime.Accelerator(0)
    config = qbruntime.ModelConfig()
    config.set_single_core_mode(core_ids=[
        qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0),
    ])
    model = qbruntime.Model(args.mxq, config)
    model.launch(accelerator)
    try:
        result = model.infer([np.ascontiguousarray(image)])
    finally:
        model.dispose()

    predicted = int(np.argmax(np.asarray(result[0]).reshape(-1)))
    print("Predicted: %d" % predicted)
    if args.expected is not None:
        print("Expected : %d" % args.expected)
        print("Result   : %s" % ("PASS" if predicted == args.expected else "FAIL"))


if __name__ == "__main__":
    main()
