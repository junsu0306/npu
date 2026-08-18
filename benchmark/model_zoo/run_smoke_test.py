#!/usr/bin/env python3
"""Fast Model Zoo smoke/latency test for one local MXQ and one image."""

import argparse
import json
import os
import sys
import time

try:
    import mblt_model_zoo.vision as vision
except Exception as exc:
    raise SystemExit("mblt_model_zoo.vision import failed: %s" % exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-class", default="ViT_Tiny_Patch16_224")
    parser.add_argument("--mxq", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--infer-mode", default="global8",
        choices=["single", "multi", "global4", "global8"],
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    for path in (args.mxq, args.image):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    if not hasattr(vision, args.model_class):
        raise ValueError("unknown Model Zoo class: %s" % args.model_class)

    model = getattr(vision, args.model_class)(
        infer_mode=args.infer_mode, local_path=args.mxq
    )
    try:
        tensor = model.preprocess(args.image)
        print("input shape=%s dtype=%s" % (tuple(tensor.shape), tensor.dtype))
        for _ in range(args.warmups):
            model(tensor)
        started = time.perf_counter()
        for _ in range(args.runs):
            output = model(tensor)
        elapsed = time.perf_counter() - started
    finally:
        model.dispose()

    result = {
        "model_class": args.model_class,
        "mxq": os.path.abspath(args.mxq),
        "image": os.path.abspath(args.image),
        "infer_mode": args.infer_mode,
        "warmups": args.warmups,
        "runs": args.runs,
        "avg_ms": elapsed * 1000.0 / args.runs,
        "fps": args.runs / elapsed,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
