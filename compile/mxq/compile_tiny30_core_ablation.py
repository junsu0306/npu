#!/usr/bin/env python3
"""Compile the same Tiny30 Patch-Slimming ONNX for core-mode ablation.

This script intentionally reuses an existing calibration list by default so
that ``single`` and ``global8`` differ only in ``inference_scheme``.
"""

import argparse
import os

from qbcompiler import mxq_compile


HERE = os.path.dirname(os.path.abspath(__file__))
NPU_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DEFAULT_ONNX = os.path.join(
    NPU_ROOT, "assets", "onnx", "tiny30__paper__rel002_npusafe.onnx"
)
DEFAULT_CALIB = os.path.join(HERE, "calib_hwc.txt")
DEFAULT_OUTPUT = os.path.join(NPU_ROOT, "assets", "mxq", "core_ablation")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default=DEFAULT_ONNX)
    parser.add_argument("--calib-list", default=DEFAULT_CALIB)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--schemes", nargs="+", default=["single", "global8"],
        choices=["single", "multi", "global", "global4", "global8"],
    )
    parser.add_argument("--preset", default="vision_transformer")
    parser.add_argument("--no-cpu-offload", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacement of an existing diagnostic MXQ.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.onnx):
        raise FileNotFoundError(args.onnx)
    if not os.path.isfile(args.calib_list):
        raise FileNotFoundError(args.calib_list)

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.onnx))[0]

    for scheme in args.schemes:
        output = os.path.join(
            args.output_dir, "%s_%s_%s.mxq" % (stem, args.preset, scheme)
        )
        if os.path.exists(output) and not args.overwrite:
            print("[SKIP] %s (use --overwrite to replace)" % output)
            continue

        print("\nCompiling core ablation")
        print("  ONNX       : %s" % args.onnx)
        print("  calibration: %s" % args.calib_list)
        print("  preset     : %s" % args.preset)
        print("  scheme     : %s" % scheme)
        print("  CPU offload: %s" % (not args.no_cpu_offload))
        print("  output     : %s" % output)

        mxq_compile(
            model=args.onnx,
            calib_data_path=args.calib_list,
            save_path=output,
            backend="onnx",
            inference_scheme=scheme,
            cpu_offload=not args.no_cpu_offload,
            config_preset=args.preset,
        )


if __name__ == "__main__":
    main()
