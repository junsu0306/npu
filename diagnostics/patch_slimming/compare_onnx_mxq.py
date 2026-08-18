#!/usr/bin/env python3
"""Compare one diagnostic MXQ output with ONNX Runtime on the exact same input."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import qbruntime

HERE = Path(__file__).resolve()
NPU_ROOT = HERE.parents[2]
sys.path.insert(0, str(NPU_ROOT / "benchmark" / "runtime"))
from run_npu import build_config  # noqa: E402


def metrics(ref, got):
    ref, got = ref.astype(np.float64).ravel(), got.astype(np.float64).ravel()
    diff = got - ref
    nr, ng = np.linalg.norm(ref), np.linalg.norm(got)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "relative_l2": float(np.linalg.norm(diff) / max(nr, 1e-12)),
        "cosine": float(np.dot(ref, got) / max(nr * ng, 1e-12)),
        "reference_range": [float(ref.min()), float(ref.max())],
        "mxq_range": [float(got.min()), float(got.max())],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--mxq", required=True)
    p.add_argument(
        "--input",
        default=str(
            NPU_ROOT
            / "assets/diagnostics/patch_slimming/input/diagnostic_input_hwc.npy"
        ),
    )
    p.add_argument("--core-mode", default="global8", choices=["single", "multi", "global", "global4", "global8"])
    p.add_argument("--output", default="")
    a = p.parse_args()

    x = np.ascontiguousarray(np.load(a.input), dtype=np.float32)
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: x[None]})[0]
    acc = qbruntime.Accelerator(0)
    model = qbruntime.Model(a.mxq, build_config(a.core_mode))
    model.launch(acc)
    try:
        got = np.asarray(model.infer([x])[0])
    finally:
        model.dispose()
    if ref.size != got.size:
        raise RuntimeError(f"output size mismatch: ONNX={ref.shape}, MXQ={got.shape}")
    result = {"onnx": a.onnx, "mxq": a.mxq, "input": a.input, "core_mode": a.core_mode,
              "output_shape_onnx": list(ref.shape), "output_shape_mxq": list(got.shape), **metrics(ref, got)}
    print(json.dumps(result, indent=2))
    if a.output:
        os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
        with open(a.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
