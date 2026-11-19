#!/usr/bin/env python3
"""Compile final NHWC ONNX models with Mobilint's vision_transformer preset.

Run inside the qbcompiler Docker. Configuration uses NPU_ONNX_DIR,
NPU_MXQ_DIR, NPU_CALIB_DIR, NPU_ONNX_FILTER and NPU_INFERENCE_SCHEME.
--dry-run validates inputs without importing qbcompiler or writing files.
Existing outputs are skipped; use a new output directory for a rebuild.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PRESET = "vision_transformer"
CPU_OFFLOAD = True
SCHEMES = ("single", "multi", "global", "global4", "global8")


def validate_calibration(directory, expected_count=1000):
    directory = Path(directory)
    profile_path = directory / "preprocess_profile.json"
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"Missing {profile_path}; regenerate with generate_hwc.py --profile timm "
            "in a new directory. Do not reuse unidentified calibration tensors."
        )
    profile = json.loads(profile_path.read_text())
    expected = {
        "profile": "timm", "resize_short": 249, "interpolation_name": "bicubic",
        "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
        "layout": "HWC", "shape": [224, 224, 3], "dtype": "float32",
        "num_tensors": expected_count,
    }
    for key, value in expected.items():
        if profile.get(key) != value:
            raise ValueError(f"Calibration metadata mismatch: {key} must be {value!r}")
    paths = sorted(directory.glob("*.npy"))
    if len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count} calibration tensors, got {len(paths)}")
    for path in paths:
        tensor = np.load(path, allow_pickle=False)
        if tensor.shape != (224, 224, 3) or tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise ValueError(f"Invalid HWC float32 calibration tensor: {path}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    scheme = os.environ.get("NPU_INFERENCE_SCHEME", "global8")
    if scheme not in SCHEMES:
        parser.error(f"NPU_INFERENCE_SCHEME must be one of {SCHEMES}")
    calib_dir = Path(os.environ.get("NPU_CALIB_DIR", ROOT / "assets/calib_hwc_timm")).resolve()
    onnx_dir = Path(os.environ.get("NPU_ONNX_DIR", ROOT / "assets/onnx")).resolve()
    mxq_dir = Path(os.environ.get("NPU_MXQ_DIR", ROOT / "assets/mxq")).resolve()
    name_filter = os.environ.get("NPU_ONNX_FILTER", "__nhwc_b1_npusafe.onnx")
    tensors = validate_calibration(calib_dir)
    models = sorted(p for p in onnx_dir.glob("*.onnx") if name_filter in p.name)
    if not models:
        raise FileNotFoundError(f"No ONNX files matching {name_filter!r} in {onnx_dir}")
    jobs = [(model, mxq_dir / f"{model.stem}_vt_{scheme}.mxq") for model in models]
    print(f"{len(tensors)} calibration tensors; {len(jobs)} ONNX models")
    print(f"Preset={CONFIG_PRESET}, scheme={scheme}, cpu_offload={CPU_OFFLOAD}")
    for model, output in jobs:
        print(f"{'[SKIP existing]' if output.exists() else '[COMPILE]'} {model.name} -> {output}")
    if args.dry_run:
        return

    # SDK is needed only for actual compilation, not host-side validation.
    from qbcompiler import mxq_compile

    mxq_dir.mkdir(parents=True, exist_ok=True)
    calib_txt = mxq_dir / "calib_hwc.txt"
    calib_txt.write_text("\n".join(str(p.resolve()) for p in tensors) + "\n")
    for model, output in jobs:
        if output.exists():
            continue
        mxq_compile(
            model=str(model),
            calib_data_path=str(calib_txt),
            save_path=str(output),
            backend="onnx",
            inference_scheme=scheme,
            cpu_offload=CPU_OFFLOAD,
            config_preset=CONFIG_PRESET,
        )
        print(f"Saved: {output}")
    print("All done. Validate new MXQs on the NPU before promoting them.")


if __name__ == "__main__":
    main()
