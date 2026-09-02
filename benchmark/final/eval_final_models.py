#!/usr/bin/env python3
"""Comprehensive ImageNet benchmark for the final Mobilint MXQ models.

The direct qbruntime path is used because the custom models accept normalized
HWC float32 tensors.  A run produces a compact Markdown/CSV summary plus the
full JSON result and one prediction array per model.

FLOPs are static ONNX estimates for Conv/MatMul/Gemm (1 MAC = 2 FLOPs).  They
are deliberately kept separate from measured NPU latency, memory and power.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import psutil
import qbruntime


NPU_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = NPU_ROOT / "assets" / "mxq_final12"
DEFAULT_ONNX_DIR = NPU_ROOT / "assets" / "onnx"
DEFAULT_VAL_DIR = NPU_ROOT.parent / "imagenet_val"

LEGACY_MEAN = np.float32([0.485, 0.456, 0.406])
LEGACY_STD = np.float32([0.229, 0.224, 0.225])
DEIT_MEAN = np.float32([0.5, 0.5, 0.5])
DEIT_STD = np.float32([0.5, 0.5, 0.5])

FINAL_NAME_RE = re.compile(
    r"^(?P<family>tiny|small)(?P<target>30|50)__(?P<variant>.+?)"
    r"__nhwc_b1_npusafe(?:_vt_[a-z0-9]+)?$"
)
VARIANT_ORDER = {
    "baseline": 0,
    "patch_slimming": 1,
    "patch_slimming_stage369_relaxed": 2,
    "reduced": 3,
}
VARIANT_LABELS = {
    "baseline": "Baseline",
    "patch_slimming": "Patch Slimming",
    "patch_slimming_stage369_relaxed": "Patch Slimming Relaxed",
    "reduced": "Reduced",
}


@dataclass
class ModelSpec:
    name: str
    path: Path
    onnx_path: Path | None
    group: str
    family: str
    target: str
    variant: str
    static_analysis: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final MXQ benchmark: ImageNet accuracy, latency distribution, "
            "CPU/NPU memory, power and static ONNX FLOPs"
        )
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="all *.mxq files are evaluated when --model is omitted",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="[NAME=]PATH",
        help="evaluate selected MXQ; repeat for multiple models",
    )
    parser.add_argument(
        "--onnx-dir",
        default=str(DEFAULT_ONNX_DIR),
        help="matching final ONNX files used only for params/FLOPs analysis",
    )
    parser.add_argument("--val-dir", default=str(DEFAULT_VAL_DIR))
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--images-per-class", type=int, default=50)
    parser.add_argument(
        "--limit", type=int, default=0, help="0 evaluates all selected images"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preprocess",
        choices=["timm", "deit", "legacy"],
        default="timm",
    )
    parser.add_argument(
        "--core-mode",
        choices=["single", "multi", "global", "global4", "global8"],
        default="global8",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--memory-sample-ms", type=float, default=100.0)
    parser.add_argument("--npu-sample-ms", type=float, default=500.0)
    parser.add_argument(
        "--skip-npu-tracker",
        action="store_true",
        help="do not collect mblt_tracker memory/power/utilization/temperature",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="default: results/final_benchmark/<timestamp>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate dataset/models and print ONNX analysis without NPU inference",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="stop after the first failed model"
    )
    args = parser.parse_args()

    if args.classes <= 0 or args.images_per_class <= 0:
        parser.error("--classes and --images-per-class must be positive")
    if args.limit < 0 or args.warmup_runs < 0:
        parser.error("--limit and --warmup-runs cannot be negative")
    if args.device_id < 0:
        parser.error("--device-id cannot be negative")
    if args.memory_sample_ms <= 0 or args.npu_sample_ms <= 0:
        parser.error("sampling intervals must be positive")
    return args


def model_metadata(stem: str) -> tuple[str, str, str, str]:
    match = FINAL_NAME_RE.match(stem)
    if not match:
        return stem, "unknown", "unknown", stem
    family = match.group("family")
    target = match.group("target")
    variant = match.group("variant")
    return family + target, family, target, variant


def model_sort_key(spec: ModelSpec) -> tuple[int, int, int, str]:
    family_order = {"tiny": 0, "small": 1}.get(spec.family, 9)
    target_order = {"30": 0, "50": 1}.get(spec.target, 9)
    return (
        family_order,
        target_order,
        VARIANT_ORDER.get(spec.variant, 8),
        spec.name,
    )


def mxq_to_onnx_stem(mxq_stem: str) -> str:
    return mxq_stem.rsplit("_vt_", 1)[0] if "_vt_" in mxq_stem else mxq_stem


def discover_models(args: argparse.Namespace) -> list[ModelSpec]:
    if args.model:
        requested = []
        for value in args.model:
            if "=" in value:
                name, raw_path = value.split("=", 1)
            else:
                raw_path = value
                name = Path(raw_path).stem
            path = Path(raw_path).expanduser().resolve()
            requested.append((name, path))
    else:
        models_dir = Path(args.models_dir).expanduser().resolve()
        if not models_dir.is_dir():
            raise NotADirectoryError(models_dir)
        requested = [(path.stem, path) for path in models_dir.glob("*.mxq")]

    if not requested:
        raise ValueError("no MXQ models selected")

    onnx_dir = Path(args.onnx_dir).expanduser().resolve()
    specs = []
    seen = set()
    for name, path in requested:
        if name in seen:
            raise ValueError("duplicate model name: %s" % name)
        seen.add(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        group, family, target, variant = model_metadata(path.stem)
        candidate = onnx_dir / (mxq_to_onnx_stem(path.stem) + ".onnx")
        onnx_path = candidate if candidate.is_file() else None
        static_analysis = analyze_onnx(onnx_path)
        specs.append(
            ModelSpec(
                name=name,
                path=path,
                onnx_path=onnx_path,
                group=group,
                family=family,
                target=target,
                variant=variant,
                static_analysis=static_analysis,
            )
        )

    specs.sort(key=model_sort_key)
    attach_baseline_reductions(specs)
    return specs


def onnx_shape(value_info: Any) -> tuple[int, ...] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        dims.append(int(dim.dim_value))
    return tuple(dims)


def analyze_onnx(path: Path | None) -> dict[str, Any]:
    """Count learned parameters and major-op FLOPs from static ONNX shapes."""
    if path is None:
        return {
            "status": "unavailable",
            "reason": "matching ONNX file not found",
            "method": "static ONNX Conv/MatMul/Gemm; 1 MAC = 2 FLOPs",
        }
    try:
        import onnx
        from onnx import shape_inference

        model = onnx.load(str(path), load_external_data=False)
        model = shape_inference.infer_shapes(
            model, strict_mode=False, data_prop=True
        )
        shapes: dict[str, tuple[int, ...]] = {}
        for value in list(model.graph.input) + list(model.graph.value_info) + list(
            model.graph.output
        ):
            shape = onnx_shape(value)
            if shape is not None:
                shapes[value.name] = shape
        for initializer in model.graph.initializer:
            shapes[initializer.name] = tuple(int(x) for x in initializer.dims)

        by_op = {"Conv": 0, "MatMul": 0, "Gemm": 0}
        missing = []
        for node in model.graph.node:
            flops = None
            if node.op_type == "MatMul":
                left = shapes.get(node.input[0])
                output = shapes.get(node.output[0])
                if left and output:
                    flops = 2 * math.prod(output) * left[-1]
            elif node.op_type == "Gemm":
                left = shapes.get(node.input[0])
                output = shapes.get(node.output[0])
                if left and output:
                    flops = 2 * math.prod(output) * left[-1]
            elif node.op_type == "Conv":
                weight = shapes.get(node.input[1])
                output = shapes.get(node.output[0])
                if weight and output:
                    # ONNX weight shape is [C_out, C_in/group, kH, kW].
                    flops = 2 * math.prod(output) * math.prod(weight[1:])
            else:
                continue

            if flops is None:
                missing.append(node.name or node.output[0])
            else:
                by_op[node.op_type] += int(flops)

        parameter_count = sum(
            math.prod(initializer.dims) for initializer in model.graph.initializer
        )
        total_flops = sum(by_op.values())
        return {
            "status": "ok" if not missing else "partial",
            "onnx": str(path),
            "onnx_file_bytes": path.stat().st_size,
            "parameter_count": int(parameter_count),
            "major_flops_per_image": int(total_flops),
            "major_gflops_per_image": total_flops / 1e9,
            "flops_by_op": by_op,
            "unresolved_major_nodes": missing,
            "method": "static ONNX Conv/MatMul/Gemm; 1 MAC = 2 FLOPs",
            "excludes": "elementwise ops, normalization, activation and softmax",
            "caution": (
                "theoretical graph FLOPs, not compiler-fused or hardware-executed ops"
            ),
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "onnx": str(path),
            "reason": "%s: %s" % (type(exc).__name__, exc),
            "method": "static ONNX Conv/MatMul/Gemm; 1 MAC = 2 FLOPs",
        }


def attach_baseline_reductions(specs: list[ModelSpec]) -> None:
    baselines = {
        spec.group: spec
        for spec in specs
        if spec.variant == "baseline"
        and spec.static_analysis.get("major_flops_per_image")
    }
    for spec in specs:
        baseline = baselines.get(spec.group)
        current_flops = spec.static_analysis.get("major_flops_per_image")
        current_params = spec.static_analysis.get("parameter_count")
        if baseline is None or not current_flops:
            continue
        baseline_flops = baseline.static_analysis["major_flops_per_image"]
        baseline_params = baseline.static_analysis.get("parameter_count")
        spec.static_analysis["baseline_model"] = baseline.name
        spec.static_analysis["major_flops_reduction_pct"] = (
            100.0 * (baseline_flops - current_flops) / baseline_flops
        )
        if baseline_params and current_params:
            spec.static_analysis["parameter_reduction_pct"] = (
                100.0 * (baseline_params - current_params) / baseline_params
            )


def collect_samples(
    val_dir: Path,
    images_per_class: int,
    num_classes: int,
    seed: int,
    limit: int,
) -> list[tuple[Path, int]]:
    if not val_dir.is_dir():
        raise NotADirectoryError(val_dir)
    class_dirs = sorted(
        path for path in val_dir.iterdir() if path.is_dir() and path.name.isdigit()
    )
    if len(class_dirs) < num_classes:
        raise ValueError(
            "requested %d classes, but only %d numeric class directories exist"
            % (num_classes, len(class_dirs))
        )
    if len(class_dirs) > num_classes:
        class_dirs = sorted(random.Random(seed).sample(class_dirs, num_classes))

    samples = []
    for class_dir in class_dirs:
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if len(images) < images_per_class:
            raise ValueError(
                "%s has %d images; %d required"
                % (class_dir, len(images), images_per_class)
            )
        class_rng = random.Random(seed + int(class_dir.name))
        selected = class_rng.sample(images, images_per_class)
        samples.extend((path, int(class_dir.name)) for path in selected)
        if limit and len(samples) >= limit:
            return samples[:limit]
    return samples


def sample_manifest_hash(samples: Iterable[tuple[Path, int]], val_dir: Path) -> str:
    digest = hashlib.sha256()
    for path, label in samples:
        try:
            display_path = path.relative_to(val_dir)
        except ValueError:
            display_path = path
        digest.update(("%s\t%d\n" % (display_path.as_posix(), label)).encode())
    return digest.hexdigest()


def write_sample_manifest(
    path: Path, samples: Iterable[tuple[Path, int]], val_dir: Path
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for image_path, label in samples:
            try:
                image_path = image_path.relative_to(val_dir)
            except ValueError:
                pass
            handle.write("%s\t%d\n" % (image_path.as_posix(), label))


def preprocess(path: Path, mode: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot read image: %s" % path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resize_short = 256 if mode == "legacy" else 249
    interpolation = cv2.INTER_LINEAR if mode == "legacy" else cv2.INTER_CUBIC
    height, width = image.shape[:2]
    scale = float(resize_short) / min(height, width)
    image = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=interpolation,
    )
    height, width = image.shape[:2]
    top, left = (height - 224) // 2, (width - 224) // 2
    image = image[top : top + 224, left : left + 224]
    image = image.astype(np.float32) / 255.0
    if mode == "legacy":
        image = (image - LEGACY_MEAN) / LEGACY_STD
    else:
        image = (image - DEIT_MEAN) / DEIT_STD
    return np.ascontiguousarray(image, dtype=np.float32)


def build_config(core_mode: str) -> qbruntime.ModelConfig:
    config = qbruntime.ModelConfig()
    cluster, core = qbruntime.Cluster, qbruntime.Core
    if core_mode == "single":
        config.set_single_core_mode(
            core_ids=[qbruntime.CoreId(cluster.Cluster0, core.Core0)]
        )
    elif core_mode == "multi":
        config.set_multi_core_mode(clusters=[cluster.Cluster0, cluster.Cluster1])
    elif core_mode == "global":
        config.set_global_core_mode(clusters=[cluster.Cluster0, cluster.Cluster1])
    elif core_mode == "global4":
        config.set_global4_core_mode(clusters=[cluster.Cluster0, cluster.Cluster1])
    elif core_mode == "global8":
        config.set_global8_core_mode()
    else:
        raise ValueError("unsupported core mode: %s" % core_mode)
    return config


class ProcessMemoryMonitor:
    """Track peaks without retaining an ever-growing time-series in RAM."""

    def __init__(self, interval_ms: float):
        self.interval = interval_ms / 1000.0
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.sample_count = 0
        self.peaks: dict[str, int] = {}
        self.stages: list[dict[str, Any]] = []
        self.error: str | None = None
        self.thread: threading.Thread | None = None

    def _sample(self, stage: str | None = None) -> dict[str, Any]:
        info = self.process.memory_info()
        full = self.process.memory_full_info()
        system = psutil.virtual_memory()
        sample = {
            "stage": stage,
            "timestamp": time.time(),
            "rss_bytes": int(info.rss),
            "vms_bytes": int(info.vms),
            "uss_bytes": int(getattr(full, "uss", 0)),
            "pss_bytes": int(getattr(full, "pss", 0)),
            "system_used_bytes": int(system.used),
            "system_available_bytes": int(system.available),
        }
        with self.lock:
            self.sample_count += 1
            for key in (
                "rss_bytes",
                "vms_bytes",
                "uss_bytes",
                "pss_bytes",
                "system_used_bytes",
            ):
                self.peaks[key] = max(self.peaks.get(key, 0), sample[key])
            current_min = self.peaks.get("system_available_min_bytes")
            self.peaks["system_available_min_bytes"] = (
                sample["system_available_bytes"]
                if current_min is None
                else min(current_min, sample["system_available_bytes"])
            )
            if stage is not None:
                self.stages.append(sample)
        return sample

    def start(self) -> None:
        self._sample("before_model_construct")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                self._sample()
            except Exception as exc:
                self.error = "%s: %s" % (type(exc).__name__, exc)
                return

    def mark(self, stage: str) -> None:
        self._sample(stage)

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self._sample("monitor_stopped")
        start = self.stages[0]
        return {
            "status": "ok" if self.error is None else "partial",
            "error": self.error,
            "sample_interval_ms": self.interval * 1000.0,
            "sample_count": self.sample_count,
            "rss_start_mb": start["rss_bytes"] / 2**20,
            "rss_peak_mb": self.peaks.get("rss_bytes", 0) / 2**20,
            "rss_peak_delta_mb": (
                self.peaks.get("rss_bytes", 0) - start["rss_bytes"]
            )
            / 2**20,
            "uss_peak_mb": self.peaks.get("uss_bytes", 0) / 2**20,
            "pss_peak_mb": self.peaks.get("pss_bytes", 0) / 2**20,
            "vms_peak_mb": self.peaks.get("vms_bytes", 0) / 2**20,
            "system_used_peak_mb": self.peaks.get("system_used_bytes", 0) / 2**20,
            "system_available_min_mb": self.peaks.get(
                "system_available_min_bytes", 0
            )
            / 2**20,
            "stage_snapshots": self.stages,
        }


def load_npu_tracker(skip: bool) -> tuple[Any | None, str | None]:
    if skip:
        return None, "disabled by --skip-npu-tracker"
    try:
        from mblt_tracker import NPUDeviceTracker

        return NPUDeviceTracker, None
    except Exception as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)


def start_npu_tracker(
    tracker_class: Any | None,
    unavailable_reason: str | None,
    interval_ms: float,
    device_id: int,
) -> tuple[Any | None, dict[str, Any]]:
    if tracker_class is None:
        return None, {"status": "unavailable", "reason": unavailable_reason}
    try:
        tracker = tracker_class(interval=interval_ms / 1000.0, npu_id=device_id)
        tracker.start()
        return tracker, {"status": "running"}
    except Exception as exc:
        return None, {
            "status": "error",
            "reason": "%s: %s" % (type(exc).__name__, exc),
        }


def stop_npu_tracker(
    tracker: Any | None, state: dict[str, Any]
) -> dict[str, Any]:
    if tracker is None:
        return state
    try:
        tracker.stop()
        metric = json_safe(tracker.get_metric())
        samples = metric.get("samples", 0) if isinstance(metric, dict) else 0
        return {
            "status": "ok" if samples else "no_samples",
            "sample_count": samples,
            "metrics": metric,
        }
    except Exception as exc:
        return {
            "status": "error",
            "reason": "%s: %s" % (type(exc).__name__, exc),
        }


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            key: None
            for key in ("mean", "std", "min", "p50", "p90", "p95", "p99", "max")
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def evaluate_model(
    spec: ModelSpec,
    accelerator: qbruntime.Accelerator,
    samples: list[tuple[Path, int]],
    args: argparse.Namespace,
    tracker_class: Any | None,
    tracker_reason: str | None,
) -> tuple[dict[str, Any], np.ndarray]:
    monitor = ProcessMemoryMonitor(args.memory_sample_ms)
    monitor.start()
    model = None
    tracker = None
    tracker_state: dict[str, Any] = {"status": "not_started"}
    disposed = False
    npu_metrics: dict[str, Any] = tracker_state
    top1 = top5 = successful = 0
    predictions = np.full(len(samples), -1, dtype=np.int32)
    latencies_ms: list[float] = []
    preprocess_ms: list[float] = []
    errors: list[dict[str, Any]] = []
    evaluation_started = None

    try:
        model = qbruntime.Model(str(spec.path), build_config(args.core_mode))
        monitor.mark("after_model_construct")
        model.launch(accelerator)
        monitor.mark("after_model_launch")

        warmup_tensor = preprocess(samples[0][0], args.preprocess)
        for _ in range(args.warmup_runs):
            model.infer([warmup_tensor])
        monitor.mark("after_warmup")

        tracker, tracker_state = start_npu_tracker(
            tracker_class,
            tracker_reason,
            args.npu_sample_ms,
            args.device_id,
        )
        evaluation_started = time.perf_counter()
        for index, (image_path, label) in enumerate(samples, 1):
            try:
                started = time.perf_counter()
                tensor = preprocess(image_path, args.preprocess)
                preprocess_elapsed_ms = (time.perf_counter() - started) * 1000.0

                started = time.perf_counter()
                raw = model.infer([tensor])[0]
                latency_ms = (time.perf_counter() - started) * 1000.0

                logits = np.asarray(raw).reshape(-1)
                if logits.size < 5:
                    raise ValueError("expected at least 5 logits, got %d" % logits.size)
                top5_indices = np.argpartition(logits, -5)[-5:]
                top5_indices = top5_indices[
                    np.argsort(logits[top5_indices])[::-1]
                ]
                prediction = int(top5_indices[0])
                predictions[index - 1] = prediction
                top1 += int(prediction == label)
                top5 += int(label in top5_indices)
                successful += 1
                preprocess_ms.append(preprocess_elapsed_ms)
                latencies_ms.append(latency_ms)
            except Exception as exc:
                if len(errors) < 100:
                    errors.append(
                        {
                            "index": index - 1,
                            "image": str(image_path),
                            "label": label,
                            "error": "%s: %s" % (type(exc).__name__, exc),
                        }
                    )

            if args.progress_every and (
                index % args.progress_every == 0 or index == len(samples)
            ):
                accuracy = 100.0 * top1 / successful if successful else 0.0
                mean_ms = statistics.fmean(latencies_ms) if latencies_ms else 0.0
                print(
                    "[%s] %d/%d top1=%.3f%% avg=%.3f ms"
                    % (spec.name, index, len(samples), accuracy, mean_ms),
                    flush=True,
                )

        wall_seconds = time.perf_counter() - evaluation_started
        monitor.mark("after_evaluation")
        npu_metrics = stop_npu_tracker(tracker, tracker_state)
        tracker = None

        if not successful:
            raise RuntimeError("all inferences failed")

        result = {
            "status": "ok" if successful == len(samples) else "incomplete",
            "name": spec.name,
            "group": spec.group,
            "family": spec.family,
            "target": spec.target,
            "variant": spec.variant,
            "mxq": str(spec.path),
            "mxq_file_bytes": spec.path.stat().st_size,
            "core_mode": args.core_mode,
            "device_id": args.device_id,
            "num_requested": len(samples),
            "num_successful": successful,
            "num_errors": len(samples) - successful,
            "error_details_first_100": errors,
            "top1_acc_pct": 100.0 * top1 / successful,
            "top5_acc_pct": 100.0 * top5 / successful,
            "latency_ms": distribution(latencies_ms),
            "preprocess_ms": distribution(preprocess_ms),
            "inference_seconds": sum(latencies_ms) / 1000.0,
            "inference_fps": successful * 1000.0 / sum(latencies_ms),
            "evaluation_wall_seconds": wall_seconds,
            "end_to_end_fps": successful / wall_seconds,
            "warmup_runs": args.warmup_runs,
            "npu_telemetry": npu_metrics,
            "static_analysis": spec.static_analysis,
        }
        per_image_flops = spec.static_analysis.get("major_flops_per_image")
        if per_image_flops:
            result["static_analysis"]["major_flops_for_successful_dataset"] = (
                int(per_image_flops) * successful
            )
        return result, predictions
    finally:
        if tracker is not None:
            npu_metrics = stop_npu_tracker(tracker, tracker_state)
        if model is not None:
            try:
                model.dispose()
                disposed = True
            except Exception:
                pass
        try:
            monitor.mark("after_model_dispose" if disposed else "cleanup")
        finally:
            memory = monitor.stop()
        # The returned result is updated before the caller receives it.
        if "result" in locals():
            result["process_memory"] = memory
            result["npu_telemetry"] = npu_metrics


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def metric_value(result: dict[str, Any], *keys: str) -> Any:
    value: Any = result
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def npu_metric(result: dict[str, Any], key: str) -> Any:
    return metric_value(result, "npu_telemetry", "metrics", key)


SUMMARY_FIELDS = [
    "group",
    "variant",
    "status",
    "images",
    "errors",
    "top1_pct",
    "top5_pct",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "inference_fps",
    "end_to_end_fps",
    "preprocess_mean_ms",
    "process_rss_start_mb",
    "process_rss_peak_mb",
    "process_rss_delta_mb",
    "process_uss_peak_mb",
    "process_pss_peak_mb",
    "npu_memory_avg_mb",
    "npu_memory_p99_mb",
    "npu_memory_peak_mb",
    "npu_util_avg_pct",
    "npu_util_p99_pct",
    "npu_util_max_pct",
    "power_avg_w",
    "power_p99_w",
    "power_max_w",
    "npu_core_power_avg_w",
    "npu_core_power_max_w",
    "temperature_avg_c",
    "temperature_p99_c",
    "temperature_max_c",
    "parameters_m",
    "parameter_reduction_pct",
    "major_gflops_per_image",
    "flops_reduction_pct",
    "mxq_size_mb",
]


def summary_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "group": result.get("group"),
        "variant": result.get("variant"),
        "status": result.get("status"),
        "images": result.get("num_successful"),
        "errors": result.get("num_errors"),
        "top1_pct": result.get("top1_acc_pct"),
        "top5_pct": result.get("top5_acc_pct"),
        "latency_mean_ms": metric_value(result, "latency_ms", "mean"),
        "latency_p50_ms": metric_value(result, "latency_ms", "p50"),
        "latency_p95_ms": metric_value(result, "latency_ms", "p95"),
        "latency_p99_ms": metric_value(result, "latency_ms", "p99"),
        "inference_fps": result.get("inference_fps"),
        "end_to_end_fps": result.get("end_to_end_fps"),
        "preprocess_mean_ms": metric_value(result, "preprocess_ms", "mean"),
        "process_rss_start_mb": metric_value(
            result, "process_memory", "rss_start_mb"
        ),
        "process_rss_peak_mb": metric_value(result, "process_memory", "rss_peak_mb"),
        "process_rss_delta_mb": metric_value(
            result, "process_memory", "rss_peak_delta_mb"
        ),
        "process_uss_peak_mb": metric_value(
            result, "process_memory", "uss_peak_mb"
        ),
        "process_pss_peak_mb": metric_value(
            result, "process_memory", "pss_peak_mb"
        ),
        "npu_memory_avg_mb": npu_metric(result, "avg_memory_used_mb"),
        "npu_memory_p99_mb": npu_metric(result, "p99_memory_used_mb"),
        "npu_memory_peak_mb": npu_metric(result, "max_memory_used_mb"),
        "npu_util_avg_pct": npu_metric(result, "avg_utilization_pct"),
        "npu_util_p99_pct": npu_metric(result, "p99_utilization_pct"),
        "npu_util_max_pct": npu_metric(result, "max_utilization_pct"),
        "power_avg_w": npu_metric(result, "avg_power_w"),
        "power_p99_w": npu_metric(result, "p99_power_w"),
        "power_max_w": npu_metric(result, "max_power_w"),
        "npu_core_power_avg_w": npu_metric(result, "avg_npu_power_w"),
        "npu_core_power_max_w": npu_metric(result, "max_npu_power_w"),
        "temperature_avg_c": npu_metric(result, "avg_temperature_c"),
        "temperature_p99_c": npu_metric(result, "p99_temperature_c"),
        "temperature_max_c": npu_metric(result, "max_temperature_c"),
        "parameters_m": (
            metric_value(result, "static_analysis", "parameter_count") / 1e6
            if metric_value(result, "static_analysis", "parameter_count")
            else None
        ),
        "parameter_reduction_pct": metric_value(
            result, "static_analysis", "parameter_reduction_pct"
        ),
        "major_gflops_per_image": metric_value(
            result, "static_analysis", "major_gflops_per_image"
        ),
        "flops_reduction_pct": metric_value(
            result, "static_analysis", "major_flops_reduction_pct"
        ),
        "mxq_size_mb": (
            result["mxq_file_bytes"] / 2**20 if result.get("mxq_file_bytes") else None
        ),
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return ("%%.%df" % digits) % value
    return str(value)


def markdown_text(value: Any) -> str:
    return " ".join(str(value).splitlines()).replace("|", "\\|")


def compute_comparisons(
    results: list[dict[str, Any]], predictions: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    comparisons = []
    by_group: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("status") in ("ok", "incomplete"):
            by_group.setdefault(result["group"], []).append(result)
    for group, group_results in by_group.items():
        baseline = next(
            (result for result in group_results if result.get("variant") == "baseline"),
            None,
        )
        if baseline is None or baseline["name"] not in predictions:
            continue
        left = predictions[baseline["name"]]
        for result in group_results:
            if result is baseline or result["name"] not in predictions:
                continue
            right = predictions[result["name"]]
            valid = (left >= 0) & (right >= 0)
            agreement = float(np.mean(left[valid] == right[valid])) if np.any(valid) else None
            comparisons.append(
                {
                    "group": group,
                    "baseline": baseline["name"],
                    "model": result["name"],
                    "common_successful": int(np.sum(valid)),
                    "top1_agreement_pct": 100.0 * agreement if agreement is not None else None,
                    "top1_delta_pct_point": (
                        result["top1_acc_pct"] - baseline["top1_acc_pct"]
                    ),
                }
            )
    return comparisons


def write_summary_csv(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(summary_row(result))


def percent(value: Any, digits: int = 1) -> str:
    return fmt(value, digits) + "%" if value is not None else "-"


def multiple(value: Any, digits: int = 2) -> str:
    return fmt(value, digits) + "x" if value is not None else "-"


def report_rows(
    results: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    baselines = {
        result["group"]: summary_row(result)
        for result in results
        if result.get("variant") == "baseline"
    }
    comparison_by_model = {item["model"]: item for item in comparisons}
    rows = []
    for result in results:
        row = summary_row(result)
        baseline = baselines.get(result.get("group"))
        comparison = comparison_by_model.get(result.get("name"), {})
        row["model"] = VARIANT_LABELS.get(
            result.get("variant"), result.get("variant", "unknown")
        )
        row["top1_delta_pct_point"] = (
            0.0
            if result.get("variant") == "baseline"
            else comparison.get("top1_delta_pct_point")
        )
        row["top1_agreement_pct"] = (
            100.0
            if result.get("variant") == "baseline"
            else comparison.get("top1_agreement_pct")
        )
        row["latency_speedup_x"] = None
        row["mxq_reduction_pct"] = None
        row["npu_memory_reduction_pct"] = None
        if baseline:
            if baseline.get("latency_mean_ms") and row.get("latency_mean_ms"):
                row["latency_speedup_x"] = (
                    baseline["latency_mean_ms"] / row["latency_mean_ms"]
                )
            if baseline.get("mxq_size_mb") and row.get("mxq_size_mb"):
                row["mxq_reduction_pct"] = 100.0 * (
                    baseline["mxq_size_mb"] - row["mxq_size_mb"]
                ) / baseline["mxq_size_mb"]
            if baseline.get("npu_memory_peak_mb") and row.get("npu_memory_peak_mb"):
                row["npu_memory_reduction_pct"] = 100.0 * (
                    baseline["npu_memory_peak_mb"] - row["npu_memory_peak_mb"]
                ) / baseline["npu_memory_peak_mb"]
        rows.append(row)
    return rows


def write_summary_markdown(
    path: Path,
    run: dict[str, Any],
    results: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    rows = report_rows(results, comparisons)
    completed = sum(row.get("status") == "ok" for row in rows)
    total_errors = sum(row.get("errors") or 0 for row in rows)
    telemetry_ok = sum(
        result.get("npu_telemetry", {}).get("status") == "ok"
        for result in results
    )
    passed = (
        run.get("status") == "complete"
        and completed == len(results)
        and total_errors == 0
    )
    wall_hours = sum(result.get("evaluation_wall_seconds", 0) for result in results) / 3600.0
    environment = run.get("environment", {})
    commit = environment.get("git_commit")
    timestamp = environment.get("timestamp", "-")

    lines = [
        "# 최종 NPU 벤치마크 결과",
        "",
        "## 평가 개요",
        "",
        "| 항목 | 결과 |",
        "|---|---|",
        "| 최종 상태 | **%s** (`%s`) |"
        % ("PASS" if passed else "CHECK", run.get("status", "unknown")),
        "| 평가 모델 | %d개 중 %d개 정상, 오류 이미지 %d장 |"
        % (len(results), completed, total_errors),
        "| 데이터셋 | ImageNet validation %s장/모델 (`%s`) |"
        % (run.get("num_samples"), run.get("val_dir")),
        "| 총 추론 수 | %s회 |" % (run.get("num_samples", 0) * len(results)),
        "| 전처리 / 코어 | `%s` / `%s`, device %s |"
        % (run.get("preprocess"), run.get("core_mode"), run.get("device_id")),
        "| Warmup | 모델당 %s회 |" % run.get("warmup_runs"),
        "| NPU telemetry | %d/%d 모델 정상 수집 |" % (telemetry_ok, len(results)),
        "| 평가 시간 합계 | %.2f시간 |" % wall_hours,
        "| 실행 시각 / Git | `%s` / `%s` |"
        % (timestamp, commit[:12] if commit else "-"),
        "| 샘플 SHA-256 | `%s` |" % run.get("sample_manifest_sha256", "-"),
        "",
        "## 핵심 비교",
        "",
        "압축 모델 중 정확도가 가장 높은 모델, 가장 빠른 모델, NPU 메모리가 가장 적은 모델을 그룹별로 요약했다.",
        "",
        "| Group | Baseline Top-1 | 최고 압축 정확도 | 가장 빠른 압축 모델 | 최저 NPU 메모리 |",
        "|---|---:|---|---|---|",
    ]
    groups = []
    for row in rows:
        if row["group"] not in groups:
            groups.append(row["group"])
    for group in groups:
        group_rows = [row for row in rows if row["group"] == group]
        baseline = next((row for row in group_rows if row["variant"] == "baseline"), None)
        compressed = [
            row
            for row in group_rows
            if row["variant"] != "baseline" and row["status"] == "ok"
        ]
        if not baseline or not compressed:
            continue
        best_accuracy = max(compressed, key=lambda row: row["top1_pct"] or -math.inf)
        fastest = min(
            compressed, key=lambda row: row["latency_mean_ms"] or math.inf
        )
        lowest_memory = min(
            compressed, key=lambda row: row["npu_memory_peak_mb"] or math.inf
        )
        lines.append(
            "| %s | %s%% | %s: %s%% (%s pp) | %s: %s ms (%s) | %s: %s/%s MB (%s) |"
            % (
                group,
                fmt(baseline["top1_pct"], 3),
                best_accuracy["model"],
                fmt(best_accuracy["top1_pct"], 3),
                fmt(best_accuracy["top1_delta_pct_point"], 3),
                fastest["model"],
                fmt(fastest["latency_mean_ms"], 3),
                multiple(fastest["latency_speedup_x"], 2),
                lowest_memory["model"],
                fmt(lowest_memory["npu_memory_avg_mb"], 1),
                fmt(lowest_memory["npu_memory_peak_mb"], 1),
                percent(lowest_memory["npu_memory_reduction_pct"], 1),
            )
        )

    lines.extend(
        [
            "",
            "## 정확도 및 모델 압축",
            "",
            "| Group | Model | 상태 | Top-1 | Δ Top-1 | Top-5 | 예측 일치율 | Params M | Params 감소 | GFLOPs* | FLOPs 감소 | MXQ MB | MXQ 감소 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| %s | %s | %s | %s%% | %s | %s%% | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["group"], row["model"], row["status"],
                fmt(row["top1_pct"], 3),
                (fmt(row["top1_delta_pct_point"], 3) + " pp")
                if row["top1_delta_pct_point"] is not None else "-",
                fmt(row["top5_pct"], 3),
                percent(row["top1_agreement_pct"], 3),
                fmt(row["parameters_m"], 3),
                percent(row["parameter_reduction_pct"], 1),
                fmt(row["major_gflops_per_image"], 3),
                percent(row["flops_reduction_pct"], 1),
                fmt(row["mxq_size_mb"], 1),
                percent(row["mxq_reduction_pct"], 1),
            )
        )

    lines.extend(
        [
            "",
            "## 추론 성능",
            "",
            "`Inference`는 `model.infer()`만, `E2E`는 이미지 로딩과 전처리를 포함한다.",
            "",
            "| Group | Model | 평균 ms | P50 | P95 | P99 | Inference FPS | Baseline 대비 | E2E FPS | 전처리 평균 ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["group"], row["model"],
                fmt(row["latency_mean_ms"], 3),
                fmt(row["latency_p50_ms"], 3),
                fmt(row["latency_p95_ms"], 3),
                fmt(row["latency_p99_ms"], 3),
                fmt(row["inference_fps"], 2),
                multiple(row["latency_speedup_x"], 2),
                fmt(row["end_to_end_fps"], 2),
                fmt(row["preprocess_mean_ms"], 3),
            )
        )

    lines.extend(
        [
            "",
            "## NPU 메모리 및 사용률",
            "",
            "| Group | Model | 메모리 평균 MB | P99 MB | 피크 MB | 피크 감소 | 사용률 평균 | P99 | 최대 | Samples |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result, row in zip(results, rows):
        samples = metric_value(result, "npu_telemetry", "sample_count")
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["group"], row["model"],
                fmt(row["npu_memory_avg_mb"], 1),
                fmt(row["npu_memory_p99_mb"], 1),
                fmt(row["npu_memory_peak_mb"], 1),
                percent(row["npu_memory_reduction_pct"], 1),
                percent(row["npu_util_avg_pct"], 1),
                percent(row["npu_util_p99_pct"], 1),
                percent(row["npu_util_max_pct"], 1),
                fmt(samples, 0),
            )
        )

    lines.extend(
        [
            "",
            "## NPU 전력 및 온도",
            "",
            "Total power는 보드 전체, NPU core power는 NPU 코어 전력이다.",
            "",
            "| Group | Model | Total W 평균 | P99 | 최대 | NPU core W 평균 | 최대 | 온도 °C 평균 | P99 | 최대 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["group"], row["model"],
                fmt(row["power_avg_w"], 2),
                fmt(row["power_p99_w"], 2),
                fmt(row["power_max_w"], 2),
                fmt(row["npu_core_power_avg_w"], 2),
                fmt(row["npu_core_power_max_w"], 2),
                fmt(row["temperature_avg_c"], 1),
                fmt(row["temperature_p99_c"], 1),
                fmt(row["temperature_max_c"], 1),
            )
        )

    lines.extend(
        [
            "",
            "## 호스트 프로세스 메모리",
            "",
            "| Group | Model | RSS 시작 MB | RSS 피크 MB | RSS 증가 MB | USS 피크 MB | PSS 피크 MB |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["group"], row["model"],
                fmt(row["process_rss_start_mb"], 1),
                fmt(row["process_rss_peak_mb"], 1),
                fmt(row["process_rss_delta_mb"], 1),
                fmt(row["process_uss_peak_mb"], 1),
                fmt(row["process_pss_peak_mb"], 1),
            )
        )

    failures = [result for result in results if result.get("status") != "ok"]
    telemetry_failures = [
        result for result in results
        if result.get("npu_telemetry", {}).get("status") != "ok"
    ]
    if failures or telemetry_failures or run.get("preflight_error"):
        lines.extend(["", "## 검증 이슈", ""])
        if run.get("preflight_error"):
            lines.append(
                "- NPU preflight 실패: `%s`"
                % markdown_text(run["preflight_error"])
            )
        if failures:
            lines.append(
                "- 정상 완료되지 않은 모델: %d개 (`benchmark.json`의 `error` 확인)"
                % len(failures)
            )
        if telemetry_failures:
            reasons = sorted(
                {
                    result.get("npu_telemetry", {}).get("reason")
                    or "metric samples unavailable"
                    for result in telemetry_failures
                }
            )
            lines.append(
                "- NPU telemetry 미수집: %d개 모델 — `%s`"
                % (len(telemetry_failures), markdown_text("; ".join(reasons)))
            )

    lines.extend(
        [
            "",
            "## 측정 기준",
            "",
            "- `GFLOPs*`는 ONNX Conv/MatMul/Gemm 정적 shape로 계산했으며 `1 MAC = 2 FLOPs`를 적용했다. Elementwise, normalization, activation, softmax와 컴파일러 fusion/tiling은 포함하지 않는다.",
            "- NPU 메모리·전력·사용률·온도는 `mblt_tracker`가 `mobilint-cli status`를 주기적으로 관측한 값이다. 메모리 피크는 관측 구간의 최대 할당량이며 연산 중 초단기 내부 activation peak와는 다르다.",
            "- 정확도 비교는 모든 모델에 동일한 이미지 목록, label, `timm` 전처리, Global8 조건을 사용했다.",
            "- 상세 원본 지표와 오류 정보는 같은 폴더의 `benchmark.json`, 분석용 평탄화 지표는 `summary.csv`에 저장된다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(results: list[dict[str, Any]]) -> None:
    columns = [
        ("group", 8),
        ("variant", 32),
        ("status", 10),
        ("top1_pct", 9),
        ("top5_pct", 9),
        ("latency_mean_ms", 10),
        ("latency_p95_ms", 10),
        ("inference_fps", 9),
        ("major_gflops_per_image", 9),
    ]
    labels = {
        "group": "group",
        "variant": "variant",
        "status": "status",
        "top1_pct": "top1%",
        "top5_pct": "top5%",
        "latency_mean_ms": "avg_ms",
        "latency_p95_ms": "p95_ms",
        "inference_fps": "fps",
        "major_gflops_per_image": "GFLOPs*",
    }
    print("\nFinal summary")
    print(" ".join(labels[key].ljust(width) for key, width in columns))
    print(" ".join("-" * width for _, width in columns))
    for result in results:
        row = summary_row(result)
        print(
            " ".join(
                fmt(row.get(key), 3).ljust(width)[:width] for key, width in columns
            )
        )


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=NPU_ROOT, text=True
        ).strip()
    except Exception:
        return None


def environment_metadata(tracker_error: str | None) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "qbruntime": getattr(qbruntime, "__version__", None),
        "mblt_tracker_import_error": tracker_error,
        "command": [sys.executable] + sys.argv,
    }


def save_outputs(
    output_dir: Path,
    run: dict[str, Any],
    results: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    final: bool,
) -> None:
    comparisons = compute_comparisons(results, predictions)
    if final:
        if len(results) != run["num_models"]:
            run["status"] = "stopped_with_failures"
        elif any(result.get("status") != "ok" for result in results):
            run["status"] = "completed_with_failures"
        else:
            run["status"] = "complete"
    elif run.get("status") != "interrupted":
        run["status"] = "running"
    payload = {"run": run, "models": results, "comparisons": comparisons}
    atomic_write_json(output_dir / "benchmark_partial.json", payload)
    write_summary_csv(output_dir / "summary.csv", results)
    write_summary_markdown(output_dir / "summary.md", run, results, comparisons)
    if final:
        atomic_write_json(output_dir / "benchmark.json", payload)


def failed_result(spec: ModelSpec, exc: BaseException) -> dict[str, Any]:
    return {
        "status": "failed",
        "name": spec.name,
        "group": spec.group,
        "family": spec.family,
        "target": spec.target,
        "variant": spec.variant,
        "mxq": str(spec.path),
        "mxq_file_bytes": spec.path.stat().st_size,
        "num_successful": 0,
        "num_errors": None,
        "error": "%s: %s" % (type(exc).__name__, exc),
        "traceback": traceback.format_exc(),
        "static_analysis": spec.static_analysis,
    }


def dry_run(specs: list[ModelSpec], samples: list[tuple[Path, int]], manifest: str) -> None:
    print("Dataset samples: %d" % len(samples))
    print("Manifest SHA-256: %s" % manifest)
    print("Models: %d" % len(specs))
    for spec in specs:
        analysis = spec.static_analysis
        print(
            "  %-8s %-34s MXQ=%6.1f MiB ONNX=%s GFLOPs*=%s"
            % (
                spec.group,
                spec.variant,
                spec.path.stat().st_size / 2**20,
                "ok" if spec.onnx_path else "missing",
                fmt(analysis.get("major_gflops_per_image"), 3),
            )
        )


def main() -> int:
    args = parse_args()
    specs = discover_models(args)
    val_dir = Path(args.val_dir).expanduser().resolve()
    samples = collect_samples(
        val_dir,
        args.images_per_class,
        args.classes,
        args.seed,
        args.limit,
    )
    manifest_hash = sample_manifest_hash(samples, val_dir)
    if args.dry_run:
        dry_run(specs, samples, manifest_hash)
        return 0

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else NPU_ROOT / "results" / "final_benchmark" / timestamp
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "%s is not empty; choose a new --output-dir" % output_dir
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir()
    (output_dir / "predictions").mkdir()
    write_sample_manifest(output_dir / "samples.tsv", samples, val_dir)

    tracker_class, tracker_error = load_npu_tracker(args.skip_npu_tracker)
    run = {
        "status": "running",
        "val_dir": str(val_dir),
        "num_samples": len(samples),
        "classes": args.classes,
        "images_per_class": args.images_per_class,
        "limit": args.limit,
        "seed": args.seed,
        "sample_manifest_sha256": manifest_hash,
        "preprocess": args.preprocess,
        "core_mode": args.core_mode,
        "device_id": args.device_id,
        "warmup_runs": args.warmup_runs,
        "num_models": len(specs),
        "models": [spec.name for spec in specs],
        "environment": environment_metadata(tracker_error),
    }
    atomic_write_json(output_dir / "run_metadata.json", run)

    results: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    try:
        accelerator = qbruntime.Accelerator(args.device_id)
    except Exception as exc:
        run["preflight_error"] = "%s: %s" % (type(exc).__name__, exc)
        results = [failed_result(spec, exc) for spec in specs]
        save_outputs(output_dir, run, results, predictions, final=True)
        print_summary(results)
        print("\nNPU preflight failed: %s" % run["preflight_error"])
        print("Summary: %s" % (output_dir / "summary.md"))
        print("JSON:    %s" % (output_dir / "benchmark.json"))
        return 1

    interrupted = False
    for index, spec in enumerate(specs, 1):
        print("\n[%d/%d] %s" % (index, len(specs), spec.name), flush=True)
        try:
            result, prediction = evaluate_model(
                spec,
                accelerator,
                samples,
                args,
                tracker_class,
                tracker_error,
            )
            prediction_path = output_dir / "predictions" / (spec.name + ".npy")
            np.save(prediction_path, prediction)
            result["predictions_npy"] = str(prediction_path)
            predictions[spec.name] = prediction
        except KeyboardInterrupt as exc:
            result = failed_result(spec, exc)
            interrupted = True
        except Exception as exc:
            result = failed_result(spec, exc)

        results.append(result)
        atomic_write_json(output_dir / "models" / (spec.name + ".json"), result)
        save_outputs(output_dir, run, results, predictions, final=False)
        gc.collect()
        if interrupted or (args.fail_fast and result["status"] == "failed"):
            break

    if interrupted:
        run["status"] = "interrupted"
    save_outputs(output_dir, run, results, predictions, final=not interrupted)
    print_summary(results)
    print("\nSummary: %s" % (output_dir / "summary.md"))
    print("CSV:     %s" % (output_dir / "summary.csv"))
    result_json = output_dir / (
        "benchmark_partial.json" if interrupted else "benchmark.json"
    )
    print("JSON:    %s" % result_json)
    if interrupted:
        print("Interrupted; completed model results remain in %s" % output_dir)
        return 130
    return 1 if any(result.get("status") != "ok" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
