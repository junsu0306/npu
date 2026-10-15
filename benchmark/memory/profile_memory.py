#!/usr/bin/env python3
"""Isolated final MXQ memory profiling, matching Server_Compression's input protocol."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("tiny30", "tiny50", "small30", "small50")
VARIANTS = ("baseline", "reduced", "patch_slimming")
IDS = tuple(f"{group}__{variant}" for group in GROUPS for variant in VARIANTS)


def load_input(path=None):
    array = (np.load(path, allow_pickle=False) if path else
             np.random.default_rng(2022).standard_normal((1, 224, 224, 3)).astype(np.float32))
    if array.dtype != np.float32 or array.shape != (1, 224, 224, 3) or not np.isfinite(array).all():
        raise ValueError("Input must be finite float32 NHWC [1,224,224,3]")
    return np.ascontiguousarray(array)


def discover(args):
    selected = args.model_id or IDS
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate model-id")
    jobs = []
    for model_id in selected:
        filename = f"{model_id}__nhwc_b1_npusafe_vt_{args.core_mode}.mxq"
        path = args.models_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        jobs.append((model_id, path.resolve()))
    return jobs


def runtime_helpers():
    # Keep SDK imports lazy so --help/--dry-run also work on the training server.
    sys.path.insert(0, str(ROOT / "benchmark/final"))
    import eval_final_models as helpers
    return helpers


def telemetry(helpers, tracker_class, args, action):
    tracker, state = helpers.start_npu_tracker(tracker_class, None, args.npu_sample_ms, args.device_id)
    if tracker is None:
        raise RuntimeError(f"NPU tracker could not start: {state}")
    try:
        action()
    finally:
        result = helpers.stop_npu_tracker(tracker, state)
    return result


def run_worker(args):
    helpers = runtime_helpers()
    tracker_class, error = helpers.load_npu_tracker(False)
    if tracker_class is None:
        raise RuntimeError(f"NPU memory profiling requires mblt_tracker: {error}")
    array = load_input(args.input_npy)
    tensor = np.ascontiguousarray(array[0])  # qbruntime's batch-1 input is HWC.
    accelerator = helpers.qbruntime.Accelerator(args.device_id)
    monitor = helpers.ProcessMemoryMonitor(args.memory_sample_ms)
    monitor.start()
    model = None
    result = None
    try:
        idle = telemetry(helpers, tracker_class, args,
                         lambda: time.sleep(max(1.0, args.npu_sample_ms * 3 / 1000)))
        model = helpers.qbruntime.Model(str(args.worker_model), helpers.build_config(args.core_mode))
        monitor.mark("after_model_construct")
        model.launch(accelerator)
        monitor.mark("after_model_launch")
        for _ in range(args.warmups):
            model.infer([tensor])
        monitor.mark("after_warmup")
        # Exactly --runs calls for latency; telemetry receives its own longer pass.
        started = time.perf_counter()
        for _ in range(args.runs):
            model.infer([tensor])
        mean_ms = (time.perf_counter() - started) * 1000 / args.runs
        telemetry_runs = 0
        def infer_for_telemetry():
            nonlocal telemetry_runs
            deadline = time.perf_counter() + max(args.memory_seconds, 3 * args.npu_sample_ms / 1000)
            while time.perf_counter() < deadline:
                output = model.infer([tensor])[0]
                telemetry_runs += 1
            logits = np.asarray(output).reshape(-1)
            if logits.size != 1000 or not np.isfinite(logits).all():
                raise ValueError("Expected 1000 finite classification logits")
        active = telemetry(helpers, tracker_class, args, infer_for_telemetry)
        monitor.mark("after_measurement")
        peak = active.get("metrics", {}).get("max_memory_used_mb")
        idle_peak = idle.get("metrics", {}).get("max_memory_used_mb")
        digest = hashlib.sha256()
        with args.worker_model.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result = {
            "status": "ok" if active.get("status") == "ok" and peak is not None and peak > 0 else "partial",
            "mxq": str(args.worker_model), "mxq_sha256": digest.hexdigest(),
            "mxq_file_bytes": args.worker_model.stat().st_size,
            "device_id": args.device_id, "core_mode": args.core_mode,
            "qbruntime_version": getattr(helpers.qbruntime, "__version__", "unknown"),
            "input_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "latency_mean_ms": mean_ms, "timed_runs": args.runs,
            "telemetry_runs": telemetry_runs, "memory_seconds_requested": args.memory_seconds,
            "npu_idle_peak_reported_mb": idle_peak,
            "npu_peak_reported_mb": peak,
            "npu_delta_reported_mb": peak - idle_peak if peak is not None and idle_peak is not None else None,
            "telemetry_unit": "MB as reported by mblt_tracker/mobilint-cli; not converted to MiB",
            "idle_telemetry": idle, "active_telemetry": active,
            "npu_activation_bytes": None,
            "note": "NPU telemetry is device-wide sampled allocation, not per-layer activation peak",
        }
    finally:
        try:
            if model is not None:
                model.dispose()
                monitor.mark("after_dispose")
        finally:
            host = monitor.stop()
    result["process_memory"] = host
    result["host_rss_peak_mib"] = host["rss_peak_mb"]  # helper divides bytes by 2**20.
    result["host_uss_peak_mib"] = host["uss_peak_mb"]
    args.worker_output.write_text(json.dumps(helpers.json_safe(result), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir", type=Path, default=ROOT / "assets/mxq_final_20261006"
    )
    parser.add_argument("--model-id", choices=IDS, action="append")
    parser.add_argument("--core-mode", choices=["single", "multi", "global", "global4", "global8"], default="global8")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--input-npy", type=Path)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--memory-seconds", type=float, default=5.0)
    parser.add_argument("--memory-sample-ms", type=float, default=20.0)
    parser.add_argument("--npu-sample-ms", type=float, default=500.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmups < 0 or args.runs < 1 or args.device_id < 0:
        parser.error("invalid warmups/runs/device-id")
    if not all(np.isfinite(v) and v > 0 for v in (args.memory_seconds, args.memory_sample_ms, args.npu_sample_ms)):
        parser.error("measurement durations must be finite and positive")
    if args.worker_model:
        if args.worker_output is None or args.input_npy is None:
            parser.error("worker requires output and input")
        run_worker(args)
        return
    jobs = discover(args)
    array = load_input(args.input_npy)
    if args.dry_run:
        for model_id, path in jobs:
            print(f"{model_id}: {path}")
        print(f"{len(jobs)} models; input SHA256={hashlib.sha256(array.tobytes()).hexdigest()}")
        return
    output_dir = args.output_dir or ROOT / "results/memory_profile" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("output-dir must be new/empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir.resolve() / "input_nhwc_fp32.npy"
    np.save(input_path, array)
    protocol = {"batch_size": 1, "input_shape_nhwc": [1, 224, 224, 3], "host_dtype": "float32",
                "seed": 2022, "warmups": args.warmups, "runs": args.runs,
                "input_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    payload = {"schema_version": 1, "backend": "mobilint_npu", "status": "running",
               "created_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol, "models": []}
    for model_id, path in jobs:
        print(model_id, flush=True)
        output = output_dir.resolve() / f"{model_id}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-model", str(path),
                   "--worker-output", str(output), "--input-npy", str(input_path),
                   "--device-id", str(args.device_id), "--core-mode", args.core_mode,
                   "--warmups", str(args.warmups), "--runs", str(args.runs),
                   "--memory-seconds", str(args.memory_seconds),
                   "--memory-sample-ms", str(args.memory_sample_ms), "--npu-sample-ms", str(args.npu_sample_ms)]
        with output.with_suffix(".log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        group, variant = model_id.split("__", 1)
        row = {"model_id": model_id, "group": group, "variant": variant, "details_file": output.name}
        if completed.returncode:
            row.update(status="error", error=f"See {output.with_suffix('.log')}")
        else:
            raw = json.loads(output.read_text())
            output.with_suffix(".log").unlink(missing_ok=True)
            for key in ("status", "npu_peak_reported_mb", "npu_idle_peak_reported_mb", "npu_delta_reported_mb",
                        "host_rss_peak_mib", "host_uss_peak_mib", "latency_mean_ms", "telemetry_runs"):
                row[key] = raw.get(key)
        payload["models"].append(row)
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    payload["status"] = "complete" if all(row["status"] == "ok" for row in payload["models"]) else "incomplete"
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    fields = sorted(set().union(*(row.keys() for row in payload["models"])))
    with (output_dir / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["models"])
    print(output_dir / "summary.json")
    if payload["status"] != "complete":
        raise SystemExit("Some models failed or telemetry was incomplete; inspect summary.json and logs")


if __name__ == "__main__":
    main()
