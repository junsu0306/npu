#!/usr/bin/env python3
"""Mobilint Model Zoo based accuracy/latency/memory benchmark.

This is the npu-repository version of Orin_Compression's detailed benchmark.
It supports a local MXQ, numeric ImageNet class directories, reproducible
sampling, and optional mblt_tracker metrics.
"""

import argparse
import csv
import json
import os
import random
import statistics
import sys
import threading
import time
from datetime import datetime

import psutil
import torch

try:
    import mblt_model_zoo.vision as vision
except Exception as exc:
    raise SystemExit("mblt_model_zoo.vision import failed: %s" % exc)

try:
    from mblt_tracker import NPUDeviceTracker
except Exception:
    NPUDeviceTracker = None


NPU_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_RESULTS = os.path.join(NPU_ROOT, "results", "model_zoo")


def parse_args():
    parser = argparse.ArgumentParser(
        description="NPU accuracy + latency + CPU/NPU memory benchmark"
    )
    parser.add_argument(
        "--model-class", default="ViT_Tiny_Patch16_224",
        help="class exported by mblt_model_zoo.vision",
    )
    parser.add_argument("--mxq", required=True, help="local compiled MXQ")
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--classes", type=int, default=200)
    parser.add_argument("--images-per-class", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=1)
    parser.add_argument(
        "--infer-mode", default="global8",
        choices=["single", "multi", "global4", "global8"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="")
    parser.add_argument("--architecture", default="", help="Patch Slimming architecture.json (activation estimate)")
    parser.add_argument("--embed-dim", type=int, default=0, help="0 infers Tiny=192, Small=384")
    parser.add_argument("--heads", type=int, default=0, help="0 infers Tiny=3, Small=6")
    parser.add_argument("--mlp-hidden", type=int, default=0, help="approximate hidden width; 0 uses 4*embed_dim")
    parser.add_argument("--memory-sample-ms", type=float, default=20.0)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def collect_samples(val_dir, images_per_class, num_classes, seed):
    class_dirs = sorted(
        entry for entry in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, entry)) and entry.isdigit()
    )
    rng = random.Random(seed)
    if num_classes and num_classes < len(class_dirs):
        class_dirs = sorted(rng.sample(class_dirs, num_classes))

    samples = []
    for class_dir in class_dirs:
        folder = os.path.join(val_dir, class_dir)
        images = sorted(
            name for name in os.listdir(folder)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        class_rng = random.Random(seed + int(class_dir))
        chosen = class_rng.sample(images, min(images_per_class, len(images)))
        samples.extend((os.path.join(folder, name), int(class_dir)) for name in chosen)
    return samples


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 2)


class ProcessMemoryMonitor:
    """High-frequency process/system RAM monitor; catches peaks between images."""

    def __init__(self, interval_ms):
        self.interval = interval_ms / 1000.0
        self.process = psutil.Process(os.getpid())
        self.samples = []
        self.stop_event = threading.Event()

    def snapshot(self, stage="sample"):
        info = self.process.memory_info()
        full = self.process.memory_full_info()
        vm = psutil.virtual_memory()
        sample = {
            "stage": stage, "time": time.time(), "rss_bytes": info.rss,
            "vms_bytes": info.vms, "uss_bytes": getattr(full, "uss", 0),
            "pss_bytes": getattr(full, "pss", 0), "system_used_bytes": vm.used,
            "system_available_bytes": vm.available,
        }
        self.samples.append(sample)
        return sample

    def start(self):
        self.snapshot("before_model_load")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.snapshot()

    def mark(self, stage):
        return self.snapshot(stage)

    def stop(self):
        self.stop_event.set(); self.thread.join(); self.snapshot("after_dispose")
        return {
            "sample_interval_ms": self.interval * 1000.0,
            "sample_count": len(self.samples),
            "rss_start_bytes": self.samples[0]["rss_bytes"],
            "rss_peak_bytes": max(x["rss_bytes"] for x in self.samples),
            "uss_peak_bytes": max(x["uss_bytes"] for x in self.samples),
            "pss_peak_bytes": max(x["pss_bytes"] for x in self.samples),
            "system_used_peak_bytes": max(x["system_used_bytes"] for x in self.samples),
            "system_available_min_bytes": min(x["system_available_bytes"] for x in self.samples),
            "stage_snapshots": [x for x in self.samples if x["stage"] != "sample"],
        }


def activation_estimate(path, model_class, embed_dim, heads, mlp_hidden):
    """Architecture-based tensor-volume estimate; not a measured NPU allocation."""
    if not path:
        return {}
    with open(path) as handle:
        architecture = json.load(handle)
    schedule = [int(architecture["num_global_tokens"])] + [
        len(layer["keep_global_ids"]) for layer in architecture["layers"]
    ]
    lower = model_class.lower()
    dim = embed_dim or (384 if "small" in lower else 192)
    num_heads = heads or (6 if "small" in lower else 3)
    hidden = mlp_hidden or 4 * dim
    rows = []
    for block, (n_in, n_out) in enumerate(zip(schedule, schedule[1:])):
        counts = {
            "input": n_in * dim, "q": n_out * dim, "k": n_in * dim,
            "v": n_in * dim, "attention_scores": num_heads * n_out * n_in,
            "attention_probabilities": num_heads * n_out * n_in,
            "attention_context": n_out * dim, "mlp_hidden": n_out * hidden,
            "output": n_out * dim, "selection_matrix": n_out * n_in,
        }
        # Compiler precision is backend-dependent. Report FP32 and FP16 volumes.
        rows.append({
            "block": block, "tokens_in": n_in, "tokens_out": n_out,
            "tensor_numel": counts,
            "fp32_sum_bytes_upper_bound": 4 * sum(counts.values()),
            "fp16_sum_bytes_upper_bound": 2 * sum(counts.values()),
            "fp32_attention_matrix_bytes": 4 * num_heads * n_out * n_in * 2,
            "fp16_attention_matrix_bytes": 2 * num_heads * n_out * n_in * 2,
        })
    return {
        "source": os.path.abspath(path), "token_schedule": schedule,
        "embed_dim": dim, "heads": num_heads, "mlp_hidden_assumption": hidden,
        "batch_size": 1, "blocks": rows,
        "caution": "analytical tensor-volume upper bounds; qbruntime does not expose internal activation allocations",
    }


def tracker_metrics(tracker):
    if tracker is None:
        return {}
    tracker.stop()
    metric = tracker.get_metric()
    return {
        "npu_avg_power_w": round(metric.get("avg_power_w") or 0, 2),
        "npu_max_power_w": round(metric.get("max_power_w") or 0, 2),
        "npu_avg_util_pct": round(metric.get("avg_utilization_pct") or 0, 1),
        "npu_max_util_pct": round(metric.get("max_utilization_pct") or 0, 1),
        "npu_avg_mem_mb": round(metric.get("avg_memory_used_mb") or 0, 1),
        "npu_max_mem_mb": round(metric.get("max_memory_used_mb") or 0, 1),
        "npu_total_mem_mb": round(metric.get("total_memory_mb") or 0, 1),
        "npu_avg_temp_c": round(metric.get("avg_temperature_c") or 0, 1),
        "npu_max_temp_c": round(metric.get("max_temperature_c") or 0, 1),
        "npu_tracker_samples": metric.get("samples", 0),
    }


def main():
    args = parse_args()
    if not os.path.isfile(args.mxq):
        raise FileNotFoundError(args.mxq)
    if not os.path.isdir(args.val_dir):
        raise NotADirectoryError(args.val_dir)
    if not hasattr(vision, args.model_class):
        raise ValueError("unknown Model Zoo class: %s" % args.model_class)

    samples = collect_samples(
        args.val_dir, args.images_per_class, args.classes, args.seed
    )
    if not samples:
        raise RuntimeError("no labeled images found below %s" % args.val_dir)
    print("Samples: %d, model=%s, mode=%s" % (
        len(samples), args.model_class, args.infer_mode
    ))

    memory_monitor = ProcessMemoryMonitor(args.memory_sample_ms)
    memory_monitor.start()
    before_load = rss_mb()
    model = getattr(vision, args.model_class)(
        infer_mode=args.infer_mode, local_path=args.mxq
    )
    memory_monitor.mark("after_model_load")
    after_load = rss_mb()
    peak_mb = after_load

    tracker = None
    if NPUDeviceTracker is not None:
        try:
            tracker = NPUDeviceTracker(interval=0.5)
            tracker.start()
        except Exception as exc:
            print("[WARN] mblt_tracker unavailable: %s" % exc)
            tracker = None

    top1 = top5 = successful = 0
    latencies = []
    errors = []
    try:
        for sample_index, (image_path, label) in enumerate(samples):
            try:
                tensor = model.preprocess(image_path)
                if sample_index == 0:
                    print("Preprocessed input: shape=%s dtype=%s min=%.4f max=%.4f" % (
                        tuple(tensor.shape), tensor.dtype,
                        float(torch.as_tensor(tensor).min()),
                        float(torch.as_tensor(tensor).max()),
                    ))
                    for _ in range(args.warmup_runs):
                        model(tensor)
                    memory_monitor.mark("after_warmup")

                started = time.perf_counter()
                for _ in range(args.timed_runs):
                    raw = model(tensor)
                elapsed = time.perf_counter() - started
                latencies.append(elapsed * 1000.0 / args.timed_runs)

                if isinstance(raw, (list, tuple)):
                    if len(raw) != 1:
                        raise ValueError("expected one output, got %d" % len(raw))
                    raw = raw[0]
                logits = torch.as_tensor(raw).flatten(1)
                indices = torch.topk(logits, 5, dim=1).indices[0].tolist()
                top1 += int(indices[0] == label)
                top5 += int(label in indices)
                successful += 1
                peak_mb = max(peak_mb, rss_mb())
                if successful % 100 == 0:
                    print("%d/%d top1=%.2f%%" % (
                        successful, len(samples), 100.0 * top1 / successful
                    ))
            except Exception as exc:
                errors.append("%s: %s" % (image_path, exc))
    finally:
        model.dispose()
        memory_monitor.mark("model_disposed")

    if not successful:
        raise RuntimeError("all inferences failed: %s" % errors[:3])

    process_memory = memory_monitor.stop()
    result = {
        "model_class": args.model_class,
        "mxq": os.path.abspath(args.mxq),
        "mxq_file_bytes": os.path.getsize(args.mxq),
        "infer_mode": args.infer_mode,
        "num_requested": len(samples),
        "num_successful": successful,
        "errors": len(errors),
        "top1_acc_pct": round(100.0 * top1 / successful, 3),
        "top5_acc_pct": round(100.0 * top5 / successful, 3),
        "avg_ms": round(statistics.mean(latencies), 3),
        "std_ms": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0,
        "fps": round(1000.0 / statistics.mean(latencies), 3),
        "cpu_mem_before_load_mb": round(before_load, 1),
        "cpu_mem_model_mb": round(after_load - before_load, 1),
        "cpu_mem_peak_mb": round(peak_mb, 1),
        "process_memory_detailed": process_memory,
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "seed": args.seed,
    }
    result.update(tracker_metrics(tracker))
    result["activation_estimate"] = activation_estimate(
        args.architecture, args.model_class, args.embed_dim, args.heads, args.mlp_hidden
    )

    print(json.dumps(result, indent=2))
    if not args.no_save:
        os.makedirs(args.results_dir, exist_ok=True)
        label = args.label or os.path.splitext(os.path.basename(args.mxq))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = os.path.join(
            args.results_dir, "%s_%s_%s.json" % (label, args.infer_mode, timestamp)
        )
        with open(output, "w") as handle:
            json.dump(result, handle, indent=2)
        print("Saved: %s" % output)
        blocks = result.get("activation_estimate", {}).get("blocks", [])
        if blocks:
            activation_csv = os.path.splitext(output)[0] + "_activation.csv"
            fields = [
                "block", "tokens_in", "tokens_out",
                "fp32_attention_matrix_bytes", "fp16_attention_matrix_bytes",
                "fp32_sum_bytes_upper_bound", "fp16_sum_bytes_upper_bound",
            ]
            with open(activation_csv, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader(); writer.writerows(blocks)
            print("Activation CSV: %s" % activation_csv)


if __name__ == "__main__":
    main()
