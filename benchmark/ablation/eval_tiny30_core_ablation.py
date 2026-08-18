#!/usr/bin/env python3
"""Evaluate Tiny30 MXQs on the exact same labeled ImageNet samples.

Manifest format (whitespace separated):
    /absolute/or/relative/image.JPEG 123

Relative paths are resolved against --data-root. Results include per-model
Top-1/Top-5 and pairwise Top-1 agreement, making single/global8 regressions
visible without relying on latency-only benchmark wrappers.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import qbruntime


LEGACY_MEAN = np.float32([0.485, 0.456, 0.406])
LEGACY_STD = np.float32([0.229, 0.224, 0.225])
DEIT_MEAN = np.float32([0.5, 0.5, 0.5])
DEIT_STD = np.float32([0.5, 0.5, 0.5])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", action="append", required=True, metavar="NAME:SCHEME:PATH",
        help="Repeat for each MXQ, e.g. single:single:assets/mxq/a.mxq",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all")
    parser.add_argument(
        "--preprocess", choices=["legacy", "deit"], default="legacy",
        help="legacy reproduces the existing baseline NPU pipeline",
    )
    parser.add_argument("--output", default="results/tiny30_core_ablation.json")
    return parser.parse_args()


def parse_model(spec):
    fields = spec.split(":", 2)
    if len(fields) != 3:
        raise ValueError("--model must be NAME:SCHEME:PATH: %s" % spec)
    name, scheme, path = fields
    if scheme not in ("single", "multi", "global", "global4", "global8"):
        raise ValueError("unsupported scheme: %s" % scheme)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return name, scheme, path


def load_manifest(path, data_root, limit):
    samples = []
    with open(path) as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.rsplit(None, 1)
            if len(fields) != 2:
                raise ValueError("bad manifest line %d: %s" % (line_no, line))
            image_path, label = fields
            if not os.path.isabs(image_path):
                image_path = os.path.join(data_root, image_path)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)
            samples.append((image_path, int(label)))
            if limit and len(samples) >= limit:
                break
    if not samples:
        raise ValueError("manifest contains no samples")
    return samples


def preprocess(path, mode):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot read image: %s" % path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Keep legacy byte-for-byte compatible with the existing NPU scripts.
    resize_short = 256 if mode == "legacy" else 248
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
    image = image[top:top + 224, left:left + 224]
    image = image.astype(np.float32) / 255.0
    if mode == "legacy":
        image = (image - LEGACY_MEAN) / LEGACY_STD
    else:
        image = (image - DEIT_MEAN) / DEIT_STD
    return np.ascontiguousarray(image, dtype=np.float32)  # HWC


def build_config(scheme):
    config = qbruntime.ModelConfig()
    cluster, core = qbruntime.Cluster, qbruntime.Core
    core_map = {
        "single": [(cluster.Cluster0, core.Core0)],
        "multi": [(cluster.Cluster0, core.Core0), (cluster.Cluster0, core.Core1)],
        "global": [(cluster.Cluster0, core.Core0), (cluster.Cluster0, core.Core1),
                   (cluster.Cluster1, core.Core0), (cluster.Cluster1, core.Core1)],
        "global4": [(cluster.Cluster0, core.Core0), (cluster.Cluster0, core.Core1),
                    (cluster.Cluster1, core.Core0), (cluster.Cluster1, core.Core1)],
        "global8": [(cluster.Cluster0, core.Core0), (cluster.Cluster0, core.Core1),
                    (cluster.Cluster0, core.Core2), (cluster.Cluster0, core.Core3),
                    (cluster.Cluster1, core.Core0), (cluster.Cluster1, core.Core1),
                    (cluster.Cluster1, core.Core2), (cluster.Cluster1, core.Core3)],
    }
    config.set_single_core_mode(
        core_ids=[qbruntime.CoreId(c, k) for c, k in core_map[scheme]]
    )
    return config


def evaluate(name, scheme, path, accelerator, samples, tensors):
    model = qbruntime.Model(path, build_config(scheme))
    model.launch(accelerator)
    top1 = top5 = 0
    predictions = []
    started = time.perf_counter()
    try:
        for index, ((_, label), tensor) in enumerate(zip(samples, tensors), 1):
            logits = np.asarray(model.infer([tensor])[0]).reshape(-1)
            order = np.argsort(logits)[::-1][:5]
            prediction = int(order[0])
            predictions.append(prediction)
            top1 += int(prediction == label)
            top5 += int(label in order)
            if index % 100 == 0 or index == len(samples):
                print("[%s] %d/%d top1=%.2f%%" % (
                    name, index, len(samples), 100.0 * top1 / index
                ))
    finally:
        model.dispose()
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "scheme": scheme,
        "path": os.path.abspath(path),
        "n": len(samples),
        "top1": 100.0 * top1 / len(samples),
        "top5": 100.0 * top5 / len(samples),
        "seconds": elapsed,
        "images_per_second": len(samples) / elapsed,
        "predictions": predictions,
    }


def main():
    args = parse_args()
    specs = [parse_model(spec) for spec in args.model]
    samples = load_manifest(args.manifest, args.data_root, args.limit)
    print("Loading %d images with preprocess=%s" % (len(samples), args.preprocess))
    tensors = [preprocess(path, args.preprocess) for path, _ in samples]

    accelerator = qbruntime.Accelerator(0)
    results = [evaluate(*spec, accelerator, samples, tensors) for spec in specs]
    comparisons = []
    for left_index in range(len(results)):
        for right_index in range(left_index + 1, len(results)):
            left, right = results[left_index], results[right_index]
            agreement = np.mean(
                np.asarray(left["predictions"]) == np.asarray(right["predictions"])
            )
            comparisons.append({
                "left": left["name"], "right": right["name"],
                "top1_agreement": 100.0 * float(agreement),
            })

    payload = {
        "preprocess": args.preprocess,
        "manifest": os.path.abspath(args.manifest),
        "models": results,
        "comparisons": comparisons,
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)

    print("\nSummary")
    for result in results:
        print("  %-16s top1=%6.2f%% top5=%6.2f%% ips=%7.2f" % (
            result["name"], result["top1"], result["top5"],
            result["images_per_second"],
        ))
    for comparison in comparisons:
        print("  %s vs %s top1 agreement=%.2f%%" % (
            comparison["left"], comparison["right"],
            comparison["top1_agreement"],
        ))
    print("Saved: %s" % args.output)


if __name__ == "__main__":
    main()
