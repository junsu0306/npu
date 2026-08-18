#!/usr/bin/env python3
"""Evaluate equivalent Patch Slimming ONNX candidates on identical images."""

import argparse
import json
import os
import random
import time
from itertools import combinations

import cv2
import numpy as np
import onnxruntime as ort


PROFILES = {
    "timm": {
        "resize_short": 249,
        "interpolation": cv2.INTER_CUBIC,
        "mean": np.float32([0.5, 0.5, 0.5]),
        "std": np.float32([0.5, 0.5, 0.5]),
    },
    "legacy": {
        "resize_short": 256,
        "interpolation": cv2.INTER_LINEAR,
        "mean": np.float32([0.485, 0.456, 0.406]),
        "std": np.float32([0.229, 0.224, 0.225]),
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME:PATH",
        help="Repeat for each ONNX candidate",
    )
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--images-per-class", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess", choices=sorted(PROFILES), default="timm")
    parser.add_argument(
        "--output",
        default="results/diagnostics/patch_slimming_onnx_candidates_1000.json",
    )
    return parser.parse_args()


def parse_model(spec):
    fields = spec.split(":", 1)
    if len(fields) != 2:
        raise ValueError("--model must be NAME:PATH: %s" % spec)
    name, path = fields
    if not name:
        raise ValueError("model name is empty: %s" % spec)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return name, os.path.abspath(path)


def collect_samples(val_dir, images_per_class, num_classes, seed, limit):
    class_dirs = sorted(
        entry
        for entry in os.listdir(val_dir)
        if entry.isdigit() and os.path.isdir(os.path.join(val_dir, entry))
    )
    rng = random.Random(seed)
    if num_classes and num_classes < len(class_dirs):
        class_dirs = sorted(rng.sample(class_dirs, num_classes))

    samples = []
    for class_dir in class_dirs:
        folder = os.path.join(val_dir, class_dir)
        images = sorted(
            name
            for name in os.listdir(folder)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not images:
            continue
        class_rng = random.Random(seed + int(class_dir))
        chosen = class_rng.sample(images, min(images_per_class, len(images)))
        samples.extend((os.path.join(folder, name), int(class_dir)) for name in chosen)
        if limit and len(samples) >= limit:
            return samples[:limit]
    if not samples:
        raise ValueError("no labeled images found below %s" % val_dir)
    return samples


def preprocess(path, profile):
    cfg = PROFILES[profile]
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot read image: %s" % path)
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
    image = (image - cfg["mean"]) / cfg["std"]
    return np.ascontiguousarray(image, dtype=np.float32)


def resolve_layout(shape):
    if len(shape) != 4:
        raise ValueError("expected a four-dimensional input, got %s" % (shape,))
    if shape[-1] == 3:
        return "nhwc"
    if shape[1] == 3:
        return "nchw"
    raise ValueError("cannot infer input layout from %s" % (shape,))


def make_input(image, layout):
    if layout == "nchw":
        image = image.transpose(2, 0, 1)
    return np.ascontiguousarray(image[None], dtype=np.float32)


def main():
    args = parse_args()
    specs = [parse_model(spec) for spec in args.model]
    if len({name for name, _path in specs}) != len(specs):
        raise ValueError("model names must be unique")

    samples = collect_samples(
        args.val_dir,
        args.images_per_class,
        args.classes,
        args.seed,
        args.limit,
    )
    print(
        "Evaluating %d images, preprocess=%s, seed=%d"
        % (len(samples), args.preprocess, args.seed),
        flush=True,
    )

    models = []
    for name, path in specs:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        layout = resolve_layout(input_meta.shape)
        models.append({
            "name": name,
            "path": path,
            "session": session,
            "input_name": input_meta.name,
            "input_shape": input_meta.shape,
            "layout": layout,
            "top1_count": 0,
            "top5_count": 0,
            "seconds": 0.0,
            "predictions": [],
        })
        print("  %s: input=%s layout=%s" % (name, input_meta.shape, layout))

    pair_stats = {}
    for left, right in combinations(models, 2):
        pair_stats[(left["name"], right["name"])] = {
            "top1_agreement_count": 0,
            "exact_logits_count": 0,
            "mae_sum": 0.0,
            "relative_l2_sum": 0.0,
            "cosine_sum": 0.0,
            "min_cosine": 1.0,
            "max_abs": 0.0,
        }

    wall_started = time.perf_counter()
    for index, (image_path, label) in enumerate(samples, 1):
        image = preprocess(image_path, args.preprocess)
        logits_by_name = {}
        for model in models:
            tensor = make_input(image, model["layout"])
            started = time.perf_counter()
            logits = np.asarray(
                model["session"].run(
                    None, {model["input_name"]: tensor}
                )[0]
            ).reshape(-1)
            model["seconds"] += time.perf_counter() - started
            prediction = int(np.argmax(logits))
            top5 = np.argpartition(logits, -5)[-5:]
            model["predictions"].append(prediction)
            model["top1_count"] += int(prediction == label)
            model["top5_count"] += int(label in top5)
            logits_by_name[model["name"]] = logits.astype(np.float64, copy=False)

        for left, right in combinations(models, 2):
            key = (left["name"], right["name"])
            stats = pair_stats[key]
            a, b = logits_by_name[key[0]], logits_by_name[key[1]]
            diff = b - a
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            cosine = float(np.dot(a, b) / max(norm_a * norm_b, 1e-12))
            stats["top1_agreement_count"] += int(np.argmax(a) == np.argmax(b))
            stats["exact_logits_count"] += int(np.array_equal(a, b))
            stats["mae_sum"] += float(np.mean(np.abs(diff)))
            stats["relative_l2_sum"] += float(
                np.linalg.norm(diff) / max(norm_a, 1e-12)
            )
            stats["cosine_sum"] += cosine
            stats["min_cosine"] = min(stats["min_cosine"], cosine)
            stats["max_abs"] = max(stats["max_abs"], float(np.max(np.abs(diff))))

        if index % 100 == 0 or index == len(samples):
            progress = " ".join(
                "%s=%.2f%%" % (
                    model["name"], 100.0 * model["top1_count"] / index
                )
                for model in models
            )
            print("[%d/%d] top1 %s" % (index, len(samples), progress), flush=True)

    wall_seconds = time.perf_counter() - wall_started
    model_results = []
    for model in models:
        model_results.append({
            "name": model["name"],
            "path": model["path"],
            "input_shape": model["input_shape"],
            "layout": model["layout"],
            "n": len(samples),
            "top1": 100.0 * model["top1_count"] / len(samples),
            "top5": 100.0 * model["top5_count"] / len(samples),
            "inference_seconds": model["seconds"],
            "images_per_second": len(samples) / model["seconds"],
            "predictions": model["predictions"],
        })

    comparisons = []
    for (left, right), stats in pair_stats.items():
        comparisons.append({
            "left": left,
            "right": right,
            "n": len(samples),
            "top1_agreement": 100.0 * stats["top1_agreement_count"] / len(samples),
            "exact_logits": stats["exact_logits_count"],
            "mean_mae": stats["mae_sum"] / len(samples),
            "mean_relative_l2": stats["relative_l2_sum"] / len(samples),
            "mean_cosine": stats["cosine_sum"] / len(samples),
            "min_cosine": stats["min_cosine"],
            "max_abs": stats["max_abs"],
        })

    payload = {
        "preprocess": args.preprocess,
        "val_dir": os.path.abspath(args.val_dir),
        "sampling": {
            "classes": args.classes,
            "images_per_class": args.images_per_class,
            "limit": args.limit,
            "seed": args.seed,
        },
        "wall_seconds": wall_seconds,
        "models": model_results,
        "comparisons": comparisons,
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)

    print("\nSummary")
    for model in model_results:
        print(
            "  %-12s top1=%6.2f%% top5=%6.2f%% ips=%6.2f"
            % (
                model["name"],
                model["top1"],
                model["top5"],
                model["images_per_second"],
            )
        )
    for item in comparisons:
        print(
            "  %s vs %s: agreement=%.2f%% exact=%d/%d max_abs=%.8g"
            % (
                item["left"],
                item["right"],
                item["top1_agreement"],
                item["exact_logits"],
                item["n"],
                item["max_abs"],
            )
        )
    print("Saved: %s" % args.output)


if __name__ == "__main__":
    main()
