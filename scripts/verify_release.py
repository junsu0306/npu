#!/usr/bin/env python3
"""Verify the frozen final model/evidence package without an NPU.

Default checks use only the standard library. --onnx and --predictions add
optional graph checks and independent Top-1 recalculation, respectively.
This is an integrity check, not a replacement for actual NPU inference.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path("results/final_benchmark/final_20261015")
ONNX_DIR = Path("assets/onnx")
MXQ_DIR = Path("assets/mxq_final_20261006")
MANIFEST = Path("assets/RELEASE_MANIFEST.json")
GROUPS = ("tiny30", "tiny50", "small30", "small50")
VARIANTS = ("baseline", "reduced", "patch_slimming")
EXPECTED_IDS = {f"{group}__{variant}" for group in GROUPS for variant in VARIANTS}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_path(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe path: {relative}")
    result = root / path
    require(result.resolve().is_relative_to(root.resolve()), f"Escaping path: {relative}")
    return result


def model_paths(model_id):
    stem = f"{model_id}__nhwc_b1_npusafe"
    return ONNX_DIR / (stem + ".onnx"), MXQ_DIR / (stem + "_vt_global8.mxq")


def check_record(root):
    record = read_json(root / REPORT / "benchmark.json")
    run, models = record["run"], record["models"]
    require(run["status"] == "complete" and run["num_models"] == len(models) == 12,
            "Expected a complete 12-model run")
    require(run["num_samples"] == 50000 and run["classes"] == 1000, "Unexpected dataset size")
    require(run["preprocess"] == "timm" and run["core_mode"] == "global8", "Unexpected protocol")
    require(read_json(root / REPORT / "run_metadata.json") == run, "Run metadata differs")
    require(sha256(root / REPORT / "samples.tsv") == run["sample_manifest_sha256"],
            "Sample manifest hash differs")
    samples = [line.rsplit("\t", 1) for line in (root / REPORT / "samples.tsv").read_text().splitlines()]
    require(len(samples) == 50000 and len({p for p, _ in samples}) == 50000,
            "Sample list is incomplete or duplicated")
    require(Counter(int(label) for _, label in samples) == Counter({i: 50 for i in range(1000)}),
            "Expected 50 images in each of 1000 classes")
    ids = [f"{model['group']}__{model['variant']}" for model in models]
    require(len(set(ids)) == 12 and set(ids) == EXPECTED_IDS, "Wrong or duplicate model IDs")
    expected_onnx, expected_mxq = set(), set()
    isolated = read_json(root / REPORT / "isolated_memory/summary.json")
    require(isolated["status"] == "complete", "Isolated memory run is incomplete")
    require(len(isolated["models"]) == 12, "Expected 12 isolated memory results")
    memory = {row["model_id"]: row for row in isolated["models"]}
    require(set(memory) == EXPECTED_IDS, "Missing isolated memory model")
    with (root / REPORT / "summary.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    require(len(csv_rows) == 12, "CSV row count differs")
    rows = {f"{row['group']}__{row['variant']}": row for row in csv_rows}
    require(set(rows) == EXPECTED_IDS, "CSV model IDs differ")
    for model_id, model in zip(ids, models):
        onnx_path, mxq_path = model_paths(model_id)
        expected_onnx.add(onnx_path.name)
        expected_mxq.add(mxq_path.name)
        require(model["status"] == "ok" and model["num_successful"] == 50000
                and model["num_errors"] == 0 and model["num_requested"] == 50000,
                f"Incomplete inference: {model_id}")
        require(model["npu_telemetry"]["status"] == "ok", f"Telemetry failure: {model_id}")
        for relative, recorded_hash in (
            (onnx_path, model["static_analysis"]["onnx_sha256"]),
            (mxq_path, model["mxq_sha256"]),
        ):
            require(sha256(root / relative) == recorded_hash, f"Model differs from measured artifact: {relative}")
        detail = root / REPORT / "models" / (model["name"] + ".json")
        require(read_json(detail) == model, f"Per-model evidence differs: {model_id}")
        for key, column in (("top1_acc_pct", "top1_pct"), ("top5_acc_pct", "top5_pct")):
            require(float(rows[model_id][column]) == model[key], f"CSV accuracy differs: {model_id}")
        isolated_model = memory[model_id]
        require(isolated_model["status"] == "ok", f"Isolated run failed: {model_id}")
        require(isolated_model["npu_peak_reported_mb"] ==
                model["npu_telemetry"]["metrics"]["max_memory_used_mb"],
                f"Integrated/isolated NPU peak differs: {model_id}")
    require({p.name for p in (root / ONNX_DIR).glob("*.onnx")} == expected_onnx,
            "ONNX directory must contain exactly the final 12 models")
    require({p.name for p in (root / MXQ_DIR).glob("*.mxq")} == expected_mxq,
            "MXQ directory must contain exactly the final 12 models")
    return record, samples


def artifact_paths(root):
    paths = [path for model_id in sorted(EXPECTED_IDS) for path in model_paths(model_id)]
    paths += [p.relative_to(root) for p in sorted((root / REPORT).rglob("*")) if p.is_file()]
    return sorted(paths)


def write_manifest(root):
    """One-time packaging; refuse to replace a previously frozen manifest."""
    destination = root / MANIFEST
    require(not destination.exists(), "Manifest already exists; do not silently rebaseline evidence")
    check_record(root)
    payload = {
        "schema_version": 1,
        "release_id": "final_20261015",
        "model_set": "mxq_final_20261006",
        "source_result_commit": "3e8bb5ae8ce923e13a703d1d7f72b3eaccbb5985",
        "purpose": "Integrity of measured ONNX/MXQ artifacts and original NPU evidence, not an accuracy guarantee",
        "files": [
            {"path": path.as_posix(), "bytes": (root / path).stat().st_size, "sha256": sha256(root / path)}
            for path in artifact_paths(root)
        ],
    }
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Created {MANIFEST}: {len(payload['files'])} files")


def check_manifest(root):
    manifest = read_json(root / MANIFEST)
    entries = manifest["files"]
    paths = [entry["path"] for entry in entries]
    require(len(set(paths)) == len(paths), "Duplicate manifest entry")
    require(set(paths) == {p.as_posix() for p in artifact_paths(root)}, "Manifest inventory differs")
    for entry in entries:
        path = safe_path(root, entry["path"])
        require(path.is_file() and path.stat().st_size == entry["bytes"],
                f"Missing/size mismatch: {entry['path']}")
        require(sha256(path) == entry["sha256"], f"Hash mismatch: {entry['path']}")
    print(f"PASS: manifest, {len(entries)} unchanged model/evidence files")


def check_predictions(root, record, samples):
    import numpy as np
    labels = np.asarray([int(label) for _, label in samples])
    for model in record["models"]:
        prediction = np.load(root / REPORT / "predictions" / (model["name"] + ".npy"),
                             allow_pickle=False)
        require(prediction.shape == (50000,) and np.issubdtype(prediction.dtype, np.integer),
                f"Invalid prediction array: {model['name']}")
        require(bool(((prediction >= 0) & (prediction < 1000)).all()), "Invalid class ID")
        top1 = float((prediction == labels).mean() * 100)
        require(abs(top1 - model["top1_acc_pct"]) < 1e-9, f"Top-1 mismatch: {model['name']}")
    isolated_dir = root / REPORT / "isolated_memory"
    value = np.load(isolated_dir / "input_nhwc_fp32.npy", allow_pickle=False)
    protocol = read_json(isolated_dir / "summary.json")["protocol"]
    require(value.shape == (1, 224, 224, 3) and value.dtype == np.float32, "Invalid memory input")
    require(hashlib.sha256(value.tobytes()).hexdigest() == protocol["input_sha256"],
            "Memory input hash differs")
    print("PASS: 12/12 Top-1 values recalculated from 50,000 saved predictions; memory input hash")


def check_onnx(root):
    import numpy as np
    import onnx
    from onnx import numpy_helper
    for path in sorted((root / ONNX_DIR).glob("*.onnx")):
        model = onnx.load(path)
        onnx.checker.check_model(model)
        require(len(model.graph.input) == len(model.graph.output) == 1, f"Wrong IO count: {path.name}")
        shape = lambda item: [d.dim_value for d in item.type.tensor_type.shape.dim]
        require(shape(model.graph.input[0]) == [1, 224, 224, 3], f"Wrong input shape: {path.name}")
        require(shape(model.graph.output[0]) == [1, 1000], f"Wrong output shape: {path.name}")
        require(model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT,
                f"Expected FP32 input: {path.name}")
        require(any(op.domain in ("", "ai.onnx") and op.version == 17 for op in model.opset_import),
                f"Expected opset 17: {path.name}")
        forbidden = {"Gather", "GatherElements", "GatherND", "TopK", "NonZero"}
        require(not (forbidden & {n.op_type for n in model.graph.node}), f"Unsupported selection op: {path.name}")
        arrays = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        for node in model.graph.node:
            if node.op_type == "Constant":
                attr = next((a for a in node.attribute if a.name == "value"), None)
                if attr is not None:
                    arrays[node.output[0]] = numpy_helper.to_array(attr.t)
        for node in model.graph.node:
            if node.op_type != "MatMul" or "/slim." not in node.name:
                continue
            left = arrays.get(node.input[0])
            if left is not None and left.ndim == 2 and left.shape[0] == left.shape[1]:
                require(not np.array_equal(left, np.eye(left.shape[0], dtype=left.dtype)),
                        f"Identity selector remains: {path.name}:{node.name}")
    print("PASS: 12 ONNX graphs, fixed FP32 NHWC IO/opset, no Gather/TopK or identity selectors")


def check_git_index(root):
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")[:-1]
    forbidden = ("past/", "assets/calib", "assets/mxq/", ".vscode/", ".tools/")
    unexpected = [p for p in tracked if p.startswith(forbidden) or p.endswith((".whl", ".log"))]
    require(not unexpected, f"Non-submission files remain tracked: {unexpected[:5]}")
    required = {p.as_posix() for p in artifact_paths(root)} | {MANIFEST.as_posix()}
    require(required.issubset(tracked), f"Submission artifacts not staged/tracked: {sorted(required - set(tracked))[:5]}")
    print(f"PASS: Git index excludes local datasets/archive/SDK; {len(tracked)} tracked files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--predictions", action="store_true")
    parser.add_argument("--git-index", action="store_true")
    parser.add_argument("--write-manifest", action="store_true", help="one-time packaging; refuses overwrite")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.write_manifest:
        write_manifest(root)
    check_manifest(root)
    record, samples = check_record(root)
    print("PASS: 12 measured ONNX/MXQ pairs, full 50,000-image evidence and matching isolated NPU peaks")
    if args.predictions:
        check_predictions(root, record, samples)
    if args.onnx:
        check_onnx(root)
    if args.git_index:
        check_git_index(root)
    print("No NPU inference performed; integrity checks do not establish accuracy for newly compiled MXQs.")


if __name__ == "__main__":
    main()
