#!/usr/bin/env python3
"""Remove exact identity selection MatMuls from a Patch Slimming ONNX.

Patch Slimming uses fixed one-hot matrices on the left side of MatMul to keep a
static, Gather-free token-selection graph.  When a stage keeps every token in
the original order, that matrix is exactly an identity matrix and the MatMul is
mathematically redundant::

    eye(N) @ x == x

This production utility removes only those exact identity cases.  Non-square
or reordered selectors are preserved, and no Gather operation is introduced.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import checker, numpy_helper


def constant_arrays(model):
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    constant_nodes = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        attr = next((item for item in node.attribute if item.name == "value"), None)
        if attr is None:
            continue
        arrays[node.output[0]] = numpy_helper.to_array(attr.t)
        constant_nodes[node.output[0]] = node
    return arrays, constant_nodes


def is_exact_identity(array):
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        return False
    return np.array_equal(array, np.eye(array.shape[0], dtype=array.dtype))


def remove_identity_selections(model, name_filter):
    arrays, constant_nodes = constant_arrays(model)
    consumers = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)

    removed_matmuls = []
    replacements = {}
    removed_node_ids = set()
    removable_constants = set()
    removable_initializers = set()

    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) != 2 or len(node.output) != 1:
            continue
        if name_filter and name_filter not in node.name:
            continue
        left = arrays.get(node.input[0])
        if left is None or not is_exact_identity(left):
            continue

        replacements[node.output[0]] = node.input[1]
        removed_node_ids.add(id(node))
        removed_matmuls.append((node.name, tuple(left.shape)))

    removed_outputs = set(replacements)
    for name, node in constant_nodes.items():
        if consumers[name] and all(id(item) in removed_node_ids for item in consumers[name]):
            removable_constants.add(name)
            removed_node_ids.add(id(node))
    initializer_names = {item.name for item in model.graph.initializer}
    for name in initializer_names:
        if consumers[name] and all(id(item) in removed_node_ids for item in consumers[name]):
            removable_initializers.add(name)

    def resolve(name):
        visited = set()
        while name in replacements:
            if name in visited:
                raise RuntimeError(f"replacement cycle at {name}")
            visited.add(name)
            name = replacements[name]
        return name

    kept_nodes = []
    for node in model.graph.node:
        if id(node) in removed_node_ids:
            continue
        for index, name in enumerate(node.input):
            node.input[index] = resolve(name)
        kept_nodes.append(node)
    for output in model.graph.output:
        output.name = resolve(output.name)

    kept_initializers = [
        item for item in model.graph.initializer
        if item.name not in removable_initializers
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    return {
        "removed": removed_matmuls,
        "removed_outputs": sorted(removed_outputs),
        "removed_constant_nodes": len(removable_constants),
        "removed_initializers": len(removable_initializers),
    }


def compare_outputs(before_path, after_path, input_path):
    import onnxruntime as ort

    value = np.ascontiguousarray(np.load(input_path), dtype=np.float32)
    if value.ndim == 3:
        value = value[None]
    before = ort.InferenceSession(str(before_path), providers=["CPUExecutionProvider"])
    after = ort.InferenceSession(str(after_path), providers=["CPUExecutionProvider"])
    reference = np.asarray(
        before.run(None, {before.get_inputs()[0].name: value})[0], dtype=np.float64
    )
    candidate = np.asarray(
        after.run(None, {after.get_inputs()[0].name: value})[0], dtype=np.float64
    )
    difference = candidate - reference
    cosine = float(
        np.dot(reference.ravel(), candidate.ravel())
        / max(np.linalg.norm(reference) * np.linalg.norm(candidate), 1e-12)
    )
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-12)
        ),
        "cosine": cosine,
        "argmax_before": int(np.argmax(reference)),
        "argmax_after": int(np.argmax(candidate)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw Patch Slimming ONNX")
    parser.add_argument("--output", default="", help="Optimized ONNX output")
    parser.add_argument(
        "--verify-clean",
        action="store_true",
        help="Fail if any identity selection MatMul remains; do not write a model",
    )
    parser.add_argument(
        "--name-filter",
        default="/slim.",
        help="Only inspect MatMul node names containing this text; empty checks all",
    )
    parser.add_argument(
        "--expect-removed",
        type=int,
        default=14,
        help="Fail unless this many identity MatMuls are removed; 0 disables the check",
    )
    parser.add_argument(
        "--check-input",
        default="",
        help="Optional HWC/BHWC .npy used for before/after logits comparison",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not args.verify_clean and not args.output:
        parser.error("--output is required unless --verify-clean is used")
    output_path = Path(args.output).resolve() if args.output else None
    if output_path is not None and input_path == output_path:
        raise ValueError("--input and --output must be different; validate before replacing")

    model = onnx.load(input_path)
    stats = remove_identity_selections(model, args.name_filter)
    count = len(stats["removed"])
    if args.verify_clean:
        if count:
            names = ", ".join(name for name, _shape in stats["removed"])
            raise RuntimeError(
                f"ONNX still contains {count} identity selection MatMuls: {names}"
            )
        checker.check_model(onnx.load(input_path))
        print(f"clean: {input_path}")
        print("identity selection MatMuls: 0")
        return
    if args.expect_removed and count != args.expect_removed:
        raise RuntimeError(
            f"expected {args.expect_removed} identity selection MatMuls, found {count}; "
            "do not publish this ONNX until the export graph is reviewed"
        )

    checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)

    print(f"input : {input_path}")
    print(f"output: {output_path}")
    print(f"identity selection MatMuls removed: {count}")
    for name, shape in stats["removed"]:
        print(f"  {name}: {shape}")
    print(f"constant nodes removed: {stats['removed_constant_nodes']}")
    print(f"initializers removed: {stats['removed_initializers']}")

    if args.check_input:
        metrics = compare_outputs(input_path, output_path, Path(args.check_input).resolve())
        print("ONNX equivalence:")
        for name, value in metrics.items():
            print(f"  {name}: {value}")
        if metrics["max_abs"] != 0.0 or metrics["argmax_before"] != metrics["argmax_after"]:
            raise RuntimeError("optimized ONNX is not bit-exact for --check-input")


if __name__ == "__main__":
    main()
