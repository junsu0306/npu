#!/usr/bin/env python3
"""Rewrite fixed Patch Slimming selection matrices into compiler-friendly ONNX.

The exported Patch Slimming graph expresses a fixed token selection as

    [N_out, N_in] one-hot matrix @ [B, N_in, D]

twice per block (query and residual).  This script can either move those
Constant tensors to graph initializers, or replace the operation with an
equivalent Gather.  Identity selections are removed in Gather mode.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "assets/onnx/tiny30__patch_slimming__nhwc_b1_npusafe.onnx"


def constant_array(node):
    if node.op_type != "Constant":
        return None
    attr = next((item for item in node.attribute if item.name == "value"), None)
    return None if attr is None else numpy_helper.to_array(attr.t)


def selection_ids(array):
    """Return row-wise selected indices when array is a one-hot matrix."""
    if array is None or array.ndim != 2 or array.shape[0] == 0:
        return None
    if not np.all((array == 0) | (array == 1)):
        return None
    if not np.all(np.sum(array, axis=1) == 1):
        return None
    return np.argmax(array, axis=1).astype(np.int64)


def rewrite(model, mode):
    consumers = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)

    selections = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or "/slim." not in node.name:
            continue
        array = constant_array(node)
        ids = selection_ids(array)
        if ids is None or len(node.output) != 1:
            continue
        uses = consumers[node.output[0]]
        if len(uses) != 1:
            continue
        matmul = uses[0]
        if (
            matmul.op_type != "MatMul"
            or len(matmul.input) != 2
            or matmul.input[0] != node.output[0]
        ):
            continue
        selections[node.name] = (node, matmul, array, ids)

    if not selections:
        raise RuntimeError("No fixed /slim.* one-hot selection MatMul nodes found")

    selected_nodes = {
        id(node)
        for constant, matmul, _array, _ids in selections.values()
        for node in (constant, matmul)
    }
    replacements = {}
    rewritten_nodes = []
    initializers = list(model.graph.initializer)
    gather_count = 0
    identity_count = 0

    if mode == "initializer":
        # Only Constant representation changes. MatMul nodes remain untouched.
        selected_nodes = {id(constant) for constant, _matmul, _array, _ids in selections.values()}
        for constant, _matmul, array, _ids in selections.values():
            initializers.append(numpy_helper.from_array(array, name=constant.output[0]))
    else:
        for constant, matmul, array, ids in selections.values():
            n_out, n_in = array.shape
            source = matmul.input[1]
            output = matmul.output[0]
            is_identity = n_out == n_in and np.array_equal(ids, np.arange(n_in))
            if is_identity:
                replacements[output] = source
                identity_count += 1
                continue

            index_name = f"{constant.output[0]}__indices"
            initializers.append(numpy_helper.from_array(ids, name=index_name))
            rewritten_nodes.append(
                helper.make_node(
                    "Gather",
                    inputs=[source, index_name],
                    outputs=[output],
                    axis=1,
                    name=f"{matmul.name}__Gather",
                )
            )
            gather_count += 1

    def resolve(name):
        seen = set()
        while name in replacements:
            if name in seen:
                raise RuntimeError(f"Replacement cycle at {name}")
            seen.add(name)
            name = replacements[name]
        return name

    new_nodes = []
    gather_by_output = {node.output[0]: node for node in rewritten_nodes}
    inserted_gathers = set()
    for node in model.graph.node:
        if id(node) in selected_nodes:
            if mode == "gather" and node.op_type == "MatMul":
                gather = gather_by_output.get(node.output[0])
                if gather is not None:
                    new_nodes.append(gather)
                    inserted_gathers.add(gather.output[0])
            continue
        for index, name in enumerate(node.input):
            node.input[index] = resolve(name)
        new_nodes.append(node)

    missing = set(gather_by_output) - inserted_gathers
    if missing:
        raise RuntimeError(f"Failed to insert Gather nodes for: {sorted(missing)}")

    for output in model.graph.output:
        output.name = resolve(output.name)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)

    return {
        "selection_matmuls": len(selections),
        "gathers": gather_count,
        "identity_matmuls_removed": identity_count,
    }


def check_equivalence(before_path, after_path, input_path):
    import onnxruntime as ort

    x = np.ascontiguousarray(np.load(input_path), dtype=np.float32)
    if x.ndim == 3:
        x = x[None]
    before = ort.InferenceSession(str(before_path), providers=["CPUExecutionProvider"])
    after = ort.InferenceSession(str(after_path), providers=["CPUExecutionProvider"])
    ref = np.asarray(before.run(None, {before.get_inputs()[0].name: x})[0], dtype=np.float64)
    got = np.asarray(after.run(None, {after.get_inputs()[0].name: x})[0], dtype=np.float64)
    diff = got - ref
    cosine = float(
        np.dot(ref.ravel(), got.ravel())
        / max(np.linalg.norm(ref) * np.linalg.norm(got), 1e-12)
    )
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-12)),
        "cosine": cosine,
        "argmax_before": int(np.argmax(ref)),
        "argmax_after": int(np.argmax(got)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["gather", "initializer"], default="gather")
    parser.add_argument("--check-input", default="")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    model = onnx.load(input_path)
    stats = rewrite(model, args.mode)
    model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)

    print(f"input : {input_path}")
    print(f"output: {output_path}")
    print(f"mode  : {args.mode}")
    for name, value in stats.items():
        print(f"{name}: {value}")
    if args.check_input:
        metrics = check_equivalence(input_path, output_path, args.check_input)
        print("ONNX equivalence:")
        for name, value in metrics.items():
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
