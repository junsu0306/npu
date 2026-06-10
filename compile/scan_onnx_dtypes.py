#!/usr/bin/env python3
"""
Fixed ONNX의 비-float32 텐서를 스캔.
zeropoint=-2147483648 에러의 원인이 되는 텐서를 찾는다.

실행:
  python3 compile/scan_onnx_dtypes.py
"""

import os
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))
ONNX_DIR   = os.path.join(ASSETS_DIR, "onnx")

DTYPE_NAME = {
    0: "UNDEFINED", 1: "FLOAT32", 2: "UINT8", 3: "INT8",
    4: "UINT16", 5: "INT16", 6: "INT32", 7: "INT64",
    8: "STRING", 9: "BOOL", 10: "FLOAT16", 11: "FLOAT64",
    12: "UINT32", 13: "UINT64",
}


def scan(path: str):
    model = onnx.load(path)
    graph = model.graph

    print(f"\n{'='*70}")
    print(f"File: {os.path.basename(path)}")
    print(f"{'='*70}")

    # ── 1. initializer 비-float32 ────────────────────────────────────────────
    non_f32_inits = []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            non_f32_inits.append((init.name, init.data_type, list(init.dims), arr.flat[0] if arr.size else None))

    print(f"\n[initializer] non-float32: {len(non_f32_inits)}")
    for name, dt, dims, sample in non_f32_inits:
        print(f"  {DTYPE_NAME.get(dt, dt):10s} shape={str(dims):20s} sample={str(sample):>20}  {name[:60]}")

    # ── 2. Constant 노드 비-float32 ──────────────────────────────────────────
    non_f32_consts = []
    for n in graph.node:
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value":
                    dt   = attr.t.data_type
                    dims = list(attr.t.dims)
                    arr  = numpy_helper.to_array(attr.t)
                    val  = arr.flat[0] if arr.size else None
                    if dt != TensorProto.FLOAT:
                        non_f32_consts.append((n.name, dt, dims, val))

    print(f"\n[Constant node] non-float32: {len(non_f32_consts)}")
    for name, dt, dims, val in non_f32_consts:
        print(f"  {DTYPE_NAME.get(dt, dt):10s} shape={dims!s:20s} value={val!r:>20}  {name[:60]}")

    # ── 3. FLOAT64 한정 검사 (양자화 치명상) ──────────────────────────────────
    f64_inits = [i.name for i in graph.initializer if i.data_type == TensorProto.DOUBLE]
    f64_const = []
    for n in graph.node:
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.DOUBLE:
                    f64_const.append(n.name)

    print(f"\n[FLOAT64 summary]")
    print(f"  initializers : {len(f64_inits)}")
    print(f"  Constant nodes: {len(f64_const)}")
    if f64_inits or f64_const:
        print("  *** FLOAT64 발견 → 이게 zeropoint 오버플로우 원인 ***")

    # ── 4. ReduceSum 잔존 여부 ────────────────────────────────────────────────
    rs_info = []
    for n in graph.node:
        if n.op_type == "ReduceSum":
            axes = next((list(a.ints) for a in n.attribute if a.name == "axes"), None)
            n_inputs = len(n.input)
            rs_info.append((n.name, axes, n_inputs))
    print(f"\n[ReduceSum 잔존]: {len(rs_info)}")
    for rname, axes, ni in rs_info:
        style = "opset9(attr)" if axes is not None else "opset13(input)"
        print(f"  {style}  axes={axes}  inputs={ni}  {rname.split('main/')[-1]}")

    # ── 5. Cast 잔존 여부 ──────────────────────────────────────────────────────
    noop_casts, int64_casts = [], []
    for n in graph.node:
        if n.op_type == "Cast":
            for attr in n.attribute:
                if attr.name == "to":
                    if attr.i == TensorProto.FLOAT:
                        noop_casts.append(n.name)
                    elif attr.i == TensorProto.INT64:
                        int64_casts.append(n.name)
    print(f"\n[no-op Cast(float32) 잔존]: {len(noop_casts)}")
    print(f"[Cast(→int64) 잔존]: {len(int64_casts)}")
    for n in int64_casts:
        print(f"  {n.split('main/')[-1]}")

    # ── 6. float32 Constant 값 (range=0 위험 체크) ────────────────────────────
    f32_consts = []
    for n in graph.node:
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t)
                    f32_consts.append((n.name, list(attr.t.dims),
                                       float(arr.min()), float(arr.max()), arr.size))
    print(f"\n[float32 Constant]: {len(f32_consts)}")
    for name, dims, mn, mx, sz in f32_consts:
        flag = "RANGE=0" if (mx - mn) == 0 else "ok"
        print(f"  {flag}  shape={dims}  [{mn:.6f}, {mx:.6f}]  {name[:60]}")

    # ── 7. float32 initializer range=0 위험 체크 ─────────────────────────────
    risky_inits = []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init)
        if arr.max() == arr.min():
            risky_inits.append((init.name, list(init.dims), float(arr.min()), arr.size))
    print(f"\n[float32 initializer range=0]: {len(risky_inits)}")
    for name, dims, val, sz in risky_inits[:20]:
        print(f"  shape={dims}  val={val:.6f}  size={sz}  {name[:60]}")

    # ── 8. 전체 노드 op_type 분포 ─────────────────────────────────────────────
    from collections import Counter
    ctr = Counter(n.op_type for n in graph.node)
    print(f"\n[op_type 분포] total={len(graph.node)}")
    for op, cnt in ctr.most_common():
        print(f"  {op:25s}: {cnt}")


MODELS = [
    "efficientvit_b0_original_r224_nchw.onnx",
    "efficientvit_b0_original_r224_nchw_fixed.onnx",
]

for fname in MODELS:
    p = os.path.join(ONNX_DIR, fname)
    if os.path.exists(p):
        scan(p)
    else:
        print(f"\n[SKIP] {fname}")
