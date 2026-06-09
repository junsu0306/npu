#!/usr/bin/env python3
"""
original EfficientViT ONNX의 qbcompiler 비호환 노드 수정.

수정 내용:
  1. ReduceSum(axes=[-1], keepdims=1)
       → MatMul(X, ones)  (qbcompiler가 확실히 지원)
     - 음수 축(-1), opset13 두 번째 input 방식 모두 qbcompiler가 처리 못 함
     - ReduceSum(X, axis=-1, keepdims=1) ≡ MatMul(X, ones_col)
       X: [B,H,S,D] → ones: [D,1] → output: [B,H,S,1]

  2. no-op Cast (float32 → float32) 제거
     - context_module 끝 Reshape → Cast(float32) → Conv 구간에서
       Cast를 우회해 Reshape 출력을 Conv에 직접 연결

사용:
  python3 compile/fix_original_onnx.py

출력:
  assets/onnx/efficientvit_b0_original_r224_nchw_fixed.onnx
  assets/onnx/efficientvit_b1_original_r224_nchw_fixed.onnx
"""

import os
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))
ONNX_DIR   = os.path.join(ASSETS_DIR, "onnx")


def fix_model(src_path: str, dst_path: str):
    model = onnx.load(src_path)
    graph = model.graph

    nodes        = list(graph.node)
    initializers = {w.name: w for w in graph.initializer}
    new_inits    = []   # 추가할 initializer
    remove_nodes = set()  # 제거할 노드 인덱스
    rewire       = {}   # {old_output_name: new_output_name}

    # ── Constant 노드 출력 맵 (axes 텐서 생성 노드 추적용) ─────────────────────
    const_outputs = {}  # output_name → Constant 노드 인덱스
    for i, n in enumerate(nodes):
        if n.op_type == "Constant":
            for o in n.output:
                const_outputs[o] = i

    # ── 1. ReduceSum → MatMul 교체 ─────────────────────────────────────────────
    ones_cache = {}   # dim → initializer name (재사용)
    new_nodes  = []

    for i, n in enumerate(nodes):
        if n.op_type != "ReduceSum":
            new_nodes.append(n)
            continue

        data_in  = n.input[0]
        data_out = n.output[0]

        # shape 조회
        vi_map = {vi.name: vi for vi in list(graph.value_info) + list(graph.input)}
        vi = vi_map.get(data_in)
        if vi is None or not vi.type.tensor_type.shape:
            print(f"  [WARN] {n.name}: shape 정보 없음 → 스킵")
            new_nodes.append(n)
            continue

        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        last_dim = dims[-1]  # reduce 대상 차원

        # ones 초기화 (한 번만 생성)
        ones_name = f"__ones_{last_dim}_1__"
        if ones_name not in ones_cache:
            ones_arr  = np.ones((last_dim, 1), dtype=np.float32)
            ones_init = numpy_helper.from_array(ones_arr, name=ones_name)
            new_inits.append(ones_init)
            ones_cache[ones_name] = ones_name

        # MatMul 노드 생성 (output: [B,H,S,1] = keepdims=1 유지)
        matmul_node = helper.make_node(
            "MatMul",
            inputs=[data_in, ones_name],
            outputs=[data_out],
            name=n.name.replace("ReduceSum", "MatMul_ones"),
        )
        new_nodes.append(matmul_node)
        print(f"  ReduceSum→MatMul: {n.name.split('main/')[-1]}  "
              f"[{dims}] axis=-1 (dim={last_dim})")

        # axes Constant 노드도 제거 (opset13 방식일 경우)
        if len(n.input) > 1:
            axes_input = n.input[1]
            if axes_input in const_outputs:
                axes_idx = const_outputs[axes_input]
                remove_nodes.add(axes_idx)

    # ── 2. no-op Cast (float32 → float32) 제거 ────────────────────────────────
    # Reshape → Cast(to=float32) → Conv 패턴에서 Cast를 우회
    final_nodes = []
    cast_bypass = {}  # {cast_output: cast_input}

    for n in new_nodes:
        if n.op_type == "Cast":
            # to dtype 확인
            to_dtype = None
            for attr in n.attribute:
                if attr.name == "to":
                    to_dtype = attr.i
            if to_dtype == TensorProto.FLOAT:  # 1 = float32
                cast_bypass[n.output[0]] = n.input[0]
                print(f"  no-op Cast 제거: {n.name.split('main/')[-1]}  float32→float32")
                continue  # 이 Cast 노드 건너뜀
        final_nodes.append(n)

    # Cast를 참조하는 후속 노드의 입력 이름 교체
    for n in final_nodes:
        for j, inp in enumerate(n.input):
            if inp in cast_bypass:
                n.input[j] = cast_bypass[inp]

    # remove_nodes 처리 (이미 new_nodes 생성 시 ReduceSum 교체됐으므로
    # const_outputs 기반 Constant 노드만 추가 제거)
    final_nodes = [n for i, n in enumerate(final_nodes)
                   if n.name not in {nodes[k].name for k in remove_nodes}]

    # ── 그래프 재구성 ─────────────────────────────────────────────────────────
    del graph.node[:]
    graph.node.extend(final_nodes)

    for init in new_inits:
        graph.initializer.append(init)

    # 검증
    onnx.checker.check_model(model)
    onnx.save(model, dst_path)
    print(f"  → 저장: {dst_path}\n")


MODELS = [
    ("efficientvit_b0_original_r224_nchw.onnx",
     "efficientvit_b0_original_r224_nchw_fixed.onnx"),
    ("efficientvit_b1_original_r224_nchw.onnx",
     "efficientvit_b1_original_r224_nchw_fixed.onnx"),
]

for src_name, dst_name in MODELS:
    src = os.path.join(ONNX_DIR, src_name)
    dst = os.path.join(ONNX_DIR, dst_name)
    if not os.path.exists(src):
        print(f"[SKIP] 없음: {src}")
        continue
    print(f"\n{'='*60}")
    print(f"수정 중: {src_name}")
    print(f"{'='*60}")
    fix_model(src, dst)

print("완료. compile_fp16_original.py 의 ONNX 경로를 _fixed 버전으로 변경 후 재컴파일.")
