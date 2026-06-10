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

  3. INT64 Constant 노드 → initializer 변환
     - Reshape shape 등 INT64 Constant 노드를 initializer로 변환
     - (Slice 파라미터는 step4에서 속성으로 이동하므로 최종적으로 제거됨)

  4. Slice INT64 입력 → 노드 속성으로 이동
     - Slice의 starts/ends/axes(shape=[1] INT64)가 quantizer에서 float로 오해
     - 단일 원소 → min==max → range=0 → scale=0 → zeropoint=-2147483648 오버플로우
     - 속성으로 내장하면 INT64 입력 텐서 자체가 그래프에서 사라짐
     - 모델 opset을 9로 낮춤 (opset9 Slice는 속성 방식 지원)

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
    after_cast = []
    cast_bypass = {}  # {cast_output: cast_input}

    for n in new_nodes:
        if n.op_type == "Cast":
            to_dtype = None
            for attr in n.attribute:
                if attr.name == "to":
                    to_dtype = attr.i
            if to_dtype == TensorProto.FLOAT:  # 1 = float32
                cast_bypass[n.output[0]] = n.input[0]
                print(f"  no-op Cast 제거: {n.name.split('main/')[-1]}  float32→float32")
                continue
        after_cast.append(n)

    for n in after_cast:
        for j, inp in enumerate(n.input):
            if inp in cast_bypass:
                n.input[j] = cast_bypass[inp]

    # remove_nodes 처리
    after_cast = [n for n in after_cast
                  if n.name not in {nodes[k].name for k in remove_nodes}]

    # ── 3. INT64 Constant 노드 → initializer 변환 ─────────────────────────────
    # Reshape shape / Slice starts·ends·axes 등이 Constant(INT64) 노드로 존재.
    # qbcompiler quantizer가 이 출력 텐서를 float로 오해
    # → range=0 → scale=0 → zeropoint=-2147483648 오버플로우.
    # initializer로 올리면 dtype 체크 후 양자화 건너뜀.
    final_nodes = []
    n_converted = 0

    for n in after_cast:
        if n.op_type == "Constant":
            converted = False
            for attr in n.attribute:
                if attr.name == "value" and attr.t.data_type != TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t)
                    init = numpy_helper.from_array(arr, name=n.output[0])
                    new_inits.append(init)
                    n_converted += 1
                    converted = True
                    break
            if not converted:
                final_nodes.append(n)
        else:
            final_nodes.append(n)

    print(f"  INT64 Constant→initializer: {n_converted}개")

    # ── 4. Slice INT64 입력 → 노드 속성으로 이동 ─────────────────────────────────
    # shape=[1] INT64 initializer: 단일 원소 → min==max → range=0 → scale=0
    # → zeropoint=-2147483648 오버플로우.
    # starts/ends/axes를 노드 속성으로 내장하면 INT64 입력 텐서가 그래프에서 사라짐.
    new_init_lookup = {i.name: numpy_helper.to_array(i) for i in new_inits}
    slice_params_used = set()
    step4_nodes = []

    for n in final_nodes:
        if n.op_type != "Slice":
            step4_nodes.append(n)
            continue

        def get_arr(idx):
            if idx >= len(n.input) or not n.input[idx]:
                return None
            return new_init_lookup.get(n.input[idx])

        starts_arr = get_arr(1)
        ends_arr   = get_arr(2)
        axes_arr   = get_arr(3)

        if starts_arr is None or ends_arr is None:
            print(f"  [WARN] Slice {n.name}: 파라미터 조회 실패 → 유지")
            step4_nodes.append(n)
            continue

        for j in range(1, len(n.input)):
            if n.input[j]:
                slice_params_used.add(n.input[j])

        starts = starts_arr.flatten().astype(int).tolist()
        ends   = ends_arr.flatten().astype(int).tolist()
        attrs  = dict(starts=starts, ends=ends)
        if axes_arr is not None:
            attrs["axes"] = axes_arr.flatten().astype(int).tolist()

        # INT64 inputs 없이 data만 입력받는 Slice 노드
        step4_nodes.append(helper.make_node(
            "Slice",
            inputs=[n.input[0]],
            outputs=list(n.output),
            name=n.name,
            **attrs,
        ))
        print(f"  Slice→attrs: {n.name.split('/')[-1]}"
              f"  starts={starts} ends={ends} axes={attrs.get('axes')}")

    final_nodes = step4_nodes
    new_inits = [i for i in new_inits if i.name not in slice_params_used]
    print(f"  Slice INT64 initializer 제거: {len(slice_params_used)}개 "
          f"(Reshape shape 잔여: {len(new_inits)}개)")

    # ── 그래프 재구성 ─────────────────────────────────────────────────────────
    del graph.node[:]
    graph.node.extend(final_nodes)

    for init in new_inits:
        graph.initializer.append(init)

    # Slice가 opset9 속성 방식으로 바뀌었으므로 모델 opset을 9로 낮춤
    # (opset13 Slice는 속성 방식 미지원 → checker 오류 방지)
    for oi in model.opset_import:
        if oi.domain == "":
            oi.version = 9
            break

    try:
        onnx.checker.check_model(model)
        print("  ONNX 검증 통과")
    except Exception as e:
        print(f"  [WARN] ONNX 검증 경고 (qbcompiler 컴파일에는 무방할 수 있음): {e}")
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
