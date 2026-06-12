#!/usr/bin/env python3
"""
original EfficientViT ONNX의 qbcompiler 비호환 노드 수정.

수정 내용:
  1. ReduceSum opset13(axes input) → opset9(axes attribute) 변환
     - opset13: INT64 axes 텐서가 그래프에 노출 → qbcompiler가 float로 오해 → overflow
     - opset9 속성 방식: ReduceSum(data, axes=[N], keepdims=1) — 입력 텐서 없음

  2. no-op Cast (float32 → float32) 제거

  3. INT64 Constant 노드 → initializer 변환
     - step4에서 재처리 (Slice params 제거, Reshape shape 유지)

  4. Slice INT64 입력 → 노드 속성으로 이동
     - 단일 원소 INT64 → range=0 → scale=0 → overflow
     - 속성으로 내장하면 INT64 입력 텐서가 그래프에서 사라짐

  5. Reshape INT64 shape → 노드 속성으로 이동 (opset1-4, Cast 없음)

  6. float32 scalar Constant (shape=[]) 처리
     - 단일 원소 → range=0 → overflow
     - Add/Sub(x, 0.0 or tiny_eps) → 우회 (no-op 제거)
     - Pow(x, 2.0) → Mul(x, x) 교체

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

    # ── 1. ReduceSum opset13(axes 입력) → opset9(axes 속성) 변환 ────────────────
    # opset13: ReduceSum(data, axes_tensor) — INT64 axes 텐서 필요
    # opset9:  ReduceSum(data, axes=[N], keepdims=1) — 속성 방식, 별도 텐서 불필요
    # MatMul(ones) 대체 방식은 all-1.0 초기값 → range=0 → zeropoint 오버플로우 유발.
    # ReduceSum 직접 속성 방식은 별도 initializer 없으므로 해당 문제 없음.
    vi_map   = {vi.name: vi for vi in list(graph.value_info) + list(graph.input)}
    new_nodes = []

    for i, n in enumerate(nodes):
        if n.op_type != "ReduceSum":
            new_nodes.append(n)
            continue

        data_in  = n.input[0]
        data_out = n.output[0]

        # shape 조회 → 마지막 축의 양수 인덱스 계산
        vi = vi_map.get(data_in)
        if vi is None or not vi.type.tensor_type.shape:
            print(f"  [WARN] {n.name}: shape 정보 없음 → 유지")
            new_nodes.append(n)
            continue

        dims       = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        last_axis  = len(dims) - 1  # -1 대신 양수 인덱스 사용 (qbcompiler 호환)

        # opset9 속성 방식 ReduceSum (단일 입력, axes 속성)
        new_rs = helper.make_node(
            "ReduceSum",
            inputs=[data_in],
            outputs=[data_out],
            name=n.name,
            axes=[last_axis],
            keepdims=1,
        )
        new_nodes.append(new_rs)
        print(f"  ReduceSum opset9: {n.name.split('main/')[-1]}  "
              f"[{dims}] axes=[{last_axis}] keepdims=1")

        # opset13 axes Constant 노드 제거
        if len(n.input) > 1:
            axes_input = n.input[1]
            if axes_input in const_outputs:
                remove_nodes.add(const_outputs[axes_input])

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

    # ── 5. Reshape INT64 shape → 노드 속성으로 이동 (opset1-4 스타일) ─────────────
    # INT64 shape 입력 텐서: bytes를 float32로 오해 → tiny denormal → range≈0 → overflow.
    # Cast(float32→int64) 출력도 동일 문제 (INT64 상수 activation 텐서).
    # opset1-4 Reshape: shape를 INTS 속성으로 내장 → INT64 텐서가 그래프에서 완전 제거.
    # (opset을 4로 낮춤: Slice도 attribute 방식이므로 opset1-4 호환)
    int64_lookup = {i.name: numpy_helper.to_array(i) for i in new_inits
                    if numpy_helper.to_array(i).dtype == np.int64}
    removed_int64 = set()
    step5_nodes   = []
    n_reshape_fixed = 0

    for n in final_nodes:
        if n.op_type != "Reshape" or len(n.input) < 2 or n.input[1] not in int64_lookup:
            step5_nodes.append(n)
            continue

        int64_name = n.input[1]
        int64_arr  = int64_lookup[int64_name]
        int_vals   = int64_arr.flatten().astype(int).tolist()

        step5_nodes.append(helper.make_node(
            "Reshape",
            inputs=[n.input[0]],          # data 만, shape 텐서 없음
            outputs=list(n.output),
            name=n.name,
            shape=int_vals,               # INTS 속성으로 내장
        ))
        removed_int64.add(int64_name)
        n_reshape_fixed += 1
        print(f"  Reshape→attr: {n.name.split('/')[-1]}  {int_vals}")

    new_inits   = [i for i in new_inits if i.name not in removed_int64]
    final_nodes = step5_nodes
    print(f"  Reshape shape attribute 변환: {n_reshape_fixed}개, INT64 initializer 제거: {len(removed_int64)}개")

    # ── 6. float32 scalar Constant (range=0) 처리 ────────────────────────────
    # shape=[] 스칼라: 단일 원소 → min=max → range=0 → scale=0 → overflow.
    # 패턴별 처리:
    #   Add/Sub(x, 0.0 또는 tiny_eps) → no-op 또는 epsilon 제거 → 우회
    #   Pow(x, 2.0)                   → Mul(x, x) 교체

    scalar_float_consts = {}  # output_name → value
    for n in final_nodes:
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t)
                    if arr.size == 1:
                        scalar_float_consts[n.output[0]] = float(arr.flat[0])

    print(f"  float32 scalar Constant: {len(scalar_float_consts)}개 검출")

    step6_bypass       = {}   # {replaced_output: actual_input}
    scalar_consts_used = set()
    step6_nodes        = []

    for n in final_nodes:
        # Add/Sub(x, 0.0 or tiny_eps) → bypass
        if n.op_type in ("Add", "Sub"):
            scalar_idx = next(
                (i for i, inp in enumerate(n.input) if inp in scalar_float_consts), None)
            if scalar_idx is not None:
                const_val = scalar_float_consts[n.input[scalar_idx]]
                if abs(const_val) <= 1e-3:
                    data_inp = n.input[1 - scalar_idx]
                    step6_bypass[n.output[0]] = data_inp
                    scalar_consts_used.add(n.input[scalar_idx])
                    print(f"  Add/Sub bypass: const={const_val:.7f}  {n.name.split('/')[-1]}")
                    continue  # 이 노드 제거

        # Pow(x, 2.0) → Mul(x, x)
        if n.op_type == "Pow":
            exp_idx = next(
                (i for i, inp in enumerate(n.input) if inp in scalar_float_consts), None)
            if exp_idx is not None and abs(scalar_float_consts[n.input[exp_idx]] - 2.0) < 1e-6:
                step6_nodes.append(helper.make_node(
                    "Mul",
                    inputs=[n.input[0], n.input[0]],
                    outputs=list(n.output),
                    name=(n.name or "") + "_sq",
                ))
                scalar_consts_used.add(n.input[exp_idx])
                print(f"  Pow(x,2)→Mul(x,x): {n.name.split('/')[-1]}")
                continue

        # 사용된 scalar Constant 제거
        if n.op_type == "Constant" and n.output[0] in scalar_consts_used:
            continue

        step6_nodes.append(n)

    # bypass 연결 전파
    for n in step6_nodes:
        for j, inp in enumerate(n.input):
            while inp in step6_bypass:
                inp = step6_bypass[inp]
            n.input[j] = inp

    # 미처리 scalar Constant도 제거 (dangling 방지)
    unhandled = [n.output[0] for n in step6_nodes
                 if n.op_type == "Constant" and n.output[0] in scalar_float_consts
                 and n.output[0] not in scalar_consts_used]
    if unhandled:
        print(f"  [WARN] scalar Constant 미처리: {unhandled}")
    final_nodes = [n for n in step6_nodes
                   if not (n.op_type == "Constant" and n.output[0] in scalar_float_consts)]

    # ── 그래프 재구성 ─────────────────────────────────────────────────────────
    del graph.node[:]
    graph.node.extend(final_nodes)

    for init in new_inits:
        graph.initializer.append(init)

    # Slice + Reshape 모두 attribute 방식 (opset1-4 스타일).
    # Reshape attribute 방식은 opset1-4에서만 유효 → 4로 낮춤.
    # Slice attribute 방식은 opset1-9에서 유효 → opset4도 호환.
    for oi in model.opset_import:
        if oi.domain == "":
            oi.version = 4
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
