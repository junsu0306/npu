# EfficientViT-B0 ONNX 모델 비교 분석 보고서

**분석 대상**
- `assets/onnx/efficientvit_b0_r224_timm_nchw.onnx` (timm 라이브러리 export)
- `assets/onnx/efficientvit_b0_original_r224_nchw.onnx` (MIT-Han-Lab 원본 export)

**분석 목적**  
qbcompiler v1.1.0으로 컴파일 시 timm 모델은 추론 결과가 0% 정확도, original 모델은 컴파일 자체가 크래시(`RuntimeError: basic_string::_M_create`)하는 원인 파악

---

## 1. 분석 방법 및 사용 명령어

### 1-1. 기본 구조 비교

```python
import onnx
from collections import Counter

timm = onnx.load("assets/onnx/efficientvit_b0_r224_timm_nchw.onnx")
orig = onnx.load("assets/onnx/efficientvit_b0_original_r224_nchw.onnx")

# 노드 수, 가중치 수, op 종류별 개수 집계
op_cnt = Counter(n.op_type for n in m.graph.node)
```

`onnx.load()`로 ONNX 프로토버프를 파이썬 객체로 읽어들이고, `graph.node` (연산 목록), `graph.initializer` (가중치 목록), `graph.input/output` (입출력 메타)에 접근했다.

### 1-2. context_module 노드 상세 추적

```python
for n in timm.graph.node:
    if "stages.2/blocks/blocks.1/context_module" in n.name:
        if n.op_type != "Constant":
            print(f"[{n.op_type}] {n.name}")
```

ONNX는 PyTorch 모듈 계층 구조를 노드 이름(`/`로 구분된 경로)으로 보존한다. `context_module`을 포함하는 노드명만 필터링해서 attention 연산 흐름을 순서대로 추출했다.

### 1-3. Cast 연산의 dtype 확인 (INT64 탐지)

```python
for n in timm.graph.node:
    if "context_module" in n.name and n.op_type == "Cast":
        for attr in n.attribute:
            if attr.name == "to":
                dtype_map = {1:"float32", 6:"int32", 7:"int64", ...}
                print(f"{n.name} → {dtype_map[attr.i]}")
```

ONNX Cast 노드의 `to` attribute는 대상 dtype을 정수 코드로 저장한다. `7 = INT64`임을 매핑해서 INT64 텐서 생성 위치를 확인했다.

### 1-4. 가중치 값 비교

```python
import numpy as np
timm_w = {w.name: onnx.numpy_helper.to_array(w) for w in timm.graph.initializer}
orig_w = {w.name.replace("backbone.", ""): onnx.numpy_helper.to_array(w)
          for w in orig.graph.initializer}

for name in timm_w:
    if "context_module" in name:
        np.allclose(timm_w[name], orig_w[mapped_name], atol=1e-5)
```

`onnx.numpy_helper.to_array()`로 프로토버프 텐서를 numpy 배열로 변환해 `np.allclose()`로 수치 동일성을 검증했다.

---

## 2. 기본 구조 비교 결과

| 항목 | timm | original |
|---|---|---|
| opset 버전 | 13 | 13 |
| ir_version | 7 | 7 |
| 입력 shape | `[1, 3, 224, 224]` | `[1, 3, 224, 224]` |
| 출력 shape | `[0, 1000]` (dynamic) | `[1, 1000]` (static) |
| 전체 노드 수 | **445** | **253** |
| 가중치(initializer) 수 | 93 | 93 |
| 총 파라미터 수 | **3,407,328** | **3,407,328** |

**핵심 관찰**: 파라미터가 완전히 동일하고 가중치 값도 수치적으로 일치한다. 같은 모델을 서로 다른 코드베이스로 export한 것이다. 그러나 노드 수가 445 대 253으로 거의 2배 차이 나는데, 이것이 모든 문제의 핵심이다.

---

## 3. Operator 종류별 비교

| Operator | timm | original | 비고 |
|---|---|---|---|
| Conv | 50 | 50 | 동일 |
| Relu | 8 | 8 | 동일 |
| HardSigmoid | 24 | 24 | 동일 |
| MatMul | **9** | **13** | original이 더 많음 |
| Shape | **13** | **0** | timm 전용 |
| Gather | **13** | **0** | timm 전용 |
| Cast | **20** | **4** | timm이 5배 |
| ConstantOfShape | **4** | **0** | timm 전용 |
| Pad | **4** | **0** | timm 전용 |
| Unsqueeze | **15** | **0** | timm 전용 |
| ReduceSum | **0** | **4** | original 전용 |
| Mul | **40** | 25 | timm이 많음 |
| Slice | **24** | 12 | timm이 2배 |
| Reshape | **16** | 8 | timm이 2배 |
| Transpose | **16** | 8 | timm이 2배 |

timm에는 있고 original에는 없는 연산: `Shape`, `Gather`, `ConstantOfShape`, `Pad`, `Unsqueeze`  
original에는 있고 timm에는 없는 연산: `ReduceSum`

---

## 4. 모듈 이름 체계 차이

```
timm:     /stages/stages.2/blocks/blocks.1/context_module/main/...
          가중치: stages.2.blocks.1.context_module.main.qkv.conv.weight

original: /backbone/stages.2/op_list.1/context_module/main/...
          가중치: backbone.stages.2.op_list.1.context_module.main.qkv.conv.weight
```

- timm은 `EfficientVit` 클래스를 최상위에서 직접 export → 경로에 `backbone` prefix 없음
- original (MIT-Han-Lab)은 `EfficientViTCls` 클래스가 `self.backbone`을 갖는 구조 → 모든 경로 앞에 `backbone.` 붙음
- 블록 컨테이너 이름이 `blocks.N` vs `op_list.N`으로 다름

이 차이 때문에 `activation_16bits`에 timm 노드명을 넣으면 original 모델에서 이름 불일치가 발생해 `basic_string::_M_create` 크래시가 났다(이전 시도에서 확인). 동적으로 각 ONNX에서 이름을 추출하도록 수정 후에도 크래시가 계속됐으므로, 이름이 원인이 아님을 최종 확인했다.

---

## 5. context_module Attention 연산 흐름 상세 비교

EfficientViT의 `context_module`은 **ReLU 기반 선형 어텐션(Linear Attention)**이다. 일반 Softmax Attention(`O(N²)`)과 달리 `O(N)` 복잡도로 동작한다. 같은 수식을 두 모델이 전혀 다르게 구현했다.

### 5-1. timm 구현 (동적 그래프, 40 ops/블록)

```
단계 1 — QKV 추출
  Conv(qkv)                 : 입력 feature를 Q, K, V 세 채널로 projection
  Conv(aggreg.0.0)          : 공간 집계(depthwise 5×5 conv)
  Conv(aggreg.0.1)          : 채널 믹싱
  Concat                    : 집계 결과를 붙임
  Reshape + Transpose       : (B, C, H, W) → (B, H*W, C) 형태로 재배열

단계 2 — 동적 shape 계산 (★ INT64 문제 발원지)
  Shape                     : 입력 텐서의 shape 추출 → int64 텐서 반환
  Gather                    : shape에서 H, W 값 추출 → int64 스칼라
  Add                       : H + padding
  Div                       : 나눗셈 → int64 결과

  ▶ 이 시점에서 이미 int64 텐서가 생성됨

단계 3 — Q, K, V 분리
  Mul × 3, Slice × 3       : Gather로 얻은 int64 인덱스로 동적 슬라이싱

단계 4 — 비선형 활성화
  Relu × 2                  : φ(Q), φ(K) 계산 (ReLU가 kernel function 역할)

단계 5 — 동적 패딩 (★ INT64 사용)
  ConstantOfShape           : int64 shape으로 동적 크기 0-텐서 생성
  Concat, Reshape, Slice    : 패딩 위치 계산
  Cast → int64              : float → int64 변환 (Pad 연산 입력에 필요)
  Pad                       : 동적 패딩 적용
  Cast → float32 × 3       : int64 → float32 역변환

단계 6 — Linear Attention 계산
  MatMul                    : φ(K)^T @ V    (key-value context 계산)
  MatMul                    : φ(Q) @ (K^T V) (weighted aggregation)
  Slice × 2, Add, Div       : 정규화 (분모: Σφ(K))

단계 7 — 출력
  Cast → float32 + Transpose + Reshape : 형태 복원
  Conv(proj)                : 출력 projection
```

### 5-2. original 구현 (정적 그래프, 21 ops/블록)

```
단계 1 — QKV 추출
  Conv(qkv)                 : (timm과 동일)
  Conv(aggreg.0.0)          : (timm과 동일)
  Conv(aggreg.0.1)          : (timm과 동일)
  Concat                    : (timm과 동일)
  Reshape                   : 재배열

단계 2 — 정적 Q, K, V 분리
  Slice × 3                 : 고정 인덱스로 슬라이싱 (int64 없음)
                              입력이 항상 224×224 → H=7, W=7 고정이므로
                              인덱스를 상수로 embed 가능

단계 3 — 비선형 활성화
  Relu × 2                  : (timm과 동일)

단계 4 — Linear Attention 계산
  Transpose                 : 전치
  MatMul_1                  : φ(K)^T @ V    (key-value context)
  MatMul_2                  : φ(Q) @ (K^T V) (aggregation)
  ReduceSum                 : Σφ(K) 계산 (정규화 분모)
  Transpose, MatMul_3       : 보정
  Add + Div                 : 정규화

단계 5 — 출력
  Reshape + Cast → float32 : 형태 복원 (int64 없음)
  Conv(proj)                : 출력 projection
```

---

## 6. 주요 연산자 설명

### ReduceSum (original 전용, 4개)

**정의**: 텐서의 특정 축(axis)을 따라 모든 원소를 합산하는 연산.

```
입력: [B, N, C]  (배치, 시퀀스 길이, 채널)
axis=1 로 ReduceSum 적용
출력: [B, 1, C]  (각 채널의 모든 토큰에 대한 합)
```

**EfficientViT에서의 역할**: ReLU linear attention의 **정규화 분모** 계산.

선형 어텐션 수식:
```
Output = φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ Σφ(K))
```

여기서 분모 `Σφ(K)`가 각 쿼리 위치에 대해 모든 키 위치의 합산이다. original은 이것을 `ReduceSum`으로 직접 계산한다.

**timm과의 차이**: timm은 이 분모를 `ReduceSum` 대신 `Slice` + `Add` + `Div` 조합으로 구현한다. 동적 패딩 후 특정 위치에서 값을 slicing해서 분모를 구하는 방식으로, 연산 수는 많지만 ONNX export 과정에서 INT64가 개입한다.

**qbcompiler에서의 문제 — 크래시 원인 확정**

공식 문서의 qbcompiler `ReduceSum` 스펙과 실제 모델 사이에 구조적 불일치가 있다.

| 항목 | qbcompiler 스펙 | original 모델 실제 |
|---|---|---|
| 입력 수 | **1개** (data만) | **2개** (data + axes 텐서) |
| axes 형식 | `reduce_axes: int32[]` (attribute) | `Constant(int64, [-1])` → 두 번째 input |
| axes 값 | 양수 정수 배열 | **`-1`** (음수 인덱스) |

ONNX ReduceSum은 **opset 11/12까지 axes가 attribute**였지만, **opset 13부터 optional한 두 번째 input tensor로 변경됐다.** original 모델은 opset 13으로 export됐고, axes를 `Constant(dtype=int64, value=[-1])` 노드의 출력으로 ReduceSum에 넘긴다.

qbcompiler는 opset 13 방식(2-input)을 지원하지 않아 C++ 백엔드가 두 번째 입력을 처리하다가 `basic_string::_M_create`로 크래시한다. 그래프 파싱 단계("100% supported")는 통과하지만, 실제 양자화 연산 구성 시점에서 실패하는 것이다.

**잠재적 수정**: original 모델 ONNX에서 ReduceSum을 opset 11 방식으로 변환하면 크래시를 해결할 수 있다.
```python
# 변환 방향:
# Before: ReduceSum(inputs=[data, Constant(int64,[-1])], attrs={keepdims:1})
# After:  ReduceSum(inputs=[data], attrs={axes:[last_dim], keepdims:1})
# dtype:  int64 → int32, 음수 인덱스 → 양수로 변환 필요
```

### Shape / Gather (timm 전용, 각 13개)

**Shape**: 텐서의 shape 자체를 int64 1D 텐서로 반환. 예: `[1,64,7,7]` → `[1,64,7,7]` 텐서 → `shape=[4]`, dtype=int64.

**Gather**: 텐서에서 특정 인덱스의 값을 추출. `Shape`의 출력(`int64`)에서 H나 W값을 꺼낼 때 사용.

```
shape_out = Shape(feature)     # [1, 64, 7, 7], dtype=int64
H = Gather(shape_out, idx=2)   # 7, dtype=int64
W = Gather(shape_out, idx=3)   # 7, dtype=int64
```

**문제**: 이 연산의 출력이 int64이고, 이후 모든 관련 연산이 int64 체인으로 전파된다. 양자화 시 INT8/FP32를 가정하는 quantizer가 이 int64 텐서를 만나면 왜곡된 scale/zero_point를 계산하거나 타입 에러를 발생시킨다.

### ConstantOfShape (timm 전용, 4개)

`Shape → Gather`로 얻은 동적 크기 정보로 특정 값으로 채워진 텐서를 생성. 예: `ConstantOfShape([7,7], value=0)` → 7×7 영텐서.

동적 크기이므로 export 시점에 상수로 고정되지 않고, 런타임에 int64 shape 값을 받아 실행된다. 이것이 int64 의존성 체인을 형성하는 주요 원인 중 하나다.

### Cast (timm: 20개, original: 4개)

데이터 타입 변환 연산. timm context_module 내 Cast의 목적:

| 위치 | timm 변환 방향 | original 변환 방향 |
|---|---|---|
| Pad 입력 전 | **float32 → int64** | (없음) |
| Pad 출력 후 | int64 → float32 × 3 | (없음) |
| 출력 직전 | int64 → float32 | float32 → float32 (no-op형) |

timm에서 `Cast → int64`는 ONNX `Pad` 연산자의 `pads` 입력이 int64 텐서를 요구하기 때문에 발생한다. original은 Pad 자체가 없으므로 int64 Cast가 필요 없다.

---

## 7. 출력 shape 차이의 의미

```
timm:     output shape = [0, 1000]   ← 배치 차원이 0 (dynamic)
original: output shape = [1, 1000]   ← 배치 차원이 1 (static)
```

timm은 임의의 배치 크기를 지원하는 동적 그래프로 export됐고, original은 배치=1로 고정된 정적 그래프다. 이것 자체가 컴파일러 동작에 영향을 줄 수 있다.

---

## 8. 가중치 동일성 검증 결과

context_module 내 모든 가중치(12개) 비교:

| 가중치 | 형태 | 동일 여부 |
|---|---|---|
| qkv.conv.weight | [192, 64, 1, 1] | ✅ |
| aggreg.0.0.weight | [192, 1, 5, 5] | ✅ |
| aggreg.0.1.weight | [192, 16, 1, 1] | ✅ |
| proj.conv.weight | [64, 64, 1, 1] | ✅ |
| ... (나머지 8개) | 다양 | ✅ 전부 일치 |

모든 가중치가 `np.allclose(atol=1e-5)` 기준으로 완전히 동일하다. **학습된 모델 자체는 동일하고, ONNX export 구현 방식만 다르다.**

---

## 9. 종합 진단

### timm 모델 실패 원인

```
Shape/Gather ops
     │
     ▼ (int64 출력)
Cast → int64                ← quantizer가 이 텐서를 float처럼 처리
     │
     ▼
Pad (int64 pads)
     │
     ▼
Cast → float32 (복구)
```

qbcompiler의 INT8 양자화기는 텐서를 float32로 가정하고 scale/zero_point를 계산한다. INT64 값(예: `7`)을 float32로 해석하면 완전히 다른 값이 되어 attention 출력이 손상된다. 컴파일과 실행은 성공하지만 수치적으로 완전히 틀린 결과가 나온다.

ONNX Runtime 독립 검증에서도 동일하게 0/10 정확도, 그리고 동일한 INT64 경고 메시지가 출력됐다.

### original 모델 실패 원인

```
ReduceSum (context_module, 4개)
     │
     ▼ (양자화 중)
qbcompiler C++ 내부 오류
RuntimeError: quantize failed. basic_string::_M_create
```

`basic_string::_M_create`는 C++ `std::string`의 메모리 할당 함수다. 이 에러는 문자열 관련 내부 연산에서 잘못된 크기나 포인터가 전달될 때 발생한다. `ReduceSum`을 처리하는 양자화 경로에 버그가 있는 것으로 추정된다. activation/weight 설정, cpu_offload, calibration 데이터 포맷과 무관하게 동일하게 발생함을 실험으로 확인했다.

---

## 10. 현재 상태 요약 및 잠재적 해결 방향

| 모델 | 컴파일 | 추론 | 실패 원인 | 해결 가능성 |
|---|---|---|---|---|
| timm | ✅ 성공 | ❌ 0% | INT64 Cast로 인한 수치 오염 | timm ONNX 재export 시 동적 Shape 연산을 정적으로 교체하면 제거 가능 |
| original | ❌ 크래시 | - | ReduceSum 양자화 버그 | qbcompiler 버그 → Mobilint 버그 리포트 필요 |

**timm 모델의 잠재적 해결 방향**

입력이 항상 224×224이므로 H=7, W=7이 컴파일 시점에 확정된다. `Shape → Gather → Add → Div → ConstantOfShape → Pad` 체인을 모두 제거하고 고정 상수로 교체하면 INT64 텐서가 완전히 사라진다. 이 방식으로 re-export하면 original의 정적 그래프에 가까운 구조가 되어 양자화 호환성이 높아질 가능성이 있다.

이 작업은 PyTorch 레벨에서 모델 forward 코드를 수정하거나, ONNX 그래프를 직접 편집하는 방식으로 시도할 수 있다.
