# EfficientViT NPU 컴파일 실패 분석 보고서

> **목적**: EfficientViT 모델을 Mobilint Aries2 NPU에 배포하는 과정에서 발생한 양자화 실패 원인 분석 및 해결 시도 기록  
> **작성일**: 2026-05-23  
> **현재 상태**: 파싱·캘리브레이션 통과, 양자화 단계 미해결

---

## 목차

1. [배경 지식: 전체 파이프라인 이해](#1-배경-지식-전체-파이프라인-이해)
2. [대상 모델 및 환경](#2-대상-모델-및-환경)
3. [최종 에러 증상](#3-최종-에러-증상)
4. [문제의 근원: PyTorch → ONNX 변환 시 발생한 버그](#4-문제의-근원-pytorch--onnx-변환-시-발생한-버그)
5. [ONNX 모델 직접 해부](#5-onnx-모델-직접-해부)
6. [파싱 에러 해결 과정 (5회 시도)](#6-파싱-에러-해결-과정-5회-시도)
7. [양자화 에러 해결 시도](#7-양자화-에러-해결-시도)
8. [현재 상태 종합](#8-현재-상태-종합)
9. [남은 선택지](#9-남은-선택지)
10. [참고: 진단 스크립트 목록](#10-참고-진단-스크립트-목록)

---

## 1. 배경 지식: 전체 파이프라인 이해

### NPU 배포 파이프라인 전체 흐름

```
[PyTorch 모델]
      │
      │  torch.onnx.export()
      ▼
[ONNX 파일 (.onnx)]          ← 중간 표현. 연산 그래프를 파일로 저장
      │
      │  mxq_compile()
      ▼
┌─────────────────────────────────────┐
│  qbcompiler (Mobilint 제공 도구)    │
│                                     │
│  1단계: 파싱      ← ONNX 그래프 읽기│
│  2단계: 캘리브레이션 ← 통계 수집    │
│  3단계: 양자화    ← float → int8    │
│  4단계: 코드생성  ← NPU 바이너리    │
└─────────────────────────────────────┘
      │
      ▼
[.mxq 파일]                  ← NPU에서 직접 실행 가능한 바이너리
      │
      ▼
[Aries2 NPU 하드웨어에서 추론]
```

### 양자화(Quantization)란?

신경망은 보통 32비트 부동소수점(float32)으로 학습됩니다.  
NPU는 전력 효율을 위해 **8비트 정수(int8)** 로 연산합니다.

```
float32 값 (예: 0.732)
    │
    │  scale = (max - min) / 255
    │  zero_point = round(-min / scale)
    │
    ▼
int8 값 (예: 94)
```

이 변환 과정에서 `scale`이 0이 되면 **0으로 나누기** 가 발생해 오버플로우가 납니다.

### 캘리브레이션(Calibration)이란?

양자화 파라미터(scale, zero_point)를 결정하기 위해 실제 데이터를 통과시켜 각 레이어의 **값 범위(min, max)** 를 측정하는 단계입니다.  
이 프로젝트에서는 ImageNet 이미지 100장을 사용했습니다.

---

## 2. 대상 모델 및 환경

| 항목 | 내용 |
|------|------|
| 모델 | `efficientvit_b0_r224_nhwc.onnx`, `efficientvit_b1_r224_nhwc.onnx` |
| 입력 형식 | NHWC (Batch, Height, Width, Channel) — NPU 친화적 레이아웃 |
| 입력 크기 | `(1, 224, 224, 3)` |
| 컴파일 도구 | Mobilint `qbcompiler` → `mxq_compile()` |
| 타깃 하드웨어 | Mobilint Aries2 NPU |
| 실행 환경 | Docker 컨테이너 내부 |
| 캘리브레이션 데이터 | `calib_nhwc_final/imagenet_nhwc.txt` (ImageNet 100장) |
| 현재 패치 파일 | `efficientvit_b0_r224_nhwc_patched.onnx` (v5 패치 적용) |

### EfficientViT 모델 구조 (핵심 부분)

EfficientViT는 CNN + 경량 Attention을 결합한 비전 트랜스포머 계열 모델입니다.  
문제가 되는 부분은 `stages.2`의 **context_module** — 즉 글로벌/로컬 Attention 레이어입니다.

```
backbone
└── stages.2
    └── op_list.1
        └── context_module
            └── main
                ├── Q, K, V 계산 (MatMul)
                ├── Attention Score 계산 (MatMul)
                └── [문제 구간] Slice_4, Slice_5
                    ├── Slice_4: 로컬 토큰 16개 추출
                    └── Slice_5: 글로벌 토큰 1개 추출
```

---

## 3. 최종 에러 증상

어떤 방법을 써도 컴파일 3단계(양자화)에서 항상 동일한 에러가 발생했습니다.

```
RuntimeError: Error. quantize failed. zeropoint=-2147483648 is out of range
```

**이 에러의 의미**:

`-2147483648` = `INT32_MIN` = 32비트 정수의 최솟값입니다.  
이 값은 연산 중 **오버플로우**가 발생했음을 의미합니다.

```
zero_point 계산식:
  zero_point = round(-min_val / scale)

정상 케이스:
  min=-1.0, max=1.0 → scale = 2.0/255 ≈ 0.00784
  zero_point = round(1.0 / 0.00784) = 128  ← 정상 범위

문제 케이스:
  scale = 0  (range=0인 상수 텐서)
  zero_point = round(-min_val / 0) = ±∞
  ±∞를 INT32로 캐스팅 → -2147483648 (INT32_MIN)
  → "zeropoint=-2147483648 is out of range"
```

즉, **어딘가에 값의 범위가 0인 텐서가 있고 양자화기가 그것을 양자화하려 시도**하고 있다는 뜻입니다.

---

## 4. 문제의 근원: PyTorch → ONNX 변환 시 발생한 버그

### 4-1. EfficientViT에서 Attention 토큰을 분리하는 코드

EfficientViT의 `ops.py`에서 Q, K, V와 로컬/글로벌 토큰을 분리하는 코드:

```python
# ops.py — context_module attention 내부
qkv = self.qkv(x)                    # shape: (B, C, H*W)
q, k, v = torch.split(qkv, self.dim, dim=1)

# 로컬/글로벌 분리 — 문제가 되는 부분
local_tokens = attn_map[:, :, :-1]   # 마지막 제외
global_token = attn_map[:, :, -1:]   # 마지막 하나만  ← 열린 끝 슬라이스
```

Python에서 `array[-1:]`는 "마지막 원소부터 끝까지"를 의미합니다.

### 4-2. PyTorch Tracer가 이 슬라이스를 ONNX로 변환하는 방법

`torch.onnx.export()` (trace 기반)는 Python 슬라이스를 ONNX `Slice` 노드로 변환합니다.  
이때 "끝까지"를 표현하기 위해 **INT64의 최댓값(INT64_MAX)**을 `ends` 입력으로 삽입합니다.

```
Python:    attn_map[:, :, -1:]
             ↓  torch.onnx.export
ONNX:      Slice(data=attn_map, starts=[-1], ends=[9223372036854775807], axes=[2])
                                                    ^^^^^^^^^^^^^^^^^^^
                                                    INT64_MAX = 2^63 - 1
```

**추가로 발견된 버그**: 이 `ends` 상수(INT64_MAX)가 ONNX 파일에 **FLOAT64** 타입으로 저장됨.  
ONNX Slice 노드 스펙은 `starts`/`ends`가 **INT64**이어야 한다고 명시되어 있어 타입 불일치 발생.

export 당시 이미 경고가 출력되었으나 무시됨:
```
UserWarning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op.
```

### 4-3. 소스 수정 시도 (효과 없음)

처음에는 소스를 수정하면 해결될 것이라 판단하여:

```python
# 수정 전
global_token = attn_map[:, :, -1:]          # 끝 인덱스 생략

# 수정 후
global_token = attn_map[:, :, -1:self.dim]  # 명시적 끝 인덱스
```

그리고 `ops.py:961`의 동적 분기 문제도 수정:
```python
# 수정 전 — TracerWarning 발생
if H * W > self.dim:
    ...

# 수정 후 — 정적 처리
# (트레이싱 시점에 고정되도록 변경)
```

**→ 소스 수정 후 재export했으나 동일한 에러가 계속 발생했습니다.**  
이미 export된 ONNX 파일에도 동일한 문제가 있었고, 재export 후에도 INT64_MAX가 다시 삽입되었습니다.

---

## 5. ONNX 모델 직접 해부

소스 수정이 효과 없자 이미 생성된 ONNX 파일을 파이썬으로 직접 파싱하여 분석했습니다.

### 5-1. inspect_slice.py 실행 결과

```
[INIT] /backbone/stages.2/op_list.1/context_module/main/Constant_17_output_0
      dtype=DOUBLE  value=[-1.0]          ← INT64 -1을 float으로 저장

[INIT] /backbone/stages.2/op_list.1/context_module/main/Constant_28_output_0
      dtype=DOUBLE  value=[9.223372036854776e+18]   ← INT64_MAX를 float으로 저장

[SLICE] Slice_4:
        data   = MatMul_1_output
        starts = Constant_2_output_0  (=0)
        ends   = Constant_17_output_0 (=-1.0, dtype=DOUBLE)

[SLICE] Slice_5:
        data   = MatMul_1_output
        starts = Constant_17_output_0 (=-1.0, dtype=DOUBLE)
        ends   = Constant_28_output_0 (=9.22e+18, dtype=DOUBLE)
```

### 5-2. 실제 모델 그래프 구조 (b1 기준)

```
MatMul_1_output: shape = (1, 16, 17, 196)
                          │   │   │    └─ HW (14×14 feature map)
                          │   │   └─── 토큰 수 (로컬 16개 + 글로벌 1개)
                          │   └─────── Attention heads
                          └─────────── Batch

          ┌─────────────────────────┐
          │     MatMul_1_output     │
          │   (1, 16, 17, 196)      │
          └────────────┬────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
    [Slice_4]                  [Slice_5]
  starts=0, ends=-1           starts=-1, ends=INT64_MAX
  = [0 : 16] 번째 토큰         = [-1 : 끝] = 마지막 토큰
  출력: (1, 16, 16, 196)        출력: (1, 16, 1, 196)
  → 로컬 attention 처리         → 글로벌 attention 처리
```

**핵심 문제 정리**:
- `Constant_17` = `-1` (float64): "마지막 이전까지"의 경계
- `Constant_28` = `9.223e+18` (float64): INT64_MAX, "끝까지"를 표현하려 한 값
- 두 상수 모두 ONNX Slice spec 위반 (INT64이어야 하는데 FLOAT64로 저장)
- `Constant_28`의 9.22e+18은 양자화기가 float 텐서로 인식하면 min=max=9.22e+18 → range=0 → scale=0

---

## 6. 파싱 에러 해결 과정 (5회 시도)

초기에는 양자화 에러보다 먼저 **파싱 에러**가 발생했습니다.  
ONNX 파일을 직접 수정하는 패치 스크립트를 총 5회 작성·실행했습니다.

### 시도 1 — Constant_28을 `Constant_17 × 2`로 교체 (patch_onnx_v1)

**발상**: `-1`이 마지막 인덱스이므로, 그 다음인 `-1 × 2 = -2` 이후로 슬라이스하면 글로벌 토큰 1개만 나올 것

```
Slice_5 = [-1 : -2]
```

**결과**: Python에서 `array[-1:-2]`는 빈 배열입니다. (시작이 끝보다 크거나 같으면 빈 슬라이스)

```
에러:
[W] Error merging shape info: Slice_5_output_0  source:{1,8,0,196}  target:{1,8,1,196}
[E] Non-zero status code: Div node. Can broadcast 0 by 0 or 1. 16 is invalid.
```
dim2 = 0 → 이후 Div 노드에서 "16으로 나누기"가 불가능해 실패.

---

### 시도 2 — `correct_end = starts + 1` 공식 사용 (patch_onnx_v2)

**발상**: `starts = -1`이면 `ends = starts + 1 = -1 + 1 = 0`으로 설정하면 1개만 추출

```
Slice_5 = [-1 : 0]
```

**결과**: 마찬가지로 빈 슬라이스. Python에서 `-1 < 0`이므로 원소가 없음.

```
에러: 동일한 dim2=0 에러
```

---

### 시도 3 — ends를 큰 양수 65536으로 교체 (patch_onnx_v3)

**발상**: ONNX Runtime은 `ends`가 실제 dim_size를 초과하면 자동으로 clamp함.  
즉 `[-1:65536]`은 `[-1:]`과 동일하게 동작.

```
Slice_5 = [-1 : 65536]
         → ONNX Runtime: clamp(65536, max=17) → 17
         → [-1 : 17] = [16 : 17] = 마지막 1개 ← 정상
```

**결과**: ✅ 파싱 통과, ✅ 캘리브레이션 100/100 통과  
**그러나** ❌ 양자화 단계에서 여전히 동일한 에러 발생

---

### 시도 4 — Constant_17을 음수(-1)에서 양수(1)로 변경 (patch_onnx_v4)

**발상**: 음수 인덱스(-1) 자체가 문제일 수 있으므로 양수(1)로 통일

```
Constant_17: -1 → 1
Constant_28: INT64_MAX → 2

Slice_4 = [0 : 1]      ← 기대: [0:-1] = 16개
Slice_5 = [1 : 2]      ← 기대: 1개
```

**결과**: Slice_4가 1개만 반환 → 다음 Conv 레이어와 채널 불일치

```
에러:
Input channels C is not equal to kernel channels * group.
C: 16  kernel channels: 256  group: 1
```

이 실패를 통해 `MatMul_1_output`의 dim2=17 (16+1 구조) 확인됨.  
Constant_17은 음수 그대로 유지해야 함을 확인.

---

### 시도 5 — Constant_28만 INT64 dtype으로 교체, value=17 (patch_onnx_v5)

**발상**:
1. FLOAT64 dtype 자체가 양자화기를 혼란시키는 원인일 수 있음
2. 실제 dim_size(=17)로 교체하면 clamp 없이 정확히 동작

```python
# 패치 내용
Constant_17: 그대로 유지 (-1, float64)  ← Slice_4의 ends, Slice_5의 starts
Constant_28: float64(9.22e18) → int64(17)  ← Slice_5의 ends만 교체

→ Slice_5 = [-1 : 17]  (실제 dim이 17이므로 [-1:] = 마지막 1개)
```

**결과**: ✅ 파싱 통과, ✅ 캘리브레이션 통과  
**여전히** ❌ 동일한 양자화 에러

---

### 파싱 단계 시도 요약표

| 버전 | 패치 내용 | 파싱 | 캘리브레이션 | 양자화 |
|------|-----------|------|-------------|--------|
| 원본 | 패치 없음 | ❌ | - | - |
| v1 | ends = starts×2 | ❌ (빈 슬라이스) | - | - |
| v2 | ends = starts+1 | ❌ (빈 슬라이스) | - | - |
| v3 | ends = 65536 | ✅ | ✅ | ❌ |
| v4 | starts=1, ends=2 | ❌ (채널 불일치) | - | - |
| v5 | ends = int64(17) | ✅ | ✅ | ❌ |

---

## 7. 양자화 에러 해결 시도

파싱 문제는 v3/v5로 해결되었으나, 그 이후 **양자화 단계의 에러가 독립적으로 존재**함을 확인하고 본격 조사.

### 7-1. 핵심 발견: 양자화 파라미터가 전부 무시되고 있었음

기존 `compile.py`에서 사용하던 설정:

```python
mxq_compile(
    model=model_path,
    ...
    is_quant_ch=True,              # ← 채널별 양자화
    is_asym_quant=True,            # ← 비대칭 양자화
    quantize_method="MaxPercentile",
    quantize_percentile=0.9999,
)
```

**`mxq_compile` 실제 시그니처 확인**:

```python
def mxq_compile(model, ..., calibration_config=None, bit_config=None, **kwargs):
    ...
```

`is_quant_ch`, `is_asym_quant`, `quantize_method`, `quantize_percentile` 모두 `**kwargs`로 흡수됩니다.  
`compat.py`(API 호환 레이어)를 분석한 결과, 이 키들에 대한 매핑이 **전혀 없습니다**.  
즉 이 설정들은 **완전히 무시**되고 있었습니다.

**이유**: `qbcompiler`가 구버전 API에서 신버전 API로 교체되면서,  
구버전 파라미터들이 자동 변환 없이 `**kwargs`로만 들어오고 무시되도록 변경됨.

### 7-2. 올바른 API: CalibrationConfig 객체 사용

`models.py`에서 발견한 실제 설정 클래스:

```python
class CalibrationConfig:
    method: int
    # 0 = WChALayer       (채널별 가중치, 레이어별 활성화)
    # 1 = WChAMulti       (채널별 가중치, 다중 활성화)
    # 2 = WChALayerZeropoint  (위 + zero_point 명시)
    # 3 = WChAMultiZeropoint  (위 + zero_point 명시)

    mode: int
    # 0 = Max             (최댓값 기준)
    # 1 = MaxPercentile   (백분위 기준, 이상치 제거)
    # 2 = Histogram       (히스토그램 기준)

    act_scale_min: float = 0.0005   # 활성화 scale의 최솟값 (0 나누기 방지)
    weight_scale_min: float = 1e-6  # 가중치 scale의 최솟값
    max_percentile.percentile: float = 0.9999
```

### 7-3. CalibrationConfig 적용 및 검증

```python
calib_config = CalibrationConfig(method=0, mode=1)
# method=0: Zeropoint 이름이 없는 방식 → zero_point 계산 안 함을 기대
# mode=1:   MaxPercentile → 이상치로 인한 큰 range 억제
# act_scale_min 기본값 0.0005 → scale이 0이 되는 것 방지 기대
```

**`debug_config.py`로 실제 JSON 확인**:

```json
{
  "calibration": {
    "method": 0,
    "actScaleMin": 0.0005,
    "mode": 1
  }
}
```

**`intercept_compile.py`로 C++ 바이너리에 실제 전달 확인**:

```
[INTERCEPT] C++ compile config:
{
  "calibration": {"method": 0, "actScaleMin": 0.0005, "mode": 1},
  ...
}
```

C++까지 올바르게 전달됨을 확인. **그럼에도 동일한 에러 발생.**

### 7-4. 모든 양자화 옵션 시도 결과

| 시도 | 설정 | 기대 효과 | 결과 |
|------|------|-----------|------|
| 구버전 파라미터 | `is_asym_quant=True` | 비대칭 양자화 | ❌ (파라미터 무시됨) |
| 구버전 파라미터 | `quantize_method="Percentile"` | 범위 클리핑 | ❌ (파라미터 무시됨) |
| method 변경 | `CalibrationConfig(method=0)` | zero_point 없는 방식 | ❌ 동일 에러 |
| method 변경 | `CalibrationConfig(method=2)` | zero_point 명시 방식 | ❌ 동일 에러 |
| mode 변경 | `mode=0` (Max) | 단순 최댓값 기준 | ❌ 동일 에러 |
| fallback 확장 | Slice, Gather, Constant 추가 | 인덱스 연산 CPU 오프로딩 | ❌ 동일 에러 |
| fallback_nodes | `Constant_11_output_0` 추가 | 문제 노드 CPU 우회 | ❌ 동일 에러 |

### 7-5. 텐서 전수 검사 (문제 텐서 특정 시도)

**활성화 텐서 검사 (find_bad_tensors.py)**:
- shape inference 실행 후 float 타입 중간 텐서: **187개** 수집
- zero-range (min == max) 텐서: **0개**
- range > 1e5 (극단적 범위) 텐서: **0개**

**가중치 텐서 검사 (check_weights.py)**:
- NaN / Inf 포함 가중치: **없음**
- range < 1e-9 이면서 절댓값 > 10인 가중치: **없음** (scale=0이면서 값 존재 케이스)
- range > 1e4 (극단적 범위) 가중치: **없음**

**→ Python/ONNX Runtime 수준에서는 문제 있는 텐서가 탐지되지 않음.**  
C++ 양자화기가 인식하는 텐서 목록과 우리가 검사한 목록이 다를 가능성 있음.

---

## 8. 현재 상태 종합

### 단계별 성공/실패

```
1단계: ONNX 파싱     ✅  (v3/v5 패치 적용 후 통과)
2단계: 캘리브레이션  ✅  (ImageNet 100장, 100/100 완료)
3단계: 양자화        ❌  (모든 시도에서 동일 에러)
4단계: 코드생성      -   (3단계 실패로 도달 불가)
```

### 에러 발생 메커니즘 (추정)

```
[어떤 텐서 T]
  min_val = max_val = C  (상수 텐서, range = 0)
       │
       ▼
  scale = (max - min) / 255 = 0 / 255 = 0
       │
       ▼
  zero_point = round(-min_val / scale)
             = round(-C / 0)
             = round(±∞)
             = 오버플로우
             → INT32 캐스팅 → -2147483648 (INT32_MIN)
       │
       ▼
  "zeropoint=-2147483648 is out of range"
```

### 원인으로 의심되지만 확인하지 못한 것

**의심 1. Constant_17 (float64, value=-1.0) 텐서**
- 여전히 FLOAT64로 저장됨 (v5 패치에서 Constant_28만 교체, Constant_17은 그대로)
- 양자화기가 이 float 상수 텐서를 양자화하려 시도할 수 있음
- min = max = -1.0 → range = 0 → scale = 0 → 오버플로우
- v5에서 Constant_28은 INT64로 바꿨는데도 에러 지속 → Constant_17이 남은 원인일 수 있음

**의심 2. C++ 양자화기의 내부 동작 불명확**
- 에러 메시지에 문제 노드/텐서 이름이 포함되지 않음
- Python 레벨에서 187개 float 텐서 전수 검사에서 zero-range 없었음
- 가중치 전수 검사에서도 문제 없었음
- **즉, C++ 내부에서 어떤 텐서에 오버플로우가 발생하는지 특정 불가**

**의심 3. method=0이 실제로 zero_point를 생략하는지 불명확**
- 문서상 method 0, 1은 "Zeropoint" 미언급 / method 2, 3은 "Zeropoint" 명시
- 그러나 method=0에서도 동일한 zero_point 관련 에러 발생
- → C++ 구현상 모든 method에서 zero_point를 계산하는 것으로 추정

**의심 4. actScaleMin이 특정 텐서 타입에 적용되지 않음**
- 0.0005가 기본값으로 설정되어 있어 scale=0이 방지될 것으로 기대
- 그러나 에러가 계속 발생 → 가중치 상수나 인덱스 상수에는 이 하한이 적용 안 될 가능성

---

## 9. 남은 선택지

### Option A: context_module 전체 CPU 오프로딩 (단기 해결책)

문제의 `context_module` 관련 노드를 모두 `fallback_nodes`에 지정하여 NPU 대신 CPU에서 실행.  
양자화 자체를 우회하므로 에러가 사라질 것으로 예상.

```python
# compile.py 수정 예시
fallback_nodes = [
    # context_module 관련 모든 노드 패턴
    "/backbone/stages.2/op_list.1/context_module/",  # 전체 블록
]
```

- **장점**: 빠르게 컴파일 통과 가능, `.mxq` 파일 획득
- **단점**: attention 연산이 NPU 대신 CPU에서 실행 → 추론 속도 저하, 전력 효율 감소
- **불확실성**: `fallback_nodes`에 정확히 어떤 이름 형식이 필요한지 시도해봐야 함

### Option B: Constant_17도 INT64로 교체 (patch_onnx_v6)

v5에서 Constant_28만 INT64로 교체했고 Constant_17은 float64로 남아있음.  
Constant_17 (-1, float64)도 INT64(-1)로 교체하면 float 상수 텐서가 사라짐.

```python
# patch_onnx_v6 (미시도)
Constant_17: float64(-1.0) → int64(-1)    ← 이번에 추가
Constant_28: int64(17)                     ← v5에서 이미 적용
```

- **장점**: float dtype 상수 전부 제거 → 가장 깨끗한 ONNX 그래프
- **단점**: 시도해봐야 알 수 있음 (근본 원인이 dtype 문제가 맞다면 해결)

### Option C: PyTorch 소스에서 올바르게 재export

PyTorch 소스(`ops.py`)를 다음과 같이 수정하고 처음부터 재export:

```python
# 수정 1: 열린 슬라이스 → 명시적 끝 인덱스
global_token = attn_map[:, :, -1:self.num_heads]   # 명시적

# 수정 2: 동적 분기 제거
# if H * W > self.dim: 를 torch.jit.script 또는 정적 처리

# export 옵션
torch.onnx.export(
    model, dummy_input, output_path,
    opset_version=13,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    do_constant_folding=True,
)
```

- **장점**: 근본 원인 제거, 향후 다른 변환 도구와도 호환
- **단점**: PyTorch 소스 및 export 환경 필요, 재export 후 NHWC 변환도 다시 필요

### Option D: Mobilint 지원팀 문의

현재 분석 내용과 패치 파일, 에러 로그를 지원팀에 제출.

- **제출 내용**:
  - 이 보고서 전체
  - `efficientvit_b1_r224_nhwc_patched.onnx` (v5 패치)
  - `compile.py` 현재 버전
  - 에러 로그 전문
- **장점**: 컴파일러 버그라면 공식 패치 기대 가능
- **단점**: 응답까지 시간 소요

---

## 10. 참고: 진단 스크립트 목록

| 스크립트 | 목적 | 결과 |
|----------|------|------|
| `inspect_slice.py` | ONNX Slice 노드 및 이상 상수 탐지 | INT64_MAX, FLOAT64 타입 발견 |
| `patch_onnx_v1.py` ~ `v5.py` | ONNX Slice 상수 교체 패치 | v3/v5에서 파싱 통과 |
| `find_bad_tensors.py` | 187개 float 텐서 zero-range 검사 | 이상 없음 |
| `check_weights.py` | 가중치 NaN/Inf/극단값 검사 | 이상 없음 |
| `debug_config.py` | CalibrationConfig JSON 출력 확인 | 올바른 JSON 생성 확인 |
| `intercept_compile.py` | C++에 전달되는 config 인터셉트 | C++까지 올바르게 전달 확인 |

---

## 현재 compile.py 전체 (참고)

```python
import os
import numpy as np
from qbcompiler import mxq_compile
from qbcompiler.configs.models import CalibrationConfig

work_dir = "/workspace/npu"
calib_txt_path = os.path.join(work_dir, "calib_nhwc_final", "imagenet_nhwc.txt")
input_node_name = "input.1_nhwc"
feed_dict = { input_node_name: np.random.randn(1, 224, 224, 3).astype(np.float32) }
in_dformats = { input_node_name: "NHWC" }
models = ["efficientvit_b0_r224_nhwc_patched.onnx", "efficientvit_b1_r224_nhwc_patched.onnx"]

problem_nodes = [
    "/backbone/stages.2/op_list.1/context_module/main/Constant_11_output_0"
]

calib_config = CalibrationConfig(method=0, mode=1)
# method=0: WChALayer (가장 단순한 양자화 방식, Zeropoint 없는 방식으로 기대)
# mode=1:   MaxPercentile (이상치 억제)
# act_scale_min 기본값 0.0005 → scale=0 방지 목적

for model_name in models:
    model_path = os.path.join(work_dir, model_name)
    if not os.path.exists(model_path):
        print("skip", model_name)
        continue
    save_name = model_name.replace(".onnx", ".mxq")
    print(f"=== Compiling {model_name} ===")
    try:
        mxq_compile(
            model=model_path,
            calib_data_path=calib_txt_path,
            backend="onnx",
            save_path=save_name,
            device="cpu",
            target_device="aries2",
            cpu_offload=True,
            feed_dict=feed_dict,
            in_dformats=in_dformats,
            fallback_optypes=[
                "ReduceMean", "GlobalAveragePool", "HardSigmoid", "Div",
                "Transpose", "Reshape", "MatMul", "InputConstant",
                "Constant", "Slice", "Gather"
            ],
            fallback_nodes=problem_nodes,
            calibration_config=calib_config,
        )
        print("+++ SUCCESS:", save_name)
    except Exception as e:
        print("=== COMPILE EXCEPTION ===")
        import traceback; traceback.print_exc()
```

---

*작성일: 2026-05-23*
