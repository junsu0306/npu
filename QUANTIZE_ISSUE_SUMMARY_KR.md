# EfficientViT NPU 컴파일 — 실패 분석 및 해결 보고서

> **목적**: EfficientViT 모델을 Mobilint Aries2 NPU에 배포하는 과정에서 발생한 양자화 실패 원인 분석 및 최종 해결  
> **작성일**: 2026-05-27  
> **최종 상태**: ✅ MXQ 생성 성공

---

## 목차

1. [배경: 전체 파이프라인 이해](#1-배경-전체-파이프라인-이해)
2. [대상 모델 및 환경](#2-대상-모델-및-환경)
3. [1차 시도: NHWC 변환 경로 (실패)](#3-1차-시도-nhwc-변환-경로-실패)
4. [실패 원인 심층 분석](#4-실패-원인-심층-분석)
5. [2차 시도: timm 직접 export 경로 (성공)](#5-2차-시도-timm-직접-export-경로-성공)
6. [왜 NCHW 경로는 바로 됐나: 근본 원인 정리](#6-왜-nchw-경로는-바로-됐나-근본-원인-정리)
7. [최종 파이프라인 및 파일 구성](#7-최종-파이프라인-및-파일-구성)
8. [교훈](#8-교훈)

---

## 1. 배경: 전체 파이프라인 이해

### NPU 배포 흐름

```
[PyTorch 모델]
      │  torch.onnx.export()
      ▼
[ONNX 파일 (.onnx)]
      │  mxq_compile()
      ▼
┌──────────────────────────────────┐
│  qbcompiler                      │
│  1단계: 파싱    (그래프 분석)    │
│  2단계: 캘리브  (값 범위 측정)  │
│  3단계: 양자화  (float → int8)  │
└──────────────────────────────────┘
      │
      ▼
[.mxq 파일]  →  Aries2 NPU에서 실행 (qbruntime)
```

### 양자화란?

NPU는 전력 효율을 위해 float32 대신 **int8**로 연산합니다.

```
scale     = (max - min) / 255
zero_point = round(-min / scale)
```

`scale = 0` 이 되면 0으로 나누기 → `zero_point = ±∞` → int32로 캐스팅 시 `-2147483648 (INT32_MIN)` 이 됩니다.

---

## 2. 대상 모델 및 환경

| 항목 | 내용 |
|------|------|
| 모델 | EfficientViT-B0 / B1 (224×224, ImageNet-1k) |
| 출처 | `timm/efficientvit_b0.r224_in1k` |
| 컴파일 도구 | Mobilint `qbcompiler` → `mxq_compile()` |
| 런타임 | Mobilint `qbruntime` |
| 타깃 하드웨어 | Mobilint Aries2 NPU (Jetson 탑재) |
| 실행 환경 | Docker 컨테이너 (컴파일) / Jetson (추론) |

---

## 3. 1차 시도: NHWC 변환 경로 (실패)

### 3-1. 시도한 파이프라인

```
timm PyTorch 모델
      │  torch.onnx.export()
      ▼
efficientvit_b0_r224.onnx  (NCHW, 표준)
      │  NHWC 변환 도구
      ▼
efficientvit_b0_r224_nhwc.onnx  (NHWC 레이아웃으로 변환)
      │  mxq_compile()
      ▼
❌ 양자화 단계에서 실패
```

NPU가 NHWC 입력을 선호한다는 판단 하에, 별도의 NHWC 변환 도구를 거쳐 ONNX를 변환한 후 컴파일 시도.

### 3-2. 발생한 에러

파싱·캘리브레이션은 통과하지만 양자화 단계에서 항상 동일한 에러:

```
RuntimeError: Error. quantize failed. zeropoint=-2147483648 is out of range
```

`-2147483648 = INT32_MIN` → **zero_point 오버플로우** 발생.

### 3-3. 해결 시도 목록 (모두 실패)

**ONNX 그래프 패치 (5회)**

| 버전 | 내용 | 결과 |
|------|------|------|
| v1 | `ends = starts × 2` | ❌ 빈 슬라이스 (dim=0) |
| v2 | `ends = starts + 1` | ❌ 빈 슬라이스 (dim=0) |
| v3 | `ends = 65536` | ✅ 파싱 통과, ❌ 양자화 실패 |
| v4 | `Constant_17 = 1, Constant_28 = 2` | ❌ Conv 채널 불일치 |
| v5 | `Constant_28 = int64(17)` | ✅ 파싱 통과, ❌ 양자화 실패 |

**양자화 설정 변경 (모두 실패)**

| 시도 | 내용 |
|------|------|
| 구버전 파라미터 `is_asym_quant`, `quantize_method` 등 | `**kwargs`로 흡수되어 **완전 무시**됨 (API 변경) |
| `CalibrationConfig(method=0, mode=1)` | 올바른 JSON이 C++까지 전달 확인, 그럼에도 동일 에러 |
| `fallback_optypes`에 Slice, Gather, Constant 추가 | 동일 에러 |
| 활성화 텐서 187개 전수 검사 | zero-range 없음 |
| 가중치 전수 검사 | NaN/Inf/극단값 없음 |

---

## 4. 실패 원인 심층 분석

### 4-1. NHWC 변환 도구가 만든 버그

`inspect_slice.py`로 `efficientvit_b0_r224_nhwc.onnx` 를 직접 분석한 결과:

```
[INIT] Constant_28_output_0
  dtype = DOUBLE (float64)        ← ❌ 잘못된 타입
  value = [9.223372036854776e+18] ← INT64_MAX를 float으로 저장

[INIT] Constant_17_output_0
  dtype = DOUBLE (float64)        ← ❌ 잘못된 타입
  value = [-1.0]
```

**ONNX 스펙상 Slice 노드의 `starts`/`ends` 입력은 반드시 INT64이어야 합니다.**  
그런데 NHWC 변환 도구가 그래프를 재파싱·재구성하는 과정에서 이 상수들을 **FLOAT64(DOUBLE)로 잘못 저장**했습니다.

### 4-2. FLOAT64 상수가 양자화를 망가뜨리는 경로

```
NHWC 변환 도구
  INT64_MAX (9223372036854775807)
       │  float64로 저장
       ▼
  FLOAT64 상수 텐서  ← 양자화기가 "float 텐서"로 인식
       │
       ▼
  min = max = 9.22e18  (상수이므로 range = 0)
       │
       ▼
  scale = (max - min) / 255 = 0
       │
       ▼
  zero_point = round(-min / scale) = round(-9.22e18 / 0) = ±∞
       │
       ▼
  int32 캐스팅 → -2,147,483,648 (INT32_MIN)
       │
       ▼
  "zeropoint=-2147483648 is out of range"
```

### 4-3. 왜 패치로는 해결이 안 됐나

- v5 패치에서 `Constant_28`은 INT64(17)로 바꿨지만 `Constant_17`은 여전히 FLOAT64(-1.0)로 남아 있었음
- 즉 **두 상수 중 하나만 고쳤던 것**
- 또한 NHWC 변환 도구가 다른 곳에도 유사한 dtype 오염을 발생시켰을 가능성 있음
- 컴파일러의 에러 메시지에 노드 이름이 포함되지 않아 정확한 원인 텐서를 특정하기 어려웠음

---

## 5. 2차 시도: timm 직접 export 경로 (성공)

### 5-1. 핵심 아이디어

> NHWC 변환 도구를 완전히 제거하고,  
> timm에서 표준 NCHW ONNX로 직접 export한다.

`torch.onnx.export`는 Slice 상수를 **INT64로 올바르게 저장**합니다.  
NHWC 변환이라는 중간 단계가 없으면 dtype 오염이 발생하지 않습니다.

### 5-2. 새 파이프라인

```
timm.create_model('efficientvit_b0.r224_in1k', pretrained=True)
      │  torch.onnx.export(..., opset_version=13)
      ▼
efficientvit_b0_r224_nchw.onnx
  → Slice 상수 dtype 확인: FLOAT64 없음 ✅

      │  mxq_compile(in_dformats={"input": "NCHW"}, cpu_offload=True)
      ▼
efficientvit_b0_r224_nchw.mxq  ✅
```

### 5-3. 달라진 compile 설정

| 항목 | 1차 (실패) | 2차 (성공) |
|------|-----------|-----------|
| ONNX 생성 방법 | NHWC 변환 도구 경유 | `torch.onnx.export` 직접 |
| 입력 이름 | `input.1_nhwc` | `input` |
| `in_dformats` | `"NHWC"` | `"NCHW"` |
| 입력 shape | `(1, 224, 224, 3)` | `(1, 3, 224, 224)` |
| Slice 상수 dtype | FLOAT64 (버그) | INT64 (정상) |
| ONNX 패치 필요 | 5회 시도 | **불필요** |
| 양자화 결과 | ❌ 항상 실패 | ✅ 성공 |

### 5-4. 전처리 변경 (컴파일·추론 동일하게 유지)

```python
# 1차: NHWC 캘리브레이션 데이터
array shape: (224, 224, 3)  HWC

# 2차: NCHW 캘리브레이션 데이터
array shape: (3, 224, 224)  CHW  ← HWC를 .transpose(2,0,1) 으로 변환
```

---

## 6. 왜 NCHW 경로는 바로 됐나: 근본 원인 정리

**결론: 문제는 EfficientViT 모델 자체가 아니라 NHWC 변환 도구였습니다.**

```
[원인 체인]

NHWC 변환 도구가 ONNX 그래프 재파싱
      │
      ▼
INT64 상수 노드를 재생성할 때 dtype을 FLOAT64로 잘못 지정
      │
      ▼
qbcompiler 양자화기가 FLOAT64 텐서를 "양자화 대상 float 텐서"로 인식
      │
      ▼
상수 텐서이므로 min = max → range = 0 → scale = 0
      │
      ▼
zero_point = -min / scale = ±∞  →  INT32 오버플로우
      │
      ▼
"zeropoint=-2147483648 is out of range"
```

**왜 NCHW 직접 export는 이 문제가 없었나?**

`torch.onnx.export`는 Python의 `array[-1:]` 슬라이스를 ONNX로 변환할 때  
`ends = INT64_MAX`를 **INT64 타입 상수**로 정확히 저장합니다.  
qbcompiler는 INT64 상수를 float 텐서가 아닌 **정수 인덱스**로 인식하므로 양자화 대상에서 제외됩니다.

---

## 7. 최종 파이프라인 및 파일 구성

### 파일 역할

| 파일 | 실행 환경 | 역할 |
|------|-----------|------|
| `export_timm_to_onnx.py` | 로컬 (timm 설치) | timm → NCHW ONNX 변환 + 캘리브레이션 데이터 생성 |
| `compile_nchw.py` | Docker 컨테이너 | NCHW ONNX → MXQ 컴파일 |
| `inference_npu.py` | Jetson (qbruntime) | MXQ 파일 NPU 추론 |

### 전체 실행 순서

```bash
# [로컬] ONNX export + 캘리브레이션 데이터 생성
pip install timm onnx
python3 export_timm_to_onnx.py
# 출력: efficientvit_b0_r224_nchw.onnx, calib_nchw.txt, calib_nchw/

# [Docker] MXQ 컴파일
python3 /workspace/npu/compile_nchw.py
# 출력: efficientvit_b0_r224_nchw.mxq

# [Jetson] NPU 추론
python3 inference_npu.py --model efficientvit_b0_r224_nchw.mxq --image cat.jpg
python3 inference_npu.py --model efficientvit_b0_r224_nchw.mxq --benchmark
```

### qbruntime API (infer_resnet50.py 기반)

```python
import qbruntime

acc    = qbruntime.Accelerator(0)
config = qbruntime.ModelConfig()
config.set_single_core_mode(core_ids=[
    qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0),
])
model  = qbruntime.Model("efficientvit_b0_r224_nchw.mxq", config)
model.launch(acc)

# 입력: CHW (3, 224, 224) float32 — NCHW 컴파일이므로 채널 먼저
result = model.infer([img_chw])
model.dispose()
```

---

## 8. 교훈

**1. ONNX 중간 변환 도구는 텐서 dtype을 오염시킬 수 있다**  
특히 `INT64` 인덱스 상수를 재생성할 때 `FLOAT64`로 잘못 저장하는 버그가 있었음.  
가능하면 원본 프레임워크에서 직접 export하는 것이 안전하다.

**2. 양자화 에러 메시지에는 노드 이름이 없다**  
`zeropoint=-2147483648` 에러는 어떤 텐서가 문제인지 알려주지 않는다.  
Python 레벨 진단(187개 활성화 텐서 + 가중치 전수 검사)에서 문제가 없었던 이유는  
**C++ 양자화기가 Python에서 보이지 않는 FLOAT64 상수 노드를 별도로 처리**하기 때문.

**3. 구버전 API 파라미터는 조용히 무시된다**  
`is_asym_quant`, `quantize_method` 등 구버전 kwargs는 `**kwargs`로 흡수되어 아무 효과가 없었음.  
신버전 API `CalibrationConfig` 객체를 사용해야 한다.

**4. NHWC 변환은 필수가 아니다**  
`mxq_compile(in_dformats={"input": "NCHW"}, cpu_offload=True)` 설정으로  
표준 NCHW ONNX를 그대로 컴파일할 수 있다.

---

*작성일: 2026-05-27*
