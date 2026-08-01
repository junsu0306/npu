# ONNX → MXQ 변환 파이프라인

Mobilint **qbcompiler v1.1.0**으로 ONNX 모델을 Aries2 NPU에서 실행 가능한 **MXQ** 형식으로 변환하는 전체 과정을 설명합니다.

---

## 파이프라인 개요

```
ImageNet 이미지
    │
    ▼  [LOCAL] gen_calib_hwc.py
HWC .npy 텐서 (assets/calib_hwc/)
    │
    ▼  [DOCKER] compile_hwc.py
MXQ 파일 (assets/mxq/)
    │
    ▼  [NPU] run_npu.py
추론 결과 + 벤치마크
```

> **환경 구분**
> - 캘리브레이션 데이터 생성(`gen_calib_hwc.py`) → **로컬 (Jetson Orin)**
> - MXQ 컴파일(`compile_hwc.py`) → **qbcompiler Docker 컨테이너 내부**
> - NPU 추론(`run_npu.py`) → **드라이버 설치된 Jetson Orin**

---

## 디렉토리 구조

```
npu/
├── assets/
│   ├── calibration_data/       # 원본 ImageNet 이미지 (~38 MB)
│   ├── calib_hwc/              # 생성된 HWC .npy 텐서 (~578 MB, git 제외)
│   ├── onnx/
│   │   └── vit_tiny_prune50_global _reduced.onnx   # 입력 모델 (11 MB)
│   ├── mxq/                    # 컴파일 출력 (git 제외)
│   │   └── vit_tiny_prune50_global _reduced_global8.mxq
│   └── qbcompiler-1.1.0+aries2-py3-none-any.whl
│
├── compile/
│   ├── gen_calib_hwc.py        # Step 2: 로컬에서 실행
│   ├── compile_hwc.py          # Step 3: Docker 안에서 실행
│   ├── calib_hwc.txt           # 자동 생성 파일 목록 (git 제외)
│   └── compile_resnet50.py     # ResNet50 sanity-check용
│
├── run_npu.py                  # Step 4: NPU 추론 + 벤치마크
├── test_npu.py                 # Top-5 정확도 테스트
├── infer_onnx.py               # ONNX Runtime CPU 추론 (변환 전 검증용)
└── infer_resnet50.py           # qbruntime 사용 예시
```

---

## Step 1 — ONNX 모델 준비

변환할 ONNX 파일을 `assets/onnx/`에 배치합니다. `compile_hwc.py`가 이 디렉토리를 자동 스캔하므로 추가 설정이 필요 없습니다.

컴파일 전에 ONNX Runtime으로 모델 출력을 검증합니다 (입력 포맷: NCHW):

```bash
python3 infer_onnx.py \
  --model "assets/onnx/vit_tiny_prune50_global _reduced.onnx" \
  --image val_example.JPEG
```

---

## Step 2 — 캘리브레이션 데이터 생성 `[LOCAL]`

Post-Training Quantization(PTQ)은 activation 범위 측정을 위한 캘리브레이션 데이터가 필요합니다. `gen_calib_hwc.py`는 ImageNet 이미지를 전처리하여 **HWC float32 텐서**로 저장합니다.

```bash
cd /home/airlab_compression/npu
source aries_env/bin/activate
python3 compile/gen_calib_hwc.py
```

출력 예시:
```
Generating 1000 calibration tensors from assets/calibration_data → assets/calib_hwc
Done: 1000 files saved to assets/calib_hwc
Sample stats: min=-2.118  max=2.640  mean=0.021  std=0.972
```

### 전처리 상세

```python
MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])

# 1. BGR → RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. 단변을 256으로 비율 유지 리사이즈
scale = 256 / min(h, w)
img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))

# 3. 중앙 224×224 크롭
img = img[(h-224)//2:(h-224)//2+224,
          (w-224)//2:(w-224)//2+224]

# 4. float32 정규화
img = img.astype(np.float32) / 255.0

# 5. ImageNet 평균/표준편차 정규화 → HWC (224, 224, 3)
img = (img - MEAN) / STD

np.save(f"{dst}/{saved:04d}.npy", img)   # shape: (224, 224, 3)
```

> **HWC 포맷 이유**: qbcompiler `backend="onnx"`는 기본적으로 HWC 입력을 가정합니다. CHW로 변환(transpose)하지 않고 저장합니다.

---

## Step 3 — MXQ 컴파일 `[DOCKER]`

### Docker 실행

```bash
docker run -it --rm \
  -v /home/airlab_compression/npu:/workspace/npu \
  <qbcompiler-image> bash

# 컨테이너 내부
pip install /workspace/npu/assets/qbcompiler-1.1.0+aries2-py3-none-any.whl
python3 /workspace/npu/compile/compile_hwc.py
```

### compile_hwc.py 동작

`assets/onnx/*.onnx`를 자동 스캔하여 순서대로 컴파일합니다. 이미 `.mxq`가 존재하는 모델은 건너뜁니다.

```python
# 상단 설정값 — 변경 후 재컴파일 필요
INFERENCE_SCHEME = "global8"   # "single" | "multi" | "global" | "global4" | "global8"

mxq_compile(
    model=onnx_path,
    calib_data_path=CALIB_TXT,
    save_path=mxq_path,               # e.g. vit_tiny_..._global8.mxq
    backend="onnx",
    inference_scheme=INFERENCE_SCHEME,
    quantization_method=0,            # WChALayer
    quantization_mode=2,              # Histogram
    hist_search_type=2,               # KL-divergence
    quantization_output=0,            # Layer
)
```

출력 파일명에 scheme이 자동 포함됩니다: `{stem}_{scheme}.mxq`

### 재컴파일 방법

`INFERENCE_SCHEME`을 변경한 경우 기존 파일을 삭제 후 재실행합니다:

```bash
rm "assets/mxq/vit_tiny_prune50_global _reduced_global8.mxq"
python3 /workspace/npu/compile/compile_hwc.py
```

---

## Step 4 — NPU 추론 실행 `[NPU]`

### 기본 실행

```bash
# warmup 5회, 측정 20회, 결과 results/ 에 저장
python3 run_npu.py \
  --mxq "assets/mxq/vit_tiny_prune50_global _reduced_global8.mxq" \
  --image val_example.JPEG

# 정밀 측정
python3 run_npu.py \
  --mxq "assets/mxq/vit_tiny_prune50_global _reduced_global8.mxq" \
  --image val_example.JPEG \
  --runs 100 --warmups 10 --no-save
```

출력 예시:
```
모델  : assets/mxq/vit_tiny_prune50_global _reduced_global8.mxq
이미지: val_example.JPEG  → shape=(224, 224, 3)
코어  : global8, warmup=5, runs=20

총 시간 : 0.4821s
평균 레이턴시: 24.11ms
FPS     : 41.49

Top-5 예측:
  1. [ 340] zebra                                        52.31%
  2. [ 292] tiger, Panthera tigris                       18.44%
  ...
```

### qbruntime 핵심 코드

```python
import qbruntime

acc    = qbruntime.Accelerator(0)          # NPU 디바이스 0번
config = qbruntime.ModelConfig()

# global8: Cluster0/1 각 4코어 = 총 8코어
C, Co = qbruntime.Cluster, qbruntime.Core
config.set_single_core_mode(core_ids=[
    qbruntime.CoreId(C.Cluster0, Co.Core0),
    qbruntime.CoreId(C.Cluster0, Co.Core1),
    qbruntime.CoreId(C.Cluster0, Co.Core2),
    qbruntime.CoreId(C.Cluster0, Co.Core3),
    qbruntime.CoreId(C.Cluster1, Co.Core0),
    qbruntime.CoreId(C.Cluster1, Co.Core1),
    qbruntime.CoreId(C.Cluster1, Co.Core2),
    qbruntime.CoreId(C.Cluster1, Co.Core3),
])

model = qbruntime.Model(mxq_path, config)
model.launch(acc)                          # NPU에 모델 로드

output = model.infer([img])               # img: (224,224,3) HWC float32
model.dispose()                           # 리소스 해제
```

### Top-5 정확도 테스트

```bash
python3 test_npu.py \
  --model "assets/mxq/vit_tiny_prune50_global _reduced_global8.mxq" \
  --labels imagenet_classes.txt \
  --n 50
```

---

## 양자화 파라미터 상세

| 파라미터 | 현재 값 | 의미 |
|---|---|---|
| `quantization_method` | `0` (WChALayer) | 가중치: 채널별(per-channel), activation: 레이어별(per-layer) |
| `quantization_mode` | `2` (Histogram) | activation 범위를 히스토그램으로 측정. `0=Max`보다 아웃라이어에 강건 |
| `hist_search_type` | `2` (KL-divergence) | KL 발산을 최소화하는 클리핑 범위 탐색. `mode=2`일 때만 적용 |
| `quantization_output` | `0` (Layer) | 출력을 레이어 단위로 양자화 |

### 전체 옵션

**quantization_method**
- `0` WChALayer — 가중치 per-channel, activation per-layer
- `1` WChAMulti — 가중치 per-channel, activation per-channel
- `2` WChALayerZeropoint — WChALayer + zero-point 허용
- `3` WChAMultiZeropoint — WChAMulti + zero-point 허용

**quantization_mode**
- `0` Max — activation 최댓값 기준
- `1` MaxPercentile — 백분위수 기준
- `2` Histogram — 히스토그램 분포 기반 (가장 정밀)

**hist_search_type** (mode=2일 때만 적용)
- `0` Percentile
- `1` MSE
- `2` KL-divergence

**quantization_output**
- `0` Layer — 레이어 단위
- `1` Ch — 채널 단위 (더 정밀하나 느림)
- `2` Sigmoid

---

## Inference Scheme

`inference_scheme`은 **컴파일 타임**에 결정되며 변경 시 재컴파일이 필요합니다.

| scheme | 코어 수 | 용도 |
|---|---|---|
| `single` | 1 (기본값) | 단일 코어, 낮은 전력 소비 |
| `multi` | 2 | Cluster0 내 2코어 |
| `global` / `global4` | 4 | 양쪽 Cluster 각 2코어 |
| **`global8`** | **8** | **전체 8코어, 최고 처리량 (현재 사용)** |
| `all` | 전체 | 가용 코어 자동 사용 |

`compile_hwc.py` 상단의 `INFERENCE_SCHEME` 변수 한 줄만 수정하면 됩니다.

---

## 입력 포맷 정리

| 스크립트 | 포맷 | Shape |
|---|---|---|
| `gen_calib_hwc.py` 출력 | HWC | `(224, 224, 3)` |
| `compile_hwc.py` 캘리브레이션 입력 | HWC | `(224, 224, 3)` |
| `run_npu.py` / `test_npu.py` NPU 입력 | HWC | `(224, 224, 3)` |
| `infer_onnx.py` CPU 검증 입력 | NCHW | `(1, 3, 224, 224)` |

---

## 빠른 명령어 참조

```bash
# ── 처음부터 전체 파이프라인 ──────────────────────────────────────────

# [로컬] 캘리브레이션 데이터 생성
source aries_env/bin/activate
python3 compile/gen_calib_hwc.py

# [Docker] MXQ 컴파일
docker run -it --rm -v /home/airlab_compression/npu:/workspace/npu <image> bash
pip install /workspace/npu/assets/qbcompiler-1.1.0+aries2-py3-none-any.whl
python3 /workspace/npu/compile/compile_hwc.py

# [로컬] NPU 추론
python3 run_npu.py \
  --mxq "assets/mxq/vit_tiny_prune50_global _reduced_global8.mxq" \
  --image val_example.JPEG --runs 50 --warmups 10

# ── inference_scheme 변경 후 재컴파일 ────────────────────────────────

# compile_hwc.py 에서 INFERENCE_SCHEME 수정 후
rm "assets/mxq/vit_tiny_prune50_global _reduced_global8.mxq"
python3 /workspace/npu/compile/compile_hwc.py

# ── 정확도 테스트 ────────────────────────────────────────────────────

python3 test_npu.py \
  --model "assets/mxq/vit_tiny_prune50_global _reduced_global8.mxq" \
  --labels imagenet_classes.txt \
  --n 50
```
