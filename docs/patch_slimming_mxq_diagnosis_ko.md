# Patch Slimming ONNX→MXQ 정확도 저하 진단

작성 기준일: 2026-08-18

이 문서는 `tiny30__patch_slimming` 모델의 ONNX 정확도는 정상이지만 Mobilint MXQ
정확도만 크게 낮아지는 문제를 정리한다. 별도 x86 Docker 컴파일 서버에서 재현 모델을
컴파일한 뒤, NPU 서버로 MXQ를 가져와 원인을 판별하는 절차도 포함한다.

## 1. 현재 결론

문제 범위는 전처리나 PyTorch→ONNX 변환이 아니라 **실제 Patch Slimming 그래프의
ONNX→MXQ 컴파일 또는 양자화 단계**로 좁혀졌다.

- Patch Slimming ONNX의 이미지 분류 정확도는 정상이다.
- 동일 입력에서 PyTorch와 진단 ONNX 출력은 일치한다.
- 합성 진단 모델 `diag_ps_01`~`diag_ps_09`의 ONNX와 MXQ 출력도 모두 유사하다.
- 하지만 실제 Patch Slimming 그래프만 ONNX와 MXQ 출력이 크게 다르다.
- 같은 컴파일 계열의 baseline과 reduced MXQ는 정상 범위다.

따라서 "Mobilint NPU가 rectangular attention 또는 token selection을 전혀 지원하지
않는다"고 결론 내릴 수는 없다. 실제 모델에 들어간 one-hot selection MatMul의 표현,
특수 shape, 실제 activation 범위 및 calibration의 조합을 분리해서 확인해야 한다.

## 2. 측정 결과

### 2.1 ImageNet 5,000장 정확도

동일한 ImageNet validation 5,000장, `timm` 전처리, Global8 기준이다.

| 모델 | Top-1 | Top-5 |
|---|---:|---:|
| Mobilint 공식 Model Zoo ViT-Tiny | 74.46% | 92.20% |
| custom baseline MXQ | 73.98% | 91.78% |
| reduced MXQ | 71.62% | 90.80% |
| patch_slimming MXQ | **9.14%** | **19.76%** |

전처리는 짧은 변 249 resize, bicubic, 224 center crop, RGB,
`mean=std=(0.5, 0.5, 0.5)`이다. baseline과 reduced가 이 설정에서 정상인 점도
Patch Slimming의 9.14%를 단순 전처리 오류로 설명하기 어렵게 한다.

### 2.2 동일 입력 ONNX↔MXQ logits 비교

고정 입력 `assets/diagnostics/patch_slimming/input/diagnostic_input_hwc.npy`를 사용했다.

| 실제 그래프 | cosine | relative L2 |
|---|---:|---:|
| baseline | 0.981961 | 0.189666 |
| reduced | 0.994729 | 0.106104 |
| patch_slimming | **0.343667** | **1.250309** |

Patch Slimming은 일반적인 양자화 오차 범위를 넘어 출력 방향 자체가 크게 달라진다.
결과 JSON은 `results/diagnostics/tiny30_*_actual_graph.json`에 생성되어 있다.

### 2.3 합성 진단 모델 결과

| 진단 | 검사 내용 | cosine | relative L2 |
|---|---|---:|---:|
| 01 | stem + head control | 0.998235 | 0.059498 |
| 02 | contiguous selection | 0.997766 | 0.066803 |
| 03 | scattered selection | 0.997881 | 0.066637 |
| 04 | square attention | 0.999887 | 0.018049 |
| 05 | rectangular attention | 0.999909 | 0.022555 |
| 06 | rectangular attention + residual + MLP | 0.999593 | 0.033278 |
| 07 | 3-block stack | 0.999796 | 0.030200 |
| 08 | 6-block stack | 0.999706 | 0.032933 |
| 09 | 12-block stack | 0.999411 | 0.053133 |

합성 진단은 모두 통과했다. 다만 이 진단은 실제 모델과 다음이 다르다.

- CLS token과 positional embedding이 없다.
- 마지막 출력이 1 token까지 줄지 않고 61 tokens에서 끝난다.
- 실제 pruning 일정과 다르다.
- 선택 행렬이 ONNX initializer로 저장된다.
- 임의 가중치와 임의 입력을 사용하므로 실제 activation 분포를 재현하지 않는다.
- 평균 pooling head를 사용하며 실제 모델의 CLS/Squeeze/head 경로와 다르다.

## 3. 실제 Patch Slimming 그래프의 특징

실제 그래프는 고정 one-hot 행렬 `P`를 이용해 토큰 선택을 다음처럼 표현한다.

```text
selected = P @ tokens
P shape = [N_out, N_in]
tokens shape = [1, N_in, 192]
selected shape = [1, N_out, 192]
```

query 경로와 residual 경로에 각각 한 번씩 들어가므로 12개 블록에서 총 24회의
selection MatMul이 존재한다.

```text
block 0 : 197 -> 151
block 1 : 151 -> 151  (identity selection)
block 2 : 151 -> 151  (identity selection)
block 3 : 151 -> 141
block 4 : 141 -> 131
block 5 : 131 -> 111
block 6 : 111 -> 111  (identity selection)
block 7 : 111 -> 111  (identity selection)
block 8 : 111 -> 111  (identity selection)
block 9 : 111 -> 111  (identity selection)
block 10: 111 -> 111  (identity selection)
block 11: 111 -> 1    (CLS만 선택)
```

MXQ 문자열 메타데이터를 확인하면 컴파일러는 selection MatMul을 내부적으로
`reshape/matmul/conv2d` 형태로 낮춘다. 이 lowering과 activation quantization의
상호작용이 현재 가장 유력한 문제 지점이다.

마지막 `111→1` 행렬이 CLS가 아닌 다른 토큰을 고른다는 가설도 검사했다. 마지막 선택
행렬을 0~110번 토큰으로 각각 바꾼 ONNX 출력과 현재 Patch MXQ 출력을 비교했지만,
MXQ와 일치하는 대체 토큰은 없었다. 정상 CLS인 0번 토큰이 오히려 cosine 기준 가장
가까웠다. 따라서 단순한 off-by-one 또는 잘못된 token index 문제일 가능성은 낮다.

## 4. 현재 가설과 우선순위

### 가설 A: one-hot Constant MatMul lowering/양자화 문제 — 가장 유력

실제 모델의 선택 행렬은 `Constant` node이며 컴파일러에서 convolution 형태로
변환된다. 고정 선택은 본질적으로 `Gather`이므로 MatMul 양자화가 필요하지 않다.
24회의 selection MatMul 중 14회는 identity라 계산 자체도 불필요하다.

### 가설 B: `N_out=1` 특수 shape 문제

마지막 블록에서 query 길이와 출력 토큰 수가 1이 된다. 기존 합성 진단에는 이 경우가
없다. 단순히 잘못된 토큰을 읽는 문제는 아니지만, `1×111` MatMul 또는
`Q=1, K/V=111` attention을 컴파일러가 다르게 처리할 가능성은 남아 있다.

### 가설 C: 실제 activation 범위에서 발생하는 양자화 누적 오차

합성 진단은 임의 가중치에서 통과했지만 실제 모델은 학습된 가중치, 블록별로 다른
MLP width와 실제 activation 분포를 가진다. calibration sample이나 preset이 실제
Patch Slimming 분포를 제대로 포착하지 못했을 수 있다.

### 가설 D: 기존 MXQ artifact 또는 컴파일 조건 불일치

현재 MXQ가 현재 ONNX와 다른 revision에서 만들어졌거나, 진단 MXQ와 compiler version,
calibration data, preset, CPU offload 설정이 달랐을 가능성을 배제할 수 없다. 별도
서버에서 원본까지 같은 조건으로 다시 컴파일하면 이 가설을 확인할 수 있다.

## 5. 준비된 등가 ONNX 3종

컴파일 서버에서는 아래 세 모델을 **동일 compiler, calibration, preset, inference
scheme**으로 컴파일한다.

| 구분 | 파일 | 목적 |
|---|---|---|
| 원본 control | `assets/onnx/tiny30__patch_slimming__nhwc_b1_npusafe.onnx` | 기존 MXQ artifact/설정 재현 |
| initializer | `assets/experiments/patch_slimming/onnx/tiny30__patch_slimming__nhwc_b1_npusafe__selection_init.onnx` | Constant와 initializer 표현 차이 분리 |
| gather | `assets/experiments/patch_slimming/onnx/tiny30__patch_slimming__nhwc_b1_npusafe__selection_gather.onnx` | selection MatMul lowering 제거 |

`initializer` 모델은 24개 선택 행렬을 Constant node에서 graph initializer로 옮겼다.
MatMul은 그대로 유지한다.

`gather` 모델은 다음처럼 변환했다.

- non-identity selection MatMul 10개를 `Gather(axis=1)`로 교체
- identity selection MatMul 14개를 제거
- 나머지 attention, residual, MLP, head와 가중치는 변경하지 않음

고정 진단 입력에서 원본과 두 후보의 ONNX Runtime 출력은 모두 완전히 동일했다.

```text
max_abs       = 0.0
relative_l2   = 0.0
cosine        = 1.0
argmax_before = argmax_after
```

파일 무결성 확인용 SHA-256은 다음과 같다.

```text
772d9c0fc973bbfc5f43090b83c537ad5100fb66f74eec574dc75df4becfe3cf  tiny30__patch_slimming__nhwc_b1_npusafe.onnx
8692811b4f8a4be5d76f674405fd1ae9ffd88caaf51bb2d82df7454a427a4299  tiny30__patch_slimming__nhwc_b1_npusafe__selection_init.onnx
9ace430037c738b01355543a6cc6f7e5dca72ad189d013bc587abc4a9628a6e0  tiny30__patch_slimming__nhwc_b1_npusafe__selection_gather.onnx
```

필요하면 다음 명령으로 두 후보를 다시 생성할 수 있다.

```bash
python3 diagnostics/patch_slimming/rewrite_selections.py \
  --mode initializer \
  --output assets/experiments/patch_slimming/onnx/tiny30__patch_slimming__nhwc_b1_npusafe__selection_init.onnx \
  --check-input assets/diagnostics/patch_slimming/input/diagnostic_input_hwc.npy

python3 diagnostics/patch_slimming/rewrite_selections.py \
  --mode gather \
  --output assets/experiments/patch_slimming/onnx/tiny30__patch_slimming__nhwc_b1_npusafe__selection_gather.onnx \
  --check-input assets/diagnostics/patch_slimming/input/diagnostic_input_hwc.npy
```

## 6. 별도 Docker 서버 컴파일 절차

### 6.1 calibration 데이터 생성

현재 `assets/calib_hwc`에는 1,000개 tensor가 있지만
`preprocess_profile.json`이 없어 생성 당시 profile을 증명할 수 없다. 이번 원인 분리
실험에서는 새 디렉터리에 `timm` calibration을 생성하는 것을 권장한다.

```bash
cd /workspace/npu

python3 compile/calibration/generate_hwc.py \
  --profile timm \
  --src-dir /path/to/imagenet/calibration_images \
  --output-dir assets/calib_hwc_timm \
  --n-images 1000

cat assets/calib_hwc_timm/preprocess_profile.json
```

세 ONNX 모두 반드시 같은 calibration 디렉터리를 사용한다.

### 6.2 세 모델 컴파일

현재 `compile/mxq/compile_hwc.py`의 공통 설정은 다음과 같다.

```text
backend          = onnx
config_preset    = vision_transformer
inference_scheme = global8
cpu_offload      = True
```

각 모델을 개별적으로 컴파일하면 로그가 섞이지 않아 provenance를 남기기 쉽다.

```bash
NPU_CALIB_DIR="$PWD/assets/calib_hwc_timm" \
NPU_ONNX_DIR="$PWD/assets/onnx" \
NPU_MXQ_DIR="$PWD/assets/experiments/patch_slimming/mxq" \
NPU_INFERENCE_SCHEME=global8 \
NPU_ONNX_FILTER='tiny30__patch_slimming__nhwc_b1_npusafe.onnx' \
python3 compile/mxq/compile_hwc.py 2>&1 | tee compile_original.log

NPU_CALIB_DIR="$PWD/assets/calib_hwc_timm" \
NPU_ONNX_DIR="$PWD/assets/experiments/patch_slimming/onnx" \
NPU_MXQ_DIR="$PWD/assets/experiments/patch_slimming/mxq" \
NPU_INFERENCE_SCHEME=global8 \
NPU_ONNX_FILTER='__selection_init.onnx' \
python3 compile/mxq/compile_hwc.py 2>&1 | tee compile_selection_init.log

NPU_CALIB_DIR="$PWD/assets/calib_hwc_timm" \
NPU_ONNX_DIR="$PWD/assets/experiments/patch_slimming/onnx" \
NPU_MXQ_DIR="$PWD/assets/experiments/patch_slimming/mxq" \
NPU_INFERENCE_SCHEME=global8 \
NPU_ONNX_FILTER='__selection_gather.onnx' \
python3 compile/mxq/compile_hwc.py 2>&1 | tee compile_selection_gather.log
```

이미 같은 이름의 MXQ가 있으면 스크립트가 `[SKIP]`하므로, 별도 빈 output 작업
디렉터리에서 실행하거나 기존 파일과 혼동되지 않게 먼저 이름을 바꾼다. 기존 결과를
삭제할 필요는 없다.

예상 출력 파일명은 다음과 같다.

```text
assets/experiments/patch_slimming/mxq/tiny30__patch_slimming__nhwc_b1_npusafe_vt_global8.mxq
assets/experiments/patch_slimming/mxq/tiny30__patch_slimming__nhwc_b1_npusafe__selection_init_vt_global8.mxq
assets/experiments/patch_slimming/mxq/tiny30__patch_slimming__nhwc_b1_npusafe__selection_gather_vt_global8.mxq
```

원본 control은 현재 NPU 서버의 MXQ를 덮어쓰지 말고 `__recompiled` 같은 이름으로
가져오는 것이 안전하다.

### 6.3 함께 기록할 정보

```bash
python3 -c 'import qbcompiler; print(qbcompiler.__version__)'
sha256sum assets/onnx/tiny30__patch_slimming*.onnx
sha256sum assets/experiments/patch_slimming/onnx/*.onnx
sha256sum assets/experiments/patch_slimming/mxq/*.mxq
cat assets/calib_hwc_timm/preprocess_profile.json
```

다음도 로그에 남긴다.

- qbcompiler Docker image 이름과 digest
- `vision_transformer` preset 사용 여부
- `cpu_offload` 값
- inference scheme
- calibration 파일 수
- 컴파일 warning/error 전체

## 7. NPU 서버에서 우선 실행할 빠른 검증

MXQ를 가져오면 5,000장 평가 전에 고정 입력 하나로 ONNX와 비교한다.

```bash
python3 diagnostics/patch_slimming/compare_onnx_mxq.py \
  --onnx assets/experiments/patch_slimming/onnx/tiny30__patch_slimming__nhwc_b1_npusafe__selection_init.onnx \
  --mxq assets/experiments/patch_slimming/mxq/tiny30__patch_slimming__nhwc_b1_npusafe__selection_init_vt_global8.mxq \
  --core-mode global8 \
  --output results/diagnostics/patch_selection_init.json

python3 diagnostics/patch_slimming/compare_onnx_mxq.py \
  --onnx assets/experiments/patch_slimming/onnx/tiny30__patch_slimming__nhwc_b1_npusafe__selection_gather.onnx \
  --mxq assets/experiments/patch_slimming/mxq/tiny30__patch_slimming__nhwc_b1_npusafe__selection_gather_vt_global8.mxq \
  --core-mode global8 \
  --output results/diagnostics/patch_selection_gather.json
```

cosine이 기존 `0.343667`보다 크게 개선되는 후보만 5,000장 평가한다. 정상 baseline과
비슷한 `0.98` 전후면 가장 유망하다.

```bash
python3 benchmark/ablation/eval_tiny30_core_ablation.py \
  --model patch_init:global8:assets/experiments/patch_slimming/mxq/tiny30__patch_slimming__nhwc_b1_npusafe__selection_init_vt_global8.mxq \
  --model patch_gather:global8:assets/experiments/patch_slimming/mxq/tiny30__patch_slimming__nhwc_b1_npusafe__selection_gather_vt_global8.mxq \
  --val-dir /media/airlab_compression/nvme_storage/imagenet_val \
  --classes 1000 \
  --images-per-class 5 \
  --seed 42 \
  --preprocess timm \
  --output results/patch_slimming_selection_rewrite_5000.json
```

## 8. 결과 해석

| 결과 | 해석 |
|---|---|
| 재컴파일한 원본부터 정상 | 기존 MXQ가 오래된 artifact이거나 compiler/calibration 설정이 달랐음 |
| initializer만 정상 | Constant node와 initializer를 compiler가 다르게 처리하는 문제 |
| gather만 정상 | one-hot selection MatMul lowering 또는 그 activation 양자화가 원인 |
| initializer와 gather 모두 정상 | Constant 표현과 MatMul이 각각 영향을 주었을 수 있으므로 gather를 최종 후보로 사용 |
| 세 모델 모두 같은 수준으로 실패 | 실제 activation calibration, `Q=1` attention 또는 다른 실제 그래프 조합을 추가 진단 |
| 고정 입력 cosine은 정상인데 정확도만 낮음 | calibration coverage와 이미지별 activation outlier를 우선 조사 |

세 모델 모두 실패할 경우 다음 진단은 `111→1` selection 단독,
`Q=1/KV=111` 마지막 attention block, 실제 가중치를 사용한 block-prefix MXQ 순서로
진행한다. 합성 모델을 더 크게 만드는 것보다 실제 ONNX를 블록별 prefix로 잘라 최초로
출력이 어긋나는 블록을 찾는 편이 정보량이 많다.

## 9. 현재 배제하거나 가능성이 낮아진 원인

- 잘못된 `timm` 이미지 전처리
- Patch Slimming 학습 모델 자체의 정확도 문제
- PyTorch→ONNX export의 일반적인 오류
- rectangular attention 연산 자체의 전면적인 미지원
- 단순한 마지막 CLS token off-by-one 선택
- Global8 core 설정만의 문제

이 항목들은 절대적으로 불가능하다는 뜻이 아니라, 현재 증거상 우선순위가 낮다는
의미다.
