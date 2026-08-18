# Patch Slimming MXQ 최종 진단

이 문서가 Patch Slimming ONNX→MXQ 문제의 유일한 전용 문서다. 현재 결론, 재현 결과,
남은 검사와 실행 명령만 관리한다.

## 1. 현재 결론

1. Patch Slimming float ONNX는 정상이다. ImageNet 1,000장 기준 Top-1 `67.30%`,
   Top-5 `88.30%`이며 그래프 변환 후보와 logits가 완전히 같다.
2. identity selection MatMul 14개는 안전하게 제거할 수 있다. 제거 후 legacy
   calibration MXQ의 Top-1이 `9.30%`에서 `50.00%`로 회복됐다.
3. 실제 selection MatMul 10개를 static Slice+Concat으로 바꿔도 Top-1은 `49.90%`로
   추가 개선이 없었다. ONNX 표면의 selection 연산 표현은 남은 손실의 주원인이 아니다.
4. 지금까지 만든 실험 MXQ는 모두 잘못된 legacy calibration을 사용했다. 따라서
   `50%`를 최종 NPU 성능이나 불가피한 Patch Slimming 손실로 결론 내릴 수 없다.
5. 다음 작업은 올바른 `timm` calibration으로 `no_identity` ONNX를 재컴파일하는 것이다.

현재 권장 ONNX:

```text
assets/experiments/patch_slimming/onnx/
tiny30__patch_slimming__nhwc_b1_npusafe__selection_no_identity.onnx
```

이 모델은 float 원본과 완전히 같고, 불필요한 identity MatMul만 제거한 가장 단순한
후보다.

## 2. 확인된 결과

모든 정확도는 같은 ImageNet 표본(1,000 classes × 1 image, seed 42)과 `timm`
runtime 전처리를 사용했다.

| 모델 | Top-1 | Top-5 | 비고 |
|---|---:|---:|---|
| Patch Slimming float ONNX | 67.30% | 88.30% | 정상 reference |
| 기존 Patch MXQ | 9.30% | 19.10% | identity 포함, legacy calibration |
| no identity MXQ | 50.00% | 76.40% | identity 14개 제거, legacy calibration |
| no identity + final Slice MXQ | 50.20% | 76.20% | legacy calibration |
| all Slice+Concat MXQ | 49.90% | 76.40% | selection MatMul 제거, legacy calibration |

동일한 고정 입력의 ONNX↔MXQ 비교:

| MXQ | MAE | relative L2 | cosine |
|---|---:|---:|---:|
| 기존 Patch | 1.1390 | 1.2503 | 0.343667 |
| no identity | 0.8337 | 0.9308 | 0.663482 |
| no identity + final Slice | 0.8351 | 0.9318 | 0.662778 |
| all Slice+Concat | 0.8347 | 0.9342 | 0.663396 |

identity 제거는 분명한 개선을 만들었지만 정상 모델에서 기대하는 cosine 약 `0.98`
수준에는 도달하지 못했다.

## 3. Calibration 오류

기존 `assets/calib_hwc`는 `timm`이 아니라 예전 `legacy` 전처리로 생성됐다.

| 항목 | 기존 legacy | 올바른 timm |
|---|---|---|
| resize short side | 256 | 249 |
| interpolation | bilinear | bicubic |
| mean | `[0.485, 0.456, 0.406]` | `[0.5, 0.5, 0.5]` |
| std | `[0.229, 0.224, 0.225]` | `[0.5, 0.5, 0.5]` |
| 일반적인 값 범위 | 약 `-2.12~2.64` | `-1~1` |

원본 calibration 이미지의 첫 파일을 다시 전처리해 비교한 결과:

```text
assets/calib_hwc/0000.npy vs legacy: max_abs = 0.0
assets/calib_hwc/0000.npy vs timm:   max_abs = 3.2524
```

기존 파일이 legacy라는 것이 bit-exact하게 확인됐다. `assets/calib_hwc`는 최종 실험에
재사용하지 않는다.

## 4. 반드시 진행할 최종 실험

### 4.1 timm calibration 생성

컴파일 서버에서 같은 원본 이미지 1,000장에 올바른 전처리를 적용한다. 기존 파일과
섞이지 않도록 새 폴더를 사용한다.

```bash
cd /workspace/npu

python3 compile/calibration/generate_hwc.py \
  --profile timm \
  --src-dir assets/calibration_data \
  --output-dir assets/calib_hwc_timm \
  --n-images 1000

cat assets/calib_hwc_timm/preprocess_profile.json
```

다음을 확인한다.

```text
profile     = timm
layout      = HWC
shape       = [224, 224, 3]
dtype       = float32
num_tensors = 1000
mean/std    = [0.5, 0.5, 0.5]
```

### 4.2 no identity ONNX 컴파일 — GPU 2

기존 MXQ와 섞이지 않도록 `mxq_timm_calib`에 저장한다.

```bash
cd /workspace/npu

CUDA_VISIBLE_DEVICES=2 \
NPU_CALIB_DIR="$PWD/assets/calib_hwc_timm" \
NPU_ONNX_DIR="$PWD/assets/experiments/patch_slimming/onnx" \
NPU_MXQ_DIR="$PWD/assets/experiments/patch_slimming/mxq_timm_calib" \
NPU_INFERENCE_SCHEME=global8 \
NPU_ONNX_FILTER='tiny30__patch_slimming__nhwc_b1_npusafe__selection_no_identity.onnx' \
python3 compile/mxq/compile_hwc.py \
  2>&1 | tee compile_no_identity_timm_gpu2.log
```

예상 출력:

```text
assets/experiments/patch_slimming/mxq_timm_calib/
tiny30__patch_slimming__nhwc_b1_npusafe__selection_no_identity_vt_global8.mxq
```

### 4.3 고정 입력 비교

새 MXQ를 NPU 서버로 가져온 뒤 먼저 한 입력의 logits를 비교한다.

```bash
python3 diagnostics/patch_slimming/compare_onnx_mxq.py \
  --onnx assets/experiments/patch_slimming/onnx/tiny30__patch_slimming__nhwc_b1_npusafe__selection_no_identity.onnx \
  --mxq assets/experiments/patch_slimming/mxq_timm_calib/tiny30__patch_slimming__nhwc_b1_npusafe__selection_no_identity_vt_global8.mxq \
  --core-mode global8 \
  --output results/diagnostics/patch_no_identity_timm_calib.json
```

cosine이 기존 `0.663482`에서 얼마나 개선됐는지 확인한다. 약 `0.98` 이상이면 정확도
회복 가능성이 높다.

### 4.4 ImageNet 1,000장 평가

```bash
python3 benchmark/ablation/eval_tiny30_core_ablation.py \
  --model patch_no_identity_timm:global8:assets/experiments/patch_slimming/mxq_timm_calib/tiny30__patch_slimming__nhwc_b1_npusafe__selection_no_identity_vt_global8.mxq \
  --val-dir /media/airlab_compression/nvme_storage/imagenet_val \
  --classes 1000 \
  --images-per-class 1 \
  --seed 42 \
  --preprocess timm \
  --output results/diagnostics/patch_no_identity_timm_calib_1000.json
```

1,000장에서 ONNX `67.30%`에 근접한 후보만 5,000장 또는 전체 50,000장을 평가한다.

## 5. timm calibration 후에도 낮을 때

다음 순서까지만 확인한다.

1. 같은 Docker, calibration, preset으로 baseline을 다시 컴파일한다.
2. baseline은 정상인데 Patch만 낮으면 `single`과 `global8`을 비교한다.
3. CPU offload on/off와 SDK가 지원하는 최고 activation/mixed precision을 비교한다.
4. 실제 가중치의 block-prefix ONNX/MXQ를 만들어 최초로 오차가 급증하는 Transformer
   block을 찾는다.
5. 특정 block이 확인되면 그 prefix 모델과 컴파일 로그를 Mobilint에 전달한다.

합성 진단 모델 `diag_ps_01~09`는 모두 정상에 가까웠다. 따라서 무작위 가중치의 작은
합성 모델을 더 만드는 것보다 실제 가중치의 block-prefix 비교가 우선이다.

## 6. 최종 판정 기준

다음 조건을 모두 만족해야만 추가 하락을 회피하기 어렵다고 판단한다.

- timm calibration으로 재컴파일했다.
- 같은 조건의 baseline MXQ는 정상이다.
- 최고 지원 정밀도에서도 Patch만 낮다.
- core mode와 CPU offload 설정을 바꿔도 결과가 같다.
- block-prefix 검사에서 단일 잘못된 op가 아니라 양자화 오차 누적으로 확인된다.
- Mobilint SDK에 지원되는 우회 설정이 없음을 확인했다.

이 경우 결론은 다음처럼 제한해서 표현한다.

> 현재 Mobilint qbcompiler/PTQ 지원 범위에서는 Patch Slimming 배포 시 추가 정확도
> 하락을 피하기 어렵다.

`Patch Slimming 모델 자체의 정확도가 50%`라고 표현하면 안 된다. float ONNX는 이미
`67.30%`를 보였기 때문이다.

## 7. 파일 위치

| 경로 | 용도 |
|---|---|
| `assets/onnx/tiny30__patch_slimming__nhwc_b1_npusafe.onnx` | 원본 float ONNX |
| `assets/experiments/patch_slimming/onnx/` | 등가 그래프 변환 후보 |
| `assets/experiments/patch_slimming/mxq/` | legacy calibration 실험 MXQ |
| `assets/experiments/patch_slimming/mxq_timm_calib/` | 새 timm calibration MXQ |
| `assets/diagnostics/patch_slimming/` | 고정 입력과 합성 진단 자산 |
| `diagnostics/patch_slimming/compare_onnx_mxq.py` | ONNX↔MXQ logits 비교 |
| `diagnostics/patch_slimming/rewrite_selections.py` | 등가 ONNX 생성 |
| `benchmark/ablation/eval_tiny30_core_ablation.py` | MXQ 정확도 평가 |

주요 결과 JSON:

```text
results/diagnostics/patch_slimming_original_vs_supported_rewrites_mxq_1000_timm_seed42.json
results/diagnostics/patch_slimming_slice_concat_mxq_1000_timm_seed42.json
results/diagnostics/patch_selection_no_identity.json
results/diagnostics/patch_selection_slice_concat.json
```
