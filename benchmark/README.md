# NPU Benchmark

이 폴더는 **컴파일이 끝난 MXQ를 실제 Mobilint NPU에서 실행**하는 코드만 모읍니다.

> AugReg ViT-Tiny는 `--preprocess timm`을 사용한다. 과거 `legacy`
> 전처리를 사용하면 baseline Top-1이 약 74%에서 44%로 하락한다.
> 상세 절차는
> [전처리·컴파일·벤치마크 가이드](../docs/preprocessing_compile_benchmark_guide_ko.md)를
> 참고한다.

## 파일 역할

| 파일 | 실행 방식 | 용도 |
|---|---|---|
| `final/eval_final_models.py` | direct `qbruntime` | 최종 MXQ 전체 ImageNet 정확도·속도·메모리·전력·FLOPs 통합 평가 |
| `model_zoo/run_smoke_test.py` | Model Zoo wrapper | 이미지 1장으로 로딩·속도 빠른 확인 |
| `model_zoo/run_model_zoo_benchmark.py` | Model Zoo wrapper | ImageNet Top-1/Top-5, latency, CPU/NPU 메모리·전력 |
| `ablation/eval_tiny30_core_ablation.py` | direct `qbruntime` | 동일 샘플에서 Tiny30 single/global8 정확도와 예측 일치율 비교 |
| `runtime/run_npu.py` | direct `qbruntime` | 이미지 1장 latency/FPS 및 Top-5 |
| `runtime/test_npu.py` | direct `qbruntime` | calibration 이미지 Top-5 점검 |
| `runtime/infer_resnet50.py` | direct `qbruntime` | ResNet50 최소 실행 예시 |
| `onnx/infer_onnx.py` | ONNX Runtime | MXQ 변환 전 CPU reference 추론 |

## 최종 모델 종합 평가

`final/eval_final_models.py`는 `assets/mxq_final12`의 최종 MXQ 16개만 자동으로
찾아 ImageNet validation 50,000장 전체를 순차 평가한다. 대응하는
`assets/onnx` 모델의 정적 shape를 이용해 주요 연산 FLOPs와 parameter 수도 함께
계산한다.

실행 전 모델, ONNX, 데이터셋 구성을 확인한다.

```bash
python3 benchmark/final/eval_final_models.py --dry-run
```

전체 최종 평가 명령은 다음과 같다. `--output-dir`를 생략하면 실행 시각별 폴더가
자동 생성되어 기존 결과를 덮어쓰지 않는다.

```bash
cd /media/airlab_compression/nvme_storage/npu

python3 benchmark/final/eval_final_models.py \
  --models-dir assets/mxq_final12 \
  --onnx-dir assets/onnx \
  --val-dir /media/airlab_compression/nvme_storage/imagenet_val \
  --classes 1000 \
  --images-per-class 50 \
  --preprocess timm \
  --core-mode global8 \
  --device-id 0 \
  --warmup-runs 10
```

`assets/mxq_final12`에는 Tiny/Small × 30%/50% × baseline/patch slimming/
stage369 relaxed/reduced의 16개 MXQ가 있다. 위 명령은 총 800,000회 추론하므로
오래 걸릴 수 있지만, 모델 하나가 끝날 때마다 중간 결과를 저장한다.

결과는 `results/final_benchmark/<실행시각>/` 아래에 생성된다.

| 결과 | 내용 |
|---|---|
| `summary.md` | 정확도, latency, FPS, 메모리, 전력, GFLOPs, MXQ 크기를 한 표로 확인 |
| `summary.csv` | 분석·그래프 작성용 평탄화된 지표 |
| `benchmark.json` | 환경, 세부 percentile, telemetry, 비교 결과를 포함한 전체 결과 |
| `benchmark_partial.json` | 실행 중에도 갱신되는 복구용 중간 결과 |
| `models/*.json` | 모델별 독립 결과 |
| `predictions/*.npy` | 모델별 Top-1 예측값 |
| `samples.tsv` | 실제 평가한 이미지와 정답 label 목록 |

기록 지표는 다음과 같다.

- 전체 50,000장 Top-1/Top-5 및 baseline 대비 정확도 차이·예측 일치율
- NPU inference latency 평균, 표준편차, min, p50, p90, p95, p99, max와 FPS
- 이미지 디코딩·전처리를 포함한 end-to-end FPS
- 프로세스 RSS/USS/PSS/VMS peak와 모델 시작 대비 RSS 증가량
- `mblt_tracker` 기반 NPU memory/utilization/power/temperature 평균·p99·최대
- MXQ/ONNX 크기와 ONNX initializer parameter 수
- ONNX Conv/MatMul/Gemm 기반 이론적 GFLOPs와 동일 그룹 baseline 대비 절감률

GFLOPs는 `1 MAC = 2 FLOPs`로 계산한 이미지 한 장당 정적 주요 연산량이다.
elementwise, normalization, activation, softmax는 제외되며, 컴파일러 fusion·tiling을
반영한 실제 NPU 연산량이 아니다. 반면 latency와 NPU telemetry는 실제 장치 측정값이다.

NPU telemetry가 `unavailable`이면 현재 Python 환경에서 다음을 먼저 확인한다.

```bash
python3 -c "from mblt_tracker import NPUDeviceTracker; print('tracker ok')"
mobilint-cli status
```

현재 저장소 환경처럼 NumPy 2.x와 오래된 pandas가 섞여 ABI 오류가 발생하면 같은
Python 환경에서 의존성을 다시 맞춘다.

```bash
python3 -m pip install -r requirements-npu.txt --upgrade
```

특정 모델만 먼저 확인할 때는 `--model`을 반복한다.

```bash
python3 benchmark/final/eval_final_models.py \
  --model assets/mxq_final12/tiny30__baseline__nhwc_b1_npusafe_vt_global8.mxq \
  --model assets/mxq_final12/tiny30__patch_slimming__nhwc_b1_npusafe_vt_global8.mxq \
  --val-dir /media/airlab_compression/nvme_storage/imagenet_val \
  --images-per-class 50 \
  --preprocess timm \
  --core-mode global8
```

Model Zoo 방식은 `model.preprocess()`까지 포함하여 기존 `Orin_Compression`의 평가 경로를
재현합니다. Direct 방식은 전처리를 명시적으로 고정하여 코어별 MXQ 차이를 분리합니다.

## Tiny30 빠른 실행

```bash
cd /media/airlab_compression/nvme_storage/npu

python3 benchmark/runtime/run_npu.py \
  --mxq assets/mxq/tiny30__baseline__nhwc_b1_npusafe_vt_global8.mxq \
  --image val_example.JPEG \
  --preprocess timm \
  --core-mode global8 \
  --warmups 10 \
  --runs 100
```

## Tiny30 5,000장 정확도 벤치마크

```bash
python3 benchmark/ablation/eval_tiny30_core_ablation.py \
  --model baseline:global8:assets/mxq/tiny30__baseline__nhwc_b1_npusafe_vt_global8.mxq \
  --model patch_slimming:global8:assets/mxq/tiny30__patch_slimming__nhwc_b1_npusafe_vt_global8.mxq \
  --model reduced:global8:assets/mxq/tiny30__reduced__nhwc_b1_npusafe_vt_global8.mxq \
  --val-dir /media/airlab_compression/nvme_storage/imagenet_val \
  --classes 1000 \
  --images-per-class 5 \
  --seed 42 \
  --preprocess timm \
  --output results/tiny30_global8_imagenet_val_5000_timm.json
```

`val-dir` 아래의 폴더 이름은 `00000`~`00999`처럼 ImageNet class index여야 합니다.

## 실제 NPU 메모리와 Patch Slimming activation 상세 측정

`run_model_zoo_benchmark.py`를 local MXQ에 적용할 때는 해당 MXQ가 Model Zoo
wrapper의 `uint8` 입력 계약으로 컴파일됐는지 먼저 확인한다. 현재 custom
Tiny30 MXQ는 정규화된 HWC `float32` 입력이므로 정확도는 위 direct runtime
명령으로 측정한다. 아래 명령은 Model Zoo 입력 계약으로 컴파일된 MXQ에만
사용한다.

`model_zoo/run_model_zoo_benchmark.py`는 기존 Orin 벤치마크보다 다음 메모리 지표를
추가로 기록한다.

- MXQ 파일 크기
- 모델 load 전/후와 warmup 후, dispose 후의 stage별 메모리 snapshot
- 20 ms 주기 프로세스 RSS·USS·PSS peak
- 시스템 RAM used peak와 available minimum
- `mblt_tracker`가 제공될 때 NPU memory 평균/최대, utilization, power, temperature
- Patch Slimming architecture 기반 블록별 activation tensor-volume 추정

```bash
cd /media/airlab_compression/nvme_storage/npu

python3 benchmark/model_zoo/run_model_zoo_benchmark.py \
  --model-class ViT_Tiny_Patch16_224 \
  --mxq /path/to/model-zoo-compatible.mxq \
  --val-dir /path/to/imagenet_val \
  --architecture /path/to/Server_Compression/output/ps_final_12way_20260806/tiny30__paper__rel002/architecture.json \
  --embed-dim 192 \
  --heads 3 \
  --classes 200 \
  --images-per-class 5 \
  --infer-mode global8 \
  --warmup-runs 10 \
  --timed-runs 1 \
  --memory-sample-ms 20 \
  --label tiny30_paper_rel002_memory
```

Tiny30의 구조적 pruning으로 MLP hidden width가 768과 다르면 `--mlp-hidden`으로
대표값을 명시할 수 있다. 블록별 폭이 서로 다를 때 이 값은 근사치이므로 attention
matrix 항목을 우선 비교하고, 정확한 블록별 MLP activation은 Server_Compression의
PyTorch profiler 결과를 사용한다.

중요: `mblt_tracker`의 `npu_max_mem_mb`는 실제 장치 측정값이지만,
`activation_estimate`는 qbruntime이 내부 tensor를 공개하지 않기 때문에
architecture와 dtype 가정으로 계산한 tensor-volume 상한이다. JSON에서도 두 값을
서로 다른 필드로 저장한다. NPU 컴파일러가 적용한 buffer reuse, tiling, INT8/FP16
혼합 정밀도까지 포함한 실제 activation allocation으로 해석하면 안 된다.
`--architecture`를 사용하면 결과 JSON과 함께 `_activation.csv`도 생성되어 블록별
토큰 수, FP16/FP32 attention matrix 크기 및 tensor-volume 상한을 바로 비교할 수 있다.

## 의존성

NPU SDK 환경에 다음 패키지가 필요합니다.

```text
mblt-model-zoo
mblt-tracker       # 선택 사항
psutil
torch
opencv-python      # direct qbruntime 평가에 필요
numpy
qbruntime
```
