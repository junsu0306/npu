# 최종 NPU 배포 및 측정 방법론

## 1. 모델 출처와 책임 범위

서버 실험 `final_20261006`에서 생성한 ONNX를 입력으로 받는다. 서버 단계는 목표 구조의 추가 학습 없는 사본을 reducing해 CUDA peak 예산을 정한 후, global L2 FFN pruning + logit KL KD + 측정 기반 메모리 항으로 30epoch 학습하고 reducing한다. 이어 Statistical Patch Slimming 검색 및 30epoch fine-tuning을 수행한다.

이 저장소는 해당 학습을 재구현하지 않는다. Baseline / Reduced / Patch Slimming의 최종 ONNX를 MXQ로 변환하고 NPU에서 평가한다. 30%·50%는 전체 모델 파라미터 감소 목표이며, NPU 메모리 30%·50% 감소를 보장하는 조건이 아니다.

최종 실측 대상은 `assets/mxq_final_20261006/`의 12개이며, 대응 ONNX와 SHA-256을 `assets/RELEASE_MANIFEST.json` 및 원본 `benchmark.json`으로 연결한다.

## 2. Calibration과 컴파일

구현: `compile/calibration/generate_hwc.py`, `compile/mxq/compile_hwc.py`.

1. 사용자가 제공한 이미지 경로를 정렬하고 읽을 수 있는 이미지 1,000장을 사용한다.
2. OpenCV BGR → RGB, aspect ratio 유지 bicubic resize, 단변 249, 정수 크기 `round`, 중앙 224×224 crop.
3. float32로 255를 나눈 후 채널별 mean/std 모두 0.5를 적용한다.
4. `(224,224,3)` HWC NPY와 `preprocess_profile.json`을 생성한다.
5. qbcompiler `backend=onnx`, `config_preset=vision_transformer`, `inference_scheme=global8`, `cpu_offload=True`로 컴파일한다.

최종 컴파일 경로는 preset을 넘기고 수동 `quantization_*` 인자를 함께 넘기지 않는다. 수동 histogram/KL 설정을 최종 실험 설정으로 해석하지 않는다. Preset 내부 bit width/scale 결정은 설치된 SDK 구현에 의존하며, 모델 전체를 균일 INT8이라고 단정하지 않는다.

컴파일 시 CUDA GPU는 호스트의 양자화/컴파일 작업에 사용한다. `CUDA_VISIBLE_DEVICES=1`과 실제 NPU의 `--device-id 0`은 서로 다른 장치 설정이다.

최종 MXQ에는 실제 NPU 평가와 해시 증거가 있다. 그러나 이번 보관 기록에는 최종 컴파일의 calibration 이미지별 해시, 전체 compiler 환경 lock, 확정 quantizer config dump가 없으므로 **동일 설정으로 재컴파일한 MXQ가 byte-exact하게 재생성된다고 보장하지 않는다.** 전처리 메타데이터 없는 기존 tensor를 최종 calibration으로 추정해 사용하지 않는다.

## 3. 정적 토큰 선택

고정 패치 ID로 만든 `P[N_out,N_in] @ X[B,N_in,D]`를 적용해 `Y[B,N_out,D]`를 얻는다. 제거 위치를 0으로 남기는 마스킹이 아니라 출력의 토큰 축이 실제 작아진다.

Query 입력과 residual에는 같은 선택을 적용하고 key/value에는 해당 block의 선택 전 토큰을 사용한다. 남길 위치와 차원은 export 전에 고정한다. Identity 선택은 no-op으로 생략하고 마지막 CLS는 static Slice/Squeeze로 추출한다. 자세한 계약은 [ONNX_CONTRACT.md](ONNX_CONTRACT.md)에 있다.

이 변환은 컴파일 가능한 고정 그래프를 제공하지만, 선택 MatMul 계산 비용이나 컴파일러의 양자화 경계가 없어지는 것은 아니다. FP32 수치 동등성과 NPU 양자화 정확도는 별도로 검증해야 한다.

## 4. 전체 정확도 및 지연시간

구현: `benchmark/final/eval_final_models.py`.

- 입력 데이터: ImageNet validation 1,000 class × 50장 = 50,000장.
- 각 class 폴더의 파일을 고정 seed 42로 선택하며, 12개 모델에 동일한 순서·정답을 적용한다.
- 폴더는 정렬된 class 순서로 label을 매긴다. 기록에서는 `00000`~`00999` 폴더를 사용했다. 폴더 순서가 모델의 ImageNet label 순서와 맞아야 한다.
- 이미지 목록과 label은 `samples.tsv`, 목록 해시는 `run_metadata.json`에 저장한다.
- HWC FP32 입력, batch 1, `global8`, NPU device 0, 모델별 warmup 10회.
- 각 이미지 logits의 Top-1/Top-5를 정답과 비교한다. 오류 이미지 수를 별도 기록한다.
- `model.infer()` 호출 시간을 측정해 평균/percentile과 FPS를 계산한다. 이미지 로딩·전처리를 포함한 E2E FPS는 별도 지표다.
- 모델별 Top-1 예측 배열을 보존한다. 이 배열로 Top-1을 재검산할 수 있지만 **Top-5 전체 예측은 저장하지 않아 Top-5는 기록된 집계값으로 검증**한다.

샘플 목록 해시는 이미지 내용 해시가 아니다. 같은 파일명이라도 내용이 바뀌면 동일한 결과를 보장할 수 없다. 최종 결과를 새 데이터셋의 결과로 덮어쓰지 않는다.

## 5. 메모리·전력과 연산량

| 지표 | 측정/계산 | 해석 |
|---|---|---|
| Host RSS/USS/PSS | psutil, 최종 평가에서 20ms 주기 | 실행 프로세스의 CPU 메모리, MiB |
| NPU memory | mblt_tracker → mobilint-cli, 500ms 주기 | 장치 전체에서 관측한 MB 값의 평균/최대 |
| 전력 | 같은 telemetry 구간 | 보드 전체 전력과 NPU core 전력을 구분 |
| 이론 GFLOPs | ONNX Conv/MatMul/Gemm 정적 shape, 1 MAC=2 FLOPs | elementwise/norm/softmax 및 compiler fusion/tiling 제외 |
| ONNX parameter 수 | 정적 분석 | 학습 weight 계수; 실제 MXQ 저장량/메모리와 다름 |

Telemetry는 레이어별 순간 activation peak가 아니다. 다른 프로세스가 같은 NPU를 사용하면 장치 전체 값에 섞인다. 측정 중 독점 사용을 권장하며 idle 상태를 확인한다. Host 프로세스 메모리와 NPU 메모리를 더해 단일 GPU/CUDA 지표와 비교하지 않는다.

구현 `benchmark/memory/profile_memory.py`는 모델마다 새 프로세스로 실행한다. 최종 독립 측정은 **고정 random NHWC FP32 입력(seed 2022), warmup 10회, latency 측정 100회**, 이후 별도 10초 반복 추론 구간에서 telemetry를 수집했다. 메모리 반복 횟수는 100회로 고정된 것이 아니며 실제 `telemetry_runs`를 기록한다.

독립 측정의 idle은 12개 모두 0 MB였고 통합 평가와 동일한 NPU peak를 재현했다. 이것은 해당 관측 구간의 일치이며 모든 짧은 순간의 내부 allocation을 포착했다는 뜻은 아니다.

## 6. 성공 조건과 한계

실행 완료, 모델/이미지 수, 오류 0건, telemetry 정상, 모델 해시를 함께 검사한다. 원본 보고서의 `PASS`를 정확도 목표 달성으로 해석하지 않는다. Patch Slimming은 최종 측정에서 속도·메모리 이득과 동시에 큰 정확도 손실을 보였다.

원본 결과의 실행 시작 HEAD는 `4575bb8…`, worktree는 `dirty`이며 결과 커밋은 `3e8bb5a…`이다. 원본 metadata를 수정해 깨끗한 실행으로 바꾸지 않는다. 정리 후 추가한 경로/입력 검증은 새 실행의 안전장치이며, 과거 측정을 새 코드로 다시 실행한 결과로 표현하지 않는다.
