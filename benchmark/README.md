# 최종 Mobilint NPU 벤치마크

최종 평가는 `assets/mxq_final_20261006/`의 12개 MXQ만 대상으로 한다.

- Family: Tiny, Small
- 목표 압축률: 30%, 50%
- Variant: Baseline, Patch Slimming, Reduced
- 입력: ImageNet validation 1,000 classes × 50장 = 모델당 50,000장
- 총 추론: 12 × 50,000 = 600,000회
- 입력 계약: HWC `(224, 224, 3)` float32, batch 1
- 전처리: bicubic short-side 249, center crop 224, `mean=std=(0.5, 0.5, 0.5)`
- 실행 모드: Mobilint Aries device 0, Global8

구형 `assets/mxq/`는 최종 모델 저장소가 아니며 Git에서 제외한다. 과거 benchmark
결과도 최종 결과와 섞지 않는다.

## 사전 확인

```bash
cd /media/airlab_compression/nvme_storage/npu

mobilint-cli status
python3 -c "import qbruntime; from mblt_tracker import NPUDeviceTracker; print('runtime/tracker OK')"
python3 benchmark/final/eval_final_models.py --dry-run
```

드라이런 결과는 모델 12개, 데이터셋 50,000장, 모든 모델의 대응 ONNX `ok`여야
한다. 이번 최종 평가의 샘플 목록 SHA-256은
`a461792c70ea36081d54a238c95d595ca6c87f8e8c49a42238d3db5ea2029703`이다.

## 12개 모델 스모크 테스트

전체 평가 전에 모든 MXQ를 실제 장치에 올려 한 장씩 추론한다.

```bash
python3 benchmark/final/eval_final_models.py \
  --models-dir assets/mxq_final_20261006 \
  --onnx-dir assets/onnx \
  --val-dir /media/airlab_compression/nvme_storage/imagenet_val \
  --classes 1000 --images-per-class 50 --limit 1 \
  --preprocess timm --core-mode global8 --device-id 0 \
  --warmup-runs 2 --progress-every 1 \
  --memory-sample-ms 20 --npu-sample-ms 10 \
  --output-dir /tmp/npu_final_20261015_smoke
```

12개 모두 `status=ok`이고 NPU telemetry도 `status=ok`인지 확인한다.

## ImageNet 전체 통합 평가

```bash
python3 benchmark/final/eval_final_models.py \
  --models-dir assets/mxq_final_20261006 \
  --onnx-dir assets/onnx \
  --val-dir /media/airlab_compression/nvme_storage/imagenet_val \
  --classes 1000 --images-per-class 50 \
  --preprocess timm --core-mode global8 --device-id 0 \
  --warmup-runs 10 --progress-every 1000 \
  --memory-sample-ms 20 --npu-sample-ms 500 \
  --output-dir results/final_benchmark/final_20261015 \
  --fail-fast
```

평가기는 각 모델이 끝날 때마다 중간 결과를 원자적으로 갱신한다. `--fail-fast` 때문에
모델 로드나 추론 오류가 발생하면 다음 모델로 넘어가지 않는다.

수집 항목은 다음과 같다.

- ImageNet Top-1/Top-5, baseline 대비 정확도 차이와 예측 일치율
- `model.infer()` latency 평균·표준편차·min·p50·p90·p95·p99·max, inference FPS
- 이미지 디코딩과 전처리를 포함한 end-to-end FPS
- 프로세스 RSS/USS/PSS/VMS peak와 모델 시작 대비 RSS 증가량
- NPU memory/utilization, 보드 total power, NPU core power, temperature의 평균·p99·최대
- MXQ/ONNX 파일 크기와 SHA-256, ONNX parameter 수
- ONNX Conv/MatMul/Gemm 기준 GFLOPs와 동일 그룹 baseline 대비 절감률

## 독립 메모리 재검증

통합 평가와 별도로 모델마다 새 프로세스를 만들어 동일한 고정 random 입력으로
메모리와 latency를 다시 측정한다. 모델 간 allocator 상태가 섞이지 않는 검증이다.

```bash
python3 benchmark/memory/profile_memory.py \
  --models-dir assets/mxq_final_20261006 \
  --core-mode global8 --device-id 0 \
  --warmups 10 --runs 100 --memory-seconds 10 \
  --memory-sample-ms 20 --npu-sample-ms 500 \
  --output-dir results/final_benchmark/final_20261015/isolated_memory
```

NPU 메모리는 `mblt_tracker`/`mobilint-cli`가 보고한 장치 전체 MB 값이다. 런타임이
모델별 내부 activation peak를 공개하지 않으므로, 초단기 activation peak나 레이어별
메모리로 해석하지 않는다. 측정 중에는 다른 NPU 프로세스를 실행하지 않는다.

## 최종 결과 파일

| 파일 | 내용 |
|---|---|
| `summary.md` | 사람이 읽는 최종 표와 측정 기준 |
| `summary.csv` | 모델별 평탄화 지표 |
| `benchmark.json` | 환경, percentile, telemetry, 비교를 포함한 원본 결과 |
| `run_metadata.json` | 평가 조건과 모델 목록 |
| `models/*.json` | 모델별 상세 결과 |
| `predictions/*.npy` | 모델별 50,000개 Top-1 예측값 |
| `samples.tsv` | 평가 이미지와 정답 label의 고정 목록 |
| `isolated_memory/summary.csv` | 독립 프로세스 메모리/latency 비교 |

최종 판정은 `summary.md`의 상태가 `PASS`, 모델 12/12가 `ok`, 오류 이미지 0장,
telemetry 12/12 정상인 경우에만 통과다.

## 지표 해석 시 주의점

`GFLOPs*`는 정적 ONNX shape로 Conv/MatMul/Gemm만 계산하고 `1 MAC = 2 FLOPs`를
적용한 이론값이다. Elementwise, normalization, activation, softmax 및 컴파일러의
fusion/tiling은 포함하지 않는다. 실제 하드웨어 성능 비교에는 측정 latency/FPS를
사용한다.

`timm` 전처리는 이름만 지정하는 옵션이 아니라 정확도 계약이다. 과거 `legacy`
전처리(단변 256, bilinear, ImageNet mean/std)를 사용하면 같은 MXQ에서도 정확도가
크게 달라지므로 최종 결과와 비교할 수 없다.
