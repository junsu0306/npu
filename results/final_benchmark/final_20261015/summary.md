# 최종 NPU 벤치마크 결과

## 평가 개요

| 항목 | 결과 |
|---|---|
| 최종 상태 | **PASS** (`complete`) |
| 평가 모델 | 12개 중 12개 정상, 오류 이미지 0장 |
| 데이터셋 | ImageNet validation 50000장/모델 (`/media/airlab_compression/nvme_storage/imagenet_val`) |
| 총 추론 수 | 600000회 |
| 최종 모델 폴더 | `/media/airlab_compression/nvme_storage/npu/assets/mxq_final_20261006` |
| 전처리 / 코어 | `timm` / `global8`, device 0 |
| Warmup | 모델당 10회 |
| NPU telemetry | 12/12 모델 정상 수집 |
| 평가 시간 합계 | 1.86시간 |
| 실행 시작 시각 / 시작 HEAD | `2026-10-15T17:22:26.273756+09:00` / `4575bb8f73b5` |
| 실행 시 worktree | `dirty` |
| Runtime / CLI | qbruntime `v1.1.0` / mobilint-cli `v1.2.0` |
| Aries driver / firmware | `1.11.0` / `1.2.5` |
| 샘플 SHA-256 | `a461792c70ea36081d54a238c95d595ca6c87f8e8c49a42238d3db5ea2029703` |

## 핵심 비교

압축 모델 중 정확도가 가장 높은 모델, 가장 빠른 모델, NPU 메모리가 가장 적은 모델을 그룹별로 요약했다.

| Group | Baseline Top-1 | 최고 압축 정확도 | 가장 빠른 압축 모델 | 최저 NPU 메모리 |
|---|---:|---|---|---|
| tiny30 | 74.226% | Reduced: 70.880% (-3.346 pp) | Patch Slimming: 2.408 ms (1.33x) | Patch Slimming: 7.0/7.0 MB (41.7%) |
| tiny50 | 74.226% | Reduced: 62.442% (-11.784 pp) | Patch Slimming: 2.388 ms (1.33x) | Patch Slimming: 5.0/5.0 MB (58.3%) |
| small30 | 80.486% | Reduced: 79.324% (-1.162 pp) | Patch Slimming: 4.585 ms (1.28x) | Patch Slimming: 23.0/23.0 MB (34.3%) |
| small50 | 80.486% | Reduced: 73.616% (-6.870 pp) | Patch Slimming: 4.204 ms (1.38x) | Patch Slimming: 18.0/18.0 MB (48.6%) |

## 결과 해석

- **tiny30**: Reduced는 Top-1 `70.880%` (`-3.346 pp`), latency `4.0%` 감소, NPU 메모리 `8.3%` 감소. Patch Slimming은 Top-1 `53.528%` (`-20.698 pp`), latency `24.6%` 감소, NPU 메모리 `41.7%` 감소.
- **tiny50**: Reduced는 Top-1 `62.442%` (`-11.784 pp`), latency `6.1%` 감소, NPU 메모리 `16.7%` 감소. Patch Slimming은 Top-1 `41.164%` (`-33.062 pp`), latency `24.7%` 감소, NPU 메모리 `58.3%` 감소.
- **small30**: Reduced는 Top-1 `79.324%` (`-1.162 pp`), latency `5.4%` 감소, NPU 메모리 `20.0%` 감소. Patch Slimming은 Top-1 `70.094%` (`-10.392 pp`), latency `22.1%` 감소, NPU 메모리 `34.3%` 감소.
- **small50**: Reduced는 Top-1 `73.616%` (`-6.870 pp`), latency `8.8%` 감소, NPU 메모리 `31.4%` 감소. Patch Slimming은 Top-1 `61.082%` (`-19.404 pp`), latency `27.8%` 감소, NPU 메모리 `48.6%` 감소.
- 압축 모델 중 정확도 보존이 가장 좋은 구성은 **small30 Reduced**로, Top-1 `79.324%`와 baseline 대비 `-1.162 pp`를 기록했다.
- NPU 할당 메모리는 압축 여부와 관계없이 같지 않았다. 통합 평가와 독립 프로세스 재검증 모두에서 각 압축 모델의 감소가 동일하게 재현됐다.
- Patch Slimming은 모든 그룹에서 가장 빠르고 메모리가 가장 작았지만 정확도 손실도 Reduced보다 컸다. 특히 Tiny50은 `-33.062 pp`이므로 정확도 우선 최종 모델로 보기 어렵다.
- 50% 목표는 Tiny와 Small 모두 정확도 손실이 커졌다. 정확도 우선이라면 이번 결과에서는 **Small30 Reduced**, Tiny가 필요하다면 **Tiny30 Reduced**가 상대적으로 안전한 선택이다.

## 정확도 및 모델 압축

| Group | Model | 상태 | Top-1 | Δ Top-1 | Top-5 | 예측 일치율 | Params M | Params 감소 | GFLOPs* | FLOPs 감소 | MXQ MB | MXQ 감소 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny30 | Baseline | ok | 74.226% | 0.000 pp | 92.270% | 100.000% | 5.717 | 0.0% | 2.507 | 0.0% | 7.6 | 0.0% |
| tiny30 | Patch Slimming | ok | 53.528% | -20.698 pp | 78.384% | 57.954% | 4.003 | 30.0% | 0.990 | 60.5% | 5.7 | 24.0% |
| tiny30 | Reduced | ok | 70.880% | -3.346 pp | 90.744% | 80.150% | 4.003 | 30.0% | 1.834 | 26.9% | 6.0 | 20.6% |
| tiny50 | Baseline | ok | 74.226% | 0.000 pp | 92.270% | 100.000% | 5.717 | 0.0% | 2.507 | 0.0% | 7.6 | 0.0% |
| tiny50 | Patch Slimming | ok | 41.164% | -33.062 pp | 66.934% | 44.200% | 2.862 | 49.9% | 0.750 | 70.1% | 4.6 | 39.1% |
| tiny50 | Reduced | ok | 62.442% | -11.784 pp | 85.314% | 69.646% | 2.862 | 49.9% | 1.385 | 44.7% | 4.9 | 35.7% |
| small30 | Baseline | ok | 80.486% | 0.000 pp | 95.852% | 100.000% | 22.051 | 0.0% | 9.198 | 0.0% | 24.5 | 0.0% |
| small30 | Patch Slimming | ok | 70.094% | -10.392 pp | 90.184% | 75.032% | 15.434 | 30.0% | 4.245 | 53.9% | 18.1 | 25.9% |
| small30 | Reduced | ok | 79.324% | -1.162 pp | 95.102% | 87.130% | 15.434 | 30.0% | 6.594 | 28.3% | 18.3 | 25.2% |
| small50 | Baseline | ok | 80.486% | 0.000 pp | 95.852% | 100.000% | 22.051 | 0.0% | 9.198 | 0.0% | 24.5 | 0.0% |
| small50 | Patch Slimming | ok | 61.082% | -19.404 pp | 84.134% | 65.582% | 11.023 | 50.0% | 2.998 | 67.4% | 13.8 | 43.5% |
| small50 | Reduced | ok | 73.616% | -6.870 pp | 92.124% | 80.160% | 11.023 | 50.0% | 4.859 | 47.2% | 13.9 | 43.2% |

## 추론 성능

`Inference`는 `model.infer()`만, `E2E`는 이미지 로딩과 전처리를 포함한다.

| Group | Model | 평균 ms | P50 | P95 | P99 | Inference FPS | Baseline 대비 | E2E FPS | 전처리 평균 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny30 | Baseline | 3.192 | 3.174 | 3.375 | 3.559 | 313.29 | 1.00x | 99.06 | 6.722 |
| tiny30 | Patch Slimming | 2.408 | 2.389 | 2.570 | 2.732 | 415.24 | 1.33x | 120.83 | 5.710 |
| tiny30 | Reduced | 3.065 | 3.047 | 3.244 | 3.413 | 326.32 | 1.04x | 105.01 | 6.282 |
| tiny50 | Baseline | 3.170 | 3.153 | 3.346 | 3.510 | 315.41 | 1.00x | 102.79 | 6.381 |
| tiny50 | Patch Slimming | 2.388 | 2.370 | 2.555 | 2.727 | 418.83 | 1.33x | 120.66 | 5.741 |
| tiny50 | Reduced | 2.976 | 2.960 | 3.146 | 3.311 | 336.03 | 1.07x | 106.31 | 6.258 |
| small30 | Baseline | 5.884 | 5.874 | 6.133 | 6.314 | 169.96 | 1.00x | 69.75 | 8.234 |
| small30 | Patch Slimming | 4.585 | 4.569 | 4.807 | 4.978 | 218.09 | 1.28x | 83.62 | 7.178 |
| small30 | Reduced | 5.569 | 5.556 | 5.817 | 6.007 | 179.58 | 1.06x | 73.09 | 7.904 |
| small50 | Baseline | 5.821 | 5.809 | 6.074 | 6.254 | 171.79 | 1.00x | 71.59 | 7.939 |
| small50 | Patch Slimming | 4.204 | 4.186 | 4.419 | 4.587 | 237.87 | 1.38x | 87.12 | 7.085 |
| small50 | Reduced | 5.310 | 5.295 | 5.531 | 5.707 | 188.31 | 1.10x | 76.95 | 7.489 |

## NPU 메모리 및 사용률

| Group | Model | 메모리 평균 MB | P99 MB | 피크 MB | 피크 감소 | 사용률 평균 | P99 | 최대 | Samples |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny30 | Baseline | 12.0 | 12.0 | 12.0 | 0.0% | 24.0% | 27.0% | 27.7% | 1011 |
| tiny30 | Patch Slimming | 7.0 | 7.0 | 7.0 | 41.7% | 21.0% | 22.8% | 23.8% | 829 |
| tiny30 | Reduced | 11.0 | 11.0 | 11.0 | 8.3% | 24.3% | 26.6% | 28.4% | 954 |
| tiny50 | Baseline | 12.0 | 12.0 | 12.0 | 0.0% | 25.0% | 27.7% | 29.6% | 974 |
| tiny50 | Patch Slimming | 5.0 | 5.0 | 5.0 | 58.3% | 20.3% | 21.9% | 22.6% | 830 |
| tiny50 | Reduced | 10.0 | 10.0 | 10.0 | 16.7% | 23.8% | 26.4% | 27.1% | 942 |
| small30 | Baseline | 35.0 | 35.0 | 35.0 | 0.0% | 34.7% | 41.3% | 43.6% | 1435 |
| small30 | Patch Slimming | 23.0 | 23.0 | 23.0 | 34.3% | 31.8% | 37.3% | 38.2% | 1197 |
| small30 | Reduced | 28.0 | 28.0 | 28.0 | 20.0% | 34.5% | 41.9% | 44.0% | 1370 |
| small50 | Baseline | 35.0 | 35.0 | 35.0 | 0.0% | 35.6% | 42.7% | 44.8% | 1398 |
| small50 | Patch Slimming | 18.0 | 18.0 | 18.0 | 48.6% | 29.8% | 35.2% | 36.6% | 1149 |
| small50 | Reduced | 24.0 | 24.0 | 24.0 | 31.4% | 34.7% | 41.3% | 43.3% | 1301 |

## NPU 전력 및 온도

Total power는 보드 전체, NPU core power는 NPU 코어 전력이다.

| Group | Model | Total W 평균 | P99 | 최대 | NPU core W 평균 | 최대 | 온도 °C 평균 | P99 | 최대 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny30 | Baseline | 16.88 | 17.20 | 17.26 | 6.99 | 7.12 | 45.1 | 48.0 | 48.0 |
| tiny30 | Patch Slimming | 16.61 | 16.68 | 16.73 | 6.82 | 6.94 | 43.4 | 44.0 | 44.0 |
| tiny30 | Reduced | 16.80 | 16.90 | 16.92 | 6.94 | 7.01 | 43.5 | 44.0 | 44.0 |
| tiny50 | Baseline | 16.86 | 16.98 | 17.05 | 6.99 | 7.09 | 43.5 | 44.0 | 44.0 |
| tiny50 | Patch Slimming | 16.57 | 16.64 | 16.95 | 6.79 | 7.04 | 43.5 | 44.0 | 44.0 |
| tiny50 | Reduced | 16.74 | 16.83 | 16.87 | 6.90 | 6.95 | 43.5 | 44.0 | 45.0 |
| small30 | Baseline | 17.43 | 17.74 | 17.97 | 7.38 | 7.64 | 44.4 | 45.0 | 45.0 |
| small30 | Patch Slimming | 17.01 | 17.21 | 17.71 | 7.04 | 7.51 | 43.9 | 45.0 | 45.0 |
| small30 | Reduced | 17.18 | 17.47 | 17.57 | 7.19 | 7.38 | 43.8 | 45.0 | 45.0 |
| small50 | Baseline | 17.43 | 17.77 | 17.84 | 7.38 | 7.63 | 43.5 | 44.0 | 44.0 |
| small50 | Patch Slimming | 16.88 | 17.05 | 17.42 | 6.95 | 7.38 | 42.7 | 44.0 | 44.0 |
| small50 | Reduced | 17.07 | 17.32 | 17.39 | 7.11 | 7.33 | 43.0 | 44.0 | 44.0 |

## 호스트 프로세스 메모리

| Group | Model | RSS 시작 MB | RSS 피크 MB | RSS 증가 MB | USS 피크 MB | PSS 피크 MB |
|---|---|---:|---:|---:|---:|---:|
| tiny30 | Baseline | 240.0 | 332.5 | 92.5 | 321.8 | 322.9 |
| tiny30 | Patch Slimming | 279.0 | 335.5 | 56.5 | 332.3 | 333.4 |
| tiny30 | Reduced | 279.0 | 377.9 | 98.9 | 374.6 | 375.7 |
| tiny50 | Baseline | 279.9 | 351.9 | 72.1 | 339.5 | 340.6 |
| tiny50 | Patch Slimming | 280.4 | 378.8 | 98.3 | 375.6 | 376.7 |
| tiny50 | Reduced | 280.3 | 334.9 | 54.6 | 323.7 | 324.8 |
| small30 | Baseline | 281.1 | 337.4 | 56.3 | 341.9 | 342.9 |
| small30 | Patch Slimming | 281.6 | 349.1 | 67.5 | 342.1 | 343.2 |
| small30 | Reduced | 282.1 | 346.4 | 64.2 | 338.9 | 340.0 |
| small50 | Baseline | 282.7 | 384.8 | 102.1 | 377.4 | 378.4 |
| small50 | Patch Slimming | 281.6 | 338.3 | 56.7 | 327.0 | 328.1 |
| small50 | Reduced | 281.8 | 341.4 | 59.5 | 332.5 | 333.6 |

## 아티팩트 무결성

| Group | Model | MXQ bytes | MXQ SHA-256 | ONNX bytes | ONNX SHA-256 |
|---|---|---:|---|---:|---|
| tiny30 | Baseline | 7929396 | `96de47b4f6aa3698ba76049f3208acdb5201b48e2221b0ecaa73cb5314b2aa54` | 22952270 | `0d8655a8e7641e9023c6464f7c97ca172b447709f5c2b3abb6c91d83a19b156b` |
| tiny30 | Patch Slimming | 6022812 | `90a9b4ddd7eaf6492539eab8f4af02ea8420e46453d92ef2df3f5721ad1d1113` | 16379897 | `08d4014de88f6f156dd88e312e240bf71e3aff0af473b2fd33aba03d7e5a5fa1` |
| tiny30 | Reduced | 6297268 | `0c651c7c4b202aa400671924fa5ef7be51c21900df82d1606653c7c834f03f0c` | 16096190 | `bc628a54be653609a895d279eb43826827eeaaddc3cf8ea23c3855108551cff6` |
| tiny50 | Baseline | 7929396 | `2c740f6034989d5512d3e77a1df693f15c3c2076346565f018e3b77cbbff73ff` | 22952270 | `0d8655a8e7641e9023c6464f7c97ca172b447709f5c2b3abb6c91d83a19b156b` |
| tiny50 | Patch Slimming | 4826234 | `8626d858b2fb43e3023c226be212173d0d358e4e4bf88c9827b072cabfc116f4` | 11815319 | `90086de3e74482a4fdbc25b0065e6d2dce9895cb1e417aa9af63d3f77302f65c` |
| tiny50 | Reduced | 5100498 | `52be73fb93398a53f8e0c6c9235ffa1c9320e36616b003cda71e7031064a6cb1` | 11531612 | `aa122558ccb0ea5b3cd081d6b8ed1129948edb95bcbcb6e1a63753859b9fe169` |
| small30 | Baseline | 25669056 | `f6d45ea2ce23104d184bbb42b09cd8515e6244eb25485367485715784f58fd48` | 88285310 | `9812da2a85262ee515dfb7947f5ced1186a0f447005e7a7988c6bb996726cdc0` |
| small30 | Patch Slimming | 19027674 | `bbd41a00058593de61acce35bd1121882551b861d17866db0c7e10ffa36b879a` | 62152349 | `0146dde675d79f36d95529ddda67ace59dca3a7a5f718fb02f4039716ab3b6a1` |
| small30 | Reduced | 19197292 | `7411c8c7441d1244fdfb8b88c15294f419a7a3cc1e2234dd827a2ed4dd39d0ab` | 61819358 | `1bf7f5fade8b32de735151f01f714e63d98e815ef9204fc2562ff3e978bad5cd` |
| small50 | Baseline | 25669056 | `590a31cfb6dd8dcbfab5ba30d455d7eaa2f7c0d690552435f7e66b0afb31034f` | 88285310 | `9812da2a85262ee515dfb7947f5ced1186a0f447005e7a7988c6bb996726cdc0` |
| small50 | Patch Slimming | 14501449 | `668741deb2709383a9caa521f3bf06d3d1a9393f999eec47558173b662d3ef19` | 45030777 | `ec63a02ee86783a268a4cfb0f001aa16fdbfb549128a9a2c8d0d794952789ce3` |
| small50 | Reduced | 14579718 | `3719c948c08f3e79a6bdbfc9a4917d334781c46dffda0483eaf5e9a4a25617a1` | 44175422 | `0caf8659b7232bedff19e42a2f798b8b508338a80c8381047630d422671183fc` |

## 독립 프로세스 메모리 재검증

각 모델을 새 프로세스에서 고정 random FP32 입력으로 실행했다. Idle/Active는 장치 전체에서 `mobilint-cli`가 보고한 MB이며, host 값은 프로세스 MiB이다.

| Group | Model | 상태 | Idle MB | Active peak MB | Δ NPU MB | Host RSS peak MiB | Host USS peak MiB | 평균 ms | Telemetry runs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| tiny30 | Baseline | ok | 0.0 | 12.0 | 12.0 | 141.6 | 122.2 | 3.155 | 3186 |
| tiny30 | Reduced | ok | 0.0 | 11.0 | 11.0 | 141.9 | 122.9 | 3.043 | 3288 |
| tiny30 | Patch Slimming | ok | 0.0 | 7.0 | 7.0 | 138.3 | 119.0 | 2.415 | 4074 |
| tiny50 | Baseline | ok | 0.0 | 12.0 | 12.0 | 137.9 | 118.9 | 3.141 | 3185 |
| tiny50 | Reduced | ok | 0.0 | 10.0 | 10.0 | 139.6 | 120.2 | 2.966 | 3407 |
| tiny50 | Patch Slimming | ok | 0.0 | 5.0 | 5.0 | 138.2 | 118.9 | 2.466 | 4229 |
| small30 | Baseline | ok | 0.0 | 35.0 | 35.0 | 139.7 | 120.4 | 5.664 | 1752 |
| small30 | Reduced | ok | 0.0 | 28.0 | 28.0 | 144.0 | 125.1 | 5.434 | 1827 |
| small30 | Patch Slimming | ok | 0.0 | 23.0 | 23.0 | 139.1 | 120.4 | 4.585 | 2197 |
| small50 | Baseline | ok | 0.0 | 35.0 | 35.0 | 137.7 | 118.4 | 5.736 | 1756 |
| small50 | Reduced | ok | 0.0 | 24.0 | 24.0 | 140.0 | 121.1 | 5.197 | 1940 |
| small50 | Patch Slimming | ok | 0.0 | 18.0 | 18.0 | 141.5 | 122.4 | 4.180 | 2453 |

독립 측정은 통합 평가와 동일한 NPU 메모리 peak를 12/12 모델에서 재현했다. Idle은 모든 모델에서 0 MB였으며, 프로세스 분리로 모델 간 allocator 잔존 영향을 배제했다.

## 측정 기준

- `GFLOPs*`는 ONNX Conv/MatMul/Gemm 정적 shape로 계산했으며 `1 MAC = 2 FLOPs`를 적용했다. Elementwise, normalization, activation, softmax와 컴파일러 fusion/tiling은 포함하지 않는다.
- NPU 메모리·전력·사용률·온도는 `mblt_tracker`가 `mobilint-cli status`를 주기적으로 관측한 값이다. 메모리 피크는 관측 구간의 최대 할당량이며 연산 중 초단기 내부 activation peak와는 다르다.
- 정확도 비교는 모든 모델에 동일한 이미지 목록, label, `timm` 전처리, Global8 조건을 사용했다.
- 상세 원본 지표와 오류 정보는 같은 폴더의 `benchmark.json`, 분석용 평탄화 지표는 `summary.csv`에 저장된다.
