# 최종 결과: final_20261015

유일한 제출 기준은 [2026-10-15 원본 보고서](../results/final_benchmark/final_20261015/summary.md)다. 과거 진단 결과나 일부 이미지 평가를 최종 성능으로 혼용하지 않는다.

## 결과와 근거

| 항목 | 원본 |
|---|---|
| 모델별 Top-1/Top-5, latency 분포, telemetry | [benchmark.json](../results/final_benchmark/final_20261015/benchmark.json) |
| 분석용 수치 표 | [summary.csv](../results/final_benchmark/final_20261015/summary.csv) |
| 환경·명령어·샘플 목록 해시 | [run_metadata.json](../results/final_benchmark/final_20261015/run_metadata.json) |
| 실제 평가 순서/정답 | [samples.tsv](../results/final_benchmark/final_20261015/samples.tsv) |
| 독립 프로세스 메모리 | [isolated_memory/summary.csv](../results/final_benchmark/final_20261015/isolated_memory/summary.csv) |
| 모델·결과 파일 동일성 | [RELEASE_MANIFEST.json](../assets/RELEASE_MANIFEST.json) |

12개 모델 × 50,000장, 오류 이미지 0건, NPU telemetry 12/12 정상이다. Top-1 예측값과 정답 목록을 이용한 오프라인 재검산도 `scripts/verify_release.py --predictions`로 수행할 수 있다.

## 비교 요약

아래 감소율과 정확도 차이는 모두 **동일 그룹 baseline 대비**다.

| 그룹 | Reduced Top-1 | Patch Top-1 | Reduced / Patch 지연시간 감소 | Reduced / Patch NPU peak 감소 |
|---|---:|---:|---:|---:|
| Tiny30 | 70.880% (-3.346%p) | 53.528% (-20.698%p) | 4.0% / 24.6% | 8.3% / 41.7% |
| Tiny50 | 62.442% (-11.784%p) | 41.164% (-33.062%p) | 6.1% / 24.7% | 16.7% / 58.3% |
| Small30 | 79.324% (-1.162%p) | 70.094% (-10.392%p) | 5.4% / 22.1% | 20.0% / 34.3% |
| Small50 | 73.616% (-6.870%p) | 61.082% (-19.404%p) | 8.8% / 27.8% | 31.4% / 48.6% |

Tiny baseline Top-1은 74.226%, Small baseline은 80.486%다. Baseline을 두 그룹에서 반복 측정했으므로 family 내 Top-1은 같지만 latency에는 차이가 있다.

독립 측정에서도 Tiny baseline/reduced30/reduced50/patch30/patch50은 **12/11/10/7/5 MB**, Small은 **35/28/24/23/18 MB**였다. Idle은 모두 0 MB였다.

## 제출 시 설명할 결론

1. FFN reducing은 실제 NPU 메모리와 latency를 낮췄다. Small30 Reduced는 비교한 압축 구성 중 가장 높은 정확도를 기록했다.
2. 정적 Patch Slimming은 모든 그룹에서 더 빠르고 메모리도 더 작았다. 그러나 reduced보다 Top-1이 크게 낮아져 정확도·자원 사이의 trade-off가 명확하다.
3. 최종 결과를 “정확도 저하 없이 압축 성공”으로 표현하지 않는다. `PASS`는 실행·기록이 끝났다는 뜻이다.
4. 서버 FP32와 NPU 결과를 비교할 때 서버의 PIL/timm 전처리와 이 저장소의 OpenCV 단변 249 프로필 차이를 명시한다. 해당 차이와 양자화/compiler 영향을 이번 결과만으로 분리할 수 없다.
5. NPU peak는 telemetry의 장치 전체 관측값이며 CUDA 메모리 peak나 레이어별 activation이 아니다. ONNX GFLOPs는 이론값이며 하드웨어 실행 연산량이 아니다.

## 보존 원칙

최종 모델 24개와 원본 결과 파일은 정리 과정에서 내용을 바꾸지 않는다. 실행 당시 절대경로/dirty 상태도 증거로 보존한다. 환경 독립적인 안내는 문서에 별도로 적고, 재실험은 새 출력 폴더에 기록한다.
