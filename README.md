# Mobilint NPU Deployment & Benchmark

ViT-Tiny / ViT-Small 압축 모델을 **ONNX → MXQ 컴파일 → 실제 Mobilint NPU 평가**로 연결하는 과제 제출용 프로젝트입니다. 최종 기준은 **2026-10-15에 완료한 `final_20261015` 실험**입니다.

- [최종 결과 원본](results/final_benchmark/final_20261015/summary.md) · [결과 해석](docs/RESULTS.md)
- [방법론과 측정 기준](docs/METHODOLOGY.md) · [실행 명령](docs/RUNBOOK.md)
- [폴더 구조](docs/REPOSITORY_LAYOUT.md) · [제출/Git 안내](docs/GIT_RELEASE.md)
- [산출물 SHA-256 목록](assets/RELEASE_MANIFEST.json)
- [제출 패키지 검증 기록](docs/VALIDATION.md)

## 프로젝트 범위

| 프로젝트 | 역할 |
|---|---|
| `Server_Compression` | 목표 구조 기반 메모리 예산, global L2 FFN pruning + logit KD + 메모리 항, reducing, Statistical Patch Slimming, 학습·FP32 평가·ONNX export |
| **이 저장소 `npu`** | calibration tensor 생성, qbcompiler MXQ 컴파일, 실제 NPU 정확도·지연시간·메모리·전력 평가 |

학습과 ONNX 생성은 서버 프로젝트에서 완료한 산출물을 사용합니다. 이 저장소에서 다시 학습하거나 `Orin_Compression`을 별도로 실행하지 않습니다. 컴파일용 CUDA 서버와 실제 NPU 실행 장치는 서로 다른 환경입니다.

## 최종 대상과 파이프라인

Tiny/Small × 목표 파라미터 감소 30%/50% × Baseline/Reduced/Patch Slimming = **ONNX 12개 + MXQ 12개**입니다. Baseline은 같은 family의 원본 모델을 30/50 비교 그룹마다 배치한 것으로 서로 다른 압축 모델이 아닙니다.

```text
Server_Compression 최종 ONNX (FP32, NHWC, batch 1)
  → ImageNet calibration 1,000장 → 정규화된 HWC FP32 tensor
  → qbcompiler: vision_transformer preset, global8, CPU offload 허용
  → MXQ
  → Mobilint Aries: ImageNet validation 50,000장/모델
  → 정확도·latency·host/NPU 메모리·전력·정적 ONNX 연산량
  → 모델별 독립 프로세스로 NPU 메모리 재검증
```

Patch Slimming은 입력마다 패치를 선택하지 않습니다. 학습 단계에서 고정한 패치 ID를 **상수 0/1 선택행렬 MatMul**로 표현해 토큰 텐서 크기를 줄입니다. Identity 선택은 생략하고 마지막 CLS는 static Slice/Squeeze로 처리합니다. [ONNX 계약](docs/ONNX_CONTRACT.md)

## 최종 NPU 실측 결과

동일한 ImageNet validation **50,000장**, HWC FP32 입력, batch 1, `global8`, 모델별 warmup 10회 조건입니다. 총 600,000장 평가에서 **12/12 정상 완료, 이미지 처리 오류 0건**, telemetry도 12/12 수집했습니다.

| 그룹 | 모델 | Top-1 (%) | Top-5 (%) | 평균 추론 (ms) | NPU peak (MB) |
|---|---|---:|---:|---:|---:|
| Tiny30 | Baseline | 74.226 | 92.270 | 3.192 | 12 |
| Tiny30 | Reduced | 70.880 | 90.744 | 3.065 | 11 |
| Tiny30 | Patch Slimming | 53.528 | 78.384 | 2.408 | 7 |
| Tiny50 | Baseline | 74.226 | 92.270 | 3.170 | 12 |
| Tiny50 | Reduced | 62.442 | 85.314 | 2.976 | 10 |
| Tiny50 | Patch Slimming | 41.164 | 66.934 | 2.388 | 5 |
| Small30 | Baseline | 80.486 | 95.852 | 5.884 | 35 |
| Small30 | Reduced | 79.324 | 95.102 | 5.569 | 28 |
| Small30 | Patch Slimming | 70.094 | 90.184 | 4.585 | 23 |
| Small50 | Baseline | 80.486 | 95.852 | 5.821 | 35 |
| Small50 | Reduced | 73.616 | 92.124 | 5.310 | 24 |
| Small50 | Patch Slimming | 61.082 | 84.134 | 4.204 | 18 |

출처: [원본 JSON](results/final_benchmark/final_20261015/benchmark.json), [CSV](results/final_benchmark/final_20261015/summary.csv). 위 값은 서버 FP32 정확도나 공식 모델 성능표가 아닌 **MXQ의 실제 NPU 평가값**입니다.

- Patch Slimming은 baseline 대비 **추론 지연시간 22.1~27.8%, NPU peak 34.3~58.3% 감소**했습니다. 독립 프로세스 측정에서도 같은 메모리 값이 12/12 재현됐습니다.
- 반면 정확도 손실이 큽니다. Patch Slimming의 reduced 대비 Top-1 추가 하락은 Tiny30 **17.352%p**, Tiny50 **21.278%p**, Small30 **9.230%p**, Small50 **12.534%p**입니다.
- 압축 모델 중 정확도가 가장 높은 구성은 Small30 Reduced **79.324%**입니다. 완료 상태 `PASS`는 평가 실행 성공이며, 정확도 보존/목표 달성을 의미하지 않습니다.
- NPU peak는 장치 전체 telemetry의 관측 최대값입니다. CUDA allocator peak, 프로세스 RSS 또는 레이어별 live activation peak와 같은 지표가 아닙니다.

## 환경과 전처리

측정 환경: Python 3.10.12, qbruntime v1.1.0, mobilint-cli v1.2.0, Aries driver 1.11.0 / firmware 1.2.5, NumPy 2.2.6, OpenCV 4.13.0. 원본 기록은 [run_metadata.json](results/final_benchmark/final_20261015/run_metadata.json)에 있습니다.

`timm`은 이 저장소의 전처리 프로필 이름입니다. **OpenCV RGB → bicubic 단변 249 → center crop 224 → mean/std 모두 0.5**를 뜻합니다. 서버의 PIL/timm 전처리와 바이트 단위로 동일하다는 뜻은 아닙니다.

```bash
python3 -m pip install -r requirements-npu.txt
python3 scripts/verify_release.py
```

`verify_release.py`의 기본 검증은 Python 표준 라이브러리만 사용하므로 SDK나 NPU 없이 가능합니다. 실제 추론은 Mobilint SDK의 `qbruntime`과 장치가 필요합니다. 재컴파일은 별도 Docker의 `qbcompiler`와 CUDA가 필요합니다. 시스템 CUDA/드라이버/SDK를 이 저장소가 자동 변경하지 않습니다.

전체 재현 명령은 [RUNBOOK](docs/RUNBOOK.md)을 따릅니다. **기존 최종 모델과 `final_20261015` 결과를 덮어쓰지 않고** 새 실험 디렉터리에 출력합니다.

## 제출물 구성

```text
compile/                        calibration / MXQ / ONNX 보조 검증
benchmark/                      final / memory / runtime / onnx
scripts/verify_release.py        파일·결과·Git 포함 대상 검증
tests/                          NPU 없이 실행하는 회귀 테스트
assets/onnx/                    최종 ONNX 12개
assets/mxq_final_20261006/        실제 평가한 MXQ 12개
assets/RELEASE_MANIFEST.json     모델 + 원본 결과 파일 해시
results/final_benchmark/
  final_20261015/                원본 JSON/CSV/MD, 예측값, 샘플 목록, 메모리 재검증
docs/                           최종 방법론·실행·결과·제출 문서
past/                           로컬 보존 자료, Git/제출 제외
```

ImageNet 이미지/calibration tensor와 SDK wheel·매뉴얼은 배포물에 포함하지 않습니다. 데이터와 SDK는 사용 권한이 있는 환경에서 별도 준비합니다. 기존 Git 이력을 재작성하지 않으므로 과거 커밋에는 제외한 파일이 남아 있을 수 있습니다. **실제 외부 제출에는 현재 커밋의 파일만 export**하고, Git 저장소 자체를 전달할 때는 [제출 안내](docs/GIT_RELEASE.md)를 확인하세요.
