# 제출용 폴더 구조

| 경로 | 역할 |
|---|---|
| `README.md` | 목적, 파이프라인, 최종 NPU 12개 결과 |
| `compile/calibration/generate_hwc.py` | HWC FP32 calibration 생성 |
| `compile/mxq/compile_hwc.py` | 최종 preset/global8 MXQ 컴파일 |
| `compile/onnx/remove_identity_selections.py` | identity 선택 점검/보조 최적화 |
| `benchmark/final/eval_final_models.py` | 전체 ImageNet 정확도·성능·telemetry |
| `benchmark/memory/profile_memory.py` | 새 프로세스 기반 메모리 재검증 |
| `benchmark/runtime/run_npu.py` | 단일 이미지 NPU 확인 |
| `benchmark/onnx/infer_onnx.py` | 단일 이미지 ONNX CPU reference |
| `scripts/verify_release.py` | 최종 파일·결과·Git 포함 대상 검증 |
| `tests/` | NPU 없이 실행하는 회귀 테스트 |
| `assets/onnx/` | 최종 ONNX 12개 |
| `assets/mxq_final_20261006/` | 실제 평가한 MXQ 12개 |
| `assets/labels/` | ImageNet class 표시명 |
| `assets/RELEASE_MANIFEST.json` | 모델/원본 결과 해시 |
| `results/final_benchmark/final_20261015/` | 최종 실험 원본, 수정하지 않음 |
| `docs/` | 방법론, ONNX 계약, 실행, 결과, 제출 안내 |

`assets/mxq/`, calibration 이미지/NPY, 새 실행 결과, 로그, SDK 설치물과 `past/`는 Git 제출 대상이 아니다. `past/submission_cleanup_20261020/`에는 정리 전 문서·코드·SDK·로컬 자료를 분류해 보존했다. 이 폴더는 실행 파이프라인의 의존성이 아니며 외부 제출에 포함하지 않는다.

모델 파일명/최종 결과 경로는 기존 기록과 대응하도록 유지했다. 파일 이동으로 바뀐 label 경로는 실행 코드에 반영했다. 최종 측정 JSON 내부의 원래 서버 절대경로는 증거이므로 수정하지 않는다.
