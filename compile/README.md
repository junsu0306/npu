# 최종 컴파일 도구

| 경로 | 역할 |
|---|---|
| `calibration/generate_hwc.py` | 이미지 1,000장 → 정규화된 HWC FP32 NPY + 프로필 |
| `mxq/compile_hwc.py` | `vision_transformer` preset, `global8`, CPU offload 허용으로 MXQ 생성 |
| `onnx/remove_identity_selections.py` | identity 선택 점검/보조 도구; 최종 ONNX에 재적용 불필요 |

실행 명령은 [RUNBOOK](../docs/RUNBOOK.md), 설정은 [ONNX → MXQ](../docs/onnx_to_mxq.md), 그래프 규칙은 [ONNX 계약](../docs/ONNX_CONTRACT.md)을 따른다.

컴파일 Docker에는 Mobilint `qbcompiler`와 CUDA가 준비되어 있어야 한다. 이미지·SDK는 별도 준비하며, 원본 최종 ONNX/MXQ를 덮어쓰지 않는다. 최종 calibration 기본 경로는 `assets/calib_hwc_timm`이고 재컴파일 기본 출력은 Git 제외 경로인 `assets/mxq/`다.
