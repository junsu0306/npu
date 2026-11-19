# 최종 모델 자산

| 경로 | 제출 | 내용 |
|---|---|---|
| `onnx/` | 포함 | 서버에서 export한 최종 ONNX 12개 |
| `mxq_final_20261006/` | 포함 | 2026-10-15 실제 평가한 MXQ 12개 |
| `labels/` | 포함 | ImageNet class 표시명 |
| `RELEASE_MANIFEST.json` | 포함 | 모델 및 원본 결과 파일 SHA-256/크기 |
| `calibration_data/` | 제외 | 사용 권한이 있는 이미지, 로컬에서 별도 준비 |
| `calib_hwc_timm/` | 제외 | 재생성하는 calibration tensor/프로필 |
| `mxq/` | 제외 | 새 컴파일·검증용 임시 산출물 |

모델명은 `{tiny30,tiny50,small30,small50}__{baseline,reduced,patch_slimming}__nhwc_b1_npusafe`를 기준으로 한다. MXQ에는 `_vt_global8`이 붙는다.

모델을 교체하면 기록된 NPU 성능을 그대로 연결할 수 없다. `python3 scripts/verify_release.py`로 실제 평가한 파일과 동일한지 확인한다. SDK wheel·매뉴얼·ImageNet 이미지/NPY는 제출물에 포함하지 않는다.
