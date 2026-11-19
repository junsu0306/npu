# 최종 ONNX → MXQ 컴파일 설정

실행 순서는 [RUNBOOK](RUNBOOK.md)의 calibration/컴파일 절을 따른다. 이 문서는 최종 코드의 설정과 해석만 정리한다.

| 설정 | 값 |
|---|---|
| 입력 ONNX | `assets/onnx/` 12개, FP32 NHWC batch 1, opset 17 |
| Calibration | HWC float32 `(224,224,3)` × 1,000 |
| 전처리 | RGB, OpenCV bicubic 단변 249, center crop 224, mean/std 모두 0.5 |
| Backend | `onnx` |
| Preset | `vision_transformer` |
| Inference scheme | `global8` |
| CPU offload | `True` |
| 출력명 | `{onnx_stem}_vt_global8.mxq` |
| 재컴파일 출력 | `assets/mxq/<새 실험명>/`, 최종 모델 폴더와 분리 |

컴파일은 `qbcompiler` Docker의 CUDA GPU에서, 실제 추론은 Mobilint NPU에서 수행한다. SDK 버전에 따라 preset의 내부 설정이 달라질 수 있다. 최종 진입점은 preset만 전달하고 수동 histogram/KL 인자를 함께 지정하지 않는다. 실제 quantizer 설정을 확인하지 않고 “모든 연산 INT8” 혹은 특정 scaling 알고리즘으로 단정하지 않는다.

`compile_hwc.py`는 `preprocess_profile.json`의 프로필/단변/정규화/shape/dtype/개수와 tensor의 finite 여부를 확인한다. 출처 불명 calibration은 실패 처리한다. 같은 MXQ 출력이 있으면 건너뛰므로 실패 후 재시도할 때는 새 출력 폴더를 사용한다.

최종 ONNX는 [ONNX_CONTRACT.md](ONNX_CONTRACT.md)의 정적 선택 규칙을 따른다. CPU offload가 켜졌다는 이유로 임의의 동적 연산을 사용할 수 있다고 가정하지 않는다.

새 MXQ는 컴파일 성공만으로 최종본이 되지 않는다. 파일 존재/크기 → 실제 NPU smoke → 전체 정확도/telemetry → 해시·보고서 보관 순서로 검증한다. 최종 보관 모델의 무결성은 `python3 scripts/verify_release.py`로 확인한다.
