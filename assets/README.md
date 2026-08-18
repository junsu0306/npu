# 모델 및 데이터 자산

| 경로 | 내용 |
|---|---|
| `onnx/` | production ONNX |
| `mxq/` | production MXQ와 `core_ablation/` 결과 |
| `calibration_data/` | calibration 원본 이미지 |
| `calib_hwc*/` | 컴파일 입력용 HWC tensor; Git 제외 |
| `diagnostics/<주제>/` | 재현용 입력, reference, 진단 ONNX/MXQ |
| `experiments/<주제>/` | production 모델을 변형한 ONNX와 재컴파일 MXQ 후보 |

production 파일을 `diagnostics`나 `experiments`에 복사하지 않는다. 진단 결과를
production으로 채택할 때만 검증과 파일명 정리를 거쳐 `onnx/` 또는 `mxq/`로 옮긴다.

Patch Slimming 진단 자산은 다음처럼 나뉜다.

```text
diagnostics/patch_slimming/
├── input/
├── reference/
├── onnx/
├── mxq/
└── manifest.json

experiments/patch_slimming/
├── onnx/
└── mxq/
```
