# 컴파일 도구

역할별 진입점만 이 폴더에 둔다. 진단 모델 생성과 비교 도구는
`diagnostics/`에서 관리한다.

| 폴더 | 용도 | 주요 진입점 |
|---|---|---|
| `calibration/` | 이미지에서 calibration tensor 생성 | `generate_hwc.py` |
| `onnx/` | 컴파일 전 ONNX 정리 | `fix_outputs.py` |
| `mxq/` | qbcompiler로 MXQ 생성 | `compile_hwc.py` |

일반적인 AugReg ViT-Tiny 컴파일 흐름은 다음과 같다.

```bash
python3 compile/calibration/generate_hwc.py \
  --profile timm \
  --src-dir assets/calibration_data \
  --output-dir assets/calib_hwc_timm \
  --n-images 1000

NPU_CALIB_DIR="$PWD/assets/calib_hwc_timm" \
NPU_ONNX_DIR="$PWD/assets/onnx" \
NPU_MXQ_DIR="$PWD/assets/mxq" \
NPU_INFERENCE_SCHEME=global8 \
python3 compile/mxq/compile_hwc.py
```

`NPU_ONNX_DIR`와 `NPU_MXQ_DIR`을 지정하면 production 파일과 진단·실험 파일을
섞지 않고 같은 컴파일 설정을 재사용할 수 있다.
