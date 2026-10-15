# ONNX → MXQ 컴파일 기준

MXQ 컴파일은 NPU 실행 서버가 아니라 `qbcompiler`가 설치된 별도 Docker 환경에서
진행한다. 이 문서는 현재 `compile/mxq/compile_hwc.py`의 실제 설정만 기록한다.

## 입력 계약

최종 AugReg ViT 계열의 calibration과 런타임 전처리는 반드시 같아야 한다.

| 항목 | 값 |
|---|---|
| 색상 | BGR로 읽은 뒤 RGB 변환 |
| resize | aspect ratio 유지, short side 249, bicubic |
| crop | center crop 224×224 |
| dtype/layout | float32, HWC `(224, 224, 3)` |
| normalize | `[0,1]` 변환 후 mean `(0.5,0.5,0.5)`, std `(0.5,0.5,0.5)` |

ImageNet mean/std를 쓰는 `legacy` profile은 과거 결과 재현 전용이며 최종 모델에
사용하지 않는다.

```bash
python3 compile/calibration/generate_hwc.py \
  --profile timm \
  --src-dir assets/calibration_data \
  --output-dir assets/calib_hwc \
  --n-images 1000
```

생성 후 아래를 확인한다.

```bash
python3 - <<'PY'
import json
import numpy as np
from pathlib import Path

root = Path("assets/calib_hwc")
files = sorted(root.glob("*.npy"))
x = np.load(files[0], allow_pickle=False)
profile = json.loads((root / "preprocess_profile.json").read_text())
print("count:", len(files))
print("shape/dtype:", x.shape, x.dtype)
print("finite:", bool(np.isfinite(x).all()))
print("profile:", profile)
PY
```

기대값은 1,000개, `(224, 224, 3)`, `float32`, finite, profile `timm`이다.

## Patch Slimming ONNX 규칙

최종 Patch Slimming export는 실제 선택 연산을 유지하되, 계산 결과를 바꾸지 않는
identity selection matmul은 제거한다. dynamic `Gather`로 되돌리지 않는다. 정확한
export 규칙과 ONNX 검증 체크리스트는
[`patch_slimming_mxq_diagnosis_ko.md`](patch_slimming_mxq_diagnosis_ko.md)를 따른다.

컴파일 전에 ONNX Runtime 정확도와 다음 항목을 확인한다.

- 입력 shape/dtype이 컴파일 계약과 일치하는가
- 출력이 `(1, 1000)` finite logits인가
- 최종 ONNX의 실제 token selection 경로가 보존됐는가
- 불필요한 identity selector가 제거됐는가
- baseline/patch/reduced 파일명이 최종 규칙과 일치하는가

## Docker에서 GPU 2번으로 컴파일

호스트 저장소가 컨테이너의 `/workspace/npu`에 mount됐다고 가정한다. 출력은 먼저
Git에서 제외되는 `assets/mxq/` staging 폴더에 생성한다. 검증 전 최종 폴더를 직접
덮어쓰지 않는다.

```bash
cd /workspace/npu

CUDA_VISIBLE_DEVICES=2 \
NPU_ONNX_DIR=/workspace/npu/assets/onnx \
NPU_MXQ_DIR=/workspace/npu/assets/mxq \
NPU_CALIB_DIR=/workspace/npu/assets/calib_hwc \
NPU_ONNX_FILTER=_npusafe \
NPU_INFERENCE_SCHEME=global8 \
python3 /workspace/npu/compile/mxq/compile_hwc.py 2>&1 | tee compile_final.log
```

현재 컴파일 설정은 다음과 같다.

- backend: `onnx`
- config preset: `vision_transformer`
- CPU offload: enabled
- inference scheme: `global8`
- 출력 이름: `{onnx_stem}_vt_global8.mxq`

`compile_hwc.py`는 동일한 출력 파일이 있으면 건너뛴다. 재컴파일이 필요할 때는 새
staging 출력 디렉터리를 지정하는 편이 안전하다.

```bash
mkdir -p /workspace/npu/assets/mxq/rebuild_20261015

CUDA_VISIBLE_DEVICES=2 \
NPU_ONNX_DIR=/workspace/npu/assets/onnx \
NPU_MXQ_DIR=/workspace/npu/assets/mxq/rebuild_20261015 \
NPU_CALIB_DIR=/workspace/npu/assets/calib_hwc \
NPU_ONNX_FILTER=_npusafe \
NPU_INFERENCE_SCHEME=global8 \
python3 /workspace/npu/compile/mxq/compile_hwc.py
```

## 컴파일 후 검증과 승격

컴파일 성공만으로 최종 모델로 간주하지 않는다. NPU 서버에서 다음 순서로 검증한다.

1. `mobilint-cli status`에서 장치·드라이버·펌웨어 확인
2. `benchmark/final/eval_final_models.py --limit 1`로 모든 MXQ 로드/출력 확인
3. ONNX와 MXQ의 출력 shape 및 finite 여부 확인
4. ImageNet 50,000장 전체 정확도와 latency/telemetry 측정
5. SHA-256과 최종 보고서를 함께 보관

검증 완료 파일만 날짜가 붙은 최종 폴더로 승격한다. 현재 최종 세트는
`assets/mxq_final_20261006/`의 12개이며, 재현 명령과 보고서는
[`benchmark/README.md`](../benchmark/README.md)에 있다.
