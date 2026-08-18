# Mobilint ViT 전처리·컴파일·벤치마크 가이드

이 문서는 다음 모델 계열을 ONNX에서 Mobilint MXQ로 컴파일하고 정확도와 성능을
재현 가능하게 측정하는 절차를 설명한다.

```text
timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k
```

현재 저장소의 `tiny30__baseline`, `tiny30__reduced`, `tiny30__patch_slimming`은 이
ViT-Tiny 계열에서 파생되었다. 가장 중요한 원칙은 아래 한 줄이다.

> 학습, ONNX 확인, PTQ calibration, MXQ 추론에서 이미지 전처리의 의미가 같아야 한다.

## 1. 이번 검증에서 확인된 결론

동일한 ImageNet validation 이미지 5,000장(1,000클래스 × 5장, seed 42)을 Global8로
측정한 결과는 다음과 같다.

| 모델 | 전처리 | Top-1 | Top-5 |
|---|---|---:|---:|
| Mobilint 공식 Model Zoo ViT-Tiny | Model Zoo 내장 | 74.46% | 92.20% |
| 커스텀 baseline MXQ | `timm` | 73.98% | 91.78% |
| 커스텀 baseline MXQ | `legacy` | 43.94% | 67.06% |
| reduced MXQ | `timm` | 71.62% | 90.80% |
| patch_slimming MXQ | `timm` | 9.14% | 19.76% |

baseline은 MXQ 자체가 고장 난 것이 아니었다. 전처리를 원본 timm 설정에 맞추자
Mobilint 공식 모델과 Top-1 차이가 0.48%p로 줄었다. 반면 patch_slimming은 올바른
전처리에서도 정확도가 낮으므로 ONNX 변환, 구조 치환 또는 양자화를 별도로 점검해야 한다.

결과 파일은 다음 위치에 있다.

- `results/model_zoo/mobilint_vit_tiny_patch16_224_augreg_in21k_ft_in1k_5000_global8_20260818_131922.json`
- `results/tiny30_global8_imagenet_val_5000_timm.json`

## 2. 전처리란 무엇인가

전처리는 JPEG/PNG 파일을 모델 입력 텐서로 바꾸는 모든 연산을 뜻한다. 단순히
이미지 크기를 224로 바꾸는 것만을 의미하지 않는다.

```text
JPEG 바이트
  -> 이미지 디코딩
  -> BGR/RGB 채널 순서 결정
  -> 종횡비를 유지한 resize
  -> resize 보간법 결정
  -> 중앙 224×224 crop
  -> 픽셀 범위 변환
  -> mean/std 정규화
  -> HWC/NHWC 또는 CHW/NCHW 배치
  -> float32/uint8 dtype 결정
```

각 항목이 모델이 학습될 때 사용한 조건과 달라지면 모델은 학습 중 보지 못한 수치
분포를 받는다. 코드가 오류 없이 실행되고 FPS도 정상으로 보이지만 정확도만 크게
하락할 수 있다. 이번 baseline의 `74.04% -> 43.94%` 하락이 그 사례다.

### 2.1 이 ViT-Tiny의 올바른 설정

원본 timm 모델 설정은 다음과 같다.

| 항목 | 값 |
|---|---|
| 입력 크기 | `3 × 224 × 224` |
| resize | 짧은 변을 약 249로, 종횡비 유지 |
| 보간법 | bicubic |
| crop | 중앙 `224 × 224`, `crop_pct=0.9` |
| RGB mean | `(0.5, 0.5, 0.5)` |
| RGB std | `(0.5, 0.5, 0.5)` |
| 정규화 후 범위 | 대략 `[-1, 1]` |

정규화 수식은 채널별로 다음과 같다.

```text
x_float = x_uint8 / 255
x_normalized = (x_float - 0.5) / 0.5
```

이 저장소에서는 이 설정을 `timm` 전처리 프로필이라고 부른다. 과거 코드의
`legacy` 프로필은 짧은 변 256, bilinear, ImageNet mean/std
`(0.485, 0.456, 0.406)/(0.229, 0.224, 0.225)`를 사용한다. ResNet에는 흔한
설정이지만 이 AugReg ViT-Tiny에는 맞지 않는다.

원본 설정은 Hugging Face의
[`config.json`](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/blob/main/config.json)이나
다음 코드로 확인할 수 있다.

```python
import timm
from timm.data import resolve_model_data_config

model = timm.create_model(
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    pretrained=True,
)
print(resolve_model_data_config(model))
```

### 2.2 custom MXQ와 Model Zoo MXQ의 입력 차이

두 MXQ의 외부 인터페이스가 같다고 가정하면 안 된다.

| 경로 | 전처리 호출 | 실제 확인된 입력 |
|---|---|---|
| 이 저장소의 custom ONNX/MXQ | 저장소의 `timm` 전처리 | HWC `float32`, 정규화된 값 |
| Mobilint 공식 Model Zoo MXQ | `model.preprocess()` | HWC `uint8` |

Mobilint 공식 MXQ는 Model Zoo wrapper의 전처리 설정과 함께 사용해야 한다. Model Zoo가
반환한 `uint8`을 custom MXQ에 그대로 넣거나, 반대로 custom용 정규화 float를 공식
MXQ에 직접 넣으면 안 된다. 파일 확장자가 둘 다 `.mxq`라는 사실은 입력 의미가
같다는 뜻이 아니다.

## 3. 환경 구성

저장소 루트는 다음과 같다.

```bash
cd /media/airlab_compression/nvme_storage/npu
```

### 3.1 custom MXQ 실행 환경

기본 Python에서 다음 항목을 확인한다.

```bash
python3 - <<'PY'
import cv2
import numpy
import qbruntime

print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("qbruntime:", qbruntime.__version__)
PY

ls -l /dev/aries0
```

현재 장치에서 확인한 기본 runtime은 `qbruntime v1.1.0`이다.

### 3.2 Mobilint Model Zoo 격리 환경

Model Zoo 최신 버전은 더 새로운 qbruntime을 설치할 수 있다. 기존 runtime을 덮어쓰지
않도록 별도 가상환경을 권장한다.

```bash
python3 -m venv --system-site-packages .venv-model-zoo
.venv-model-zoo/bin/python -m pip install mblt-model-zoo
```

이 장치에서는 tracker의 선택 의존성 때문에 일반 설치가 실패할 수 있었다. 정확도
평가에 tracker가 필요하지 않다면 다음 격리 설치로 재현할 수 있다.

```bash
.venv-model-zoo/bin/python -m pip install --no-deps \
  'mblt-model-zoo==2.3.0' \
  'mobilint-qb-runtime==1.3.2'

.venv-model-zoo/bin/python -m pip install \
  faster-coco-eval cityscapesScripts 'requests>=2.32.0'
```

이 가상환경은 공식 Model Zoo 평가에만 사용한다. custom MXQ 평가 명령은 기본
`python3`를 사용한다.

## 4. ONNX 확인

현재 Tiny30 ONNX 입력은 모두 `[1, 224, 224, 3]`, 즉 NHWC batch 입력이다. 모델마다
다를 수 있으므로 파일명만 보고 결정하지 말고 입력 메타데이터를 확인한다.

```bash
python3 - <<'PY'
import onnx

path = "assets/onnx/tiny30__baseline__nhwc_b1_npusafe.onnx"
model = onnx.load(path, load_external_data=False)
for value in model.graph.input:
    shape = [d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim]
    print(value.name, shape)
PY
```

ONNX Runtime reference 추론은 다음과 같이 실행한다. 스크립트가 NHWC/NCHW를 자동으로
판별한다.

```bash
python3 benchmark/onnx/infer_onnx.py \
  --model assets/onnx/tiny30__baseline__nhwc_b1_npusafe.onnx \
  --image val_example.JPEG \
  --preprocess timm \
  --layout auto \
  --labels imagenet_classes.txt
```

MXQ 정확도가 낮을 때는 반드시 ONNX CPU 결과를 먼저 확인한다.

- ONNX도 낮다: 학습 checkpoint, pruning, reduce, ONNX export 또는 전처리 문제
- ONNX는 정상이고 MXQ만 낮다: calibration, 양자화 또는 compiler 문제
- 단일 이미지만 다르다: Top-1 한 장으로 판단하지 말고 동일 manifest의 Top-1/Top-5와
  prediction agreement를 확인

## 5. PTQ calibration 데이터 생성

PTQ(Post-Training Quantization)는 실제 입력 값의 범위를 calibration 이미지로 측정한다.
라벨은 필요 없지만 운영 데이터와 비슷한 이미지 분포가 필요하다. 무엇보다
calibration 전처리와 runtime 전처리가 동일해야 한다.

이 ViT-Tiny용 calibration을 별도 디렉터리에 생성한다.

```bash
python3 compile/gen_calib_hwc.py \
  --profile timm \
  --src-dir assets/calibration_data \
  --output-dir assets/calib_hwc_timm \
  --n-images 1000
```

출력은 다음과 같다.

```text
assets/calib_hwc_timm/
  0000.npy
  0001.npy
  ...
  0999.npy
  preprocess_profile.json
```

검증 명령은 다음과 같다.

```bash
python3 - <<'PY'
import json
import numpy as np

root = "assets/calib_hwc_timm"
x = np.load(root + "/0000.npy")
print("shape/dtype:", x.shape, x.dtype)
print("range:", float(x.min()), float(x.max()))
print(json.load(open(root + "/preprocess_profile.json")))
PY
```

기대 shape/dtype은 `(224, 224, 3) / float32`이고 값 범위는 대략 `[-1, 1]`이다.

`compile/gen_calib_hwc.py`의 기본 프로필은 현재 모델에 맞는 `timm`이다.
과거 결과를 재현할 때만 `--profile legacy`를 명시한다. 실험 기록을 명확히
남기기 위해 권장 명령에서는 `--profile timm`을 생략하지 않는다.

## 6. ONNX에서 MXQ로 컴파일

컴파일은 `qbcompiler`가 설치된 Aries2 compiler 컨테이너에서 실행한다.

```bash
docker run -it --rm \
  -v /media/airlab_compression/nvme_storage/npu:/workspace/npu \
  <qbcompiler-image> bash
```

컨테이너 내부에서 wheel과 calibration 경로를 지정한다.

```bash
pip install /workspace/npu/assets/qbcompiler-1.1.0+aries2-py3-none-any.whl
cd /workspace/npu

NPU_CALIB_DIR=/workspace/npu/assets/calib_hwc_timm \
NPU_ONNX_FILTER=tiny30__baseline__nhwc_b1_npusafe \
NPU_INFERENCE_SCHEME=global8 \
python3 compile/compile_hwc.py
```

다른 모델은 filter만 변경한다.

```bash
NPU_CALIB_DIR=/workspace/npu/assets/calib_hwc_timm \
NPU_ONNX_FILTER=tiny30__reduced__nhwc_b1_npusafe \
NPU_INFERENCE_SCHEME=global8 \
python3 compile/compile_hwc.py
```

컴파일 설정은 현재 다음과 같다.

| 항목 | 값 | 의미 |
|---|---|---|
| backend | `onnx` | ONNX 입력 |
| config preset | `vision_transformer` | transformer용 양자화/활성 정밀도 설정 |
| inference scheme | `global8` | 2개 cluster의 8개 local core 공동 사용 |
| CPU offload | `True` | 미지원 연산을 CPU subgraph로 분리 |
| calibration layout | HWC | custom ONNX 입력 의미와 일치 |

`compile_hwc.py`는 같은 출력 파일이 이미 있으면 건너뛴다. 재컴파일 전 기존 MXQ를
삭제하는 대신 먼저 백업하는 편이 안전하다.

```bash
mv assets/mxq/<기존파일>.mxq assets/mxq/<기존파일>.legacy-preprocess.mxq
```

그 뒤 compiler 컨테이너에서 다시 실행한다. calibration 프로필이 달라졌으면 결과
파일명이나 실험 라벨에도 `timm`/`legacy`를 기록해 혼동을 막는다.

## 7. NPU smoke test와 성능 측정

### 7.1 Global8 모델 한 장 확인

```bash
python3 benchmark/runtime/run_npu.py \
  --mxq assets/mxq/tiny30__baseline__nhwc_b1_npusafe_vt_global8.mxq \
  --image val_example.JPEG \
  --labels imagenet_classes.txt \
  --preprocess timm \
  --core-mode global8 \
  --warmups 10 \
  --runs 100
```

여기서 확인할 항목은 다음과 같다.

- `Model constructed`, `Model launched`, `Model disposed`가 모두 출력되는가
- 입력 shape가 `(224, 224, 3)`인가
- 전처리가 `timm`으로 출력되는가
- latency/FPS가 반복 실행에서 안정적인가
- Top-5가 ONNX reference와 대체로 일치하는가

### 7.2 코어 모드 일치

MXQ를 컴파일한 scheme과 runtime core mode는 반드시 같아야 한다.

| MXQ scheme | 실행 옵션 |
|---|---|
| `single` | `--core-mode single` |
| `multi` | `--core-mode multi` |
| `global4` | `--core-mode global4` |
| `global8` | `--core-mode global8` |

Global8 MXQ를 `set_single_core_mode()`로 실행하면 “cores ... not covered by the MXQ
file” 오류가 발생한다. 현재 runtime 스크립트는 각 모드별 qbruntime API를 사용하도록
수정되어 있다.

### 7.3 모델 3개 latency 비교

```bash
for mxq in \
  assets/mxq/tiny30__baseline__nhwc_b1_npusafe_vt_global8.mxq \
  assets/mxq/tiny30__patch_slimming__nhwc_b1_npusafe_vt_global8.mxq \
  assets/mxq/tiny30__reduced__nhwc_b1_npusafe_vt_global8.mxq
do
  python3 benchmark/runtime/run_npu.py \
    --mxq "$mxq" \
    --image val_example.JPEG \
    --preprocess timm \
    --core-mode global8 \
    --warmups 20 \
    --runs 500
done
```

이 측정은 같은 이미지 한 장을 반복하므로 순수 single-image latency/FPS 비교용이다.
정확도 측정과 혼동하지 않는다.

## 8. 정확도 벤치마크

현재 ImageNet validation 경로는 다음 구조다.

```text
/media/airlab_compression/nvme_storage/imagenet_val/
  00000/  # class index 0, 50 images
  00001/  # class index 1, 50 images
  ...
  00999/  # class index 999, 50 images
```

숫자 폴더명은 곧 정답 class index다.

### 8.1 클래스당 5장, 총 5,000장

다음 명령은 각 클래스에서 seed 42로 5장씩 뽑는다. 모든 모델에 같은 표본을 사용한다.

```bash
python3 benchmark/ablation/eval_tiny30_core_ablation.py \
  --model baseline:global8:assets/mxq/tiny30__baseline__nhwc_b1_npusafe_vt_global8.mxq \
  --model patch_slimming:global8:assets/mxq/tiny30__patch_slimming__nhwc_b1_npusafe_vt_global8.mxq \
  --model reduced:global8:assets/mxq/tiny30__reduced__nhwc_b1_npusafe_vt_global8.mxq \
  --val-dir /media/airlab_compression/nvme_storage/imagenet_val \
  --classes 1000 \
  --images-per-class 5 \
  --seed 42 \
  --preprocess timm \
  --output results/tiny30_global8_imagenet_val_5000_timm.json
```

결과에는 모델별 Top-1, Top-5, inference throughput, 예측 class index와 모델 간 Top-1
agreement가 저장된다. 이미지 디코딩은 스트리밍하므로 5만 장을 RAM에 모두 올리지 않는다.

### 8.2 전체 50,000장

전체 평가 시 `--images-per-class 50`만 바꾼다.

```bash
python3 benchmark/ablation/eval_tiny30_core_ablation.py \
  --model baseline:global8:assets/mxq/tiny30__baseline__nhwc_b1_npusafe_vt_global8.mxq \
  --model reduced:global8:assets/mxq/tiny30__reduced__nhwc_b1_npusafe_vt_global8.mxq \
  --val-dir /media/airlab_compression/nvme_storage/imagenet_val \
  --classes 1000 \
  --images-per-class 50 \
  --seed 42 \
  --preprocess timm \
  --output results/tiny30_global8_imagenet_val_50000_timm.json
```

정확도 비교에서는 다음 조건을 고정한다.

- 동일 이미지 목록과 label
- 동일 전처리 프로필
- 동일 core mode
- 동일 qbruntime major/minor 환경
- 오류 이미지 수 0건 확인
- 모델마다 Top-1과 Top-5를 모두 보고
- latency 측정과 accuracy 측정을 별도 결과로 관리

## 9. Mobilint 공식 Model Zoo 원본 평가

Mobilint의 `ViT_Tiny_Patch16_224` 저장소는 base model로
`timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`를 명시한다. Model Zoo 엔진에서
`model_path=""`로 생성하면 Hugging Face Hub에서 MXQ를 자동으로 받아
`~/.mblt_model_zoo`에 캐시한다.

### 9.1 다운로드 및 한 장 smoke test

```bash
.venv-model-zoo/bin/python - <<'PY'
import numpy as np
from mblt_model_zoo.vision import MBLT_Engine

model = MBLT_Engine(
    model_cls="ViT_Tiny_Patch16_224",
    model_type="DEFAULT",
    model_path="",
    core_mode="global8",
)
try:
    x = model.preprocess("val_example.JPEG")
    y = model(x)
    raw = y[0] if isinstance(y, (list, tuple)) else y
    print("input:", np.asarray(x).shape, np.asarray(x).dtype)
    print("top5:", np.argsort(np.asarray(raw).reshape(-1))[::-1][:5])
finally:
    model.dispose()
PY
```

현재 다운로드된 파일은 다음 위치에 있다.

```text
/home/airlab_compression/.mblt_model_zoo/aries/vit_tiny_patch16_224.mxq
```

### 9.2 동일 조건 5,000장 평가

```bash
.venv-model-zoo/bin/python benchmark/model_zoo/run_model_zoo_benchmark.py \
  --model-class ViT_Tiny_Patch16_224 \
  --mxq /home/airlab_compression/.mblt_model_zoo/aries/vit_tiny_patch16_224.mxq \
  --val-dir /media/airlab_compression/nvme_storage/imagenet_val \
  --classes 1000 \
  --images-per-class 5 \
  --infer-mode global8 \
  --warmup-runs 3 \
  --timed-runs 1 \
  --seed 42 \
  --label mobilint_vit_tiny_patch16_224_augreg_in21k_ft_in1k_5000
```

이 명령에서는 직접 작성한 OpenCV 전처리 대신 반드시 `model.preprocess()`를 사용한다.
현재 숫자 class 폴더 구조를 지원하기 위해 공식 `mblt-model-zoo val` CLI 대신 이
저장소의 wrapper를 사용한다.

## 10. 정확도가 이상할 때 확인 순서

### 증상: baseline이 약 44%

가장 먼저 `--preprocess legacy`가 사용됐는지 확인한다.

```bash
python3 benchmark/runtime/run_npu.py --help
```

AugReg ViT에서는 `--preprocess timm`이어야 한다. calibration도 timm profile인지
`preprocess_profile.json`으로 확인한다.

### 증상: 모든 모델의 class가 이상하게 쏠림

다음을 순서대로 확인한다.

1. BGR을 RGB로 변환했는가
2. HWC/NHWC와 CHW/NCHW를 혼동하지 않았는가
3. 입력 dtype이 MXQ가 기대하는 dtype인가
4. mean/std가 원본 timm config와 같은가
5. output class 순서와 폴더 label index가 같은가

### 증상: ONNX는 정상인데 MXQ만 낮음

1. calibration tensor의 profile과 runtime profile 비교
2. calibration tensor의 shape/dtype/range 확인
3. 동일 ONNX를 single/global8로 각각 컴파일하여 core mode 영향 분리
4. 양자화 preset과 CPU offload 설정 확인
5. 작은 동일 manifest로 ONNX/MXQ prediction agreement 계산

### 증상: patch_slimming만 낮음

현재 올바른 timm 전처리에서도 5,000장 Top-1이 9.14%다. 전처리 문제로 설명되지
않는다. 다음 순서로 확인한다.

1. patch_slimming ONNX CPU Top-1/Top-5
2. PyTorch checkpoint와 ONNX의 동일 이미지 logits 비교
3. selection matrix와 token index 순서 확인
4. dynamic gather를 고정 matmul로 바꾸는 과정 확인
5. ONNX가 정상일 때 calibration/quantization 단계 비교

## 11. 가상환경과 Git

`.venv`, `venv`, `env`, `aries_env` 같은 가상환경은 보통 Git에 커밋하지 않는다.
가상환경에는 OS/CPU/Python 버전에 종속된 binary, 수천 개의 dependency 파일, cache가
포함되기 때문이다. 재현에 필요한 것은 가상환경 자체가 아니라 requirements/lock
파일과 설치 명령이다.

현재 `.venv-model-zoo`는 약 50MB, 2,300개 이상의 파일이지만 Git에 이미 commit된
상태는 아니었다. `.gitignore`가 빠져 `?? .venv-model-zoo/`로 보였을 뿐이다. 이제
다음 패턴이 `.gitignore`에 추가되어 있다.

```gitignore
/.venv/
/.venv-*/
/venv/
/env/
/aries_env/
```

무시 여부는 다음처럼 확인한다.

```bash
git check-ignore -v .venv-model-zoo/bin/python
git status --short
```

가상환경이 이미 Git index에 올라간 저장소라면 `.gitignore`만 추가해서는 빠지지 않는다.
그 경우에만 다음 명령으로 index 추적을 해제한다. 로컬 파일은 삭제되지 않는다.

```bash
git rm -r --cached .venv-model-zoo
```

현재 저장소의 `.venv-model-zoo`는 추적된 적이 없으므로 위 `git rm`은 필요 없다.

## 12. 최종 체크리스트

컴파일 전:

- 원본 모델의 data config를 확인했다.
- ONNX 입력 layout과 shape를 확인했다.
- ONNX CPU 정확도가 정상이다.
- `timm` profile calibration을 별도 디렉터리에 생성했다.
- `preprocess_profile.json`을 확인했다.

컴파일 시:

- `NPU_CALIB_DIR`이 올바른 calibration 디렉터리를 가리킨다.
- `NPU_ONNX_FILTER`가 정확히 한 모델을 선택한다.
- `NPU_INFERENCE_SCHEME=global8`을 확인했다.
- 기존 MXQ를 실수로 재사용하지 않았다.

벤치마크 시:

- custom MXQ에는 `--preprocess timm`을 사용한다.
- 공식 Model Zoo MXQ에는 `model.preprocess()`를 사용한다.
- MXQ scheme과 `--core-mode`가 일치한다.
- 모든 모델에 같은 `--seed`, `--classes`, `--images-per-class`를 사용한다.
- 결과 JSON에서 `num_successful`, `errors`, Top-1, Top-5를 확인한다.
