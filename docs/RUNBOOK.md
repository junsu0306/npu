# 최종 모델 실행 안내

모든 상대경로는 `npu` 저장소 루트 기준이다. 컴파일 서버와 실제 NPU 서버는 구분한다. 아래 `/data/...` 경로는 각 서버의 실제 데이터 경로로 바꾼다. **최종 보관 결과를 재실험 출력으로 덮어쓰지 않는다.**

## 1. 파일 검증 (NPU 불필요)

```bash
python3 scripts/verify_release.py
```

추가 라이브러리가 있는 환경에서는:

```bash
python3 scripts/verify_release.py --onnx --predictions
python3 -m unittest discover -s tests -v
```

배포된 12개 ONNX/MXQ의 SHA-256, 원본 결과, 샘플 목록, 정확도 CSV와 독립 메모리 결과를 검사한다. `--predictions`는 NumPy, `--onnx`는 NumPy/ONNX가 필요하다. 컴파일이나 추론을 실행하지 않는다.

## 2. 컴파일 Docker에서 calibration 생성

`qbcompiler`가 설치된 Mobilint Docker에서 실행한다. SDK 설치 파일은 별도 공급받는다. 원본 ImageNet calibration 이미지 최소 1,000장을 준비한다. 생성기는 경로를 정렬하고 읽을 수 있는 앞 1,000장을 사용한다.

```bash
python3 compile/calibration/generate_hwc.py \
  --profile timm \
  --src-dir /data/imagenet_calibration \
  --output-dir assets/calib_hwc_timm \
  --n-images 1000
```

출력 폴더는 새 폴더여야 한다. 생성 파일이 1,000개보다 적으면 실패한다. 컴파일러 진입점은 profile/shape/dtype/finite/개수를 확인한다. `preprocess_profile.json`을 임의로 만들어 출처 불명 tensor에 붙이지 않는다.

최종 MXQ의 원본 calibration 목록/컴파일 전체 환경이 완전히 보존되지는 않았으므로 새 컴파일은 별도 실험으로 검증한다. 기존 최종 모델은 해시로 확인해 바로 NPU에서 재평가할 수 있다.

## 3. CUDA GPU 1번으로 12개 컴파일

```bash
CUDA_VISIBLE_DEVICES=1 \
NPU_CALIB_DIR="$PWD/assets/calib_hwc_timm" \
NPU_ONNX_DIR="$PWD/assets/onnx" \
NPU_MXQ_DIR="$PWD/assets/mxq/rebuild_final" \
NPU_ONNX_FILTER="__nhwc_b1_npusafe.onnx" \
NPU_INFERENCE_SCHEME=global8 \
python3 compile/mxq/compile_hwc.py
```

- `assets/mxq/rebuild_final`은 새 출력용 예시이며 Git에서 제외한다.
- 실제 평가한 `assets/mxq_final_20261006`에 덮어쓰지 않는다.
- 같은 출력 파일이 있으면 컴파일을 건너뛴다. 실패 후 재시도는 새 폴더를 지정해 불완전 파일이 건너뛰어지지 않게 한다.
- `CUDA_VISIBLE_DEVICES=1`은 Docker에 노출된 GPU의 CUDA 번호다. 컨테이너 노출 설정이 다르면 `nvidia-smi`의 UUID와 함께 확인한다.
- 컴파일 상세 설정은 [onnx_to_mxq.md](onnx_to_mxq.md)를 참조한다.

## 4. 실제 NPU 장치 사전 확인

```bash
python3 -m pip install -r requirements-npu.txt
mobilint-cli status
python3 -c "import qbruntime; from mblt_tracker import NPUDeviceTracker; print('runtime/tracker OK')"

python3 benchmark/final/eval_final_models.py \
  --models-dir assets/mxq_final_20261006 \
  --onnx-dir assets/onnx \
  --val-dir /data/imagenet_val \
  --classes 1000 --images-per-class 50 \
  --dry-run
```

Python 3.10 이상을 사용한다. `qbruntime`/드라이버/펌웨어는 장치에 맞는 SDK로 별도 제공한다. `--dry-run`은 NPU 없이 파일·데이터셋·ONNX 분석만 확인하며 MXQ 실행 검증이 아니다. 평가 dataset은 class별 하위 폴더이고 정렬된 폴더 순서가 ImageNet label 순서와 일치해야 한다.

## 5. 12개 모델 smoke test

```bash
python3 benchmark/final/eval_final_models.py \
  --models-dir assets/mxq_final_20261006 \
  --onnx-dir assets/onnx \
  --val-dir /data/imagenet_val \
  --classes 1000 --images-per-class 50 --limit 1 \
  --preprocess timm --core-mode global8 --device-id 0 \
  --warmup-runs 2 --progress-every 1 \
  --memory-sample-ms 20 --npu-sample-ms 500 \
  --output-dir results/reproduction_smoke \
  --fail-fast
```

새로 컴파일한 모델을 검사할 때는 `--models-dir assets/mxq/rebuild_final`로 바꾼다. Smoke 성공은 로드/추론 확인이며 정확도 확인이 아니다.

## 6. ImageNet 전체 평가

```bash
python3 benchmark/final/eval_final_models.py \
  --models-dir assets/mxq_final_20261006 \
  --onnx-dir assets/onnx \
  --val-dir /data/imagenet_val \
  --classes 1000 --images-per-class 50 \
  --preprocess timm --core-mode global8 --device-id 0 \
  --warmup-runs 10 --progress-every 1000 \
  --memory-sample-ms 20 --npu-sample-ms 500 \
  --output-dir results/reproduction_full \
  --fail-fast
```

오류 이미지/모델 수와 telemetry를 확인한다. 기존 결과와 같은 데이터셋이면 `samples.tsv`의 해시도 비교한다. 출력 폴더가 비어 있지 않으면 실패하므로 재실행할 때 새 이름을 사용한다. 최종 보관 `final_20261015`는 덮어쓰지 않는다.

## 7. 독립 프로세스 메모리 재검증

```bash
python3 benchmark/memory/profile_memory.py \
  --models-dir assets/mxq_final_20261006 \
  --core-mode global8 --device-id 0 \
  --warmups 10 --runs 100 --memory-seconds 10 \
  --memory-sample-ms 20 --npu-sample-ms 500 \
  --output-dir results/reproduction_memory
```

같은 NPU에서 다른 모델을 동시에 실행하지 않는다. 이 측정은 고정 random 입력으로 수행하며 이미지 정확도 평가가 아니다. Idle/active 값은 장치 전체 telemetry MB다.

## 8. 단일 이미지 reference

```bash
python3 benchmark/runtime/run_npu.py \
  --mxq assets/mxq_final_20261006/tiny30__reduced__nhwc_b1_npusafe_vt_global8.mxq \
  --image /data/example.jpg --preprocess timm --core-mode global8 \
  --warmups 10 --runs 50 --no-save
```

ONNX CPU reference는 선택적으로 `requirements-onnx.txt`를 설치한 뒤 실행한다.

```bash
python3 benchmark/onnx/infer_onnx.py \
  --model assets/onnx/tiny30__reduced__nhwc_b1_npusafe.onnx \
  --image /data/example.jpg --preprocess timm --layout nhwc
```

단일 이미지 Top-5 출력은 전체 validation Top-1/Top-5 성능을 대체하지 않는다.
