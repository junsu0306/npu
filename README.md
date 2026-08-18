# Mobilint NPU Deployment

ONNX 변환물의 MXQ 컴파일부터 실제 NPU 정확도·성능 측정까지 한 코드베이스에서 관리합니다.

## 디렉터리

- `compile/`: calibration 생성, ONNX 정리, MXQ 컴파일을 역할별 하위 폴더로 관리
- `benchmark/model_zoo/`: Model Zoo 전처리를 포함한 정확도·성능 벤치마크
- `benchmark/runtime/`: direct `qbruntime` 실행
- `benchmark/ablation/`: 코어 모드 등 원인 분리 실험
- `benchmark/onnx/`: MXQ 이전 ONNX Runtime reference
- `diagnostics/`: 모델별 재현 모델 생성 및 ONNX↔MXQ 비교 도구
- `assets/onnx`, `assets/mxq`: production 모델
- `assets/diagnostics/`: 고정 입력, reference, 진단 ONNX/MXQ
- `assets/experiments/`: 수정한 ONNX 및 재컴파일 MXQ 후보
- `results/`: 생성된 벤치마크 결과
- `docs/`: 상세 구현 및 실행 문서

처음 사용하는 경우 [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md)와
[benchmark/README.md](benchmark/README.md)를 먼저 확인합니다.

ViT-Tiny의 전처리, calibration, ONNX → MXQ 컴파일, Mobilint Model Zoo
비교 평가는
[docs/preprocessing_compile_benchmark_guide_ko.md](docs/preprocessing_compile_benchmark_guide_ko.md)를
기준으로 실행합니다. AugReg ViT-Tiny는 ImageNet 표준 mean/std가
아닌 `mean=std=(0.5, 0.5, 0.5)` 전처리가 필요합니다.

Patch Slimming ONNX는 정상인데 MXQ 정확도만 낮아지는 문제의 측정 근거, 원인 가설,
별도 Docker 서버에서 실행할 3-way 재컴파일 실험은
[docs/patch_slimming_mxq_diagnosis_ko.md](docs/patch_slimming_mxq_diagnosis_ko.md)에
정리되어 있습니다.

## 환경

일반 Python 의존성은 다음과 같이 설치합니다.

```bash
pip install -r requirements-npu.txt
```

`qbruntime`은 실제 NPU 장치의 Mobilint SDK 환경에서, `qbcompiler`는 컴파일러
컨테이너에서 각각 제공되어야 합니다.
