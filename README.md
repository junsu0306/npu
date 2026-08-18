# Mobilint NPU Deployment

ONNX 변환물의 MXQ 컴파일부터 실제 NPU 정확도·성능 측정까지 한 코드베이스에서 관리합니다.

## 디렉터리

- `compile/`: calibration 생성, ONNX 정리, MXQ 컴파일
- `benchmark/model_zoo/`: Model Zoo 전처리를 포함한 정확도·성능 벤치마크
- `benchmark/runtime/`: direct `qbruntime` 실행
- `benchmark/ablation/`: 코어 모드 등 원인 분리 실험
- `benchmark/onnx/`: MXQ 이전 ONNX Runtime reference
- `assets/`: ONNX, MXQ, calibration 데이터
- `results/`: 생성된 벤치마크 결과
- `docs/`: 상세 구현 및 실행 문서

처음 사용하는 경우 [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md)와
[benchmark/README.md](benchmark/README.md)를 먼저 확인합니다.

## 환경

일반 Python 의존성은 다음과 같이 설치합니다.

```bash
pip install -r requirements-npu.txt
```

`qbruntime`은 실제 NPU 장치의 Mobilint SDK 환경에서, `qbcompiler`는 컴파일러
컨테이너에서 각각 제공되어야 합니다.
