# NPU 코드베이스 구조

```text
npu/
├── assets/
│   ├── calibration_data/   원본 calibration 이미지
│   ├── calib_hwc/          컴파일 입력용 HWC NPY
│   ├── onnx/               production ONNX
│   ├── mxq/                production MXQ
│   └── mxq_final12/        최종 비교 대상 MXQ 16개
├── compile/
│   ├── calibration/        calibration tensor 생성
│   ├── onnx/               컴파일 전 ONNX 정리
│   └── mxq/                ONNX/Torch → MXQ 컴파일
├── benchmark/              실제 NPU 실행, 정확도 및 성능 측정
│   ├── model_zoo/          Model Zoo wrapper 기반 통합 벤치마크
│   ├── final/              최종 모델 정확도·속도·메모리·전력·FLOPs 종합 평가
│   ├── runtime/            direct qbruntime 실행 예제
│   ├── ablation/           single/global8 비교 실험
│   └── onnx/               MXQ 변환 전 ONNX CPU reference
├── results/                벤치마크 JSON/CSV/log 출력
└── docs/                   변환·실행·구현 문서
```

## 전체 흐름

1. `compile/calibration/generate_hwc.py`: calibration tensor 생성
2. `compile/mxq/compile_hwc.py`: 일반 ONNX → MXQ 컴파일
3. `benchmark/model_zoo/run_smoke_test.py`: 실제 NPU에서 빠른 동작 검사
4. `benchmark/model_zoo/run_model_zoo_benchmark.py`: 기존 Orin Model Zoo 경로로 전체 지표 측정
5. `benchmark/final/eval_final_models.py`: 최종 custom MXQ 전체 지표 종합 평가
6. `benchmark/ablation/eval_tiny30_core_ablation.py`: direct runtime으로 모델 정확도 비교

최종 custom HWC float32 MXQ는 `benchmark/final/eval_final_models.py`로 실제
NPU/process peak와 정확도·속도를 함께 측정한다. Model Zoo 입력 계약으로 컴파일된
MXQ의 activation 분석은 `benchmark/model_zoo/run_model_zoo_benchmark.py`에
`--architecture`를 전달한다.

`Orin_Compression/scripts/run_detailed_benchmark.py`와 `run_npu_test.py`의 핵심 실행
흐름은 각각 `model_zoo/run_model_zoo_benchmark.py`, `model_zoo/run_smoke_test.py`로 이식되었습니다.
새 구현은 모델 하나와 MXQ 하나를 명시적으로 연결하므로, 하나의 local MXQ가 여러
Model Zoo wrapper에 잘못 적용되는 실수를 방지합니다.

완료된 진단 ONNX/MXQ는 저장소에 장기 보관하지 않는다. Patch Slimming의 최종 export
규칙은 `docs/patch_slimming_mxq_diagnosis_ko.md`와
`compile/onnx/remove_identity_selections.py`에 유지한다.

기존 루트의 `run_npu.py`, `test_npu.py`, `infer_onnx.py`, `infer_resnet50.py`도
`benchmark/runtime`과 `benchmark/onnx`로 이동했습니다. NPU 루트에는 실행 스크립트를
두지 않으며 신규 실험은 목적에 맞는 `benchmark/` 하위 진입점을 사용합니다.
