# 모델 및 데이터 자산

| 경로 | 내용 |
|---|---|
| `onnx/` | production ONNX |
| `mxq/` | production MXQ |
| `calibration_data/` | calibration 원본 이미지 |
| `calib_hwc*/` | 컴파일 입력용 HWC tensor; Git 제외 |

완료된 진단 artifact를 장기 보관하지 않는다. 최종 검증을 통과한 모델만 `onnx/`와
`mxq/`에 두고, 재현에 필요한 규칙과 명령은 `docs/`에 기록한다.
