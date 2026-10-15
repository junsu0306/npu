# 모델 및 데이터 자산

| 경로 | 내용 |
|---|---|
| `onnx/` | 최종 MXQ와 대응하는 production ONNX |
| `mxq_final_20261006/` | Tiny/Small × 30%/50% × 3 variants 최종 MXQ 12개 |
| `calibration_data/` | calibration 원본 이미지 |
| `calib_hwc*/` | 컴파일 입력용 HWC tensor; Git 제외 |

`assets/mxq/`는 컴파일 중간 출력용이며 Git에서 제외한다. 완료된 진단 artifact를
장기 보관하지 않고, 검증을 통과한 최종 MXQ만 날짜가 붙은 최종 폴더에 둔다.
