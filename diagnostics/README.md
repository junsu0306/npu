# 진단 도구

모델별 문제 재현 코드와 판정 스크립트를 production 컴파일·벤치마크 코드와 분리한다.

| 폴더 | 용도 |
|---|---|
| `patch_slimming/` | token selection 및 ONNX→MXQ 정확도 저하 진단 |

진단 산출물은 코드와 섞지 않고 `assets/diagnostics/<주제>/`에 저장한다. 수정한 모델
후보는 `assets/experiments/<주제>/`에 저장한다.
