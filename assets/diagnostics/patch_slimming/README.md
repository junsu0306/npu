# Patch Slimming 진단 자산

- `input/`: 모든 ONNX/MXQ 비교에 사용하는 고정 HWC 입력
- `reference/`: PyTorch 출력
- `onnx/`: `diag_ps_01`~`diag_ps_09` 합성 진단 모델
- `mxq/`: 위 ONNX를 컴파일한 Global8 모델
- `manifest.json`: 입력·ONNX·MXQ·reference의 상대 경로 매핑

생성 및 비교 코드는 `diagnostics/patch_slimming/`에 있다. 실제 Patch Slimming 모델을
수정한 후보는 이 폴더가 아니라 `assets/experiments/patch_slimming/`에서 관리한다.
