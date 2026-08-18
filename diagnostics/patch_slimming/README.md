# Patch Slimming 진단 도구

| 파일 | 역할 |
|---|---|
| `generate_probes.py` | `diag_ps_01`~`diag_ps_09` ONNX, 입력, PyTorch reference 생성 |
| `compare_onnx_mxq.py` | 같은 입력의 ONNX와 MXQ logits 비교 |
| `rewrite_selections.py` | one-hot selection MatMul을 initializer 또는 Gather로 변환 |

산출물 구조는 다음과 같다.

```text
assets/diagnostics/patch_slimming/
├── input/       고정 입력
├── reference/   PyTorch reference 출력
├── onnx/        합성 진단 ONNX
├── mxq/         합성 진단 MXQ
└── manifest.json

assets/experiments/patch_slimming/
├── onnx/        실제 모델의 등가 변환 후보
└── mxq/         별도 컴파일 서버에서 만든 후보 MXQ
```

상세 원인과 실행 순서는
[`docs/patch_slimming_mxq_diagnosis_ko.md`](../../docs/patch_slimming_mxq_diagnosis_ko.md)를
참고한다.

예시:

```bash
python3 diagnostics/patch_slimming/compare_onnx_mxq.py \
  --onnx assets/diagnostics/patch_slimming/onnx/diag_ps_09_stack12.onnx \
  --mxq assets/diagnostics/patch_slimming/mxq/diag_ps_09_stack12_vt_global8.mxq \
  --core-mode global8
```
