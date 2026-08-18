# Patch Slimming 컴파일 후보

`onnx/`에는 production Patch Slimming ONNX와 의미가 같은 그래프 수정 후보가 있다.

- `__selection_init`: 실험 완료. 기존 실패 MXQ와 동일한 출력을 내므로 원인에서 제외
- `__selection_gather`: ONNX 진단 전용. qbcompiler가 Gather를 MXQ로 변환하지 못함
- `__selection_no_identity`: identity MatMul 14개 제거, 실제 selection MatMul 10개 유지
- `__selection_no_identity_final_slice`: 위 모델에서 마지막 CLS MatMul 2개만 static Slice로 교체
- `__selection_slice_concat`: identity 14개 제거, 실제 selection 10개를 static Slice+Concat으로 교체

현재 추가 MXQ 컴파일 대상은 `selection_slice_concat`이며 Gather와 selection MatMul을
모두 포함하지 않는다. 기존 두 `no_identity` 후보 역시 Gather op를 포함하지 않는다.
ImageNet 1,000장에서 원본 ONNX와 logits가 1,000/1,000 bit-exact임을 확인했다.

별도 qbcompiler 서버에서 생성한 MXQ는 `mxq/`에 넣는다. 정확도 검증이 끝나기 전에는
`assets/mxq/`의 production 모델을 덮어쓰지 않는다.
