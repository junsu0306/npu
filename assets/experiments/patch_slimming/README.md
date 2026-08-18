# Patch Slimming 컴파일 후보

`onnx/`에는 production Patch Slimming ONNX와 의미가 같은 그래프 수정 후보가 있다.

- `__selection_init`: selection 행렬을 Constant node에서 initializer로 이동
- `__selection_gather`: selection MatMul을 Gather로 교체하고 identity MatMul 제거

별도 qbcompiler 서버에서 생성한 MXQ는 `mxq/`에 넣는다. 정확도 검증이 끝나기 전에는
`assets/mxq/`의 production 모델을 덮어쓰지 않는다.
