# 최종 ONNX 입력 및 Patch Slimming 계약

## 고정 입출력

- 입력 하나: float32 NHWC `[1,224,224,3]`.
- 출력 하나: `[1,1000]` classification logits.
- Batch 1, static shape, opset 17.
- 입력 전처리는 모델 밖에서 수행한다. [방법론](METHODOLOGY.md)의 `timm` 프로필을 사용한다.
- 파일명 `npusafe`는 고정 그래프 export 규칙을 뜻한다. 컴파일 성공이나 정확도 보존을 보증하는 이름이 아니다.

## 선택 방식

`architecture.json`에 기록한 고정 token ID로 선택행렬을 만든다. 선택행렬은 학습 파라미터가 아니라 고정 0/1 데이터다.

```text
입력 X: [B, N_in, D]
선택 P: [N_out, N_in]  (각 행에 선택 위치의 1 하나)
출력 Y = P @ X: [B, N_out, D]
```

토큰을 제거할 때 `N_out < N_in`이다. 모든 토큰을 동일 순서로 유지하는 경우에는 identity MatMul을 생성하지 않고 입력을 직접 반환한다. 마지막 CLS만 추출하는 경로는 static Slice/Squeeze를 사용한다.

Query용 normalized 입력과 residual에 동일한 선택을 적용한다. Key/value는 선택 전 입력을 사용하므로 attention은 정적으로 확정된 직사각형 shape가 될 수 있다. 동적 TopK, 입력별 token 수 변경, 선택용 Gather로 되돌리지 않는다.

## export와 보조 도구

최종 ONNX 생성은 `Server_Compression`에서 수행한다. 이 저장소의 배포된 12개 ONNX는 수정하지 않는다.

`compile/onnx/remove_identity_selections.py`는 다른 export를 점검할 때 쓰는 보조 도구다. 정확한 정방 identity selection만 제거하며 실제 토큰 수를 줄이는 선택이나 순서를 바꾸는 행렬은 유지한다. 새 파일로 출력하고 원본과 ONNX Runtime logits를 비교해야 한다. 최종 배포물에 이 도구를 다시 실행할 필요는 없다.

```bash
python3 scripts/verify_release.py --onnx --predictions
```

이 명령은 checker, 고정 입출력, Gather/TopK 부재, identity selection 부재와 저장된 결과를 검사한다. 이것만으로 새로운 MXQ의 정확도를 판단할 수는 없다. 신규 MXQ는 실제 NPU에서 따로 평가한다.

## 정확도에 대한 해석

Identity no-op을 생략하고 실제 선택을 정적 MatMul로 내보내는 것은 최종 export 규칙이다. 이번 `final_20261015`에서도 Patch Slimming 정확도 하락은 남았다. 고정 0/1 행렬이라는 사실만으로 compiler fusion이나 activation 양자화의 영향이 없다고 단정할 수 없다. 최종 결과만으로 하락 원인을 양자화/컴파일 중 하나로 확정하지 않는다.
