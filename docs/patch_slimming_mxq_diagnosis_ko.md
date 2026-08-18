# Patch Slimming 최종 ONNX 생성 가이드

이 문서가 Patch Slimming의 유일한 전용 문서다. Tiny ViT의 추가 NPU 정확도 하락은
현재 qbcompiler/PTQ 특성으로 받아들이고 진단을 종료한다. 이후 최종 체크포인트를
ONNX/MXQ로 만들 때 아래 규칙을 지켜, 이번에 확보한 수준보다 성능이 다시 떨어지는
일을 방지한다.

## 1. 최종 결론

- Patch Slimming float ONNX는 정상이다. ImageNet 1,000장 기준 Top-1 `67.30%`,
  Top-5 `88.30%`였다.
- 기존 export는 pruning하지 않는 block에도 identity selection MatMul을 넣었다.
  이를 유지한 MXQ는 Top-1 `9.30%`까지 하락했다.
- identity selection MatMul 14개만 제거하면 float logits는 bit-exact하게 유지되고,
  MXQ는 Top-1 `50.00~50.20%`, Top-5 `76.40~76.60%`로 회복됐다.
- 실제 token을 줄이는 selection MatMul을 Slice+Concat으로 바꿔도 추가 개선은 없었다.
  실제 selector는 기존 one-hot MatMul 방식으로 유지한다.
- 올바른 `timm` calibration으로 다시 컴파일해도 Top-1 `50.20%`로 같았다.
  calibration은 올바르게 사용해야 하지만 이번 추가 하락의 주원인은 아니었다.
- Gather는 qbcompiler가 MXQ로 변환하지 못하므로 사용하지 않는다.

따라서 최종 ONNX의 핵심 규칙은 하나다.

> 실제로 token 수가 줄어드는 stage에만 고정 one-hot selection MatMul을 만들고,
> 모든 token을 그대로 유지하는 stage는 입력 tensor를 직접 반환한다.

## 2. 기존 export의 문제

기존 그래프는 selection을 다음처럼 표현했다.

```text
P[N_out, N_in] @ X[B, N_in, D] -> Y[B, N_out, D]
```

`P`는 학습 weight가 아니라 0과 1로 구성된 고정 one-hot 행렬이다. 실제 pruning이면
`N_out < N_in`이고 필요한 token 행만 선택한다.

문제는 pruning하지 않는 stage에도 다음 연산을 만들었다는 점이다.

```text
I[N, N] @ X[B, N, D] == X[B, N, D]
```

float에서는 완전한 no-op이지만 MXQ에서는 별도 MatMul 및 activation 양자화 경계로
처리돼 오차가 크게 누적됐다.

확인한 Tiny 모델의 selector 구조:

| stage | token 변화 | 처리 |
|---|---:|---|
| `slim.0` | 197→151 | 실제 one-hot MatMul 유지 |
| `slim.1` | 151→151 | identity, 제거 |
| `slim.2` | 151→151 | identity, 제거 |
| `slim.3` | 151→141 | 실제 one-hot MatMul 유지 |
| `slim.4` | 141→131 | 실제 one-hot MatMul 유지 |
| `slim.5` | 131→111 | 실제 one-hot MatMul 유지 |
| `slim.6~10` | 111→111 | identity, 제거 |
| 마지막 CLS | 111→1 | static Slice 권장 |

각 block에서 normalized query 입력과 residual 입력에 같은 selection을 적용하므로
MatMul이 두 번씩 나타난다. 제거 대상은 `slim.1`, `slim.2`, `slim.6~10`의 두 branch,
총 14개다.

## 3. 학습 저장소에서 selector를 구현하는 방법

최선의 방법은 ONNX를 고치는 것이 아니라 PyTorch 모듈이 identity selector를 애초에
실행하지 않게 만드는 것이다. `keep_indices`와 token 수는 export 전에 고정돼 있어야
한다.

```python
import torch
from torch import nn


class StaticTokenSelection(nn.Module):
    """Gather 없이 고정 token selection을 수행한다."""

    def __init__(self, n_in: int, keep_indices):
        super().__init__()
        keep = torch.as_tensor(keep_indices, dtype=torch.long)

        if keep.ndim != 1 or keep.numel() == 0:
            raise ValueError("keep_indices must be a non-empty 1D sequence")
        if int(keep.min()) < 0 or int(keep.max()) >= n_in:
            raise ValueError("keep_indices is out of range")
        if keep.unique().numel() != keep.numel():
            raise ValueError("keep_indices contains duplicates")

        expected_identity = torch.arange(n_in, dtype=torch.long)
        self.is_identity = bool(
            keep.numel() == n_in and torch.equal(keep, expected_identity)
        )

        if self.is_identity:
            # Python bool 분기이므로 export graph에 MatMul이 생기지 않는다.
            self.register_buffer(
                "selector", torch.empty(0, dtype=torch.float32), persistent=False
            )
        else:
            selector = torch.zeros(keep.numel(), n_in, dtype=torch.float32)
            selector[torch.arange(keep.numel()), keep] = 1.0
            self.register_buffer("selector", selector)

    def forward(self, x):
        # x: [B, N_in, D]
        if self.is_identity:
            return x
        return torch.matmul(self.selector, x)
```

Transformer block에서는 normalized branch와 residual branch에 같은 selector를 쓴다.

```python
class SlimmingBlock(nn.Module):
    def __init__(self, n_in, keep_indices, ...):
        super().__init__()
        self.select = StaticTokenSelection(n_in, keep_indices)
        # norm, attention, mlp 등 기존 모듈 구성

    def forward(self, x):
        z = self.norm1(x)

        # 두 branch의 token 순서와 길이가 반드시 같아야 한다.
        q_input = self.select(z)
        residual = self.select(x)

        y = self.attention(q_input, key_value=z)
        x = residual + y
        x = x + self.mlp(self.norm2(x))
        return x
```

실제 attention 함수의 인자 구조는 학습 저장소 구현에 맞게 연결하되, 두 branch 모두
identity이면 직접 반환하고 실제 pruning일 때만 같은 one-hot selector를 적용한다는
원칙은 바꾸지 않는다.

마지막 CLS token은 one-hot MatMul 대신 static Slice로 가져온다.

```python
cls_token = x[:, :1, :]     # ONNX Slice
logits = self.head(cls_token.squeeze(1))
```

다음 표현은 사용하지 않는다.

```python
x[:, keep_indices]                 # Gather로 export될 수 있음
torch.index_select(x, 1, keep)     # Gather
torch.gather(x, 1, index)          # Gather
```

실제 scattered selection을 여러 Slice+Concat으로 바꾸는 것도 권장하지 않는다. ONNX
node 수만 늘고 MXQ 정확도는 개선되지 않았다.

## 4. ONNX export 규칙

이 저장소에는 학습 모델 factory와 checkpoint loader가 없으므로 export 자체는 학습
저장소에서 수행한다. 아래 설정을 유지한다.

```python
import torch


class NHWCExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image_nhwc):
        # 학습 모델이 NCHW를 받는 경우에만 이 변환을 사용한다.
        image_nchw = image_nhwc.permute(0, 3, 1, 2)
        return self.model(image_nchw)


model = load_final_patch_slimming_checkpoint(...)  # 학습 저장소의 실제 loader
model.eval()
export_model = NHWCExportWrapper(model).eval()
dummy = torch.randn(1, 224, 224, 3, dtype=torch.float32)

with torch.no_grad():
    torch.onnx.export(
        export_model,
        dummy,
        "tiny30__patch_slimming__nhwc_b1_npusafe.onnx",
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        # dynamic_axes를 지정하지 않는다: batch=1, resolution=224 고정
    )
```

필수 출력 조건:

```text
input name   = input
input shape  = [1, 224, 224, 3]
input dtype  = float32
output name  = output
output shape = [1, 1000]
opset        = 17
Gather count = 0
identity selection MatMul count = 0
```

## 5. 기존 exporter를 바로 고치기 어려울 때

기존 방식으로 raw ONNX를 만든 다음 production 후처리 도구로 정확히 단위행렬인
selector만 제거할 수 있다. 입력과 출력을 같은 경로로 지정할 수 없게 해 원본을
보존하도록 했다.

```bash
python3 compile/onnx/remove_identity_selections.py \
  --input /path/to/raw_patch_slimming.onnx \
  --output /path/to/final_patch_slimming.onnx \
  --expect-removed 14 \
  --check-input /path/to/one_timm_preprocessed_hwc.npy
```

성공 기준:

```text
identity selection MatMuls removed: 14
max_abs: 0.0
relative_l2: 0.0
cosine: 1.0
argmax_before == argmax_after
```

최종 ONNX가 이미 수정된 export인지 검사할 때는 파일을 쓰지 않는 모드를 사용한다.

```bash
python3 compile/onnx/remove_identity_selections.py \
  --input /path/to/final_patch_slimming.onnx \
  --verify-clean
```

architecture나 pruning schedule이 달라져 identity 개수가 14개가 아니면 숫자를 임의로
바꾸지 않는다. 어떤 stage가 실제 pruning이고 어떤 stage가 identity인지 먼저 출력
token schedule과 대조한 뒤 `--expect-removed` 값을 결정한다.

## 6. 정확도 유지 검증

최종 ONNX를 배포하기 전에 다음 gate를 모두 통과해야 한다.

1. `onnx.checker.check_model()` 통과.
2. PyTorch checkpoint와 ONNX의 같은 입력 logits 비교.
3. 후처리했다면 raw ONNX와 final ONNX의 logits `max_abs=0` 확인.
4. ImageNet 1,000장 `timm` 전처리 평가에서 raw/final ONNX의 Top-1 예측 일치율
   `100%` 및 logits `1000/1000` exact 확인.
5. 실제 checkpoint의 PyTorch 정확도와 final ONNX 정확도가 동일한지 확인.
6. ONNX에 Gather와 identity selection MatMul이 없는지 확인.

이번 Tiny checkpoint에서 확인한 float 기준은 Top-1 `67.30%`, Top-5 `88.30%`였다.
새 checkpoint의 목표값은 새 PyTorch 평가 결과를 기준으로 해야 하며 이 숫자를 그대로
강제하면 안 된다.

## 7. MXQ 컴파일 시 유지할 조건

- calibration과 runtime 모두 `timm` 전처리를 사용한다.
- `timm` profile은 resize short side 249, bicubic, mean/std `[0.5, 0.5, 0.5]`이다.
- HWC `float32`, shape `(224, 224, 3)` calibration tensor 1,000개를 사용한다.
- `config_preset="vision_transformer"`, `inference_scheme="global8"`을 사용한다.
- 기존 MXQ가 있으면 skip될 수 있으므로 새 출력 경로에서 컴파일한다.

올바른 calibration 생성:

```bash
python3 compile/calibration/generate_hwc.py \
  --profile timm \
  --src-dir assets/calibration_data \
  --output-dir assets/calib_hwc_timm \
  --n-images 1000
```

이번 Tiny 실험의 최종 NPU 참고값은 1,000장 기준 Top-1 `50.20%`, Top-5 `76.60%`다.
이는 현재 compiler/PTQ와 해당 checkpoint 조합의 참고값이며 다른 최종 checkpoint의
성능 보장은 아니다.

## 8. 삭제한 실험

다음 실험은 결론에 반영한 뒤 artifact와 전용 코드를 제거했다.

- initializer vs Constant
- Gather 변환
- identity 제거
- 마지막 CLS static Slice
- 전체 selection Slice+Concat
- 합성 probe `diag_ps_01~09`
- legacy/timm calibration 비교 MXQ
- core-ablation MXQ

재현에 필요한 핵심은 이 문서와
`compile/onnx/remove_identity_selections.py`에 남아 있다.
