# EfficientViT always gives wrong predictions after MXQ compilation — is this architecture supported?

Hi,

I'm trying to deploy **EfficientViT-B0/B1** (MIT-Han Lab) on a Mobilint MLA100 NPU (Jetson Orin, qbcompiler v1.1.0). The float32 ONNX model works correctly, but the compiled `.mxq` gives completely wrong predictions on every image (effectively 0% top-1 accuracy).

**What I've tried:**

- All quantization method/mode combinations (`WChALayer`, `WChAMulti`, `max`, `histogram`, `kl-divergence`) with both `backend="onnx"` and `backend="torch"` — all wrong.
- `cpu_offload=True` — predictions still wrong.
- `BitConfig` with all transformer bits set to 16 + `mixed_precision_apply=True` — compiler output showed `Attn=0`, so EfficientViT's attention wasn't recognized as an Attention block. Everything stayed at 8-bit.
- To rule out a qbcompiler-specific issue, I ran ONNX Runtime INT8 quantization on the same model. Every method (per-tensor, per-channel, MinMax, Entropy, Percentile) gave 0/10. UINT16 activations improved to 3/10. ONNX Runtime also logged warnings about `elem_type: 7` (INT64) tensors inside `context_module` — the ReLU-based linear attention has `Gather`/`Slice`/`Div` ops that produce INT64 tensors, and these seem to break INT8 quantization.
- Sanity check: ResNet50 through the same pipeline → 99% accuracy. So the pipeline itself is fine.

**My questions:**

1. **Architecture support** — Is EfficientViT's `context_module` (ReLU linear attention with dynamic `Gather`/`Slice` shape ops) actually supported on Aries2? Are there known issues with this kind of attention under INT8?

2. **INT16 activation** — Does Aries2 actually support INT16 activation at runtime? ONNX Runtime results suggest this model needs at least 16-bit activations. The v1.1.0 API has `BitConfig.LayerOverrides.activation_16bits`, but we can't tell if the hardware executes it or silently ignores it.

3. **`activation_16bits` not working** — We extracted the 141 float-op node names from `context_module` directly from the ONNX graph (e.g. `stages/stages.2/blocks/blocks.1/context_module/main/kernel_func/Relu`) and passed them to `activation_16bits` + `weight_16bits` with `backend="onnx"`. Compilation succeeded with no warnings, but the output still showed `8b=1.00` and `Average Bit Width: 8.00 bits` — no mention of 16-bit anywhere. Is the feature silently a no-op on this hardware, or are we using the wrong name format?

4. **`cpu_offload` verification** — Is there a log or runtime check to confirm `cpu_offload=True` is actually active? The PDF notes it requires a compatible qbruntime version.

Any help or pointers to working examples with attention-based models would be really appreciated. Thanks!
