# EfficientViT always gives wrong predictions after MXQ compilation — is this architecture supported?

Hi,

I'm trying to deploy **EfficientViT-B0/B1** (MIT-Han Lab) on a **Mobilint MLA100 NPU** connected to a Jetson Orin using qbcompiler v1.1.0. The float32 ONNX model runs correctly, but after compiling to `.mxq`, the model produces completely wrong predictions for every image. Top-1 accuracy is effectively 0%, even though the model isn't fully dead (different inputs give different outputs).

I've been debugging this for a while and tried pretty much everything I can think of, so I'm hoping someone here can point me in the right direction.

---

**What I tried so far:**

I tested every combination of quantization settings (`WChALayer`, `WChAMulti`, `max`, `histogram`, `kl-divergence`, etc.) with both `backend="onnx"` and `backend="torch"`. All of them produce wrong predictions. Interestingly, CHW and HWC inputs at runtime give *identical* outputs, which suggests the issue isn't about input layout — something deeper is broken.

I also tried `cpu_offload=True`. The mixed precision output showed that the attention blocks disappeared from the NPU graph (Conv=4, Undef=0), which I assumed meant they were running on CPU in FP32. But predictions were still wrong.

Then I tried `BitConfig` with all transformer bit widths set to 16 and `mixed_precision_apply=True`. The compiler output showed `Attn=0` — EfficientViT's efficient attention wasn't recognized as an Attention block at all. Everything was allocated to 8-bit anyway.

To check if this is a qbcompiler-specific issue or a general INT8 problem, I ran ONNX Runtime static INT8 quantization on the same model independently. Every method (per-tensor, per-channel, MinMax, Entropy, Percentile) gave 0/10 Top-1 match vs. float32. UINT16 activations improved it slightly to 3/10. ONNX Runtime also threw a lot of warnings like:

```
WARNING: Inference failed or unsupported type to quantize for tensor
'/stages/.../context_module/main/Div_output_0', type is tensor_type { elem_type: 7 }
```

`elem_type: 7` is INT64, coming from the `Gather`/`Slice`/`Div` ops inside EfficientViT's `context_module` (the ReLU-based linear attention). My hypothesis is that the float activations flowing through this attention are extremely sensitive to INT8 quantization error.

As a sanity check, I ran ResNet50 through the exact same pipeline and got 99% accuracy — so the pipeline itself is fine.

---

**My questions:**

1. Is EfficientViT (specifically the `context_module` with ReLU-based linear attention and dynamic Gather/Slice ops) actually a supported architecture on Aries2? Are there known issues with this kind of linear attention under INT8 quantization?

2. Does Aries2 / MLA100 actually support INT16 activation at runtime? The v1.1.0 API has `activation_16bits` in `BitConfig.LayerOverrides`, but I can't tell whether the hardware actually executes at INT16 or silently falls back to INT8. The ONNX Runtime results suggest at least 16-bit activations are needed for this model.

3. How are the layer names for `activation_16bits` supposed to look? The compiler assigns block-level names like `ConvBlock_3`, `UndefinedBlock_1` in the mixed precision output, but `activation_16bits` seems to need individual layer names inside those blocks. Where can I find the correct names?

   Also, we noticed the API location changed between versions. The official PDF documents it as a flat attribute:
   ```
   qbcompiler.configs.quantization_config.BitConfig.activation_16bits  (line 436)
   qbcompiler.configs.quantization_config.BitConfig.weight_16bits       (line 437)
   ```
   But in v1.1.0 it moved to `BitConfig.LayerOverrides.activation_16bits`. We tried both, but couldn't confirm whether `activation_16bits` was actually applied — the mixed precision output always showed `8b=1.00` regardless. Is there a log or compilation output that confirms 16-bit activation was actually used for a given layer?

4. Is there a way to verify that `cpu_offload=True` is actually active at runtime? The PDF mentions it only works if the installed qbruntime supports it — is there a version check or log I can look for?

Any help or pointers to working examples with attention-based models would be really appreciated. Thanks!
