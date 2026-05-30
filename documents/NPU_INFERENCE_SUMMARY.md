# Mobilint NPU Inference Summary

## Hardware: MOBILINT ARIES

| Item | Spec |
|------|------|
| Performance | 64 TOPS (Boost 80 TOPS) |
| Host Interface | PCIe Gen 4.0 8-lane |
| DRAM | 4CH 32-bit LPDDR4/LPDDR4X @ 4266 Mbps |
| Power | 25W TDP |
| Architecture | 2 Clusters × 4 Cores, L1/L2/L3 SRAM |
| Supported Frameworks | TensorFlow, PyTorch, ONNX, Keras, TVM |

---

## Workflow: Model → NPU

```
ONNX Model + Calibration Data (.npy)
        ↓
  [Docker] Mobilint Compiler (qbcompiler)
        ↓
    MXQ File (NPU binary)
        ↓
  Runtime Library (qbruntime / libmaccel)
        ↓
    NPU Inference
```

### Step 1 — Compile (run on GPU server, inside Docker)
```python
from qubee import mxq_compile

mxq_compile(
    model="model.onnx",
    calib_data_path="calib/",
    quantize_method="maxpercentile",   # float32 → int8
)
# Output: model.mxq
```

### Step 2 — Inference (run on NPU host)
```python
import qbruntime

acc    = qbruntime.Accelerator(device_id=0)
config = qbruntime.ModelConfig()
config.set_single_core_mode(core_ids=[
    qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0),
])
model = qbruntime.Model("model.mxq", config)
model.launch(acc)

result = model.infer([input_array])   # input: numpy array

model.dispose()
```

---

## Preprocessing for EfficientViT (NCHW)

```python
import cv2, numpy as np

MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])

def preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize: shorter side → 256
    h, w = img.shape[:2]
    scale = 256 / min(h, w)
    img = cv2.resize(img, (int(round(w*scale)), int(round(h*scale))))

    # Center crop 224×224
    h, w = img.shape[:2]
    img = img[(h-224)//2:(h-224)//2+224, (w-224)//2:(w-224)//2+224]

    # Normalize → CHW
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)   # HWC → CHW (3, 224, 224)
    return img
```

> **Key difference from ResNet50 example:**
> - ResNet50 example uses HWC layout (`cpu_offload=False`)
> - EfficientViT uses CHW/NCHW layout (`cpu_offload=True`, `in_dformats="NCHW"`)

---

## Running inference_npu.py

```bash
source aries_env/bin/activate

# Single image — NOTE: extension is case-sensitive on Linux (.JPEG not .jpeg)
python3 inference_npu.py --model efficientvit_b0_r224_nchw.mxq --image val_example.JPEG

# Benchmark (200 iterations)
python3 inference_npu.py --model efficientvit_b0_r224_nchw.mxq --benchmark

# ImageNet val accuracy
python3 inference_npu.py --model efficientvit_b0_r224_nchw.mxq --val_dir /data/imagenet/val
```

---

## Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `cv2.error: !_src.empty()` | Image file not found | Check filename case — Linux is case-sensitive. Use `val_example.JPEG` not `.jpeg` |
| Model not launching | Driver not installed | Install Mobilint NPU driver before using runtime |
| Wrong output shape | HWC vs CHW mismatch | Ensure model compiled with `in_dformats="NCHW"` matches CHW input |

---

## Compiler Internal Pipeline

```
ONNX
  → High-level Compiler (Mobilint IR, Layer Fusing)
  → Quantizer (float32 → int8 Quantized IR)
  → Low-level Compiler (target-optimized binary)
  → MXQ
```

## Aries2 Notable Features
- Global/Multi-Core scenarios (co-working & batch inference)
- Transformer support
- Int16 inference + Mixed-Precision
- Float input directly (no manual quantization of input needed)
- Improved Host-NPU communication protocol
