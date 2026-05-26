#!/usr/bin/env python3
"""ResNet50 NPU Inference Demo (qbruntime v1.1.0)"""

import cv2
import numpy as np
import qbruntime

# Preprocess
img = cv2.imread("ILSVRC2012_val_00000001.JPEG", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))[16:240, 16:240]
img = img.astype(np.float32) / 255.0
img = (img - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])

# NPU inference
acc = qbruntime.Accelerator(0)
config = qbruntime.ModelConfig()
config.set_single_core_mode(core_ids=[
    qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0),
])
model = qbruntime.Model("resnet50.mxq", config)
model.launch(acc)
result = model.infer([img])
model.dispose()

# Result
predicted = int(np.argmax(result[0]))
print(f"Predicted: {predicted}")
print(f"Expected : 65 (sea snake)")
print(f"Result   : {'PASS' if predicted == 65 else 'FAIL'}")
