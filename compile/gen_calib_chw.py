#!/usr/bin/env python3
"""
Generate CHW calibration data for cpu_offload=True compilation.

cpu_offload=True 로 컴파일할 때는 calibration 데이터가
원본 PyTorch 모델 입력 포맷인 CHW (3, 224, 224) 이어야 한다.

Run OUTSIDE Docker:
  cd /home/airlab_compression/npu
  source aries_env/bin/activate
  python3 compile/gen_calib_chw.py
"""

import glob
import os
import cv2
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "assets"))

SRC_DIR  = os.path.join(ASSETS_DIR, "calibration_data")
DST_DIR  = os.path.join(ASSETS_DIR, "calib_chw")
N_IMAGES = 1000

MEAN = np.float32([0.485, 0.456, 0.406])
STD  = np.float32([0.229, 0.224, 0.225])

os.makedirs(DST_DIR, exist_ok=True)

jpg_files = sorted(f for f in glob.glob(os.path.join(SRC_DIR, "**", "*.jpg"), recursive=True)
                   if not f.endswith(".Zone.Identifier"))
if not jpg_files:
    raise FileNotFoundError(f"No .jpg files found in {SRC_DIR}")

n = min(N_IMAGES, len(jpg_files))
print(f"Generating {n} CHW calibration tensors: {SRC_DIR} → {DST_DIR}")

saved = 0
for jpg_path in jpg_files:
    img = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [skip] cannot read: {jpg_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = 256 / min(h, w)
    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))
    h, w = img.shape[:2]
    img = img[(h - 224) // 2:(h - 224) // 2 + 224,
              (w - 224) // 2:(w - 224) // 2 + 224]

    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD             # (224, 224, 3) HWC
    img = img.transpose(2, 0, 1)         # (3, 224, 224) CHW

    np.save(os.path.join(DST_DIR, f"{saved:04d}.npy"), img)
    saved += 1
    if saved >= n:
        break

print(f"Done: {saved} files saved to {DST_DIR}")
print(f"Shape: {img.shape}  min={img.min():.3f}  max={img.max():.3f}")
