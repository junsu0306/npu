#!/usr/bin/env python3
"""Generate deterministic ONNX probes that isolate Patch Slimming MXQ errors."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "assets" / "diagnostics" / "patch_slimming"
ONNX_DIR = ARTIFACT_DIR / "onnx"
INPUT_DIR = ARTIFACT_DIR / "input"
REFERENCE_DIR = ARTIFACT_DIR / "reference"


def selection_matrix(n_in, ids):
    p = torch.zeros(len(ids), n_in)
    p[torch.arange(len(ids)), torch.as_tensor(ids)] = 1.0
    return p


class Stem(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, 16, 16)

    def forward(self, image):
        x = self.conv(image.permute(0, 3, 1, 2))
        return x.flatten(2).transpose(1, 2)  # [1,196,D]


class ProbeHead(nn.Module):
    def __init__(self, dim=192, out_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, out_dim)

    def forward(self, x):
        return self.fc(self.norm(x).mean(1))


class Attention(nn.Module):
    def __init__(self, dim=192, heads=3):
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.scale = self.dh ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.q, self.k, self.v, self.proj = [nn.Linear(dim, dim) for _ in range(4)]

    def _heads(self, x):
        b, n, d = x.shape
        return x.reshape(b, n, self.heads, self.dh).permute(0, 2, 1, 3)

    def forward(self, x, p=None, residual=False, mlp=None):
        z = self.norm(x)
        q_in = z if p is None else torch.matmul(p, z)
        r = x if p is None else torch.matmul(p, x)
        q, k, v = self._heads(self.q(q_in)), self._heads(self.k(z)), self._heads(self.v(z))
        a = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
        y = (a @ v).transpose(1, 2).reshape(q_in.shape[0], q_in.shape[1], -1)
        y = self.proj(y)
        if residual:
            y = r + y
            if mlp is not None:
                y = y + mlp(y)
        return y


class Diagnostic(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.stem, self.head = Stem(), ProbeHead()
        self.attn = Attention()
        scattered = torch.linspace(0, 195, 151).round().long().unique().tolist()
        contiguous = list(range(151))
        self.register_buffer("p_scattered", selection_matrix(196, scattered))
        self.register_buffer("p_contiguous", selection_matrix(196, contiguous))
        self.mlp = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 384), nn.GELU(), nn.Linear(384, 192))

    def forward(self, image):
        x = self.stem(image)
        if self.kind == "control_stem_head":
            y = x
        elif self.kind == "select_contiguous":
            y = torch.matmul(self.p_contiguous, x)
        elif self.kind == "select_scattered":
            y = torch.matmul(self.p_scattered, x)
        elif self.kind == "attention_square":
            y = self.attn(x)
        elif self.kind == "attention_rect":
            y = self.attn(x, self.p_scattered)
        elif self.kind == "block_rect_residual_mlp":
            y = self.attn(x, self.p_scattered, residual=True, mlp=self.mlp)
        else:
            raise ValueError(self.kind)
        return self.head(y)


class StackDiagnostic(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.stem, self.head = Stem(), ProbeHead()
        schedule = [196, 181, 166, 151, 141, 131, 121, 111, 101, 91, 81, 71, 61]
        self.blocks = nn.ModuleList([Attention() for _ in range(depth)])
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 384), nn.GELU(), nn.Linear(384, 192))
            for _ in range(depth)
        ])
        for i in range(depth):
            ids = torch.linspace(0, schedule[i] - 1, schedule[i + 1]).round().long().unique()
            self.register_buffer(f"p_{i}", selection_matrix(schedule[i], ids.tolist()))

    def forward(self, image):
        x = self.stem(image)
        for i, (block, mlp) in enumerate(zip(self.blocks, self.mlps)):
            x = block(x, getattr(self, f"p_{i}"), residual=True, mlp=mlp)
        return self.head(x)


def main():
    torch.manual_seed(20260818)
    np.random.seed(20260818)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    # Fixed normalized NHWC input. qbruntime consumes the unbatched HWC array.
    inp = np.random.uniform(-1.0, 1.0, (224, 224, 3)).astype(np.float32)
    input_path = INPUT_DIR / "diagnostic_input_hwc.npy"
    np.save(input_path, inp)
    dummy = torch.from_numpy(inp[None])

    specs = [
        ("diag_ps_01_control_stem_head", Diagnostic("control_stem_head")),
        ("diag_ps_02_select_contiguous", Diagnostic("select_contiguous")),
        ("diag_ps_03_select_scattered", Diagnostic("select_scattered")),
        ("diag_ps_04_attention_square", Diagnostic("attention_square")),
        ("diag_ps_05_attention_rect", Diagnostic("attention_rect")),
        ("diag_ps_06_block_rect_residual_mlp", Diagnostic("block_rect_residual_mlp")),
        ("diag_ps_07_stack3", StackDiagnostic(3)),
        ("diag_ps_08_stack6", StackDiagnostic(6)),
        ("diag_ps_09_stack12", StackDiagnostic(12)),
    ]
    manifest = {
        "seed": 20260818,
        "input": str(input_path.relative_to(ROOT)),
        "models": [],
    }
    for name, model in specs:
        model.eval()
        path = ONNX_DIR / f"{name}.onnx"
        with torch.no_grad():
            ref = model(dummy).cpu().numpy()
        torch.onnx.export(model, dummy, path, opset_version=17, input_names=["input"],
                          output_names=["output"], do_constant_folding=True)
        reference_path = REFERENCE_DIR / f"{name}__torch_output.npy"
        np.save(reference_path, ref)
        manifest["models"].append({
            "name": name,
            "onnx": str(path.relative_to(ROOT)),
            "mxq": str(
                (
                    ARTIFACT_DIR
                    / "mxq"
                    / f"{name}_vt_global8.mxq"
                ).relative_to(ROOT)
            ),
            "torch_reference": str(reference_path.relative_to(ROOT)),
            "output_shape": list(ref.shape),
        })
        print(f"{name}: {path.name}, output={ref.shape}, range=[{ref.min():.5f},{ref.max():.5f}]")
    manifest_path = ARTIFACT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
