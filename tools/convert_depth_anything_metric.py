#!/usr/bin/env python3
"""
Convert Depth-Anything-V2-Metric-Indoor-Small to CoreML (.mlpackage).

Output: DepthAnythingV2MetricIndoorSmallF16.mlpackage
  - Input:  image (518×518 RGB, normalised per DPT convention)
  - Output: depth (518×518 float16, metres, range 0–20m, large = far)

Place the output in drivingsim/drivingsim/ then add it to the Xcode target.

Usage:
  pip install torch torchvision coremltools huggingface_hub transformers pillow
  python tools/convert_depth_anything_metric.py

Tested on Apple Silicon macOS with Python 3.11.
"""

import os
import sys
import tempfile

import numpy as np
import torch
import coremltools as ct
from huggingface_hub import hf_hub_download
from PIL import Image


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID    = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
INPUT_SIZE  = 518           # DPT encoder input size
MAX_DEPTH   = 20.0          # metres — model trained with max_depth=20 (Hypersim)
OUT_NAME    = "DepthAnythingV2MetricIndoorSmallF16"
OUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "drivingsim", "drivingsim")


# ── Load model ────────────────────────────────────────────────────────────────

print(f"[1/4] Downloading + loading {MODEL_ID} (~500MB, may take a few minutes)...")
sys.stdout.flush()
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
print("[1/4] Processor loaded.")
sys.stdout.flush()
hf_model  = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
hf_model.eval()
print("[1/4] Model loaded.")
sys.stdout.flush()


# ── Trace via torch.jit ───────────────────────────────────────────────────────

class DepthWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values)
        depth = out.predicted_depth
        depth = depth.unsqueeze(1)
        return depth


wrapper = DepthWrapper(hf_model)
wrapper.eval()

print("[2/4] Tracing model with torch.jit (30-60s)...")
sys.stdout.flush()
dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
with torch.no_grad():
    traced = torch.jit.trace(wrapper, dummy)
print("[2/4] Trace done.")
sys.stdout.flush()


# ── Convert to CoreML ─────────────────────────────────────────────────────────

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

scale = ct.ImageType(
    name="image",
    shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
    bias=[-m / s for m, s in zip(MEAN, STD)],
    scale=[1.0 / (s * 255.0) for s in STD],
    color_layout=ct.colorlayout.RGB,
)

print("[3/4] Converting to CoreML (CPU_AND_NE compile — 5-15 min)...")
sys.stdout.flush()

mlmodel = ct.convert(
    traced,
    inputs=[scale],
    outputs=[ct.TensorType(name="depth")],
    compute_units=ct.ComputeUnit.ALL,   # ANE preferred
    convert_to="mlprogram",             # .mlpackage with Float16 weights
    minimum_deployment_target=ct.target.macOS14,
    compute_precision=ct.precision.FLOAT16,
)

# Annotate metadata.
mlmodel.short_description = "Depth-Anything-V2 Metric Indoor Small — metric depth in metres (0–20m)"
mlmodel.input_description["image"] = "518×518 RGB frame"
mlmodel.output_description["depth"] = "Metric depth map in metres (Float16, 1×1×518×518). Large = far."

print("[4/4] Saving .mlpackage...")
sys.stdout.flush()
out_path = os.path.join(OUT_DIR, OUT_NAME + ".mlpackage")
mlmodel.save(out_path)
print(f"\n✓ Done: {out_path}")
print("Next: drag DepthAnythingV2MetricIndoorSmallF16.mlpackage into Xcode target, build, run.")
