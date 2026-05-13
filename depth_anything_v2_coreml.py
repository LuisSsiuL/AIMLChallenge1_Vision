"""Minimal CoreML wrapper for DepthAnythingV2 .mlpackage.

Restored after repo reorg (commit 562079c moved files into drivingsim/).
Exposes the API the live_depth_mlmodel_wasd_*.py scripts expect:

    model = DepthAnythingV2CoreML(model_path, colormap=cv2.COLORMAP_INFERNO)
    depth = model.predict_depth(bgr_frame)   # -> float32 HxW, raw model output
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import coremltools as ct


class DepthAnythingV2CoreML:
    def __init__(self, model_path: str, colormap: Optional[int] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CoreML model not found: {model_path}")
        self.model_path = model_path
        self.colormap = colormap
        self.model = ct.models.MLModel(model_path)

        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        self.output_name = spec.description.output[0].name

        img_type = spec.description.input[0].type.imageType
        self.input_w = int(img_type.width) or 518
        self.input_h = int(img_type.height) or 518

    def predict_depth(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize(
            (self.input_w, self.input_h), Image.BICUBIC
        )
        out = self.model.predict({self.input_name: pil})
        depth = out[self.output_name]
        if isinstance(depth, Image.Image):
            depth = np.asarray(depth, dtype=np.float32)
        else:
            depth = np.asarray(depth, dtype=np.float32).squeeze()
        if depth.ndim > 2:
            depth = depth.squeeze()
        if depth.shape != frame_bgr.shape[:2]:
            depth = cv2.resize(
                depth, (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return depth
