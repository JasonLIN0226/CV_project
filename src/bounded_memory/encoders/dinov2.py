"""DINOv2 ViT-B/14 class-token features (FrameMorrow's visual encoder and teacher).

`preprocess` and the encoding loop are moved unchanged from
experiments/memory_baselines/encode_dinov2.py. The checkpoint revision is passed explicitly so a
rerun uses the same weights as the reference run.
"""
from __future__ import annotations

import numpy as np
import torch

REPO = "facebook/dinov2-base"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # the checkpoint's own normalisation constants
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(rgb):
    """Whole 96x96 frame -> 224x224, bicubic. The stock processor centre-crops, which would cut into the cue panel."""
    from PIL import Image
    x = np.stack([np.asarray(Image.fromarray(f).resize((224, 224), Image.BICUBIC), np.float32) / 255.0 for f in rgb])
    return torch.from_numpy(np.ascontiguousarray(((x - MEAN) / STD).transpose(0, 3, 1, 2)))


def load_model(revision, device):
    from transformers import AutoModel
    return AutoModel.from_pretrained(REPO, revision=revision).to(device).eval()


@torch.inference_mode()
def encode_frames(model, rgb, device, batch=64):
    """rgb [frames, H, W, 3] uint8 -> [frames, 768] float32 class tokens."""
    chunks = []
    for i in range(0, len(rgb), batch):
        pixels = preprocess(rgb[i:i + batch]).to(device)
        chunks.append(model(pixel_values=pixels).last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)
