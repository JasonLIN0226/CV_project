from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .utils import digest, file_hash


def checkpoint_components(config):
    """Resolve config and official slow processor at the same immutable revision."""
    from transformers import AutoConfig, AutoImageProcessor
    repo = config["repository"]
    kwargs = {"revision": config["revision"], "cache_dir": config["cache_dir"],
              "local_files_only": config["local_files_only"], "trust_remote_code": False}
    mc = AutoConfig.from_pretrained(repo, **kwargs)
    if mc.model_type != "ijepa":
        raise ValueError("Official backend requires an I-JEPA checkpoint")
    revision = getattr(mc, "_commit_hash", None)
    if Path(repo).is_dir():
        files = sorted(Path(repo).glob("*.safetensors")) + sorted(Path(repo).glob("*.bin"))
        if not files:
            raise ValueError("Local checkpoint has no weight files")
        revision = "local-sha256:" + digest({p.name: file_hash(p) for p in files})
    else:
        if revision is None:
            from huggingface_hub import HfApi
            revision = HfApi().model_info(repo, revision=config["revision"]).sha
        kwargs["revision"] = revision
    processor = AutoImageProcessor.from_pretrained(repo, backend="pil", **kwargs)
    if getattr(processor, "do_center_crop", False):
        raise ValueError("Center crop could remove evidence; use a no-crop official processor")
    return mc, processor, revision, kwargs


def pool_tokens(tokens, height, width, patch_size, pooling):
    ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    gh, gw = height // ph, width // pw
    if tokens.ndim != 3 or tokens.shape[1] != gh * gw:
        raise ValueError(f"Expected {gh*gw} patch tokens, got {tokens.shape}; do not guess or drop a CLS token")
    if pooling == "mean":
        return tokens.mean(1)
    if pooling != "grid2" or gh < 2 or gw < 2:
        raise ValueError("Invalid spatial pooling/grid")
    grid = tokens.reshape(len(tokens), gh, gw, -1)
    # Four explicit non-overlapping cells, row-major. Handles odd grids without overlaps.
    return torch.cat([grid[:, y0:y1, x0:x1].mean((1, 2))
                      for y0, y1 in [(0, gh//2), (gh//2, gh)]
                      for x0, x1 in [(0, gw//2), (gw//2, gw)]], dim=-1)


class SmokePixels:
    """Fixed RGB downsampling, never receives simulator metadata. NOT JEPA."""
    def __init__(self, config, device):
        self.size = config["smoke_size"]
        self.device = device
        self.metadata = {"name": "smoke_pixels_NOT_JEPA", "resolved_revision": "fixed-area-v1",
                         "processor": {"resize": self.size, "rescale": "RGB/255", "crop": False},
                         "pooling": "spatial_downsample", "dtype": "float32", "special_tokens": 0}

    @torch.inference_mode()
    def encode(self, rgb):
        x = torch.from_numpy(np.asarray(rgb).copy()).permute(0, 3, 1, 2).to(self.device, torch.float32)/255
        return F.interpolate(x, (self.size, self.size), mode="area").flatten(1).cpu().numpy()


class FrozenIJepa:
    def __init__(self, config, device, model=None, processor=None):
        self.config, self.device = config, device
        dtype = getattr(torch, config["dtype"], None)
        if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError("Encoder dtype must be float32/float16/bfloat16")
        if device.type == "cpu" and dtype != torch.float32:
            raise ValueError("Use float32 on CPU")
        if model is None:
            from transformers import IJepaModel
            repo = config["repository"]
            _, processor, revision, kwargs = checkpoint_components(config)
            # No random fallback. Missing/mismatched weights fail loudly.
            model, loading = IJepaModel.from_pretrained(repo, dtype=dtype, output_loading_info=True, **kwargs)
            if any(loading.get(k) for k in ["missing_keys", "mismatched_keys", "error_msgs"]):
                raise ValueError("Incomplete/mismatched I-JEPA weights; refusing randomly initialized parameters")
        else:
            revision = "UNTRAINED_TINY_ADAPTER_TEST_ONLY"
        if processor is None:
            raise ValueError("Image processor required")
        if getattr(processor, "do_center_crop", False):
            raise ValueError("Center crop could remove evidence; use a no-crop official processor")
        self.model = model.eval().requires_grad_(False).to(device=device, dtype=dtype)
        self.processor, self.dtype = processor, dtype
        self.metadata = {"name": config["repository"], "resolved_revision": revision,
                         "processor": processor.to_dict(), "pooling": config["pooling"],
                         "dtype": config["dtype"], "special_tokens": 0,
                         "model_config": model.config.to_dict(),
                         "parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters())}

    @torch.inference_mode()
    def encode(self, rgb):
        x = self.processor(images=[Image.fromarray(im) for im in rgb], return_tensors="pt")["pixel_values"]
        x = x.to(device=self.device, dtype=self.dtype)
        try:
            tokens = self.model(pixel_values=x).last_hidden_state
            z = pool_tokens(tokens, x.shape[-2], x.shape[-1], self.model.config.patch_size, self.config["pooling"])
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("I-JEPA CUDA OOM: lower encoder.batch_size (start with 1-2), then use a new run config.") from exc
        return z.float().cpu().numpy()


def build_encoder(c, device):
    return SmokePixels(c["encoder"], device) if c["encoder"]["backend"] == "smoke_pixels" else FrozenIJepa(c["encoder"], device)
