"""Backend, precision, and restartable random state."""

import contextlib
import os
import platform
import random
from importlib.metadata import version

import numpy as np
import torch


def select_device(requested):
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    if requested == "auto":
        return next(k for k, exists in available.items() if exists)
    if not available.get(requested):
        raise ValueError(f"Requested device unavailable: {requested}")
    return requested


class Runtime:
    def __init__(self, config):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(config.cpu_threads)
        self.device = select_device(config.device)
        supports_bf16 = self.device == "cuda" and torch.cuda.is_bf16_supported()
        if config.precision == "bf16" and not supports_bf16:
            raise ValueError("BF16 requires a supported CUDA device; use fp32 on CPU/MPS")
        self.precision = "bf16" if supports_bf16 and config.precision != "fp32" else "fp32"

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.precision == "bf16" else contextlib.nullcontext()

    def seed_all(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        elif self.device == "mps":
            torch.mps.manual_seed(seed)

    def rng_state(self):
        accelerator = (
            torch.cuda.get_rng_state_all()
            if self.device == "cuda"
            else torch.mps.get_rng_state()
            if self.device == "mps"
            else None
        )
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(),
            "accelerator": accelerator,
        }

    def restore_rng(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["cpu"])
        if self.device == "cuda":
            torch.cuda.set_rng_state_all(state["accelerator"])
        elif self.device == "mps":
            torch.mps.set_rng_state(state["accelerator"])

    def metadata(self):
        return {
            "device": self.device,
            "precision": self.precision,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if self.device == "cuda" else None,
            "versions": {p: version(p) for p in ("transformers", "numpy", "scipy", "scikit-learn")},
            "cross_backend_bitwise_identity": False,
        }
