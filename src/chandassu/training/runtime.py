"""Explicit runtime state and atomic output writes for training."""

from dataclasses import dataclass, field
from pathlib import Path
import contextlib
import gc
import random
import numpy as np
import torch
from ..evaluation.corpus_audit import json_bytes


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def select_device(override=None):
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if override is not None:
        if override not in available or not available[override]:
            raise ValueError(f"Device is unavailable: {override}")
        return override
    return next(device for device in ("cuda", "mps", "cpu") if available[device])


@dataclass
class TrainingSession:
    config: dict
    run_sha: str
    device: str
    records: dict = field(default_factory=dict)
    token: str | bool = False
    progress_every: int = 50
    model_factory: object = None

    @property
    def amp(self):
        return (
            self.device == "cuda"
            and self.config["bf16_cuda"]
            and torch.cuda.is_bf16_supported()
        )

    def amp_context(self):
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.amp
            else contextlib.nullcontext()
        )

    def seed_all(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        if self.device == "mps":
            torch.mps.manual_seed(seed)

    def rng_state(self):
        accelerator = None
        if self.device == "cuda":
            accelerator = torch.cuda.get_rng_state_all()
        if self.device == "mps":
            accelerator = torch.mps.get_rng_state()
        return dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            cpu=torch.get_rng_state(),
            accelerator=accelerator,
        )

    def restore_rng(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["cpu"])
        if self.device == "cuda":
            torch.cuda.set_rng_state_all(state["accelerator"])
        if self.device == "mps":
            torch.mps.set_rng_state(state["accelerator"])

    def synchronize_device(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        if self.device == "mps":
            torch.mps.synchronize()

    def clean_memory(self):
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        if self.device == "mps":
            torch.mps.empty_cache()
