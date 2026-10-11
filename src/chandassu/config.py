"""Flat training settings: preset first, explicit CLI overrides second."""

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class TrainConfig:
    model_type: str = "cnn"
    seed: int = 17
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    epochs: int = 60
    minimum_epochs: int = 24
    patience: int = 12
    microbatch_poems: int = 16
    effective_batch_poems: int = 16
    evaluation_batch_poems: int = 16
    dropout: float = 0.2
    gradient_clip: float = 1.0
    gradient_checkpointing: bool = False
    max_tokens: int = 256
    width: int = 64
    kernel_sizes: tuple = (3, 5, 7)
    base_model: str = "google/byt5-small"
    model_revision: str = "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
    lr_patience: int = 4
    lr_factor: float = 0.5
    min_lr: float = 0.000001
    f1_delta: float = 0.000001
    loss_delta: float = 0.0001
    device: str = "auto"
    precision: str = "auto"
    cpu_threads: int = 4
    progress_every: int = 50
    shuffle_poems: bool = False
    data_dir: str = "data/v1"
    output_dir: str = ""
    split_manifest: str | None = "splits/final_h100.json"
    calibration_fraction: float = 0.15
    split_seed: int = 142
    mode: str = "train"
    folds: int = 3
    resume: bool = False
    init_checkpoint: str | None = None

    def validate(self):
        defaults = TrainConfig()
        for spec in fields(self):
            value, expected = getattr(self, spec.name), getattr(defaults, spec.name)
            if isinstance(expected, bool) and not isinstance(value, bool):
                raise ValueError(f"{spec.name} must be boolean")
            if isinstance(expected, int) and not isinstance(expected, bool) and type(value) is not int:
                raise ValueError(f"{spec.name} must be an integer")
            if isinstance(expected, float) and (type(value) not in {int, float} or not math.isfinite(value)):
                raise ValueError(f"{spec.name} must be finite numeric")
            if isinstance(expected, str) and value is not None and not isinstance(value, str):
                raise ValueError(f"{spec.name} must be a string")
        if not isinstance(self.kernel_sizes, (tuple, list)) or any(type(k) is not int for k in self.kernel_sizes):
            raise ValueError("kernel_sizes must be an integer list")
        if self.init_checkpoint is not None and not isinstance(self.init_checkpoint, str):
            raise ValueError("init_checkpoint must be a path string")
        if self.model_type not in {"cnn", "byt5"} or self.mode not in {"train", "cross_validate"}:
            raise ValueError("Unsupported model type or execution mode")
        if not 0 <= self.seed < 2**32 or not 0 <= self.split_seed < 2**32:
            raise ValueError("Seeds must be in [0, 2**32)")
        for name in (
            "epochs",
            "minimum_epochs",
            "patience",
            "microbatch_poems",
            "effective_batch_poems",
            "evaluation_batch_poems",
            "max_tokens",
            "width",
            "cpu_threads",
            "progress_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs must not exceed epochs")
        if self.effective_batch_poems % self.microbatch_poems:
            raise ValueError("effective_batch_poems must be divisible by microbatch_poems")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid optimizer settings")
        if not 0 <= self.dropout < 1 or not 0 < self.lr_factor < 1 or not 0 < self.min_lr <= self.learning_rate:
            raise ValueError("Invalid dropout or scheduler settings")
        if self.lr_patience < 0 or self.f1_delta < 0 or self.loss_delta < 0:
            raise ValueError("Patience and checkpoint tolerances cannot be negative")
        if not self.kernel_sizes or any(k < 1 or k % 2 == 0 for k in self.kernel_sizes):
            raise ValueError("CNN kernel sizes must be positive odd integers")
        if not 0 <= self.calibration_fraction < 1:
            raise ValueError("calibration_fraction must be in [0, 1)")
        if self.mode == "cross_validate" and (self.folds < 2 or self.calibration_fraction != 0 or self.split_manifest):
            raise ValueError("Plain CV requires folds >= 2, calibration_fraction=0 and split_manifest=none")
        if self.resume and self.init_checkpoint:
            raise ValueError("Resume and initialization from another model are different operations")
        if self.device not in {"auto", "cuda", "mps", "cpu"} or self.precision not in {"auto", "bf16", "fp32"}:
            raise ValueError("Unsupported device or precision")
        if not self.output_dir:
            self.output_dir = f"runs/{self.model_type}-seed-{self.seed}"
        return self


def parser():
    result = argparse.ArgumentParser(description="Train one CNN or ByT5 model; optional plain work-grouped CV.")
    result.add_argument("--model_type", required=True, choices=("cnn", "byt5"))
    result.add_argument("--config", type=Path, help="Flat JSON preset; explicit CLI flags override it")
    defaults = TrainConfig()
    for spec in fields(defaults):
        name, value = spec.name, getattr(defaults, spec.name)
        if name == "model_type":
            continue
        kwargs = {"default": argparse.SUPPRESS}
        if name == "kernel_sizes":
            kwargs.update(type=int, nargs="+")
        elif isinstance(value, bool):
            kwargs.update(action=argparse.BooleanOptionalAction)
        else:
            kwargs.update(type=type(value) if value is not None else str)
        result.add_argument(f"--{name}", **kwargs)
    return result


def resolve(argv=None):
    supplied = vars(parser().parse_args(argv))
    model_type = supplied["model_type"]
    path = supplied.pop("config", None)
    values = asdict(TrainConfig())
    if model_type == "byt5":
        values.update(
            learning_rate=0.00003,
            epochs=32,
            minimum_epochs=8,
            patience=8,
            microbatch_poems=8,
            effective_batch_poems=16,
            evaluation_batch_poems=8,
            max_tokens=1024,
        )
    if path:
        saved = json.loads(path.read_text())
        unknown = set(saved) - set(values)
        if unknown or any(isinstance(v, dict) for v in saved.values()):
            raise ValueError(f"Config must be flat with known keys; unknown: {sorted(unknown)}")
        if saved.get("model_type", model_type) != model_type:
            raise ValueError("Preset model_type disagrees with --model_type")
        values.update(saved)
    values.update(supplied)
    if "data_dir" in supplied and "split_manifest" not in supplied and not path:
        values["split_manifest"] = None
    if values["split_manifest"] == "none":
        values["split_manifest"] = None
    if values["init_checkpoint"] == "none":
        values["init_checkpoint"] = None
    return TrainConfig(**values).validate()
