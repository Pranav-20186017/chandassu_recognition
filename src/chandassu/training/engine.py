"""One training loop with whole-poem batching and identity-checked epoch resume."""

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .. import LABEL_TO_ID
from ..artifacts import load_package, save_package
from ..data import assert_isolated, fingerprint, summary
from ..evaluation.metrics import better_checkpoint, fit_temperature, line_poem_metrics
from ..io import atomic_json, atomic_torch, code_identity, git_identity, identity, sha256
from ..models.encoding import collate_poems, encoded_examples, vocabulary_for
from ..models.factory import build_model
from .optimization import backward_window
from .runtime import Runtime


def loader(poems, vocabulary, batch_size, shuffle=False, seed=0):
    return DataLoader(
        encoded_examples(poems, vocabulary),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_poems,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.inference_mode()
def predict_logits(model, batches, runtime):
    model.eval()
    logits, labels = [], []
    for ids, y in batches:
        with runtime.autocast():
            values = model(ids.to(runtime.device))
        logits.append(values.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def class_weights(poems):
    counts = np.bincount([LABEL_TO_ID[p.label] for p in poems], minlength=4)
    if not counts.all():
        raise ValueError("Fitting set must contain all four classes")
    return torch.tensor(len(poems) / (4 * counts), dtype=torch.float32)


def write_history(folder, history):
    atomic_json(folder / "history.json", history)
    with (folder / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_fit(config, parts, folder, model_factory=None):
    """Optimize fit only, select on validation, then calibrate on separate poems.

    Resume starts from the last atomically completed epoch. Mid-epoch work is
    discarded and deterministically replayed. Old research checkpoints are not
    treated as resume files; use the export adapter for their inference packages.
    """
    config.validate()
    assert_isolated({role: poems for role, poems in parts.items() if poems})
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()) and not (folder / "run.json").exists():
        raise ValueError("Output directory contains unrelated files; choose a new output_dir")
    runtime = Runtime(config)
    runtime.seed_all(config.seed)
    settings = asdict(config)
    operational = {"resume", "output_dir", "data_dir"}
    contract = {
        "schema_version": 1,
        "config": {k: v for k, v in settings.items() if k not in operational},
        "partitions": {role: fingerprint(poems) for role, poems in parts.items()},
        "implementation": code_identity(),
        "runtime": runtime.metadata(),
    }
    if config.init_checkpoint:
        contract["initialization_manifest"] = sha256(Path(config.init_checkpoint) / "manifest.json")
    run_id = identity(contract)
    manifest_path = folder / "run.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["run_id"] != run_id:
            raise ValueError("Settings, data, implementation or runtime changed: choose a new output_dir")
        if not config.resume:
            raise ValueError("Run directory already exists: use --resume or a new output_dir")
        if (folder / "result.json").exists():
            result = json.loads((folder / "result.json").read_text())
            if sha256(folder / "best.pt") != result["checkpoint_sha256"]:
                raise ValueError("Completed checkpoint changed")
            load_package(folder / "model")
            print("Verified completed run; no optimization repeated", flush=True)
            return result
    elif config.resume:
        raise ValueError("--resume requires an existing run")
    else:
        atomic_json(
            manifest_path,
            {
                "run_id": run_id,
                "contract": contract,
                "git": git_identity(),
                "resolved_config": settings,
                "counts": {role: summary(poems) for role, poems in parts.items()},
            },
        )
        atomic_json(folder / "partitions.json", {role: [p.poem_id for p in poems] for role, poems in parts.items()})
    vocabulary = vocabulary_for(config.model_type, config.max_tokens, parts["fit"])
    if config.init_checkpoint:
        model, vocabulary, initial, _ = load_package(config.init_checkpoint)
        if initial["model_type"] != config.model_type:
            raise ValueError("Initialization model type differs")
        if config.max_tokens != vocabulary.max_tokens:
            raise ValueError("Initialization must retain the model's saved encoding limits")
        for key in ("width", "kernel_sizes", "dropout", "base_model", "model_revision"):
            if (
                key in initial
                and initial[key] != settings[key]
                and not (key == "kernel_sizes" and tuple(initial[key]) == tuple(settings[key]))
            ):
                raise ValueError(f"Initialization architecture/config differs: {key}")
        if config.model_type == "byt5":
            if config.gradient_checkpointing:
                model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                model.encoder.gradient_checkpointing_disable()
    else:
        model = model_factory(settings, vocabulary) if model_factory else build_model(settings, vocabulary)
    # Encoding validates capacity before the first optimizer step; never truncate.
    for poems in parts.values():
        for poem in poems:
            for line in poem.lines:
                vocabulary.encode(line)
    model.to(runtime.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience, min_lr=config.min_lr
    )
    weights_cpu = class_weights(parts["fit"])
    weights = weights_cpu.to(runtime.device)
    batches = loader(
        parts["fit"],
        vocabulary,
        config.microbatch_poems,
        config.shuffle_poems,
        config.seed if config.shuffle_poems else 0,
    )
    train_evaluation = loader(parts["fit"], vocabulary, config.evaluation_batch_poems)
    validation = loader(parts["selection"], vocabulary, config.evaluation_batch_poems)
    history, best, bad, first_epoch = [], None, 0, 1
    if config.resume and (folder / "last.pt").exists():
        # This is this application's own local optimizer state, not a public model download.
        state = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        if state["run_id"] != run_id:
            raise ValueError("Resume state identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, bad, first_epoch = state["history"], state["best"], state["bad"], state["epoch"] + 1
        batches.generator.set_state(state["loader_rng"])
        runtime.restore_rng(state["rng"])
        del state
    elif config.resume:
        print("No completed epoch; restarting this identified run from its seed", flush=True)
    print(
        f"{config.model_type}: {sum(p.numel() for p in model.parameters()):,} parameters; "
        f"fit={len(parts['fit']):,} selection={len(parts['selection']):,} calibration={len(parts.get('calibration', [])):,}; "
        f"seed={config.seed} {runtime.device}/{runtime.precision}",
        flush=True,
    )
    accumulation = config.effective_batch_poems // config.microbatch_poems
    for epoch in range(first_epoch, config.epochs + 1):
        if history and len(history) >= config.minimum_epochs and bad >= config.patience:
            break
        started = time.monotonic()
        model.train()
        objective, mass, window = torch.zeros((), device=runtime.device), 0.0, []
        for step, batch in enumerate(batches, 1):
            window.append(batch)
            if len(window) < accumulation and step < len(batches):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, denominator = backward_window(model, window, weights, weights_cpu, runtime.device, runtime.autocast)
            objective += loss
            mass += denominator
            norm = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            if not torch.isfinite(norm).item():
                raise FloatingPointError("Non-finite gradients; checkpoint not advanced")
            optimizer.step()
            if config.model_type == "cnn":
                with torch.no_grad():
                    model.embedding.weight[0].zero_()
            window = []
            if step % config.progress_every == 0:
                print(f"epoch {epoch}: {step}/{len(batches)} microbatches", flush=True)
        tz, ty = predict_logits(model, train_evaluation, runtime)
        vz, vy = predict_logits(model, validation, runtime)
        train_metrics, val_metrics = line_poem_metrics(tz, ty), line_poem_metrics(vz, vy)
        row = {
            "epoch": epoch,
            "weighted_train_objective": float(objective) / mass,
            "train_eval_loss": train_metrics["line"]["cross_entropy"],
            "train_poem_f1": train_metrics["poem"]["macro_f1"],
            "validation_eval_loss": val_metrics["line"]["cross_entropy"],
            "validation_poem_f1": val_metrics["poem"]["macro_f1"],
            "validation_line_f1": val_metrics["line"]["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.monotonic() - started,
        }
        if not all(np.isfinite(v) for v in row.values()):
            raise FloatingPointError("Non-finite training/validation metrics")
        if better_checkpoint(row, best, config.f1_delta, config.loss_delta):
            best, bad = dict(row), 0
            atomic_torch(
                folder / "best.pt",
                {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "run_id": run_id,
                },
            )
        else:
            bad += 1
        scheduler.step(row["validation_eval_loss"])
        history.append(row)
        atomic_torch(
            folder / "last.pt",
            {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best": best,
                "bad": bad,
                "epoch": epoch,
                "rng": runtime.rng_state(),
                "loader_rng": batches.generator.get_state(),
                "run_id": run_id,
            },
        )
        write_history(folder, history)
        print(
            f"epoch={epoch}/{config.epochs} validation poem F1={row['validation_poem_f1']:.6f} "
            f"CE={row['validation_eval_loss']:.6f} best={best['epoch']}",
            flush=True,
        )
    if best is None:
        raise RuntimeError("No completed training epoch")
    state = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    if state["run_id"] != run_id:
        raise ValueError("Selected checkpoint identity mismatch")
    model.load_state_dict(state["model"])
    del state
    calibration = {"temperature": 1.0, "resource": "Not calibrated; no calibration poems"}
    if parts.get("calibration"):
        cz, cy = predict_logits(model, loader(parts["calibration"], vocabulary, config.evaluation_batch_poems), runtime)
        calibration = fit_temperature(cz, cy)
    vz, vy = predict_logits(model, validation, runtime)
    result = {
        "run_id": run_id,
        "selected_epoch": best["epoch"],
        "epochs_run": len(history),
        "checkpoint_sha256": sha256(folder / "best.pt"),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "calibration": calibration,
        "selection_metrics": line_poem_metrics(vz, vy, calibration["temperature"]),
        "historical_test_used": False,
        "validation_scores_are_checkpoint_selected": True,
    }
    metadata = {
        "run": result,
        "hardware": runtime.metadata(),
        "seed": config.seed,
        "partitions": {
            role: {"sha256": fingerprint(poems), "poem_ids": [p.poem_id for p in poems]}
            for role, poems in parts.items()
        },
        "scope": "Four supported metres; reliable other-metre rejection has not been established",
    }
    save_package(folder / "model", model, settings, vocabulary, calibration["temperature"], metadata)
    atomic_json(folder / "result.json", result)
    return result
