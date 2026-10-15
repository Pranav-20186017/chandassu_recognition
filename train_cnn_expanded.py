"""Fresh whole-poem eight-class CNN training, resume, evaluation and local prediction.

python train_cnn_expanded.py --config configs/cnn_eight_class.json
python train_cnn_expanded.py --config configs/cnn_eight_class.json --resume
python train_cnn_expanded.py --predict runs/cnn-eight-class/model --input poem.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
import torch
from safetensors.torch import save_file
from scipy.special import softmax
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from chandassu.expanded import (
    LABELS,
    OrderedPoemCNN,
    class_weights,
    collate,
    encode,
    load_corpus,
    load_model,
)
from chandassu.io import atomic_json, atomic_torch, git_identity, identity, sha256
from chandassu.models.encoding import CharacterVocabulary
from chandassu.training.runtime import Runtime


def metrics(logits, targets):
    targets = np.asarray(targets, dtype=int)
    if np.shape(logits) != (len(targets), len(LABELS)) or not np.isfinite(logits).all():
        raise ValueError("Expected finite eight-class poem logits")
    p = softmax(logits, axis=1)
    y = p.argmax(1)
    return dict(
        observations=len(y),
        accuracy=float(accuracy_score(targets, y)),
        macro_f1=float(f1_score(targets, y, labels=range(len(LABELS)), average="macro", zero_division=0)),
        cross_entropy=float(-np.log(np.maximum(p[np.arange(len(y)), targets], 1e-12)).mean()),
        per_class=classification_report(
            targets, y, labels=range(len(LABELS)), target_names=LABELS, output_dict=True, zero_division=0
        ),
        confusion_matrix=confusion_matrix(targets, y, labels=range(len(LABELS))).tolist(),
        class_order=list(LABELS),
    )


@torch.inference_mode()
def predict(model, batches, runtime):
    model.eval()
    logits, labels = [], []
    for ids, y in batches:
        with runtime.autocast():
            logits.append(model(ids.to(runtime.device)).float().cpu().numpy())
        labels.extend(y.tolist())
    return np.concatenate(logits), labels


def better(row, best):
    if best is None:
        return True
    delta = row["validation_poem_macro_f1"] - best["validation_poem_macro_f1"]
    return delta > 1e-6 or (abs(delta) <= 1e-6 and row["validation_ce"] < best["validation_ce"] - 1e-4)


def validate(c):
    required = {
        "seed",
        "learning_rate",
        "weight_decay",
        "epochs",
        "minimum_epochs",
        "patience",
        "batch_poems",
        "max_tokens",
        "width",
        "kernel_sizes",
        "dropout",
        "device",
        "precision",
        "cpu_threads",
        "data_dir",
        "output_dir",
        "lr_patience",
        "lr_factor",
        "min_lr",
        "gradient_clip",
    }
    if set(c) != required:
        raise ValueError(f"Unexpected/missing configuration keys: {set(c) ^ required}")
    for key in ["epochs", "minimum_epochs", "patience", "batch_poems", "max_tokens", "width", "cpu_threads"]:
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 <= c["seed"] < 2**32 or c["minimum_epochs"] > c["epochs"]:
        raise ValueError("Invalid seed or epoch limits")
    if not 0 <= c["dropout"] < 1 or not 0 < c["lr_factor"] < 1:
        raise ValueError("Invalid dropout or LR factor")
    if not 0 < c["min_lr"] <= c["learning_rate"] or c["weight_decay"] < 0 or c["gradient_clip"] <= 0:
        raise ValueError("Invalid optimizer parameters")
    if not c["kernel_sizes"] or any(type(k) is not int or k <= 0 or k % 2 == 0 for k in c["kernel_sizes"]):
        raise ValueError("Convolution kernels must be positive odd integers")
    if c["device"] not in {"auto", "cpu", "cuda", "mps"} or c["precision"] not in {"auto", "fp32", "bf16"}:
        raise ValueError("Unsupported backend/precision")
    if any(isinstance(v, float) and not np.isfinite(v) for v in c.values()):
        raise ValueError("Nonfinite configuration")


def train(c, resume=False, evaluate=True):
    validate(c)
    runtime = Runtime(SimpleNamespace(**c))
    runtime.seed_all(c["seed"])
    parts, manifest = load_corpus(c["data_dir"])
    folder = Path(c["output_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    contract = dict(
        config={k: v for k, v in c.items() if k not in {"output_dir", "data_dir"}},
        data_manifest=identity(manifest),
        runtime=runtime.metadata(),
        implementation={
            str(p): sha256(p)
            for p in [
                Path(__file__),
                Path(__file__).parent / "src/chandassu/expanded.py",
                Path(__file__).parent / "src/chandassu/models/encoding.py",
                Path(__file__).parent / "src/chandassu/training/runtime.py",
            ]
        },
        selection="Validation poem Macro F1, unweighted poem CE tie-break",
        initialization="Fresh random initialization; all eight labels together",
        calibration="None: retained calibration works remain reserved and unused in this experiment",
        test_used_for_selection=False,
    )
    run_id = identity(contract)
    exists = (folder / "run.json").exists()
    if exists:
        if not resume or json.loads((folder / "run.json").read_text())["run_id"] != run_id:
            raise ValueError("Existing run: requires --resume and identical data/config/code/runtime")
    elif resume or any(folder.iterdir()):
        raise ValueError("Cannot resume missing run or overwrite unrelated files")
    else:
        atomic_json(
            folder / "run.json", dict(run_id=run_id, contract=contract, git=git_identity(), counts=manifest["counts"])
        )
        atomic_json(folder / "partitions.json", {role: [p.poem_id for p in rows] for role, rows in parts.items()})
    vocabulary = CharacterVocabulary(c["max_tokens"]).fit(line for p in parts["fit"] for line in p.lines)
    # Validate every input's capacity before optimization; never silently truncate.
    encoded = {role: encode(rows, vocabulary) for role, rows in parts.items()}
    generator = torch.Generator().manual_seed(c["seed"])
    fit = DataLoader(encoded["fit"], batch_size=c["batch_poems"], shuffle=True, collate_fn=collate, generator=generator)

    def batches(role):
        return DataLoader(encoded[role], batch_size=c["batch_poems"], collate_fn=collate)

    model = OrderedPoemCNN(len(vocabulary.tokens) + 2, c["width"], c["kernel_sizes"], c["dropout"]).to(runtime.device)
    weights = class_weights(parts["fit"]).to(runtime.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=c["lr_factor"], patience=c["lr_patience"], min_lr=c["min_lr"]
    )
    history, best, bad, first = [], None, 0, 1
    if resume and (folder / "result.json").exists():
        result = json.loads((folder / "result.json").read_text())
        load_model(folder / "model")
        if sha256(folder / "best.pt") != result["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed")
        if evaluate:
            evaluate_test(folder, parts, runtime)
        return result
    if resume and (folder / "last.pt").exists():
        state = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        if state["run_id"] != run_id:
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, bad, first = state["history"], state["best"], state["bad"], state["epoch"] + 1
        generator.set_state(state["loader_rng"])
        runtime.restore_rng(state["rng"])
        del state
    print(
        f"Fresh eight-class ordered poem CNN, {sum(p.numel() for p in model.parameters()):,} parameters; "
        f"{runtime.device}/{runtime.precision}; fit={len(parts['fit']):,}; selection={len(parts['selection']):,}",
        flush=True,
    )
    print("Fixed fit budget: 1 seed/recipe; test inference only after validation selection is saved.", flush=True)
    for epoch in range(first, c["epochs"] + 1):
        if len(history) >= c["minimum_epochs"] and bad >= c["patience"]:
            break
        start = time.monotonic()
        model.train()
        loss_sum, mass = 0.0, 0.0
        for ids, targets in fit:
            ids, targets = ids.to(runtime.device), targets.to(runtime.device)
            optimizer.zero_grad(set_to_none=True)
            with runtime.autocast():
                logits = model(ids)
                loss = nn.functional.cross_entropy(logits, targets, weight=weights)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip"])
            if not torch.isfinite(norm).item():
                raise FloatingPointError("Nonfinite gradients")
            optimizer.step()
            with torch.no_grad():
                model.embedding.weight[0].zero_()
            denominator = weights[targets].sum().item()
            loss_sum += loss.item() * denominator
            mass += denominator
        z, y = predict(model, batches("selection"), runtime)
        val = metrics(z, y)
        row = dict(
            epoch=epoch,
            weighted_train_ce=loss_sum / mass,
            validation_ce=val["cross_entropy"],
            validation_poem_macro_f1=val["macro_f1"],
            validation_accuracy=val["accuracy"],
            learning_rate=optimizer.param_groups[0]["lr"],
            seconds=time.monotonic() - start,
        )
        if not all(np.isfinite(v) for v in row.values()):
            raise FloatingPointError("Nonfinite epoch metrics")
        if better(row, best):
            best, bad = row.copy(), 0
            atomic_torch(folder / "best.pt", dict(run_id=run_id, model=model.state_dict(), epoch=epoch))
        else:
            bad += 1
        scheduler.step(row["validation_ce"])
        history.append(row)
        atomic_torch(
            folder / "last.pt",
            dict(
                run_id=run_id,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                history=history,
                best=best,
                bad=bad,
                epoch=epoch,
                rng=runtime.rng_state(),
                loader_rng=generator.get_state(),
            ),
        )
        atomic_json(folder / "history.json", history)
        print(
            f"epoch={epoch} val poem Macro F1={val['macro_f1']:.5f} CE={val['cross_entropy']:.5f} "
            f"best={best['epoch']} seconds={row['seconds']:.1f}",
            flush=True,
        )
    state = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    if state["run_id"] != run_id:
        raise ValueError("Checkpoint identity mismatch")
    model.load_state_dict(state["model"])
    model_folder = folder / "model"
    model_folder.mkdir(exist_ok=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(model_folder / "model.safetensors"),
    )
    atomic_json(model_folder / "vocabulary.json", vocabulary.tokens)
    atomic_json(
        model_folder / "config.json",
        dict(
            architecture="ordered_poem_cnn_v1",
            labels=LABELS,
            width=c["width"],
            kernel_sizes=c["kernel_sizes"],
            dropout=c["dropout"],
            max_tokens=c["max_tokens"],
            representation="Four ordered padas; Seesamu body only",
            temperature=1.0,
            calibrated=False,
            run_id=run_id,
            training_data_manifest=identity(manifest),
        ),
    )
    atomic_json(
        model_folder / "manifest.json",
        dict(
            files={
                name: sha256(model_folder / name) for name in ["model.safetensors", "config.json", "vocabulary.json"]
            }
        ),
    )
    z, y = predict(model, batches("selection"), runtime)
    result = dict(
        run_id=run_id,
        selected_epoch=best["epoch"],
        epochs_run=len(history),
        checkpoint_sha256=sha256(folder / "best.pt"),
        selection_metrics=metrics(z, y),
        validation_scores_are_checkpoint_selected=True,
        test_used_for_selection=False,
        parameter_count=sum(p.numel() for p in model.parameters()),
        calibrated=False,
    )
    atomic_json(folder / "result.json", result)
    if evaluate:
        evaluate_test(folder, parts, runtime)
    return result


def evaluate_test(folder, parts, runtime):
    model, vocabulary, config = load_model(folder / "model")
    model.to(runtime.device)
    # Test never enters the epoch loop, scheduler, checkpoint ranking or calibration.
    batches = DataLoader(encode(parts["test"], vocabulary), batch_size=32, collate_fn=collate)
    z, y = predict(model, batches, runtime)
    work = {}
    for source in sorted({p.source for p in parts["test"]}):
        ix = [i for i, p in enumerate(parts["test"]) if p.source == source]
        work[source] = metrics(z[ix], np.asarray(y)[ix])
    cohorts = {}
    for name in ["v2_existing", "new_source_annotation"]:
        ix = [i for i, p in enumerate(parts["test"]) if p.metadata["cohort"] == name]
        if ix:
            cohorts[name] = metrics(z[ix], np.asarray(y)[ix])
    # Same selected CNN on old-label holdout with eight possible predictions.
    old_ix = [i for i, p in enumerate(parts["test"]) if p.label in LABELS[:4]]
    old_y = np.asarray(y)[old_ix]
    old_pred = z[old_ix].argmax(1)
    retained = dict(
        observations=len(old_ix),
        accuracy=float(accuracy_score(old_y, old_pred)),
        macro_f1=float(f1_score(old_y, old_pred, labels=range(4), average="macro", zero_division=0)),
    )
    report = dict(
        run_id=config["run_id"],
        selected_before_test=True,
        poem=metrics(z, y),
        per_work=work,
        cohorts=cohorts,
        original_four_label_retention=retained,
        scope="Historical v2 benchmark plus new-label held-out works; source annotations, not expert gold; "
        "no claim of blind historical test, calibration, rule verification or unseen-author generalization",
    )
    atomic_json(folder / "test.json", report)
    atomic_json(
        folder / "test_predictions.json",
        [
            dict(
                poem_id=p.poem_id,
                source=p.source,
                label=p.label,
                prediction=LABELS[int(z[i].argmax())],
                probabilities=dict(zip(LABELS, map(float, softmax(z[i])))),
            )
            for i, p in enumerate(parts["test"])
        ],
    )
    print(
        f"Selected CNN TEST poem Macro F1={report['poem']['macro_f1']:.5f}, accuracy={report['poem']['accuracy']:.5f}",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cnn_eight_class.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--skip_test", action="store_true", help="Fit/select only; defer held-out inference")
    parser.add_argument("--predict", type=Path, help="Local expanded model package; bypass training")
    parser.add_argument("--input", type=Path, help="JSON object containing four ordered body lines")
    args = parser.parse_args()
    if args.predict:
        if not args.input:
            parser.error("--predict requires --input")
        model, vocabulary, _ = load_model(args.predict)
        lines = json.loads(args.input.read_text())["lines"]
        if len(lines) != 4 or any(not isinstance(t, str) or not t.strip() for t in lines):
            raise ValueError("Prediction requires four ordered padas; Seesamu body only")
        ids, _ = collate([([vocabulary.encode(t) for t in lines], 0)])
        with torch.inference_mode():
            p = torch.softmax(model(ids), 1)[0].tolist()
        print(
            json.dumps(
                dict(prediction=LABELS[int(np.argmax(p))], uncalibrated_probabilities=dict(zip(LABELS, p))),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    c = json.loads(args.config.read_text())
    if args.output_dir:
        c["output_dir"] = args.output_dir
    if args.device:
        c["device"] = args.device
    train(c, args.resume, evaluate=not args.skip_test)


if __name__ == "__main__":
    main()
