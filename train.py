"""Train CNN or ByT5 with flat, validated arguments."""

import json
import os
import sys
import tarfile
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

# Support the existing source layout directly from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from chandassu.config import resolve
from chandassu.data import Corpus, assert_isolated, fingerprint, plain_folds, summary
from chandassu.io import atomic_json, code_identity, sha256
from chandassu.training.engine import train_fit


def train_replicates(config, corpus, parts, output):
    """Repeat a fixed CNN recipe; freeze validation selection before any test inference."""
    from scripts.evaluate import main as evaluate

    output = Path(output)
    settings = asdict(config)
    for name in ("resume", "output_dir", "data_dir"):
        settings.pop(name)
    plan = {
        "config": settings,
        "data_manifest_sha256": sha256(corpus.folder / "manifest.json"),
        "partitions": {role: fingerprint(rows) for role, rows in parts.items()},
        "code_identity": code_identity(),
        "fit_budget": len(config.seeds),
        "selection": "Selected-checkpoint validation poem Macro F1, unweighted validation line CE tie-break, seed tie-break",
        "test_used_for_selection": False,
        "ensemble": "All supplied seeds, equal calibrated probabilities; membership fixed before training",
    }
    plan_path = output / "replicate_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != json.loads(json.dumps(plan)):
            raise ValueError("Replicate plan changed; use a new output_dir")
        if not config.resume:
            raise ValueError("Replicate directory exists; use --resume or a new output_dir")
    else:
        if config.resume:
            raise ValueError("--resume requires an existing replicate plan")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Replicate directory contains unrelated files; use a new output_dir")
        atomic_json(plan_path, plan)
    print(f"Fit budget: {len(config.seeds)} CNNs; one fixed LR={config.learning_rate}; no rate search", flush=True)
    members = []
    for seed in config.seeds:
        folder = output / f"seed-{seed}"
        child = replace(
            config,
            mode="train",
            seed=seed,
            output_dir=str(folder),
            resume=config.resume and (folder / "run.json").exists(),
        )
        print(f"CNN replica seed={seed}; output={folder}", flush=True)
        result = train_fit(child, parts, folder)
        history = json.loads((folder / "history.json").read_text())
        selected = next(row for row in history if row["epoch"] == result["selected_epoch"])
        members.append(
            {
                "seed": seed,
                "model_dir": f"seed-{seed}/model",
                "learning_rate": config.learning_rate,
                "selected_epoch": result["selected_epoch"],
                "validation_poem_macro_f1": selected["validation_poem_f1"],
                "validation_line_ce": selected["validation_eval_loss"],
                "calibrated_validation_metrics": result["selection_metrics"],
                "model_manifest_sha256": sha256(folder / "model/manifest.json"),
                "checkpoint_sha256": result["checkpoint_sha256"],
            }
        )
    ranked = sorted(members, key=lambda m: (-m["validation_poem_macro_f1"], m["validation_line_ce"], m["seed"]))
    selection = {
        "selection_resource": plan["selection"],
        "test_used_for_selection": False,
        "best_single": ranked[0],
        "members": ranked,
        "ensemble_members": [m["model_dir"] for m in members],
        "ensemble_note": "Optional local ensemble; membership predetermined, not optimized on test",
        "huggingface_candidate": ranked[0]["model_dir"],
    }
    selection_path = output / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Frozen validation selection changed; use a new output_dir")
    atomic_json(selection_path, selection)
    print(f"Validation-selected single CNN: seed={ranked[0]['seed']}; test evaluation starts now", flush=True)
    for member in members:
        evaluate(
            [
                "--model_dir",
                str(output / member["model_dir"]),
                "--data_dir",
                str(corpus.folder),
                "--output",
                str(output / f"seed-{member['seed']}/test.json"),
            ]
        )
    evaluate(
        [
            "--model_dirs",
            *[str(output / member["model_dir"]) for member in members],
            "--data_dir",
            str(corpus.folder),
            "--output",
            str(output / "ensemble-test.json"),
        ]
    )
    # One review bundle preserves every checkpoint, model, report and the exact prepared corpus.
    bundle = output.parent / f"{output.name}.tar.gz"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tar.gz", dir=output.parent)
    os.close(descriptor)
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            archive.add(output, arcname=f"runs/{output.name}")
            archive.add(corpus.folder, arcname=f"data/{corpus.folder.name}")
        os.replace(temporary, bundle)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"All CNNs and test reports packaged: {bundle}", flush=True)
    return selection


def main(argv=None):
    config = resolve(argv)
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2), flush=True)
    corpus = Corpus(config.data_dir)
    output = Path(config.output_dir)
    if config.mode in {"train", "replicates"}:
        parts = corpus.final_parts(config.split_manifest, config.calibration_fraction, config.split_seed)
        test = corpus.poems("test")
        assert_isolated({**parts, "test": test})
        print(
            json.dumps(
                {role: summary(rows) for role, rows in {**parts, "test": test}.items()}, ensure_ascii=False, indent=2
            ),
            flush=True,
        )
        if config.mode == "replicates":
            train_replicates(config, corpus, parts, output)
        else:
            print("Fit budget: 1 model; no automatic hyperparameter search", flush=True)
            train_fit(config, parts, output)
    else:
        development = corpus.poems("train") + corpus.poems("validation")
        folds = plain_folds(development, config.folds, config.split_seed)
        print(f"Fit budget: {len(folds)} models; fixed supplied configuration; validation results", flush=True)
        plan = [{role: [p.poem_id for p in poems] for role, poems in fold.items()} for fold in folds]
        plan_path = output / "folds.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("CV partitions changed; use a new output_dir")
        atomic_json(plan_path, plan)
        results = [
            train_fit(replace(config, mode="train"), dict(fold, calibration=[]), output / f"fold-{n}")
            for n, fold in enumerate(folds, 1)
        ]
        atomic_json(
            output / "cross_validation.json",
            {
                "folds": results,
                "score_role": "Checkpoint-selected validation; not independent test",
                "mean_poem_macro_f1": sum(r["selection_metrics"]["poem"]["macro_f1"] for r in results) / len(results),
            },
        )


if __name__ == "__main__":
    main()
