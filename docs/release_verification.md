# Release verification — 2026-10-11

- Locked macOS CPU environment: 15 tests pass; lint and formatting checks pass.
- Linux ARM64 Docker test stage: the same 15 tests and lint checks pass.
- Production CPU Docker image: exported CNN mounted read-only, health/info/predict API verified.
- Actual `train.py` CLI: one complete CNN epoch on the frozen 6,517 fitting poems,
  separate validation and calibration, then completed-run `--resume` verification.
  This smoke run lives under ignored runs/ and is not presented as a final trained model.
- Interrupted epoch checkpoint tests: CNN and tiny local ByT5 resumed weights, metrics,
  calibration and random state match uninterrupted runs. Changed seeds are rejected.
- Weighted gradient accumulation matches a full weighted update, including a partial
  final window. Padding invariance, train-only vocabulary, split/edition/duplicate
  isolation, flat overrides and invalid input/artifact checks pass.
- Export verification: all CNN and ByT5 seed-17 weight tensors exactly match the original
  checkpoint. Direct original-checkpoint and SafeTensors inference logits match exactly
  on four historical poems under the same FP32 CPU runtime. See release_export_verification.json.
- Exported CNN evaluated explicitly on all 2,234 historical poems: poem accuracy and
  Macro F1 are 1.0; line accuracy is 0.994405. See release_cnn_historical_verification.json.
- Single/ensemble probability averaging, per-seed selection, default strategy, unavailable
  seed rejection and duplicate-seed rejection are tested.
- Original light browser paste interface restored with a remembered light/dark toggle,
  single-seed/ensemble selector, all class scores and line predictions;
  screenshot in results/v1/inference_interface.jpg. Server is `inference.py`.
- All 7,449 original tracked/nonignored file hashes have a preserved on-disk copy.
  The pre-cleanup all-ref Git bundle verifies successfully.

GPU training/gradient checkpointing and the CUDA image were not executed on this Mac.
The original completed H100 evidence is retained; the new code is not claimed to recreate
that runtime bit-for-bit. Full historical ByT5 evaluation was not repeated locally;
its completed H100 reports and exact export checks are included. A Starlette test-client
transport deprecation warning does not affect the passing API checks.
