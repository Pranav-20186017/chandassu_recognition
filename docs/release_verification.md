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

## GX10 CNN update — 2026-10-11

- Three fresh fixed-LR CNN replicas on the frozen expanded v2 corpus: seeds 17/42/73,
  initial LR 0.0001, 60 epochs each, NVIDIA GB10, BF16, NVIDIA PyTorch 2.10.0a0/CUDA 13.0.
  The clean training checkout was f7939ccac30d14bab929ed7d9c91aa202c5a9c80.
- Fitting 7,575 poems, validation 1,859, separate temperature calibration 908, test 2,953.
  Original v1 roles and the seven frozen new test works are preserved; every test work
  stays outside fitting, vocabulary construction, selection and calibration.
- Seed 73, epoch 58, wins the validation F1 tie through lower validation line CE.
  Its temperature is 1.401754510250526. Selection was recorded before test evaluation.
- Best checkpoints safely loaded with weights_only=True; every exported tensor matches
  its selected checkpoint exactly. Run/code identities, manifests, split fingerprints,
  calibration assignments and checkpoint selection were verified for all three seeds.
- Mac CPU inference reproduces all single-model and ensemble test confusion matrices
  and F1 values. Seed 73: 2,950/2,953 correct, poem Macro F1 0.999084, line Macro F1
  0.993420. New-work cohort: 716/719 correct, poem Macro F1 0.996042.
- All seeds, the equal-probability ensemble, and the previous v1 seed-17 CNN make the
  same three test poem errors. No poem-level accuracy gain is claimed. Both test cohorts
  are previously evaluated benchmarks, not a new blind study.
- Local native and Transformers Auto encoding/logits match exactly on CPU; exported
  weights and temperature are preserved. Single-model and three-model inference APIs pass.
- Default local CNN is seed 73; all three packages remain available separately. Previous
  local CNN packages are backed up in the ignored archive. ByT5 weights stay unchanged.
- 24 tests and lint pass. Weights and returned research records stay out of Git. The
  staged CNN Hub package includes the selection, test and verification evidence;
  publication remains pending approval of the exact repository and payload.

Complete local review: runs/review-gx10-cnn-replicates/verification.json and
export-verification.json. Original H100 verification above remains historical evidence.
