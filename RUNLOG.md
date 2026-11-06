# Run log

One row per training or evaluation run. Fill this in as you go, not at the end —
reconstructing it later is how numbers in the report stop matching the files.

Status values: `ok`, `failed`, `superseded`.

| Date | Who | Run ID | Method | Seed | Commit | Config ver | Output path | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-05 | Anusha | base_s11 | BASE | 11 | 8fb4208 | 5 | `checkpoints/base_config5_seed11.pt`, `results/preds/base_seed11_clean.npz` | ok | 20 epochs, final_epoch. Clean test acc 97.21%, val NLL 0.1037 |
| 2026-11-05 | Anusha | base_s22 | BASE | 22 | 8fb4208 | 5 | `checkpoints/base_config5_seed22.pt`, `results/preds/base_seed22_clean.npz` | ok | 20 epochs, final_epoch. Clean test acc 97.47%, val NLL 0.0960 |
| 2026-11-05 | Anusha | base_s33 | BASE | 33 | 8fb4208 | 5 | `checkpoints/base_config5_seed33.pt`, `results/preds/base_seed33_clean.npz` | ok | 20 epochs, final_epoch. Clean test acc 97.08%, val NLL 0.1098. BASE 97.25 ± 0.20, matches Smitha's run |
| 2026-11-05 | Anusha | mc_dropout_s11 | MC Dropout | 11 | 8fb4208 | 5 | `checkpoints/mc_dropout_config5_seed11.pt`, `results/preds/mc_dropout_seed11_clean_passes{2,20}.npz` | ok | 20 epochs, final_epoch, dropout 0.1. Clean test acc 97.47% (2 passes) / 97.62% (20 passes), val NLL 0.0850 |
| 2026-11-05 | Anusha | mc_dropout_s22 | MC Dropout | 22 | 8fb4208 | 5 | `checkpoints/mc_dropout_config5_seed22.pt`, `results/preds/mc_dropout_seed22_clean_passes{2,20}.npz` | ok | 20 epochs, final_epoch, dropout 0.1. Clean test acc 97.06% / 97.46%, val NLL 0.0870 |
| 2026-11-05 | Anusha | mc_dropout_s33 | MC Dropout | 33 | 8fb4208 | 5 | `checkpoints/mc_dropout_config5_seed33.pt`, `results/preds/mc_dropout_seed33_clean_passes{2,20}.npz` | ok | 20 epochs, final_epoch, dropout 0.1. Clean test acc 96.95% / 97.34%, val NLL 0.1003 |
| 2026-11-06 | Anusha | deep_ensemble_r11 | Deep ensemble | 11 | 8fb4208 | 5 | `checkpoints/deep_ensemble_config5_run11_member{1-5}_seed{1101-1105}.pt`, `results/preds/deep_ensemble_seed11_clean_members{2,5}.npz` | ok | 5 members, 20 epochs each, final_epoch. Clean test acc 97.66% (2 members) / 97.88% (5 members) |
| 2026-11-06 | Anusha | deep_ensemble_r22 | Deep ensemble | 22 | 8fb4208 | 5 | `checkpoints/deep_ensemble_config5_run22_member{1-5}_seed{2201-2205}.pt`, `results/preds/deep_ensemble_seed22_clean_members{2,5}.npz` | ok | 5 members, 20 epochs each, final_epoch. Clean test acc 97.17% / 97.69% |
| 2026-11-06 | Anusha | deep_ensemble_r33 | Deep ensemble | 33 | 8fb4208 | 5 | `checkpoints/deep_ensemble_config5_run33_member{1-5}_seed{3301-3305}.pt`, `results/preds/deep_ensemble_seed33_clean_members{2,5}.npz` | ok | 5 members, 20 epochs each, final_epoch. Clean test acc 97.69% / 97.95% |

## Environment fingerprints (Step 1)

Each of us runs `python tests/smoke_test.py` and records the result. These must
match before any training starts.

| Person | Date | torch | torchvision | Split indices | Batch labels | Batch images | Train mean | Match? |
|---|---|---|---|---|---|---|---|---|
| Abirami | 2026-10-19 |2.14.1+cpu|0.29.1+cpu| aaed0a23690e2e3c | cae9caa451436cf3 | 521727b4f446b293 | 0.130731 | yes |
| Anusha | 2026-10-19 |2.14.1+cpu|0.29.1+cpu| aaed0a23690e2e3c | cae9caa451436cf3 | 521727b4f446b293 | 0.130731 | yes |
| Smitha | 2026-10-17 |2.14.1+cpu|0.29.1+cpu| aaed0a23690e2e3c | cae9caa451436cf3 | 521727b4f446b293 | 0.130731 | yes |

## Protocol changes

Any edit to `configs/base.yaml`. Every row needs a reason and a note on whether
earlier runs are still comparable.

| Date | Config ver | What changed | Why | Reruns needed |
|---|---|---|---|---|
| 2026-10-17 | 1 | Initial protocol | Frozen at Step 2 | - |
| 2026-10-19 | 2 | Added `model:` block - channels, kernel, pooling, head | Step 7 shared CNN needed a recorded architecture | None, no training has run |
| 2026-10-22 | 3 | `training.data_order: fixed_by_split_seed`. MC Dropout `headline_is_prefix_of_reference: true` | Record that model seed does not change minibatch order, and that the 2-pass file is the first 2 of the 20 | None, no reported training has run |
| 2026-10-30 | 4 | `training.num_workers: 0`. `fit_cnn` calls `pin_training_randomness` after model init | Keep augmentation tied to `split_seed`, not model seed; document worker init when num_workers > 0 | None, no reported training has run |
| 2026-11-02 | 5 | Added `qute.tau: [0.25, 0.5]` and `evaluation.window_stride: 1`. `training.checkpoint_selection: final_epoch` (was `min_val_nll`), added 2026-11-05 | QUTE trainer needs early-exit weights; paper gives no tau values and no window stride. Final-epoch weights match the paper and keep QUTE's frozen stage | None, no training or evaluation has run |

## Decisions we made that the paper does not specify

| Date | Decision | Reason | Goes in the report? |
|---|---|---|---|
|2026-10-17 | 50,000 / 10,000 validation split, seed 0 | Validation NLL is logged every epoch for reporting and overfitting checks (originally for checkpoint selection, dropped 2026-11-05) | Yes |
|2026-10-17 | Seeds 11, 22, 33 | Paper reports 3 runs but not which seeds | Yes |
|2026-10-17 | 15 equal-mass ECE bins | Confidences cluster near 1 | Yes, as a deviation |
| 2026-10-19 | Channel widths [8, 16, 32, 32] | Paper says "4-layer CNN" but gives no widths | Yes, as a deviation |
| 2026-10-22 | Minibatch order is fixed by split seed 0 for every model seed and ensemble member. Ensemble diversity is initialization only | A shared global RNG would also change data order when the model seed changes, which confounds the QUTE comparison | Yes |
| 2026-10-22 | Deep-ensemble member k of run R is initialized with R * 100 + k (run 11 uses 1101–1105) | The protocol names 3 run seeds and 5 members, and does not list 5 initialization seeds | Yes |
| 2026-10-30 | Adam: `training.momentum` is beta1; beta2 is fixed at 0.999 | Paper lists momentum for Adam without naming betas | Yes, as a deviation |
| 2026-10-30 | After model init, global RNG is reseeded from `data.split_seed` so train-time augmentation matches across run seeds and ensemble members | `set_seed(model_seed)` would otherwise make RandomCrop/RandomRotation differ per seed | Yes |
| 2026-11-02 | Early-exit loss weights tau = [0.25, 0.5] (relative depth of each exit) | Paper specifies w_EV (3.0, 3.5) but only cites Kaya et al. 2019 for tau, with no values | Yes, as a deviation |
| 2026-11-02 | Models predict each condition once (clean + 15 corruptions = 16 files per model per seed). Streams are built in eval from saved predictions | Avoids re-predicting clean 15×, keeps sample IDs unique per file, keeps MC Dropout clean predictions consistent | Yes |
| 2026-11-02 | Stream = clean MNIST test then one corruption. Window 100, stride 1 | Section 5 / Appendix A.3–A.4; stride is not stated in the paper | Yes, stride as a deviation |
| 2026-11-02 | Degradation event = window accuracy strictly below μ_ID − 3σ_ID, with μ and σ from that model's clean-only window accuracies. Alarm = mean window confidence < ρ | Matches the paper and "below" in the config | Yes |
| 2026-11-02 | Primary AUPRC = average precision on continuous mean confidence. Paper's 11-point ρ sweep reported separately for comparison | The two estimators give different values | Yes |
| 2026-11-02 | AUPRC is marked undefined when a stream has no degradation events, never 0 or 1 | Follows `undefined_auprc_policy: mark_undefined` | Yes |
| 2026-11-05 | All methods keep the final-epoch (20) weights; validation NLL is logged per epoch but does not select the checkpoint. Replaces min-val-NLL selection (2026-10-17) | QUTE's final stage (frozen base, epochs 19–20) is part of the method and selection could skip it; one rule for every method; matches the paper | Yes, as alignment with the paper |
