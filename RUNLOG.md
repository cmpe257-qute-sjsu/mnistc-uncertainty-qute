# Run log

One row per training or evaluation run. Fill this in as you go, not at the end —
reconstructing it later is how numbers in the report stop matching the files.

Status values: `ok`, `failed`, `superseded`.

| Date | Who | Run ID | Method | Seed | Commit | Config ver | Output path | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-05 | Anusha | base_s11 | BASE | 11 | 8fb4208 | 5 | `checkpoints/base_config5_seed11.pt`, `results/preds/base_seed11_clean.npz` | ok | 20 epochs, final_epoch. Clean test acc 97.21%, val NLL 0.1037 |
| 2026-11-05 | Anusha | base_s22 | BASE | 22 | 8fb4208 | 5 | `checkpoints/base_config5_seed22.pt`, `results/preds/base_seed22_clean.npz` | ok | 20 epochs, final_epoch. Clean test acc 97.47%, val NLL 0.0960 |
| 2026-11-05 | Anusha | base_s33 | BASE | 33 | 8fb4208 | 5 | `checkpoints/base_config5_seed33.pt`, `results/preds/base_seed33_clean.npz` | ok | 20 epochs, final_epoch. Clean test acc 97.08%, val NLL 0.1098 |
| 2026-11-05 | Anusha | mc_dropout_s11 | MC Dropout | 11 | 8fb4208 | 5 | `checkpoints/mc_dropout_config5_seed11.pt`, `results/preds/mc_dropout_seed11_clean_passes{2,20}.npz` | ok | 20 epochs, final_epoch, dropout 0.1. Clean test acc 97.47% (2 passes) / 97.62% (20 passes), val NLL 0.0850 |
| 2026-11-05 | Anusha | mc_dropout_s22 | MC Dropout | 22 | 8fb4208 | 5 | `checkpoints/mc_dropout_config5_seed22.pt`, `results/preds/mc_dropout_seed22_clean_passes{2,20}.npz` | ok | 20 epochs, final_epoch, dropout 0.1. Clean test acc 97.06% / 97.46%, val NLL 0.0870 |
| 2026-11-05 | Anusha | mc_dropout_s33 | MC Dropout | 33 | 8fb4208 | 5 | `checkpoints/mc_dropout_config5_seed33.pt`, `results/preds/mc_dropout_seed33_clean_passes{2,20}.npz` | ok | 20 epochs, final_epoch, dropout 0.1. Clean test acc 96.95% / 97.34%, val NLL 0.1003 |
| 2026-11-06 | Anusha | deep_ensemble_r11 | Deep ensemble | 11 | 8fb4208 | 5 | `checkpoints/deep_ensemble_config5_run11_member{1-5}_seed{1101-1105}.pt`, `results/preds/deep_ensemble_seed11_clean_members{2,5}.npz` | ok | 5 members, 20 epochs each, final_epoch. Clean test acc 97.66% (2 members) / 97.88% (5 members) |
| 2026-11-06 | Anusha | deep_ensemble_r22 | Deep ensemble | 22 | 8fb4208 | 5 | `checkpoints/deep_ensemble_config5_run22_member{1-5}_seed{2201-2205}.pt`, `results/preds/deep_ensemble_seed22_clean_members{2,5}.npz` | ok | 5 members, 20 epochs each, final_epoch. Clean test acc 97.17% / 97.69% |
| 2026-11-06 | Anusha | deep_ensemble_r33 | Deep ensemble | 33 | 8fb4208 | 5 | `checkpoints/deep_ensemble_config5_run33_member{1-5}_seed{3301-3305}.pt`, `results/preds/deep_ensemble_seed33_clean_members{2,5}.npz` | ok | 5 members, 20 epochs each, final_epoch. Clean test acc 97.69% / 97.95% |
| 2026-11-05 | Smitha | base-11 | BASE | 11 | 8fb4208 | 5 | `results/preds/base_seed11_clean.npz` (local) | superseded | Cross-machine reproducibility check; Anusha's BASE checkpoints are the official ones. Clean test acc 96.98% vs her 97.21% from the same commit, config and seed |
| 2026-11-05 | Smitha | base-22 | BASE | 22 | 8fb4208 | 5 | `results/preds/base_seed22_clean.npz` (local) | superseded | Cross-machine check. 97.42% vs 97.47% |
| 2026-11-05 | Smitha | base-33 | BASE | 33 | 8fb4208 | 5 | `results/preds/base_seed33_clean.npz` (local) | superseded | Cross-machine check. 97.11% vs 97.08%. CPU training is not bit-reproducible across machines; differences are about 0.2 points, the size of the seed spread |
| 2026-11-05 | Smitha | qute_s11 | QUTE | 11 | 8fb4208 | 5 | `checkpoints/qute_config5_seed11.pt` | ok | 20 epochs, final_epoch, base frozen epochs 19–20. Clean test acc 97.33% |
| 2026-11-05 | Smitha | qute_s22 | QUTE | 22 | 8fb4208 | 5 | `checkpoints/qute_config5_seed22.pt` | ok | Clean test acc 97.67% |
| 2026-11-05 | Smitha | qute_s33 | QUTE | 33 | 8fb4208 | 5 | `checkpoints/qute_config5_seed33.pt` | ok | Clean test acc 97.72%. QUTE 97.57 ± 0.21 |
| 2026-11-05 | Smitha | qute_nt_s11 | QUTE no transfer | 11 | 8fb4208 | 5 | `checkpoints/qute_no_transfer_config5_seed11.pt` | ok | Ablation: transfer off, everything else identical. Clean test acc 97.64% |
| 2026-11-05 | Smitha | qute_nt_s22 | QUTE no transfer | 22 | 8fb4208 | 5 | `checkpoints/qute_no_transfer_config5_seed22.pt` | ok | Clean test acc 97.65% |
| 2026-11-05 | Smitha | qute_nt_s33 | QUTE no transfer | 33 | 8fb4208 | 5 | `checkpoints/qute_no_transfer_config5_seed33.pt` | ok | Clean test acc 97.75%. Ablation 97.68 ± 0.06, matches QUTE on accuracy as expected (transfer targets calibration, not accuracy) |
| 2026-11-06 | Smitha | qute_eval | QUTE + no transfer | 11, 22, 33 | 25494c5 | 5 | `results/preds/conditions/qute*_seed*_{condition}.npz` | ok | 96 files, 16 conditions each, labels attached, via eval/predict_conditions.py |
| 2026-11-07 | Anusha | base_eval | BASE | 11, 22, 33 | e68df15 | 5 | `results/preds/conditions/base_seed*_{condition}.npz` | ok | 48 files, 16 conditions each, labels attached, via eval/predict_conditions.py. Uploaded to Drive predictions/ (preds_base.zip) |
| 2026-11-07 | Anusha | mc_dropout_eval | MC Dropout | 11, 22, 33 | e68df15 | 5 | `results/preds/conditions/mc_dropout_seed*_{condition}_passes{2,20}.npz` | ok | 96 files (2-pass headline + 20-pass reference), 16 conditions each, labels attached, via eval/predict_conditions.py. Uploaded to Drive predictions/ (preds_mc_dropout_passes2.zip, preds_mc_dropout_passes20.zip) |
| 2026-11-07 | Anusha | deep_ensemble_eval | Deep ensemble | 11, 22, 33 | e68df15 | 5 | `results/preds/conditions/deep_ensemble_seed*_{condition}_members{2,5}.npz` | ok | 96 files (2-member headline + 5-member reference), 16 conditions each, labels attached, via eval/predict_conditions.py. Uploaded to Drive predictions/ (preds_deep_ensemble_members2.zip, preds_deep_ensemble_members5.zip) |
| 2026-11-09 | Smitha | calibration_all | all 7 variants | 11, 22, 33 | 0186f43 | 5 | `results/tables/calibration_summary.csv`, `calibration_per_file.csv` | ok | 336 files scored with eval.metrics (15 equal-mass ECE bins). MNIST-C ECE at K=2: QUTE 0.226 ± 0.045, DE-2 0.265, MC-2 0.271, BASE 0.282 |
| 2026-11-13 | Smitha | width_sweep | BASE | 11 | ffa48c2 | 5 (channels only) | `results/tables/width_sweep.csv`, `checkpoints/width_sweep/` | ok | Val acc: 8-16-32-32 0.9682, 6-12-24-24 0.9621, 4-8-16-16 0.9554, 3-6-12-12 0.9400, 2-4-8-8 0.9142. Chosen [2,4,8,8] (1,066 params) by the pre-stated rule; wrote `configs/small_cnn.yaml` (config_version 51) |
| 2026-11-13 | Smitha | small_cnn base | BASE | 11, 22, 33 | 10a76fa | config v51 (small_cnn.yaml), worktree | `results/small_cnn/` | ok | Clean test acc: 0.9199 / 0.8976 / 0.7991 (seed 33 well below the 0.91 target; kept, as pre-registered) |
| 2026-11-13 | Smitha | small_cnn mc_dropout | MC Dropout | 11, 22, 33 | 10a76fa | config v51 | `results/small_cnn/` | ok | Passes 2 and 20 |
| 2026-11-13 | Smitha | small_cnn ensemble | Deep ensemble | 11, 22, 33 | 10a76fa | config v51 | `results/small_cnn/` | ok | Members 2 and 5 |
| 2026-11-14 | Smitha | small_cnn qute | QUTE | 11, 22, 33 | 10a76fa | config v51 | `results/small_cnn/` | ok | With weight transfer |
| 2026-11-14 | Smitha | small_cnn qute_no_transfer | QUTE (ablation) | 11, 22, 33 | 10a76fa | config v51 | `results/small_cnn/` | ok | Transfer off |
| 2026-11-14 | Smitha | small_cnn evaluation | All | 11, 22, 33 | 10a76fa | config v51 | `results/small_cnn/tables/`, `results/small_cnn/detection/` | ok | predict_conditions, summarize_calibration, run_detection_all (first run failed: scikit-learn missing from local venv; installed 1.6.1 as pinned, reran). MNIST-C mean ECE: QUTE 0.173, DE-2 0.190, MC-2 0.219, BASE 0.262. AP: DE-2 0.786, QUTE 0.719, MC-2 0.657, BASE 0.602 |
| 2026-11-14 | Anusha | cost_table | BASE, MC-2, MC-20, DE-2, DE-5, QUTE | 11 (init only) | 990b6ab | 5 | `results/tables/cost_table.csv` | ok | Command: `python -m eval.cost_table`. Apple M1, torch 2.15.0, 1 thread, batch 256, 5 warm-up batches, median of 5 interleaved runs over 10,000 random images; models built from config (cost does not depend on weights). CPU ms / 1,000 images: BASE 119.2, MC-2 267.0, DE-2 238.5, QUTE 134.7 (1.13x BASE), MC-20 2670, DE-5 596.3 |
| 2026-11-14 | Anusha | small_cnn cost_table | BASE, MC-2, DE-2, QUTE | 11 (init only) | 990b6ab | config v51 | `results/small_cnn/tables/cost_table.csv` | ok | Command: `python -m eval.cost_table --config configs/small_cnn.yaml --variants base mc_dropout_passes2 deep_ensemble_members2 qute --out results/small_cnn/tables/cost_table.csv`. Exploratory, same machine and settings as cost_table, run straight after it. Params: BASE 1,066, DE-2 2,132, QUTE 1,316. CPU ms / 1,000 images: BASE 26.5, MC-2 60.1, DE-2 53.1, QUTE 30.6 (1.16x BASE) |
| 2026-11-14 | Smitha | significance | QUTE vs DE-2, no transfer, BASE, MC-2 | 11, 22, 33 | (this commit) | v5 and config v51 | `results/tables/significance.csv`, `results/small_cnn/tables/significance.csv` | ok | eval/significance.py: pairs = (seed, corruption), 45 per comparison, clean excluded; mean of QUTE minus other, 95% percentile bootstrap CI, 10,000 resamples of pairs, bootstrap seed 0; AP and ECE, plus 11-point AUPRC as secondary. Commands: `python -m eval.significance`; `python -m eval.significance --results results/small_cnn`. Also committed `results/detection/` (v5 per-stream detection CSVs), which had not been committed before |

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
| 2026-10-19 | Channel widths [8, 16, 32, 32] | Paper says "4-layer CNN" but gives no widths. Resulting BASE clean accuracy is ~97% vs the paper's ~91%, so our base model is stronger than theirs | Yes, as a deviation |
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
| 2026-11-11 | Pre-registered extensions, defined before computing. (1) Detection delay (Abirami): per model, τ = 1st percentile of clean-only window mean confidence (1% clean false alarms); delay = samples from corruption onset to the end of the first window with mean confidence < τ, else "not detected"; plus detection rate on fully-corrupted windows; median by variant and by severity group. (2) Cost (Anusha): deployed params, FP32 size, forward passes, CPU ms per 1,000 images (batch 256, warm-up, median of 5 runs, same machine and threads). (3) Significance (Smitha): differences paired by (seed, corruption), mean and 95% bootstrap CI (10,000 resamples) for AUPRC and ECE: QUTE vs DE-2, ablation, BASE, MC-2. Severity groups: mild = translate, shear, shot_noise, spatter; severe = impulse_noise, fog, brightness, canny_edges, stripe; moderate = the other 6 | Turns the reproduction into a deployment-oriented analysis; fixing the definitions before seeing results avoids tuning to the outcome | Yes, as extensions |
| 2026-11-13 | EXPLORATORY extension (4), not part of the 2026-11-11 pre-registration: decided after seeing v5 results. Base-model strength (Smitha): does the ranking change when the base CNN is as weak as the paper's (~91% clean)? Width chosen by `experiments/width_sweep.py` before any uncertainty metric is computed: BASE, seed 11, v5 with only `model.channels` changed; candidates [8,16,32,32], [6,12,24,24], [4,8,16,16], [3,6,12,12], [2,4,8,8]; pick final-epoch validation accuracy closest to 0.91 (tie: smaller model). Then retrain every method on seeds 11/22/33 under `configs/small_cnn.yaml` (config_version 51, identical to v5 except channels and parameters) in a separate worktree, and run the unchanged evaluation code. Prediction, written before running: QUTE's AUPRC lead over DE-2 grows, and absolute AUPRC falls toward the paper's 0.63 | Tests our explanation for the gap to the paper (base CNN ~97% vs ~91%) instead of asserting it | Yes, labelled exploratory |
| 2026-11-14 | Outcome of exploratory extension (4): prediction NOT supported. Absolute AUPRC did fall toward the paper's (small-CNN QUTE 11-point 0.626 vs paper 0.63), but QUTE's lead over DE-2 did not grow: it reversed further (AP 0.719 vs 0.786, was 0.846 vs 0.854). QUTE keeps the lowest MNIST-C ECE (0.173 vs DE-2 0.190) but loses NLL and Brier narrowly. Base clean accuracy varied by seed (0.92 / 0.90 / 0.80), so gaps await the paired significance test | Recording the result against the pre-stated prediction, as for the pre-registered extensions | Yes, labelled exploratory |
| 2026-11-14 | Outcome of pre-registered extension (3). v5: QUTE's ECE is significantly lower than DE-2 (-0.039 [-0.072, -0.010]), BASE and MC-2; no AP difference is significant, including QUTE vs BASE (+0.029 [-0.025, +0.086]); transfer is not significant on either metric. Small CNN: QUTE vs DE-2 ECE no longer significant (-0.017 [-0.051, +0.019]); DE-2 ahead on detection (11-point -0.072 [-0.139, -0.006], AP borderline); QUTE beats BASE on both. Pairs share seeds, so resampling pairs is the pre-registered choice; per-seed means are in the CSVs (seed 22 often reverses the sign) | Reporting what came out, against the Nov 11 definition | Yes |
