# Baselines

Owner: Anusha. BASE, MC Dropout, and the deep ensemble. All three train the
shared CNN from `models/` and read every protocol setting from
`configs/base.yaml`. Do not hardcode epochs, batch size, learning rate, seeds,
dropout, or ensemble size in a call that is meant to be a reported run.

QUTE is not trained here.

## What each method does

| Method | Training | Prediction |
|---|---|---|
| BASE | One CNN per seed, dropout off | `predict_proba`. Confidence is the maximum softmax probability. |
| MC Dropout | One CNN per seed, dropout 0.1 after conv 1 and conv 2 | `predict_proba_mc` for 20 passes. The 2-pass file is the first 2 of those passes, averaged with `ensemble_proba`. |
| Deep ensemble | Five separate CNNs per run seed, dropout off | `stack_members`, then `ensemble_proba`. The 2-member file is the first 2 of those 5. |

Run seeds are `11`, `22`, and `33`. Training is 20 epochs, batch size 256,
Adam at learning rate `0.001`, decay `0.99` per epoch. The checkpoint kept is
the epoch with the lowest validation NLL. Validation NLL is computed with
dropout off. MC Dropout turns dropout back on only while predicting.

There is no separate dropout model. MC Dropout is
`SharedCNN(CNNConfig(dropout_rate=0.1))`.

## Data

Reported runs call Abirami's factory. Nothing is built at import time:

```python
train_loader, validation_loader, test_loader = data.get_loaders(protocol)
```

`get_loaders` reads `training.batch_size` and `training.num_workers` from the
protocol. A shuffled loader must use
`generator=torch.Generator().manual_seed(split_seed)`. After model init,
`fit_cnn` calls `pin_training_randomness` so train-time augmentation also
follows `split_seed`. If `num_workers > 0`, pass
`worker_init_fn=baselines.base.worker_init_fn(protocol)` to each DataLoader.
Ensemble diversity is initialization only.

`python -m baselines.base` stops if `data.get_loaders` is missing. This folder
does not build the 50,000 / 10,000 split.

The test loader must yield `(image, label, sample_id)` with `shuffle=False`.
For a subset, pass the original indices:

```python
IndexedDataset(Subset(dataset, val_idx), ids=val_idx)
```

## Commands

From the repo root, with the virtual environment active:

```bash
python -m baselines.base
python -m baselines.mc_dropout
python -m baselines.ensemble
```

Add `--device cpu` or `--overwrite` when needed. Without `--overwrite`, an
existing checkpoint or prediction file is left in place.

Synthetic checks, with no MNIST download:

```bash
python tests/test_base.py
python tests/test_mc_dropout.py
python tests/test_ensemble.py
```

## Files

Checkpoints go in `checkpoints/` and stay out of git.

| Method | Checkpoint | Predictions |
|---|---|---|
| BASE | `checkpoints/base_config3_seed11.pt` | `results/preds/base_seed11_clean.npz` |
| MC Dropout | `checkpoints/mc_dropout_config3_seed11.pt` | `results/preds/mc_dropout_seed11_clean_passes2.npz` and `..._passes20.npz` |
| Deep ensemble | `checkpoints/deep_ensemble_config3_run11_member1_seed1101.pt` | `results/preds/deep_ensemble_seed11_clean_members2.npz` and `..._members5.npz` |

Seeds `22` and `33` follow the same names. Prediction files store probabilities
and sample IDs before any aggregate metric. Every file records
`confidence: max_softmax_probability` from `eval/scores.py`. The 2-pass MC file
is the first 2 of the 20, flagged in the config as
`headline_is_prefix_of_reference`.

## Ensemble member seeds

`configs/base.yaml` names the three run seeds and asks for five members. It
does not list five initialization seeds. Member `k` of run `R` uses
`R * 100 + k`, counting from 1.

| Run | Members |
|---|---|
| 11 | 1101, 1102, 1103, 1104, 1105 |
| 22 | 2201, 2202, 2203, 2204, 2205 |
| 33 | 3301, 3302, 3303, 3304, 3305 |

The headline pair is 1101 and 1102 for run 11, and the same positions for the
other runs. Both this scheme and the fixed minibatch order are recorded in
`RUNLOG.md`. Each run is 5 trainings, so the full ensemble is 15 trainings.

## Not in this folder yet

Accuracy, NLL, Brier score, 15-bin equal-mass ECE, AUPRC, and the resource
table. Those wait until the prediction files exist.
