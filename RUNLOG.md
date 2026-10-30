# Run log

One row per training or evaluation run. Fill this in as you go, not at the end —
reconstructing it later is how numbers in the report stop matching the files.

Status values: `ok`, `failed`, `superseded`.

| Date | Who | Run ID | Method | Seed | Commit | Config ver | Output path | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

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

## Decisions we made that the paper does not specify

| Date | Decision | Reason | Goes in the report? |
|---|---|---|---|
|2026-10-17 | 50,000 / 10,000 validation split, seed 0 | Needed for checkpoint selection | Yes |
|2026-10-17 | Seeds 11, 22, 33 | Paper reports 3 runs but not which seeds | Yes |
|2026-10-17 | 15 equal-mass ECE bins | Confidences cluster near 1 | Yes, as a deviation |
| 2026-10-19 | Channel widths [8, 16, 32, 32] | Paper says "4-layer CNN" but gives no widths | Yes, as a deviation |
| 2026-10-22 | Minibatch order is fixed by split seed 0 for every model seed and ensemble member. Ensemble diversity is initialization only | A shared global RNG would also change data order when the model seed changes, which confounds the QUTE comparison | Yes |
| 2026-10-22 | Deep-ensemble member k of run R is initialized with R * 100 + k (run 11 uses 1101–1105) | The protocol names 3 run seeds and 5 members, and does not list 5 initialization seeds | Yes |
