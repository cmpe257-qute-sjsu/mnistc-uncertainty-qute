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

| Person | Date | torch | torchvision | Split indices | Batch labels | Batch images | Match? |
|---|---|---|---|---|---|---|---|
| Abirami | | | | | | | |
| Anusha | | | | | | | |
| Smitha | | | | | | | |

## Protocol changes

Any edit to `configs/base.yaml`. Every row needs a reason and a note on whether
earlier runs are still comparable.

| Date | Config ver | What changed | Why | Reruns needed |
|---|---|---|---|---|
| | 1 | Initial protocol | Frozen at Step 2 | — |

## Decisions we made that the paper does not specify

| Date | Decision | Reason | Goes in the report? |
|---|---|---|---|
| | 50,000 / 10,000 validation split, seed 0 | Needed for checkpoint selection | Yes |
| | Seeds 11, 22, 33 | Paper reports 3 runs but not which seeds | Yes |
| | 15 equal-mass ECE bins | Confidences cluster near 1 | Yes, as a deviation |
