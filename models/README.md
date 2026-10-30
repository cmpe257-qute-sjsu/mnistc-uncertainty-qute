# Shared model interface

Owner: Smitha. Anusha's baselines and the QUTE model both build on this, so
changes need a version bump and a note in the team chat.

## Building a model

```python
from models import CNNConfig, SharedCNN, build_model

base   = build_model(seed=11)                                  # BASE
drop   = SharedCNN(CNNConfig(dropout_rate=0.1))                # MC Dropout
member = build_model(seed=22)                                  # one ensemble member
```

`build_model(seed=...)` seeds initialization. Deep-ensemble members each need a
**different** seed — that is the only thing making them an ensemble. Never train
one model and copy it into several members.

For MC Dropout, pass `dropout_rate=0.1`. The dropout layers land after conv 1
and conv 2, the same places QUTE attaches its early exits. The paper does this
deliberately so the two methods are comparable, which is why there is no
separate dropout model to write.

## Getting predictions

Wrap every evaluation dataset in `IndexedDataset` so predictions carry their
sample IDs. Use `shuffle=False` on evaluation loaders.

```python
from torch.utils.data import DataLoader
from models import IndexedDataset, predict_proba

loader = DataLoader(IndexedDataset(test_set), batch_size=256, shuffle=False)
probs, sample_ids = predict_proba(model, loader)     # [N,10], [N]
```

| Method | Call |
|---|---|
| BASE | `predict_proba(model, loader)` |
| MC Dropout | `predict_proba_mc(model, loader, passes=2)` → mean, ids, members |
| Deep ensemble | `predict_proba` per member → `stack_members(...)` → `ensemble_proba(...)` |
| QUTE | same as deep ensemble, but members come from the two output blocks |

```python
from models import stack_members, ensemble_proba

per_member = [predict_proba(m, loader)[0] for m in members]   # K x [N,10]
stacked    = stack_members(per_member)                        # [N,K,10]
mean_probs = ensemble_proba(stacked)                          # [N,10]
```

`ensemble_proba` averages over axis 1, the member axis. This is a named
function rather than a bare `.mean()` because averaging the wrong axis still
produces something shaped like a probability vector, so the bug is silent. It
rejects a 2-D input for the same reason.

## Saving predictions

```python
from models import save_predictions

save_predictions(
    "results/preds/base_seed11_clean.npz",
    probs, sample_ids,
    method="base", seed=11, condition="clean",
    member_probs=stacked,          # optional
)
```

Filename convention: `{method}_seed{seed}_{condition}.npz`, where `condition`
is `clean` or a corruption name such as `fog`.

Each file stores `probs`, `sample_ids`, optional `member_probs`, and metadata
carrying method, seed, condition, schema and model versions, and the git commit.
The commit is recorded automatically — it is how we trace a number in the report
back to the code that produced it.

## Comparing two methods

```python
from models import align_to

aligned = align_to(other_probs, other_ids, reference_ids)
```

Never join two prediction files by position. `align_to` reorders by sample ID
and raises if anything is missing.

## Current model

| | |
|---|---|
| Layers | 4 conv (8, 16, 32, 32), max-pool after conv 1 and 2, global average pool, linear |
| Parameters | 15,466 |
| Approx FP32 size | 60.4 KiB |
| Taps | `layer1` [N,8,14,14], `layer2` [N,16,7,7] |

`model.tap_shapes()` returns the tap shapes, which QUTE needs to size its early
exits. `model.describe()` returns parameter count and config for the resource
table in Step 15.

## Open item: clean accuracy

The paper's MNIST model reports about **91%** clean accuracy, well below the 99%
a normal MNIST CNN reaches. It does not give exact channel widths, so the widths
above are our choice and are recorded as a deviation.

Once BASE trains, compare:

- **~98–99%** — our model is larger than the paper's. The overconfidence effect
  the study depends on will be weaker, so corruption detection may look
  different from the paper's numbers. Report the architecture difference. If we
  want to narrow the gap, reduce `channels` in `configs/base.yaml` rather than
  changing anything else.
- **~91%** — close to the reference setup, nothing to do.

Either outcome is fine for the project as long as we say which one we got.
Decide this with the whole team before Gate B, because changing widths after
evaluation locks means retraining everything.
