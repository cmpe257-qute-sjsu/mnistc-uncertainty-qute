# QUTE Uncertainty Estimation on MNIST / MNIST-C

CMPE 257 group project. We evaluate a QUTE-style single-pass ensemble for
uncertainty estimation against softmax confidence, MC Dropout and a small deep
ensemble, measuring calibration and accuracy-drop detection under corruption.

Paper: Ghanathe & Wilton, *QUTE: Quantifying Uncertainty in TinyML models with
Early-exit-assisted ensembles for model-monitoring*, ICML 2025, PMLR 267,
pp. 19286–19306. <https://proceedings.mlr.press/v267/ghanathe25a.html>

## Team

| Person | Owns |
|---|---|
| Abirami Adaikappan | Data, preprocessing, splits, ID+CID streams, detection pipeline |
| Anusha Tulasi | BASE / MC Dropout / deep ensemble, metrics, figures |
| Smitha Soman | Shared CNN, QUTE branches and output blocks, weight transfer, ablation |

## Setup

```bash
git clone https://github.com/cmpe257-qute-sjsu/qute-uncertainty-mnistc.git
cd qute-uncertainty-mnistc

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS:    source .venv/bin/activate

pip install -r requirements.txt
pip freeze > requirements.lock.txt
```

`requirements.lock.txt` is the environment of record. Commit it. If anyone's
results stop matching, diff that file first.

## Verify your environment

```bash
python tests/smoke_test.py
```

Paste the FINGERPRINT block into the team chat. All three of us must get
identical values before anyone starts training. If they differ, compare
torchvision versions — that is the usual cause.

## Layout

| Folder | Contents | Owner |
|---|---|---|
| `configs/` | Frozen protocol settings. `base.yaml` is authoritative. | All |
| `data/` | Loading, preprocessing, split indices, stream manifests. No image files. | Abirami |
| `models/` | Shared CNN, QUTE early exits and output blocks. | Smitha |
| `baselines/` | BASE, MC Dropout, deep ensemble. | Anusha |
| `eval/` | ECE, Brier, NLL, AUPRC, sliding-window detection. | Anusha / Abirami |
| `tests/` | Smoke test and implementation checks. | All |
| `results/` | Generated metrics and figures. | All |
| `notebooks/` | Exploration only. Nothing final depends on these. | All |

## Rules we agreed on

- `configs/base.yaml` is the only place protocol settings live. Do not override
  them inside a training script. To change one, edit the file, bump
  `config_version`, and tell the team.
- Datasets and checkpoints stay out of git. MNIST-C is a few hundred MB.
- Interface owners: Abirami owns data identity, Smitha owns the CNN definition,
  Anusha owns the metrics. Changing any of these needs a version note and a review.
- Work on a branch and open a pull request. Do not commit to `main` directly.
- Record commit hash, config version and seed for every run in `RUNLOG.md`.

## Known deviations from the paper

Track these here as they come up. Every one goes in the final report.

| Setting | Paper | Ours | Why |
|---|---|---|---|
| ECE binning | Equal-width bins | 15 equal-mass bins | Confidences cluster near 1, so equal-width bins are badly imbalanced. The paper itself criticizes ECE binning in Appendix C. |

## Things that will surprise you

- The paper's MNIST model reports about **91% clean accuracy**, not 99%. It is
  deliberately tiny. If ours hits 98–99%, our CNN is larger than theirs and the
  overconfidence effect the study depends on will be weaker.
- MNIST-C ships an **identity** set that is uncorrupted data. It is not one of
  the 15 corruptions — keep it out of corruption averages.
- QUTE trains **slower** than the baselines because of the per-batch weight copy.
- At inference QUTE drops the early exits **and** the network's original output
  block. Forgetting the second one is the easiest silent mistake here.
