"""Checks for eval/significance.py on small synthetic CSVs (no predictions needed).

    python tests/test_significance.py
"""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.significance import COMPARISONS, QUTE, SEEDS, analyse, bootstrap_ci, main  # noqa: E402

PASS, FAIL = [], []
CORRUPTIONS = ["fog", "zigzag", "stripe", "shear", "rotate"]
VARIANTS = [QUTE] + [other for other, _ in COMPARISONS]


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:  # noqa: BLE001
        FAIL.append(name)
        print(f"  FAIL  {name}: {e}")


def write_fixture(folder: Path, ece_shift=None, ap_shift=None, undefined=None):
    """Other variant = QUTE + shift (per variant), with a little noise."""
    ece_shift = ece_shift or {}
    ap_shift = ap_shift or {}
    rng = np.random.default_rng(1)
    (folder / "tables").mkdir(parents=True)
    (folder / "detection").mkdir(parents=True)
    base_ece = {(s, c): 0.2 + 0.05 * rng.random() for s in SEEDS for c in CORRUPTIONS}
    base_ap = {(s, c): 0.7 + 0.05 * rng.random() for s in SEEDS for c in CORRUPTIONS}
    with (folder / "tables" / "calibration_per_file.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed", "condition", "accuracy", "nll", "brier", "ece"])
        for v in VARIANTS:
            for s in SEEDS:
                w.writerow([v, s, "clean", 0.9, 0.3, 0.1, 0.99])  # must be ignored
                for c in CORRUPTIONS:
                    qute_value = base_ece[(s, c)]
                    value = qute_value if v == QUTE else qute_value + ece_shift.get(v, 0.0) + 1e-4 * rng.standard_normal()
                    w.writerow([v, s, c, 0.6, 2.0, 0.6, value])
    with (folder / "detection" / "detection_per_stream.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_id", "variant", "seed", "corruption", "average_precision", "sweep_auprc"])
        for v in VARIANTS:
            for s in SEEDS:
                for c in CORRUPTIONS:
                    qute_value = base_ap[(s, c)]
                    value = qute_value if v == QUTE else qute_value + ap_shift.get(v, 0.0) + 1e-4 * rng.standard_normal()
                    if undefined and (v, s, c) == undefined:
                        value = ""
                    w.writerow([f"{v}_seed{s}", v, s, c, value, value])


def rows_by(rows, comparison_other, metric_prefix):
    return next(r for r in rows if r["other_variant"] == comparison_other and r["metric"].startswith(metric_prefix))


def t_clear_effect():
    with tempfile.TemporaryDirectory() as d:
        write_fixture(Path(d), ece_shift={"base": 0.05}, ap_shift={"base": -0.1})
        rows = analyse(Path(d), n_resamples=2000)
        ece = rows_by(rows, "base", "ECE")
        assert ece["n_pairs"] == len(SEEDS) * len(CORRUPTIONS), ece["n_pairs"]
        assert abs(ece["mean_diff"] + 0.05) < 1e-3, ece["mean_diff"]  # QUTE lower ECE
        assert ece["excludes_zero"] and ece["favours"] == QUTE, ece
        ap = rows_by(rows, "base", "AUPRC (AP)")
        assert ap["mean_diff"] > 0.09 and ap["excludes_zero"] and ap["favours"] == QUTE, ap


check("a clear gap gives a CI that excludes zero and favours QUTE", t_clear_effect)


def t_no_effect():
    with tempfile.TemporaryDirectory() as d:
        write_fixture(Path(d))
        rows = analyse(Path(d), n_resamples=2000)
        ece = rows_by(rows, "deep_ensemble_members2", "ECE")
        assert ece["ci_low"] <= 0 <= ece["ci_high"], (ece["ci_low"], ece["ci_high"])
        assert not ece["excludes_zero"]


check("no gap gives a CI that covers zero", t_no_effect)


def t_clean_ignored():
    with tempfile.TemporaryDirectory() as d:
        write_fixture(Path(d), ece_shift={"base": 0.05})
        rows = analyse(Path(d), n_resamples=500)
        assert rows_by(rows, "base", "ECE")["n_pairs"] == len(SEEDS) * len(CORRUPTIONS)


check("clean rows are excluded from the pairs", t_clean_ignored)


def t_undefined_dropped():
    with tempfile.TemporaryDirectory() as d:
        write_fixture(Path(d), undefined=("base", 22, "fog"))
        rows = analyse(Path(d), n_resamples=500)
        ap = rows_by(rows, "base", "AUPRC (AP)")
        assert ap["n_pairs"] == len(SEEDS) * len(CORRUPTIONS) - 1 and ap["n_undefined_dropped"] == 1


check("an undefined AP drops only that pair, and is counted", t_undefined_dropped)


def t_reproducible():
    diffs = np.random.default_rng(3).normal(0.01, 0.02, 45)
    assert bootstrap_ci(diffs, 1000) == bootstrap_ci(diffs, 1000)


check("the bootstrap is reproducible (fixed seed)", t_reproducible)


def t_cli():
    with tempfile.TemporaryDirectory() as d:
        write_fixture(Path(d))
        out = Path(d) / "sig.csv"
        main(["--results", d, "--out", str(out), "--resamples", "200"])
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3 * len(COMPARISONS), len(rows)


check("command line writes one row per comparison and metric", t_cli)


print("\nRESULT\n------")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    sys.exit(1)
print("\nSignificance checks pass.")
