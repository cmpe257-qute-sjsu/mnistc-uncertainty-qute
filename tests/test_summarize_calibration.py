"""
Checks for eval/summarize_calibration.py.

    python tests/test_summarize_calibration.py

Writes a handful of tiny fake prediction files, so it needs no MNIST, no
checkpoints and no real predictions.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.base import load_protocol
from eval.summarize_calibration import (
    prediction_files,
    score_files,
    summarize,
    variant_name,
    write_csv,
)

PASS, FAIL = [], []
PROTOCOL = load_protocol()


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {e}")


def fake_file(directory, name, method, seed, condition, correct_frac=1.0):
    """A 10-example prediction file: confident, correct on the first correct_frac of rows."""
    n = 10
    labels = np.arange(n, dtype=np.int64) % 10
    preds = labels.copy()
    wrong = n - int(round(correct_frac * n))
    if wrong:
        preds[:wrong] = (labels[:wrong] + 1) % 10
    probs = np.full((n, 10), 0.01, dtype=np.float32)
    probs[np.arange(n), preds] = 0.91
    metadata = {"method": method, "seed": seed, "condition": condition,
                "config_version": 5, "git_commit": "abc1234"}
    np.savez(Path(directory) / name, probs=probs, sample_ids=np.arange(n, dtype=np.int64),
             labels=labels, confidence=probs.max(axis=1), metadata=np.array(json.dumps(metadata)))


def populate(directory):
    for seed in (11, 22):
        fake_file(directory, f"qute_seed{seed}_clean.npz", "qute", seed, "clean")
        fake_file(directory, f"qute_seed{seed}_fog.npz", "qute", seed, "fog", correct_frac=0.5)
        fake_file(directory, f"mc_dropout_seed{seed}_fog_passes2.npz", "mc_dropout", seed, "fog", 0.7)


print("\nNAMES AND FILES")
print("---------------")

def t_variant_names():
    assert variant_name(Path("qute_seed11_fog.npz"), 11, "fog") == "qute"
    assert variant_name(Path("qute_no_transfer_seed11_fog.npz"), 11, "fog") == "qute_no_transfer"
    assert variant_name(Path("mc_dropout_seed11_fog_passes20.npz"), 11, "fog") == "mc_dropout_passes20"
    assert variant_name(Path("deep_ensemble_seed33_shear_members5.npz"), 33, "shear") == "deep_ensemble_members5"
check("variant names keep the headline/reference suffix", t_variant_names)

def t_tmp_skipped():
    with tempfile.TemporaryDirectory() as d:
        populate(d)
        fake_file(d, "qute_seed11_zigzag.tmp.npz", "qute", 11, "zigzag")
        names = [p.name for p in prediction_files(Path(d))]
    assert "qute_seed11_zigzag.tmp.npz" not in names, names
    assert len(names) == 6, names
check("leftover .tmp.npz files are skipped", t_tmp_skipped)


print("\nSCORING")
print("-------")

def t_rows_carry_traceability():
    with tempfile.TemporaryDirectory() as d:
        populate(d)
        rows = score_files(Path(d), PROTOCOL)
    assert len(rows) == 6
    assert {r["variant"] for r in rows} == {"qute", "mc_dropout_passes2"}
    assert all(r["git_commit"] == "abc1234" and r["config_version"] == 5 for r in rows)
    fog = [r for r in rows if r["variant"] == "qute" and r["condition"] == "fog"]
    assert all(abs(r["accuracy"] - 0.5) < 1e-9 for r in fog), fog
check("rows have variant, accuracy, commit and config version", t_rows_carry_traceability)

def t_variant_filter():
    with tempfile.TemporaryDirectory() as d:
        populate(d)
        rows = score_files(Path(d), PROTOCOL, ["qute"])
    assert {r["variant"] for r in rows} == {"qute"} and len(rows) == 4
check("--variants keeps only the requested variants", t_variant_filter)

def t_variant_filter_matches_nothing():
    with tempfile.TemporaryDirectory() as d:
        populate(d)
        try:
            score_files(Path(d), PROTOCOL, ["not_a_variant"])
        except SystemExit as exc:
            assert "matched no files" in str(exc) and "qute" in str(exc), exc
            return
    raise AssertionError("an unmatched --variants was accepted")
check("an unmatched --variants stops with a clear message", t_variant_filter_matches_nothing)

def t_empty_csv_refused():
    with tempfile.TemporaryDirectory() as d:
        try:
            write_csv(Path(d) / "x.csv", [])
        except SystemExit as exc:
            assert "nothing to write" in str(exc), exc
            return
    raise AssertionError("an empty table was written")
check("writing an empty table stops with a clear message", t_empty_csv_refused)


print("\nSUMMARY")
print("-------")

def t_summary_mean_and_mnistc():
    with tempfile.TemporaryDirectory() as d:
        populate(d)
        rows = score_files(Path(d), PROTOCOL)
    summary = summarize(rows, expected_corruptions=1)
    lookup = {(r["variant"], r["condition"]): r for r in summary}
    fog = lookup[("qute", "fog")]
    assert fog["n_seeds"] == 2 and abs(fog["accuracy_mean"] - 0.5) < 1e-9 and fog["accuracy_std"] == 0.0
    mnistc = lookup[("qute", "mnistc_mean")]
    assert abs(mnistc["accuracy_mean"] - 0.5) < 1e-9, mnistc
    assert ("qute", "clean") in lookup and lookup[("qute", "clean")]["accuracy_mean"] == 1.0
check("seed means and the MNIST-C mean are computed per variant", t_summary_mean_and_mnistc)

def t_incomplete_seed_skipped_from_mnistc():
    with tempfile.TemporaryDirectory() as d:
        populate(d)
        rows = score_files(Path(d), PROTOCOL)
    summary = summarize(rows, expected_corruptions=15)
    assert not [r for r in summary if r["condition"] == "mnistc_mean"], "partial seeds were averaged"
check("a seed missing corruptions is left out of the MNIST-C mean", t_incomplete_seed_skipped_from_mnistc)


print("\nRESULT")
print("------")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name, err in FAIL:
        print(f"  {name}: {err}")
    sys.exit(1)
print("\nCalibration summary checks pass.")
