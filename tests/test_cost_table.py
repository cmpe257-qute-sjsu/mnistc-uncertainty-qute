"""
Checks for eval/cost_table.py.

    python tests/test_cost_table.py

Builds models from the configs only. Does not need MNIST, MNIST-C or checkpoints.
"""

import csv
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.base import load_protocol
from eval.cost_table import VARIANTS, build_rows, build_variant, main, time_variants

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {e}")


PROTOCOL = load_protocol()
CNN_PARAMS = int(PROTOCOL["model"]["parameters"])
BUILT = {name: build_variant(PROTOCOL, name) for name in VARIANTS}
X = torch.randn(7, 1, 28, 28)


print("\nSIZE AND PASSES")
print("---------------")

def t_base():
    assert BUILT["base"]["params"] == CNN_PARAMS and BUILT["base"]["passes"] == 1
check("BASE: one CNN, one pass", t_base)

def t_mc():
    for k in (2, 20):
        entry = BUILT[f"mc_dropout_passes{k}"]
        assert entry["params"] == CNN_PARAMS, entry["params"]
        assert entry["passes"] == k
check("MC Dropout: same parameters as BASE, K passes", t_mc)

def t_ensemble():
    for k in (2, 5):
        entry = BUILT[f"deep_ensemble_members{k}"]
        assert entry["params"] == k * CNN_PARAMS, entry["params"]
        assert entry["passes"] == k
check("Deep ensemble: K x parameters, K passes", t_ensemble)

def t_qute():
    from qute.train import build_qute_from_protocol
    model = build_qute_from_protocol(PROTOCOL, 11)
    entry = BUILT["qute"]
    assert entry["params"] == model.deployed_parameters()
    assert entry["params"] < model.training_parameters(), "early exits must not count"
    assert entry["passes"] == 1
check("QUTE: deployed parameters (no early exits), one pass", t_qute)

def t_small():
    small = load_protocol(ROOT / "configs" / "small_cnn.yaml")
    entry = build_variant(small, "base")
    assert entry["params"] == 1066, entry["params"]
check("small_cnn.yaml BASE has 1,066 parameters", t_small)


print("\nPREDICTIONS")
print("-----------")

def t_probs():
    with torch.inference_mode():
        for name, entry in BUILT.items():
            probs = entry["predict"](X)
            assert probs.shape == (7, 10), (name, probs.shape)
            assert torch.allclose(probs.sum(dim=1), torch.ones(7), atol=1e-5), name
check("every variant returns probabilities [N, 10] that sum to 1", t_probs)

def t_mc_stochastic():
    with torch.inference_mode():
        a = BUILT["mc_dropout_passes2"]["predict"](X)
        b = BUILT["mc_dropout_passes2"]["predict"](X)
    assert not torch.equal(a, b), "MC Dropout passes should be stochastic"
check("MC Dropout keeps dropout active at inference", t_mc_stochastic)

def t_unknown():
    try:
        build_variant(PROTOCOL, "qute_no_transfer")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
check("unknown variant is rejected", t_unknown)


print("\nTIMING AND OUTPUT")
print("-----------------")

def t_timing():
    subset = {k: BUILT[k] for k in ("base", "deep_ensemble_members2")}
    timings = time_variants(subset, n_images=64, batch_size=32, warmup_batches=1, repeats=3)
    assert all(len(v) == 3 and min(v) > 0 for v in timings.values()), timings
    rows = build_rows(PROTOCOL, list(subset), timings, subset, {"threads": 1})
    assert rows[0]["time_vs_base"] == 1.0
    assert rows[1]["fp32_kib"] == round(2 * CNN_PARAMS * 4 / 1024, 1)
check("timing gives one positive value per repeat; BASE ratio is 1", t_timing)

def t_cli():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "cost.csv"
        main(["--variants", "base", "qute", "--n-images", "64", "--repeats", "2",
              "--warmup-batches", "1", "--out", str(out)])
        rows = list(csv.DictReader(out.open()))
    assert [r["variant"] for r in rows] == ["base", "qute"]
    assert rows[0]["batch_size"] == "256" and rows[0]["threads"] == "1"
check("command line writes one CSV row per variant", t_cli)


print("\nRESULT")
print("------")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    sys.exit(1)
print("\nCost table checks pass.")
