"""
Step 7 acceptance checks for the shared CNN and prediction interface.

    python tests/test_shared_cnn.py

Checks the "ready when" conditions from the experiment guide:
  - probabilities are finite, nonnegative and sum to one
  - output order matches sample IDs
  - averaging happens across members, not classes

Uses synthetic data, so it runs in seconds and needs no MNIST download.
"""

import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.shared_cnn import CNNConfig, SharedCNN, build_model, enable_mc_dropout
from models.interface import (
    IndexedDataset, predict_proba, predict_proba_mc, ensemble_proba,
    stack_members, save_predictions, load_predictions, align_to,
)

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {e}")


def fake_loader(n=64, batch=16, shuffle=False):
    g = torch.Generator().manual_seed(3)
    x = torch.rand(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    ds = IndexedDataset(TensorDataset(x, y), ids=list(range(1000, 1000 + n)))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


print("\nMODEL")
print("-----")

def t_builds():
    m = SharedCNN()
    out = m(torch.zeros(4, 1, 28, 28))
    assert out.shape == (4, 10), out.shape
check("builds and produces [N,10] logits", t_builds)

def t_taps():
    m = SharedCNN()
    _, taps = m(torch.zeros(2, 1, 28, 28), return_taps=True)
    assert set(taps) == {"layer1", "layer2"}, set(taps)
    assert taps["layer1"].shape == (2, 8, 14, 14), taps["layer1"].shape
    assert taps["layer2"].shape == (2, 16, 7, 7), taps["layer2"].shape
check("taps exposed after conv layers 1 and 2", t_taps)

def t_seed():
    a = build_model(seed=11)
    b = build_model(seed=11)
    c = build_model(seed=22)
    pa = torch.cat([p.flatten() for p in a.parameters()])
    pb = torch.cat([p.flatten() for p in b.parameters()])
    pc = torch.cat([p.flatten() for p in c.parameters()])
    assert torch.equal(pa, pb), "same seed gave different weights"
    assert not torch.equal(pa, pc), "different seeds gave identical weights"
check("same seed reproduces, different seeds differ", t_seed)

def t_base_deterministic():
    m = build_model(seed=11)
    m.eval()
    x = torch.rand(8, 1, 28, 28)
    with torch.no_grad():
        assert torch.equal(m(x), m(x)), "BASE output changed between calls"
check("BASE repeats exactly in eval mode", t_base_deterministic)

def t_dropout_only_when_asked():
    plain = SharedCNN(CNNConfig(dropout_rate=0.0))
    drop = SharedCNN(CNNConfig(dropout_rate=0.1))
    n_plain = sum(1 for m in plain.modules() if isinstance(m, torch.nn.Dropout))
    n_drop = sum(1 for m in drop.modules() if isinstance(m, torch.nn.Dropout))
    assert n_plain == 0, f"BASE should have no dropout, found {n_plain}"
    assert n_drop == 2, f"expected dropout at both EE locations, found {n_drop}"
check("dropout appears only at the two EE locations", t_dropout_only_when_asked)

def t_mc_varies():
    m = SharedCNN(CNNConfig(dropout_rate=0.3))
    enable_mc_dropout(m)
    x = torch.rand(8, 1, 28, 28)
    with torch.no_grad():
        assert not torch.equal(m(x), m(x)), "MC passes did not vary"
check("MC Dropout passes vary", t_mc_varies)


print("\nPREDICTION INTERFACE")
print("--------------------")

def t_probs_contract():
    m = build_model(seed=11)
    probs, ids = predict_proba(m, fake_loader())
    assert probs.shape == (64, 10), probs.shape
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all()
    assert torch.allclose(probs.sum(1), torch.ones(64), atol=1e-5)
    assert ids.tolist() == list(range(1000, 1064)), "sample IDs wrong or reordered"
check("probs finite, nonnegative, sum to 1; IDs preserved", t_probs_contract)

def t_shuffle_safe():
    m = build_model(seed=11)
    ordered, ids_o = predict_proba(m, fake_loader(shuffle=False))
    shuffled, ids_s = predict_proba(m, fake_loader(shuffle=True))
    realigned = align_to(shuffled, ids_s, ids_o)
    assert torch.allclose(ordered, realigned, atol=1e-6), "align_to failed to restore order"
check("shuffled predictions realign by sample ID", t_shuffle_safe)

def t_ensemble_axis():
    members = [build_model(seed=s) for s in (11, 22, 33)]
    probs = [predict_proba(m, fake_loader())[0] for m in members]
    stacked = stack_members(probs)
    assert stacked.shape == (64, 3, 10), stacked.shape
    mean = ensemble_proba(stacked)
    assert mean.shape == (64, 10), mean.shape
    assert torch.allclose(mean, (probs[0] + probs[1] + probs[2]) / 3, atol=1e-6)
check("ensemble averages across members, output stays [N,10]", t_ensemble_axis)

def t_wrong_axis_rejected():
    try:
        ensemble_proba(torch.rand(64, 10))      # [N,C], missing member axis
    except ValueError:
        return
    raise AssertionError("ensemble_proba accepted a 2-D tensor")
check("ensemble_proba rejects a missing member axis", t_wrong_axis_rejected)

def t_bad_probs_rejected():
    from models.interface import _check_probs
    for bad, why in [
        (torch.full((4, 10), 0.5), "rows summing to 5"),
        (torch.full((4, 10), float("nan")), "NaN"),
    ]:
        try:
            _check_probs(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {why}")
check("malformed probability tensors are rejected", t_bad_probs_rejected)

def t_mc_shapes():
    m = SharedCNN(CNNConfig(dropout_rate=0.1))
    mean, ids, members = predict_proba_mc(m, fake_loader(), passes=2, seed=11)
    assert mean.shape == (64, 10), mean.shape
    assert members.shape == (64, 2, 10), members.shape
    assert ids.tolist() == list(range(1000, 1064))
    assert torch.allclose(mean, members.mean(dim=1), atol=1e-6)
check("MC Dropout returns mean [N,10] and members [N,K,10]", t_mc_shapes)

def t_roundtrip():
    m = build_model(seed=11)
    probs, ids = predict_proba(m, fake_loader())
    members = stack_members([probs, probs])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.npz"
        meta = save_predictions(path, probs, ids, "base", 11, "clean", member_probs=members)
        r_probs, r_ids, r_meta, r_members = load_predictions(path)
    assert torch.allclose(probs, r_probs, atol=1e-6)
    assert torch.equal(ids, r_ids)
    assert r_meta["method"] == "base" and r_meta["seed"] == 11
    assert r_meta["condition"] == "clean" and r_meta["n_members"] == 2
    assert r_members.shape == (64, 2, 10)
check("prediction files round-trip with metadata", t_roundtrip)


print("\nMODEL SIZE")
print("----------")
info = SharedCNN().describe()
print(f"  parameters      {info['parameters']:,}")
print(f"  approx FP32     {info['approx_fp32_kib']} KiB")
print(f"  feature dim     {info['feature_dim']}")
for k, v in info["taps"].items():
    print(f"  tap {k}       {v}")

print("\nRESULT")
print("------")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name, err in FAIL:
        print(f"  {name}: {err}")
    sys.exit(1)
print("\nStep 7 interface checks pass. Anusha can build baselines on this.")
print("Still to confirm: clean accuracy after real training (expect a gap vs")
print("the paper's ~91% if our channel widths are larger).")
