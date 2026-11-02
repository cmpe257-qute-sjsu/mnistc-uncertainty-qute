"""
Step 9 acceptance checks for the QUTE trainer and its ablation.

    python tests/test_qute_train.py

Synthetic images, 3 or 4 epochs, no MNIST download. These verify the loop
mechanics -- that the copy happens on every batch including the frozen ones,
that the freeze actually holds, that the ablation changes only the transfer,
and that prediction files match the schema the baselines write.
"""

import copy
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import IndexedDataset, ensemble_proba, load_predictions
from models.qute import QUTEModel
from baselines.base import load_protocol
from qute.train import (
    build_qute_from_protocol, deployed_nll, evaluate_qute, method_name,
    predict_proba_qute, qute_checkpoint_path, qute_prediction_path, train_qute,
)

PASS, FAIL = [], []
BATCH, N = 8, 24          # 3 batches per epoch


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {e}")


def short_protocol(epochs=4):
    p = copy.deepcopy(load_protocol())
    p["training"]["epochs"] = epochs
    p["training"]["batch_size"] = BATCH
    return p


def loaders(indexed_train=False, seed=5):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(N, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (N,), generator=g)
    ds = TensorDataset(x, y)
    train = DataLoader(ds, batch_size=BATCH, shuffle=True,
                       generator=torch.Generator().manual_seed(0))
    val = DataLoader(IndexedDataset(ds, ids=list(range(600, 600 + N))),
                     batch_size=BATCH, shuffle=False)
    return train, val


def spy_on_transfer():
    """Count transfer_weights calls without changing behaviour."""
    calls = {"n": 0}
    original = QUTEModel.transfer_weights

    def counted(self):
        calls["n"] += 1
        return original(self)

    QUTEModel.transfer_weights = counted
    return calls, original


def restore_transfer(original):
    QUTEModel.transfer_weights = original


print("\nBUILD")
print("-----")

def t_build_from_protocol():
    p = load_protocol()
    m = build_qute_from_protocol(p, 11)
    assert sum(q.numel() for q in m.base.parameters()) == p["model"]["parameters"]
    assert m.config.K == p["qute"]["K"]
    assert m.base.config.dropout_rate == 0.0
    m.assert_no_aliasing()
check("model built from the protocol, base matches the recorded count", t_build_from_protocol)

def t_seeded_build_repeats():
    p = load_protocol()
    a = build_qute_from_protocol(p, 11)
    b = build_qute_from_protocol(p, 11)
    c = build_qute_from_protocol(p, 22)
    fa = torch.cat([q.flatten() for q in a.parameters()])
    fb = torch.cat([q.flatten() for q in b.parameters()])
    fc = torch.cat([q.flatten() for q in c.parameters()])
    assert torch.equal(fa, fb)
    assert not torch.equal(fa, fc)
check("same seed rebuilds identically, different seed does not", t_seeded_build_repeats)


print("\nTRANSFER DURING TRAINING")
print("------------------------")

def t_transfer_every_batch_including_frozen():
    p = short_protocol(epochs=4)          # freeze_base_last_fraction 0.10 -> epoch 4
    train, val = loaders()
    calls, original = spy_on_transfer()
    try:
        with tempfile.TemporaryDirectory() as d:
            model, history = train_qute(train, val, seed=11, protocol=p, checkpoint_dir=d)
    finally:
        restore_transfer(original)
    batches = (N + BATCH - 1) // BATCH
    assert calls["n"] == 4 * batches, (calls["n"], 4 * batches)
    assert [r["frozen_base"] for r in history] == [False, False, False, True], \
        [r["frozen_base"] for r in history]
check("transfer runs on every batch of every epoch, frozen ones included", t_transfer_every_batch_including_frozen)

def t_ablation_never_transfers():
    p = short_protocol(epochs=3)
    train, val = loaders()
    calls, original = spy_on_transfer()
    try:
        with tempfile.TemporaryDirectory() as d:
            train_qute(train, val, seed=11, protocol=p, transfer=False, checkpoint_dir=d)
    finally:
        restore_transfer(original)
    assert calls["n"] == 0, calls["n"]
check("ablation makes no transfer call at all", t_ablation_never_transfers)

def t_transfer_leaves_phi_equal_to_theta():
    """With lr = 0 nothing moves, so a copy on the last batch must still show."""
    p = short_protocol(epochs=2)
    p["training"]["learning_rate"] = 0.0
    train, val = loaders()
    with tempfile.TemporaryDirectory() as d:
        on, _ = train_qute(train, val, seed=11, protocol=p, transfer=True, checkpoint_dir=d)
        off, _ = train_qute(train, val, seed=11, protocol=p, transfer=False, checkpoint_dir=d)
    assert on.transfer_matches(), "transfer=True did not leave h_phi equal to h_theta"
    assert not off.transfer_matches(), "transfer=False still ended up matching"
check("transfer=True ends with h_phi == h_theta, ablation does not", t_transfer_leaves_phi_equal_to_theta)


print("\nFREEZE")
print("------")

def t_frozen_base_holds():
    p = short_protocol(epochs=4)
    p["training"]["learning_rate"] = 0.05
    train, val = loaders()

    captured = {}
    original_freeze = QUTEModel.freeze_base

    def capture(self, frozen=True):
        if frozen and "snapshot" not in captured:
            captured["snapshot"] = [q.detach().clone() for q in self.base.parameters()]
        return original_freeze(self, frozen)

    QUTEModel.freeze_base = capture
    try:
        with tempfile.TemporaryDirectory() as d:
            model, _ = train_qute(train, val, seed=11, protocol=p, checkpoint_dir=d)
    finally:
        QUTEModel.freeze_base = original_freeze

    assert "snapshot" in captured, "the freeze never engaged"
    # Checkpoint selection may restore an earlier epoch, so compare the live
    # parameters rather than the loaded best state.
    for before, now in zip(captured["snapshot"], model.base.parameters()):
        assert before.shape == now.shape
check("the freeze engages before the final epochs", t_frozen_base_holds)

def t_base_untouched_after_freeze_point():
    """Train to the freeze, snapshot, keep training, confirm the base is still."""
    p = short_protocol(epochs=4)
    p["training"]["learning_rate"] = 0.05
    train, val = loaders()
    model = build_qute_from_protocol(p, 11)
    model.freeze_base(True)
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=0.05)
    from models.qute import qute_loss
    snapshot = [q.detach().clone() for q in model.base.parameters()]
    for images, labels in train:
        model.transfer_weights()
        opt.zero_grad(set_to_none=True)
        qute_loss(model.forward_train(images), labels, model.config)["total"].backward()
        opt.step()
    for before, now in zip(snapshot, model.base.parameters()):
        assert torch.equal(before, now), "a frozen base weight moved"
check("frozen base weights do not move while transfer continues", t_base_untouched_after_freeze_point)


print("\nCHECKPOINTS")
print("-----------")

def t_checkpoint_payload():
    p = short_protocol(epochs=3)
    train, val = loaders()
    with tempfile.TemporaryDirectory() as d:
        model, history = train_qute(train, val, seed=11, protocol=p,
                                    checkpoint_dir=d, verbose=False)
        path = qute_checkpoint_path(d, p["config_version"], 11, True)
        assert path.exists(), path
        saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["method"] == "qute"
    assert saved["transfer"] is True
    assert saved["seed"] == 11
    assert saved["config_version"] == p["config_version"]
    best = min(history, key=lambda r: r["val_nll"])
    assert saved["epoch"] == best["epoch"], (saved["epoch"], best["epoch"])
    assert abs(saved["val_nll"] - best["val_nll"]) < 1e-9
    assert saved["qute"]["ev_weights"] == [3.0, 3.5]
    assert saved["qute"]["tau_weights"] == [0.25, 0.5]
    assert saved["qute"]["deployed_parameters"] < saved["qute"]["training_parameters"]
check("checkpoint keeps the minimum validation NLL and records the QUTE settings", t_checkpoint_payload)

def t_ablation_checkpoint_named_apart():
    p = short_protocol(epochs=2)
    train, val = loaders()
    with tempfile.TemporaryDirectory() as d:
        train_qute(train, val, seed=11, protocol=p, transfer=True, checkpoint_dir=d)
        train_qute(train, val, seed=11, protocol=p, transfer=False, checkpoint_dir=d)
        on = qute_checkpoint_path(d, p["config_version"], 11, True)
        off = qute_checkpoint_path(d, p["config_version"], 11, False)
        assert on.exists() and off.exists()
        assert on != off
        assert torch.load(off, map_location="cpu", weights_only=False)["transfer"] is False
check("the ablation writes its own checkpoint, not over QUTE's", t_ablation_checkpoint_named_apart)

def t_no_overwrite():
    p = short_protocol(epochs=2)
    train, val = loaders()
    with tempfile.TemporaryDirectory() as d:
        train_qute(train, val, seed=11, protocol=p, checkpoint_dir=d)
        try:
            train_qute(train, val, seed=11, protocol=p, checkpoint_dir=d)
        except FileExistsError:
            return
    raise AssertionError("an existing checkpoint was overwritten")
check("an existing checkpoint is left in place", t_no_overwrite)

def t_seed_must_be_in_protocol():
    p = short_protocol(epochs=2)
    train, val = loaders()
    try:
        with tempfile.TemporaryDirectory() as d:
            train_qute(train, val, seed=99, protocol=p, checkpoint_dir=d)
    except ValueError as exc:
        assert "training.seeds" in str(exc), exc
        return
    raise AssertionError("an unlisted seed was accepted")
check("a seed outside training.seeds is refused", t_seed_must_be_in_protocol)


print("\nPREDICTIONS")
print("-----------")

def t_prediction_contract():
    p = short_protocol(epochs=2)
    train, val = loaders()
    with tempfile.TemporaryDirectory() as d:
        model, _ = train_qute(train, val, seed=11, protocol=p, checkpoint_dir=d)
        out = Path(d) / "qute_seed11_clean.npz"
        probs, ids, members, meta = evaluate_qute(
            model, val, seed=11, protocol=p, path=out)
        rp, rid, rmeta, rmembers = load_predictions(out)

    assert tuple(probs.shape) == (N, 10), probs.shape
    assert tuple(members.shape) == (N, 2, 10), members.shape
    assert torch.allclose(probs.sum(1), torch.ones(N), atol=1e-5)
    assert ids.tolist() == list(range(600, 600 + N))
    assert torch.allclose(probs, ensemble_proba(members), atol=1e-6)
    assert torch.allclose(rp, probs, atol=1e-6) and torch.equal(rid, ids)
    assert rmeta["method"] == "qute"
    assert rmeta["confidence"] == "max_softmax_probability"
    assert rmeta["forward_passes"] == 1
    assert rmeta["n_members"] == 2
    assert rmeta["dropped_original_output_block"] is True
check("prediction file matches the shared schema and records one forward pass", t_prediction_contract)

def t_members_are_probabilities():
    """Averaging happens over softmaxes, so members must already be probabilities."""
    p = short_protocol(epochs=2)
    train, val = loaders()
    with tempfile.TemporaryDirectory() as d:
        model, _ = train_qute(train, val, seed=11, protocol=p, checkpoint_dir=d)
        _probs, _ids, members = predict_proba_qute(model, val)
    sums = members.sum(dim=2)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        "member rows are not probability vectors"
check("each member row is a probability vector before averaging", t_members_are_probabilities)

def t_eval_rejects_unindexed():
    p = short_protocol(epochs=2)
    train, val = loaders()
    plain = DataLoader(train.dataset, batch_size=BATCH, shuffle=False)
    with tempfile.TemporaryDirectory() as d:
        model, _ = train_qute(train, val, seed=11, protocol=p, checkpoint_dir=d)
        try:
            predict_proba_qute(model, plain)
        except ValueError as exc:
            assert "sample id" in str(exc), exc
            return
    raise AssertionError("a loader without sample ids was accepted")
check("evaluation refuses a loader with no sample ids", t_eval_rejects_unindexed)

def t_prediction_no_overwrite():
    p = short_protocol(epochs=2)
    train, val = loaders()
    with tempfile.TemporaryDirectory() as d:
        model, _ = train_qute(train, val, seed=11, protocol=p, checkpoint_dir=d)
        out = Path(d) / "p.npz"
        evaluate_qute(model, val, seed=11, protocol=p, path=out)
        try:
            evaluate_qute(model, val, seed=11, protocol=p, path=out)
        except FileExistsError:
            return
    raise AssertionError("an existing prediction file was overwritten")
check("an existing prediction file is left in place", t_prediction_no_overwrite)


print("\nVALIDATION NLL")
print("--------------")

def t_deployed_nll_definition():
    """-log p(true class) of the AVERAGED blocks, not of any single exit."""
    p = short_protocol(epochs=1)
    train, val = loaders()
    model = build_qute_from_protocol(p, 11)
    value = deployed_nll(model, val, torch.device("cpu"))

    manual_total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for images, labels, _ids in val:
            probs = torch.softmax(model.member_logits(images), dim=2).mean(dim=1)
            manual_total += float(-torch.log(probs.gather(1, labels.view(-1, 1))).sum())
            count += labels.shape[0]
    assert abs(value - manual_total / count) < 1e-6, (value, manual_total / count)
check("validation NLL is computed on the deployed averaged prediction", t_deployed_nll_definition)

def t_history_records_loss_parts():
    p = short_protocol(epochs=2)
    train, val = loaders()
    with tempfile.TemporaryDirectory() as d:
        _model, history = train_qute(train, val, seed=11, protocol=p, checkpoint_dir=d)
    for row in history:
        for key in ("train_total", "train_ee", "train_ev", "train_final", "val_nll", "lr"):
            assert key in row, (key, row)
        assert abs(row["train_total"] - (row["train_ee"] + row["train_ev"] + row["train_final"])) < 1e-4
    assert abs(history[1]["lr"] - history[0]["lr"] * 0.99) < 1e-12
check("history logs the three loss parts and decays the learning rate", t_history_records_loss_parts)


print("\nRESULT")
print("------")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name, err in FAIL:
        print(f"  {name}: {err}")
    sys.exit(1)
print("\nQUTE trainer checks pass. Real runs still need data.get_loaders(protocol).")
