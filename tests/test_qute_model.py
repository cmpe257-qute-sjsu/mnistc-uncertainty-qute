"""
Step 3 acceptance checks for the QUTE architecture.

    python tests/test_qute_model.py

Synthetic tensors only -- no MNIST, runs in seconds. These check the mechanism,
not the results: shapes line up, the copy is a value transfer rather than
parameter sharing, the deployed graph excludes what it should, and the loss has
the three terms the paper specifies.

A favourable metric later does not replace any of this.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.shared_cnn import CNNConfig, SharedCNN, build_model
from models.qute import (
    EarlyExit, OutputBlock, QUTEConfig, QUTEModel,
    build_qute, ev_weights, qute_loss, tau_weights,
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


def fake_batch(n=8):
    g = torch.Generator().manual_seed(3)
    x = torch.rand(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return x, y


print("\nSHAPES")
print("------")

def t_builds():
    m = build_qute(seed=11)
    assert len(m.early_exits) == 2, len(m.early_exits)
    assert len(m.output_blocks) == 2, len(m.output_blocks)
check("builds with K = 2 exits and 2 output blocks", t_builds)

def t_transfer_shapes_match():
    m = build_qute(seed=11)
    for k, (ee, ob) in enumerate(zip(m.early_exits, m.output_blocks)):
        assert ee.h_theta.weight.shape == ob.h_phi.weight.shape, (
            k, ee.h_theta.weight.shape, ob.h_phi.weight.shape)
        assert tuple(ob.h_phi.weight.shape) == (32, 1, 3, 3), ob.h_phi.weight.shape
check("h_theta and h_phi weights are both [32,1,3,3]", t_transfer_shapes_match)

def t_pointwise_matches_channels():
    m = build_qute(seed=11)
    assert m.early_exits[0].pointwise.in_channels == 8
    assert m.early_exits[1].pointwise.in_channels == 16
    for ee in m.early_exits:
        assert ee.pointwise.out_channels == 32
check("pointwise convs map taps 8 and 16 up to 32 channels", t_pointwise_matches_channels)

def t_spatial_sizes_differ():
    m = build_qute(seed=11)
    x, _ = fake_batch()
    taps, f_d = m._base_features(x)
    assert tuple(taps[1].shape) == (8, 8, 14, 14), taps[1].shape
    assert tuple(taps[2].shape) == (8, 16, 7, 7), taps[2].shape
    assert tuple(f_d.shape) == (8, 32, 7, 7), f_d.shape
check("taps differ in spatial size, which the copy does not care about", t_spatial_sizes_differ)

def t_mismatched_channels_rejected():
    base = build_model(CNNConfig(channels=[8, 16, 32, 64]))
    m = QUTEModel(base)          # final channels 64 -> pointwise must follow
    assert tuple(m.output_blocks[0].h_phi.weight.shape) == (64, 1, 3, 3)
    assert m.early_exits[0].pointwise.out_channels == 64
check("a different final width moves both sides together", t_mismatched_channels_rejected)


print("\nWEIGHT TRANSFER")
print("---------------")

def t_transfer_copies():
    m = build_qute(seed=11)
    assert not m.transfer_matches(), "fresh init should differ before any transfer"
    m.transfer_weights()
    assert m.transfer_matches(), "h_phi does not equal h_theta after transfer"
check("transfer makes every h_phi equal its h_theta", t_transfer_copies)

def t_no_aliasing():
    m = build_qute(seed=11)
    m.transfer_weights()
    m.assert_no_aliasing()
    # Mutating theta must NOT change phi -- that is the difference between
    # a value copy and a shared parameter.
    with torch.no_grad():
        m.early_exits[0].h_theta.weight.add_(1.0)
    assert not m.transfer_matches(), "changing h_theta changed h_phi: they are aliased"
check("copy is by value, not shared storage", t_no_aliasing)

def t_phi_trains_then_is_overwritten():
    """The loop the paper describes: copy, train, overwrite."""
    m = build_qute(seed=11)
    x, y = fake_batch()
    opt = torch.optim.Adam(m.parameters(), lr=0.01)

    m.transfer_weights()
    before = m.output_blocks[0].h_phi.weight.detach().clone()

    out = m.forward_train(x)
    qute_loss(out, y, m.config)["total"].backward()
    opt.step()

    after = m.output_blocks[0].h_phi.weight.detach().clone()
    assert not torch.equal(before, after), "h_phi did not update during the step"
    assert not m.transfer_matches(), "h_phi should have drifted from h_theta"

    m.transfer_weights()
    assert m.transfer_matches(), "the next batch did not restore h_phi"
check("h_phi updates in a step, then the next transfer overwrites it", t_phi_trains_then_is_overwritten)

def t_gradients_reach_base():
    """The point of the mechanism: EV losses must shape the shared base."""
    m = build_qute(seed=11)
    x, y = fake_batch()
    m.transfer_weights()
    out = m.forward_train(x)
    # EV term alone, so the gradient can only arrive through the base network.
    ev_only = sum(F.cross_entropy(logits, y) for logits in out["ev"])
    ev_only.backward()
    grads = [p.grad for p in m.base.parameters() if p.grad is not None]
    assert grads, "no gradient reached the base network from the output blocks"
    assert any(g.abs().sum() > 0 for g in grads), "base gradients are all zero"
check("output-block loss sends gradient into the shared base", t_gradients_reach_base)


print("\nDEPLOYED PATH")
print("-------------")

def t_member_logits_shape():
    m = build_qute(seed=11)
    x, _ = fake_batch()
    with torch.no_grad():
        members = m.member_logits(x)
    assert tuple(members.shape) == (8, 2, 10), members.shape
check("member_logits returns [N, K, 10]", t_member_logits_shape)

def t_forward_returns_probabilities():
    m = build_qute(seed=11).eval()
    x, _ = fake_batch()
    with torch.no_grad():
        probs = m(x)
    assert tuple(probs.shape) == (8, 10), probs.shape
    assert torch.allclose(probs.sum(1), torch.ones(8), atol=1e-5)
    assert (probs >= 0).all()
check("forward gives probabilities that sum to 1", t_forward_returns_probabilities)

def t_mean_of_softmax_not_softmax_of_mean():
    """These differ, and the paper averages probabilities."""
    m = build_qute(seed=11).eval()
    x, _ = fake_batch()
    with torch.no_grad():
        members = m.member_logits(x)
        correct = torch.softmax(members, dim=2).mean(dim=1)
        wrong = torch.softmax(members.mean(dim=1), dim=1)
        assert torch.allclose(m(x), correct, atol=1e-6)
    assert not torch.allclose(correct, wrong, atol=1e-4), (
        "the two orderings happened to agree; the test cannot distinguish them")
check("forward averages softmaxes, not logits", t_mean_of_softmax_not_softmax_of_mean)

def t_deployed_excludes_training_parts():
    m = build_qute(seed=11).eval()
    x, _ = fake_batch()
    # Break the early exits and the original classifier. The deployed path must
    # not notice, because neither is on it.
    with torch.no_grad():
        before = m(x).clone()
        for ee in m.early_exits:
            for p in ee.parameters():
                p.mul_(0).add_(99.0)
        for p in m.base.classifier.parameters():
            p.mul_(0).add_(99.0)
        after = m(x)
    assert torch.allclose(before, after, atol=1e-6), (
        "deployment changed after breaking the early exits or the original classifier")
check("deployment ignores early exits and the original classifier", t_deployed_excludes_training_parts)

def t_parameter_counts():
    m = build_qute(seed=11)
    info = m.describe()
    base_total = sum(p.numel() for p in m.base.parameters())
    head = sum(p.numel() for p in m.base.classifier.parameters())
    blocks = sum(p.numel() for p in m.output_blocks.parameters())
    exits = sum(p.numel() for p in m.early_exits.parameters())
    assert info["deployed_parameters"] == base_total - head + blocks
    assert info["training_parameters"] == base_total + blocks + exits
    assert info["deployed_parameters"] < info["training_parameters"]
check("deployed count drops the early exits and the original head", t_parameter_counts)


print("\nLOSS")
print("----")

def t_ev_weights():
    assert ev_weights(QUTEConfig()) == [3.0, 3.5]
check("output-block weights are [3.0, 3.5]", t_ev_weights)

def t_tau_default():
    assert tau_weights(QUTEConfig()) == [0.25, 0.5]
    assert tau_weights(QUTEConfig(tau=[1.0, 1.0])) == [1.0, 1.0]
check("tau defaults to relative depth and is overridable", t_tau_default)

def t_loss_parts():
    m = build_qute(seed=11)
    x, y = fake_batch()
    out = m.forward_train(x)
    parts = qute_loss(out, y, m.config)
    assert set(parts) == {"total", "ee", "ev", "final"}
    assert torch.allclose(parts["total"], parts["ee"] + parts["ev"] + parts["final"], atol=1e-6)
    assert parts["total"].requires_grad
check("loss is EE + EV + final exit and stays differentiable", t_loss_parts)

def t_loss_weighting_applied():
    m = build_qute(seed=11)
    x, y = fake_batch()
    out = m.forward_train(x)
    parts = qute_loss(out, y, m.config)
    taus, ws = tau_weights(m.config), ev_weights(m.config)
    manual_ee = sum(t * F.cross_entropy(l, y) for t, l in zip(taus, out["ee"]))
    manual_ev = sum(w * F.cross_entropy(l, y) for w, l in zip(ws, out["ev"]))
    assert torch.allclose(parts["ee"], manual_ee, atol=1e-6)
    assert torch.allclose(parts["ev"], manual_ev, atol=1e-6)
check("the configured weights are the ones actually applied", t_loss_weighting_applied)


print("\nFINAL STAGE")
print("-----------")

def t_freeze_epoch():
    m = build_qute(seed=11)
    assert m.freeze_epoch_start(20) == 19, m.freeze_epoch_start(20)
    assert m.freeze_epoch_start(10) == 10
check("20 epochs freeze from epoch 19", t_freeze_epoch)

def t_freeze_blocks_base_only():
    m = build_qute(seed=11)
    m.freeze_base(True)
    assert all(not p.requires_grad for p in m.base.parameters())
    assert all(p.requires_grad for p in m.output_blocks.parameters())
    assert all(p.requires_grad for p in m.early_exits.parameters())
    m.freeze_base(False)
    assert all(p.requires_grad for p in m.base.parameters())
check("freezing stops the base and leaves the blocks trainable", t_freeze_blocks_base_only)

def t_frozen_base_does_not_move():
    m = build_qute(seed=11)
    x, y = fake_batch()
    m.freeze_base(True)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.05)
    snapshot = [p.detach().clone() for p in m.base.parameters()]
    for _ in range(3):
        m.transfer_weights()
        opt.zero_grad(set_to_none=True)
        qute_loss(m.forward_train(x), y, m.config)["total"].backward()
        opt.step()
    for before, now in zip(snapshot, m.base.parameters()):
        assert torch.equal(before, now), "a frozen base weight changed"
check("base weights hold still while transfer keeps running", t_frozen_base_does_not_move)


print("\nSHORT TRAINING LOOP")
print("-------------------")

def t_loop_runs():
    """Three batches of the real order: copy, forward, loss, backward, step."""
    m = build_qute(seed=11)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    x, y = fake_batch()
    losses = []
    for _ in range(3):
        m.transfer_weights()
        assert m.transfer_matches(), "transfer did not take at the start of the batch"
        opt.zero_grad(set_to_none=True)
        parts = qute_loss(m.forward_train(x), y, m.config)
        parts["total"].backward()
        opt.step()
        losses.append(parts["total"].detach().item())
    assert all(torch.isfinite(torch.tensor(v)) for v in losses), losses
    assert losses[-1] < losses[0], f"loss did not fall over three batches: {losses}"
    m.assert_no_aliasing()
check("three batches run and the loss falls", t_loop_runs)

def t_ablation_keeps_phi():
    """Transfer off: h_phi keeps its own learned values instead."""
    m = build_qute(seed=11)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    x, y = fake_batch()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        qute_loss(m.forward_train(x), y, m.config)["total"].backward()
        opt.step()
    assert not m.transfer_matches(), "h_phi tracked h_theta with transfer disabled"
check("with transfer off, h_phi keeps its own values", t_ablation_keeps_phi)


print("\nMODEL SIZE")
print("----------")
info = build_qute(seed=11).describe()
for key in ("K", "ev_weights", "tau_weights", "transfer_weight_shape",
            "training_parameters", "deployed_parameters", "deployed_approx_fp32_kib"):
    print(f"  {key:28} {info[key]}")

print("\nRESULT")
print("------")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name, err in FAIL:
        print(f"  {name}: {err}")
    sys.exit(1)
print("\nQUTE architecture checks pass. Next: the trainer in qute/train.py.")
