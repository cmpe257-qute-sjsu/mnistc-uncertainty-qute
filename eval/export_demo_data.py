"""Export the data behind the presentation demo (demo/index.html).

Reads the saved prediction files in results/preds/conditions/ and writes
demo/demo_data.js, a small file the demo page loads with a <script> tag so
it runs offline from file:// with no server. Nothing here runs a model, and
every number comes from the same functions the report uses:

  detection    eval.detection (window_means, evaluate_detection), so the
               stream, the mu - 3 sigma event rule and AP match Step 16
  calibration  eval.metrics (accuracy, nll, ece and its equal-mass bins)

What goes in the file
  detection    one display seed: window accuracy and mean confidence for
               every 20th stride-1 window (for plotting), plus exact
               precision, recall, clean false alarms and alarm delay on a
               0.00-1.00 rho grid computed on all stride-1 windows; AP for
               every seed
  calibration  every method, condition and seed: accuracy, NLL, ECE, the
               15 equal-mass reliability bins, and confidence histograms
               for correct and wrong predictions
  images       optional: the same 12 test digits under every condition,
               with each method's prediction and confidence on them

Usage (from the repo root, after predict_conditions has run):
  python -m eval.export_demo_data
  python -m eval.export_demo_data --display-seed 22 --no-images
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from data.prepare_mnistc import CORRUPTIONS, MNISTC_ROOT
from eval.detection import WINDOW_SIZE, load_predictions, window_means
from eval.metrics import _bin_index, accuracy, ece, nll

PRED_DIR = Path("results/preds/conditions")
OUT_PATH = Path("demo/demo_data.js")
SEEDS = (11, 22, 33)
CONDITIONS = ("clean",) + tuple(CORRUPTIONS)
N_BINS = 15
PLOT_EVERY = 20
RHO_GRID = np.round(np.linspace(0.0, 1.0, 101), 2)
N_IMAGES = 12
IMAGE_IDS_SEED = 0

# Display name -> filename pattern. The headline budget is K = 2;
# the K > 2 references appear in the calibration tab only.
METHODS = {
    "BASE": "base_seed{seed}_{condition}.npz",
    "MC Dropout (2)": "mc_dropout_seed{seed}_{condition}_passes2.npz",
    "Deep ensemble (2)": "deep_ensemble_seed{seed}_{condition}_members2.npz",
    "QUTE": "qute_seed{seed}_{condition}.npz",
    "QUTE, no transfer": "qute_no_transfer_seed{seed}_{condition}.npz",
}
REFERENCES = {
    "MC Dropout (20)": "mc_dropout_seed{seed}_{condition}_passes20.npz",
    "Deep ensemble (5)": "deep_ensemble_seed{seed}_{condition}_members5.npz",
}


def _r(values, digits=4):
    return [round(float(v), digits) for v in values]


def load_probs(path):
    """probs, labels and confidence ordered by sample ID (as Step 16 does)."""
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["sample_ids"])
        order = np.argsort(ids)
        return np.asarray(data["probs"], dtype=np.float64)[order], np.asarray(data["labels"])[order]


def calibration_entry(probs, labels):
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    bins = _bin_index(confidence, N_BINS, "equal_mass")
    reliability = []
    for b in range(N_BINS):
        mask = bins == b
        reliability.append([round(float(confidence[mask].mean()), 4), round(float(correct[mask].mean()), 4)])
    edges = np.linspace(0.0, 1.0, 21)
    right, _ = np.histogram(confidence[correct], bins=edges)
    wrong, _ = np.histogram(confidence[~correct], bins=edges)
    return {
        "accuracy": round(accuracy(probs, labels), 4),
        "nll": round(nll(probs, labels), 4),
        "ece": round(ece(probs, labels, n_bins=N_BINS, binning="equal_mass"), 4),
        "reliability": reliability,
        "hist_correct": right.tolist(),
        "hist_wrong": wrong.tolist(),
    }


def detection_stream(clean, corrupted, full):
    """Step 16's stream, plus the rho-grid view the demo slider needs."""
    clean_correct = (clean["predictions"] == clean["labels"]).astype(float)
    corrupted_correct = (corrupted["predictions"] == corrupted["labels"]).astype(float)
    reference = window_means(clean_correct)
    threshold = float(reference.mean() - 3.0 * reference.std())

    window_accuracy = window_means(np.concatenate((clean_correct, corrupted_correct)))
    mean_confidence = window_means(np.concatenate((clean["confidence"], corrupted["confidence"])))
    events = window_accuracy < threshold
    positives = int(events.sum())
    ap = float(average_precision_score(events, 1.0 - mean_confidence)) if positives else None
    if not full:
        return {"ap": ap}

    n_clean = len(clean_correct)
    starts = np.arange(len(events))
    clean_only = starts + WINDOW_SIZE <= n_clean
    after_switch = ~clean_only
    precision, recall, false_alarms, delay = [], [], [], []
    for rho in RHO_GRID:
        alarms = mean_confidence < rho
        tp = int(np.sum(alarms & events))
        fp = int(np.sum(alarms & ~events))
        precision.append(round(tp / (tp + fp), 4) if tp + fp else None)
        recall.append(round(tp / positives, 4) if positives else None)
        false_alarms.append(round(float(alarms[clean_only].mean()), 4))
        hits = np.flatnonzero(alarms & after_switch)
        # Samples after the switch when the first alarming window ends.
        delay.append(int(hits[0] + WINDOW_SIZE - n_clean) if len(hits) else None)

    keep = slice(0, len(events), PLOT_EVERY)
    return {
        "ap": ap,
        "threshold": round(threshold, 4),
        "events": positives,
        "window_start": starts[keep].tolist(),
        "window_accuracy": _r(window_accuracy[keep], 3),
        "mean_confidence": _r(mean_confidence[keep], 3),
        "precision": precision,
        "recall": recall,
        "clean_false_alarm_rate": false_alarms,
        "delay": delay,
    }


def encode_image(image):
    return base64.b64encode(np.asarray(image, dtype=np.uint8).reshape(28, 28).tobytes()).decode("ascii")


def find_mnistc_root():
    for root in (MNISTC_ROOT, MNISTC_ROOT / "mnist_c"):
        if (root / "shot_noise" / "test_images.npy").is_file():
            return root
    return None


def export_images(display_seed, pred_dir):
    root = find_mnistc_root()
    if root is None:
        print("MNIST-C images not found under data/mnist_c/; skipping the image strip")
        return None
    ids = np.sort(np.random.default_rng(IMAGE_IDS_SEED).choice(10000, size=N_IMAGES, replace=False))
    out = {"ids": ids.tolist(), "conditions": {}}
    for condition in CONDITIONS:
        folder = root / ("identity" if condition == "clean" else condition)
        if not (folder / "test_images.npy").is_file():
            continue
        images = np.load(folder / "test_images.npy", mmap_mode="r")
        labels = np.load(folder / "test_labels.npy", mmap_mode="r").reshape(-1)
        entry = {"images": [encode_image(images[i]) for i in ids], "labels": labels[ids].astype(int).tolist(), "methods": {}}
        for name, pattern in METHODS.items():
            path = pred_dir / pattern.format(seed=display_seed, condition=condition)
            if path.is_file():
                probs, _ = load_probs(path)
                entry["methods"][name] = {
                    "pred": probs[ids].argmax(axis=1).astype(int).tolist(),
                    "conf": _r(probs[ids].max(axis=1), 3),
                }
        out["conditions"][condition] = entry
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=Path, default=PRED_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--display-seed", type=int, default=SEEDS[0], choices=SEEDS)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args(argv)

    missing = [
        pattern.format(seed=seed, condition=condition)
        for pattern in {**METHODS, **REFERENCES}.values()
        for seed in SEEDS
        for condition in CONDITIONS
        if not (args.pred_dir / pattern.format(seed=seed, condition=condition)).is_file()
    ]
    if missing:
        raise SystemExit(f"{len(missing)} prediction files missing in {args.pred_dir}, e.g. {missing[:3]}")

    calibration = {}
    for name, pattern in {**METHODS, **REFERENCES}.items():
        calibration[name] = {}
        for condition in CONDITIONS:
            calibration[name][condition] = {}
            for seed in SEEDS:
                probs, labels = load_probs(args.pred_dir / pattern.format(seed=seed, condition=condition))
                calibration[name][condition][str(seed)] = calibration_entry(probs, labels)
        print(f"calibration: {name}")

    detection, ap_by_seed = {}, {}
    for name, pattern in METHODS.items():
        detection[name], ap_by_seed[name] = {}, {}
        for seed in SEEDS:
            clean = load_predictions(args.pred_dir / pattern.format(seed=seed, condition="clean"))
            for corruption in CORRUPTIONS:
                corrupted = load_predictions(args.pred_dir / pattern.format(seed=seed, condition=corruption))
                full = seed == args.display_seed
                result = detection_stream(clean, corrupted, full)
                ap_by_seed[name].setdefault(corruption, {})[str(seed)] = (
                    None if result["ap"] is None else round(result["ap"], 4)
                )
                if full:
                    detection[name][corruption] = result
        print(f"detection: {name}")

    payload = {
        "display_seed": args.display_seed,
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "references": list(REFERENCES),
        "conditions": list(CONDITIONS),
        "corruptions": list(CORRUPTIONS),
        "window_size": WINDOW_SIZE,
        "plot_every": PLOT_EVERY,
        "rho_grid": RHO_GRID.tolist(),
        "detection": detection,
        "detection_ap": ap_by_seed,
        "calibration": calibration,
        "images": None if args.no_images else export_images(args.display_seed, args.pred_dir),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = "window.DEMO_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out} ({len(text) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
