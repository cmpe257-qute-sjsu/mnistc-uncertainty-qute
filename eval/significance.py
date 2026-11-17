"""Paired significance for the close calls: pre-registered extension (3), RUNLOG 2026-11-11.

    python -m eval.significance                       # v5 results
    python -m eval.significance --results results/small_cnn --out results/small_cnn/tables/significance.csv

Definition (fixed before computing):
  - Pair unit: (seed, corruption). 3 seeds x 15 MNIST-C corruptions = 45 pairs per
    comparison. Clean is excluded: there is no degradation stream for it, and the
    calibration headline is the MNIST-C mean.
  - Difference per pair: QUTE minus the other method, on the same seed and corruption.
  - Report the mean difference and a 95% percentile bootstrap CI, 10,000 resamples of
    the pairs with replacement (fixed bootstrap seed, so the numbers reproduce).
  - Metrics: AUPRC as average precision (primary detection score), and ECE (15
    equal-mass bins, as in calibration_per_file.csv). The paper's 11-point AUPRC is
    reported as a secondary row so slides can quote either.
  - Comparisons: QUTE vs DE-2, QUTE vs QUTE without transfer (ablation), QUTE vs BASE,
    QUTE vs MC-2.

Reading the output: for ECE lower is better, so a negative difference favours QUTE; for
AUPRC higher is better, so a positive difference favours QUTE. "excludes_zero" says
whether the CI excludes zero. Pairs where AP is undefined (a stream with no
degradation events) are dropped and the remaining count is reported.

Transparency columns (not part of the test): the mean difference per seed, and how
many pairs favour QUTE. Pairs share a seed, so they are not fully independent;
resampling pairs is the pre-registered choice, and the per-seed means show whether one
seed drives the result.

Reads only the committed CSVs, never the prediction files:
  <results>/tables/calibration_per_file.csv
  <results>/detection/detection_per_stream.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

QUTE = "qute"
COMPARISONS = (
    ("deep_ensemble_members2", "QUTE - DE-2"),
    ("qute_no_transfer", "QUTE - QUTE without transfer"),
    ("base", "QUTE - BASE"),
    ("mc_dropout_passes2", "QUTE - MC-2"),
)
# metric key -> (source, column, higher_is_better, label)
METRICS = {
    "ap": ("detection", "average_precision", True, "AUPRC (AP)"),
    "ece": ("calibration", "ece", False, "ECE"),
    "auprc_11pt": ("detection", "sweep_auprc", True, "AUPRC (11-point, secondary)"),
}
N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
SEEDS = (11, 22, 33)


def _float(value: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def read_calibration(path: Path) -> Dict[Tuple[str, int, str], float]:
    """(variant, seed, corruption) -> ECE, MNIST-C conditions only."""
    table = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] == "clean":
                continue
            table[(row["variant"], int(row["seed"]), row["condition"])] = _float(row["ece"])
    return table


def read_detection(path: Path, column: str) -> Dict[Tuple[str, int, str], float]:
    """(variant, seed, corruption) -> the given detection column."""
    table = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            table[(row["variant"], int(row["seed"]), row["corruption"])] = _float(row[column])
    return table


def paired_differences(
    table: Dict[Tuple[str, int, str], float], other: str
) -> Tuple[np.ndarray, np.ndarray, int]:
    """QUTE minus `other` for every (seed, corruption) both have. Returns diffs, seeds, n_dropped."""
    keys = sorted((seed, corruption) for (variant, seed, corruption) in table if variant == QUTE)
    diffs, seeds, dropped = [], [], 0
    for seed, corruption in keys:
        if (other, seed, corruption) not in table:
            raise SystemExit(f"missing {other} seed {seed} {corruption}: cannot pair")
        d = table[(QUTE, seed, corruption)] - table[(other, seed, corruption)]
        if np.isnan(d):
            dropped += 1
            continue
        diffs.append(d)
        seeds.append(seed)
    return np.asarray(diffs), np.asarray(seeds), dropped


def bootstrap_ci(diffs: np.ndarray, n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def analyse(results_dir: Path, n_resamples: int = N_RESAMPLES) -> List[dict]:
    calibration = read_calibration(results_dir / "tables" / "calibration_per_file.csv")
    detection_path = results_dir / "detection" / "detection_per_stream.csv"
    rows = []
    for metric, (source, column, higher_is_better, label) in METRICS.items():
        table = calibration if source == "calibration" else read_detection(detection_path, column)
        for other, comparison in COMPARISONS:
            diffs, seeds, dropped = paired_differences(table, other)
            if len(diffs) == 0:
                raise SystemExit(f"no usable pairs for {comparison} on {label}")
            low, high = bootstrap_ci(diffs, n_resamples)
            favours = diffs > 0 if higher_is_better else diffs < 0
            rows.append({
                "comparison": comparison,
                "other_variant": other,
                "metric": label,
                "n_pairs": len(diffs),
                "n_undefined_dropped": dropped,
                "mean_diff": round(float(diffs.mean()), 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
                "excludes_zero": bool(low > 0 or high < 0),
                "favours": (QUTE if (diffs.mean() > 0) == higher_is_better else other)
                if diffs.mean() != 0 else "neither",
                "pairs_favouring_qute": int(favours.sum()),
                **{f"seed{s}_mean_diff": round(float(diffs[seeds == s].mean()), 4) if np.any(seeds == s) else ""
                   for s in SEEDS},
                "n_resamples": n_resamples,
                "bootstrap_seed": BOOTSTRAP_SEED,
            })
    return rows


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: List[dict]) -> str:
    lines = ["| Comparison | Metric | Mean diff [95% CI] | Excludes 0 | Favours | Pairs for QUTE | Seed means (11/22/33) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        seeds = " / ".join(str(r[f"seed{s}_mean_diff"]) for s in SEEDS)
        lines.append(
            f"| {r['comparison']} | {r['metric']} | {r['mean_diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
            f"| {'yes' if r['excludes_zero'] else 'no'} | {r['favours']} | {r['pairs_favouring_qute']}/{r['n_pairs']} | {seeds} |"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"),
                        help="folder holding tables/ and detection/ (default results; small CNN: results/small_cnn)")
    parser.add_argument("--out", type=Path, default=None, help="default <results>/tables/significance.csv")
    parser.add_argument("--resamples", type=int, default=N_RESAMPLES)
    args = parser.parse_args(argv)

    rows = analyse(args.results, args.resamples)
    out = args.out or args.results / "tables" / "significance.csv"
    write_csv(rows, out)
    print(markdown(rows))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
