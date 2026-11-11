"""Calibration tables from the saved condition prediction files.

    python -m eval.summarize_calibration
    python -m eval.summarize_calibration --variants qute qute_no_transfer

Reads every file in results/preds/conditions/, scores it with
eval.metrics.metrics_from_file (accuracy, NLL, Brier, ECE with the protocol's
15 equal-mass bins), and writes:

    results/tables/calibration_per_file.csv   one row per prediction file,
                                              with its git commit and config
                                              version so numbers trace back
    results/tables/calibration_summary.csv    mean and std over seeds, per
                                              variant and condition, plus an
                                              "mnistc_mean" row per variant

A "variant" is the method plus its headline/reference suffix, taken from the
file name: base, mc_dropout_passes2, mc_dropout_passes20,
deep_ensemble_members2, deep_ensemble_members5, qute, qute_no_transfer.

mnistc_mean is the average over the 15 corruptions, computed per seed first
and then averaged over seeds, so its std is the spread across seeds.

It also prints a short table: clean and MNIST-C mean for each variant.
Nothing is trained or predicted here.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from baselines.base import load_protocol
from eval.metrics import metrics_from_file

DEFAULT_PREDS = Path("results/preds/conditions")
DEFAULT_OUT = Path("results/tables")
METRICS = ("accuracy", "nll", "brier", "ece")
HEADLINE_ORDER = (
    "base", "mc_dropout_passes2", "deep_ensemble_members2", "qute", "qute_no_transfer",
    "mc_dropout_passes20", "deep_ensemble_members5",
)


def variant_name(path: Path, seed: int, condition: str) -> str:
    """mc_dropout_seed11_fog_passes2 -> mc_dropout_passes2; qute_seed11_fog -> qute."""
    stem = path.stem
    marker = f"_seed{seed}_{condition}"
    if marker not in stem:
        raise ValueError(f"{path.name}: expected '{marker}' in the file name")
    return stem.replace(marker, "", 1)


def file_identity(path: Path) -> Tuple[str, int, str]:
    """(variant, seed, condition) from the file's metadata, without scoring it."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
    seed, condition = int(metadata["seed"]), str(metadata["condition"])
    return variant_name(path, seed, condition), seed, condition


def prediction_files(preds_dir: Path) -> List[Path]:
    """Every finished prediction file. Leftover *.tmp.npz from an interrupted write are skipped."""
    return sorted(p for p in preds_dir.glob("*.npz") if not p.name.endswith(".tmp.npz"))


def score_files(preds_dir: Path, protocol: dict, variants: Optional[Sequence[str]] = None) -> List[Dict]:
    files = prediction_files(preds_dir)
    if not files:
        raise SystemExit(f"no prediction files in {preds_dir}")
    rows = []
    for path in files:
        variant, seed, condition = file_identity(path)
        if variants and variant not in variants:
            continue
        result = metrics_from_file(path, protocol=protocol)
        rows.append({
            "variant": variant, "seed": seed, "condition": condition,
            **{m: float(result[m]) for m in METRICS}, "n": int(result["n"]),
            "config_version": result.get("config_version", ""),
            "git_commit": result.get("git_commit", ""),
            "file": path.name,
        })
    if not rows:
        found = sorted({file_identity(p)[0] for p in files})
        raise SystemExit(f"--variants {list(variants)} matched no files; variants found: {found}")
    return rows


def _mean_std(values: List[float]):
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def summarize(rows: List[Dict], expected_corruptions: int) -> List[Dict]:
    by_cell = defaultdict(list)             # (variant, condition) -> rows over seeds
    by_seed = defaultdict(list)             # (variant, seed) -> corruption rows
    for row in rows:
        by_cell[(row["variant"], row["condition"])].append(row)
        if row["condition"] != "clean":
            by_seed[(row["variant"], row["seed"])].append(row)

    summary = []
    for (variant, condition), cell in sorted(by_cell.items()):
        out = {"variant": variant, "condition": condition, "n_seeds": len(cell)}
        for m in METRICS:
            out[f"{m}_mean"], out[f"{m}_std"] = _mean_std([r[m] for r in cell])
        summary.append(out)

    per_variant = defaultdict(dict)         # variant -> seed -> {metric: corruption mean}
    for (variant, seed), cell in by_seed.items():
        if len(cell) != expected_corruptions:
            print(f"  note: {variant} seed {seed} has {len(cell)} corruptions, "
                  f"expected {expected_corruptions}; mnistc_mean skipped for it")
            continue
        per_variant[variant][seed] = {m: statistics.fmean(r[m] for r in cell) for m in METRICS}
    for variant, seeds in sorted(per_variant.items()):
        out = {"variant": variant, "condition": "mnistc_mean", "n_seeds": len(seeds)}
        for m in METRICS:
            out[f"{m}_mean"], out[f"{m}_std"] = _mean_std([s[m] for s in seeds.values()])
        summary.append(out)
    return summary


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        raise SystemExit(f"nothing to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_overview(summary: List[Dict]) -> None:
    lookup = {(r["variant"], r["condition"]): r for r in summary}
    variants = [v for v in HEADLINE_ORDER if (v, "clean") in lookup]
    variants += sorted({r["variant"] for r in summary} - set(variants))
    header = f"{'variant':<24}{'':<8}" + "".join(f"{m:>16}" for m in METRICS) + "   seeds"
    print("\n" + header)
    print("-" * len(header))
    for variant in variants:
        for condition, label in (("clean", "clean"), ("mnistc_mean", "MNIST-C")):
            row = lookup.get((variant, condition))
            if row is None:
                continue
            cells = "".join(f"{row[f'{m}_mean']:>9.4f} ±{row[f'{m}_std']:.4f}" for m in METRICS)
            name = variant if condition == "clean" else ""
            print(f"{name:<24}{label:<8}{cells}   {row['n_seeds']}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Calibration tables from saved condition predictions")
    parser.add_argument("--preds-dir", default=str(DEFAULT_PREDS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--variants", nargs="+", default=None, help="default: every variant found")
    args = parser.parse_args(argv)

    protocol = load_protocol()
    rows = score_files(Path(args.preds_dir), protocol, args.variants)
    summary = summarize(rows, int(protocol["data"]["corruptions_expected"]))

    out = Path(args.out_dir)
    write_csv(out / "calibration_per_file.csv", rows)
    write_csv(out / "calibration_summary.csv", summary)
    print(f"scored {len(rows)} files")
    print_overview(summary)
    print(f"\nwrote {out / 'calibration_per_file.csv'} and {out / 'calibration_summary.csv'}")


if __name__ == "__main__":
    main()
