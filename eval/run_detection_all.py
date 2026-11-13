
"""Run detection/AUPRC on every saved condition prediction file."""

import csv
from collections import defaultdict
from pathlib import Path

from data.prepare_mnistc import CORRUPTIONS
from eval.detection import load_predictions, evaluate_detection


PRED_DIR = Path("results/preds/conditions")
OUT_DIR = Path("results/detection")


def save_csv(path, rows):
    """Write results to a CSV file."""
    if not rows:
        raise ValueError(f"No results to save: {path}")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(PRED_DIR.glob("*_clean*.npz"))

    if not clean_files:
        raise FileNotFoundError(
            f"No clean prediction files found in {PRED_DIR}"
        )

    results = []
    sweep_results = []

    for clean_path in clean_files:
        stem = clean_path.stem

        method, separator, rest = stem.partition("_seed")

        if not separator:
            raise ValueError(
                f"Unexpected filename: {clean_path.name}"
            )

        seed, separator, suffix = rest.partition("_clean")

        if not separator or not seed.isdigit():
            raise ValueError(
                f"Unexpected filename: {clean_path.name}"
            )

        # Unique model and seed identifier.
        model_id = f"{method}_seed{seed}"
        if suffix:
            model_id += suffix

        # Variant identifies the method and its compute budget,
        # without the seed.
        #
        # Examples:
        # mc_dropout_passes2
        # mc_dropout_passes20
        # deep_ensemble_members2
        # deep_ensemble_members5
        # qute
        # qute_no_transfer
        # base
        variant = method + suffix

        clean = load_predictions(clean_path)

        for corruption in CORRUPTIONS:
            corrupted_name = clean_path.name.replace(
                "_clean",
                f"_{corruption}",
                1,
            )

            corrupted_path = PRED_DIR / corrupted_name

            if not corrupted_path.exists():
                raise FileNotFoundError(
                    f"Missing prediction file: {corrupted_path}"
                )

            corrupted = load_predictions(corrupted_path)

            outcome = evaluate_detection(
                clean,
                corrupted,
            )

            # One row per model/seed/corruption stream.
            results.append({
                "model_id": model_id,
                "method": method,
                "variant": variant,
                "seed": int(seed),
                "corruption": corruption,
                "total_windows": outcome["total_windows"],
                "degradation_events": outcome["degradation_events"],
                "clean_reference_mean": outcome[
                    "clean_reference_mean"
                ],
                "clean_reference_std": outcome[
                    "clean_reference_std"
                ],
                "degradation_threshold": outcome[
                    "degradation_threshold"
                ],
                "average_precision": outcome[
                    "average_precision"
                ],
                "sweep_auprc": outcome[
                    "sweep_auprc"
                ],
            })

            # Eleven rows per stream: one for each rho value.
            for point in outcome["rho_sweep"]:
                sweep_results.append({
                    "model_id": model_id,
                    "method": method,
                    "variant": variant,
                    "seed": int(seed),
                    "corruption": corruption,
                    **point,
                })

        print(f"Completed: {model_id}")

    # Save the final detection results.
    save_csv(
        OUT_DIR / "detection_per_stream.csv",
        results,
    )

    save_csv(
        OUT_DIR / "detection_rho_sweep.csv",
        sweep_results,
    )

    # Summarize by actual variant, not model_id or method.
    # This prevents different compute budgets from mixing.
    groups = defaultdict(list)
    sweep_groups = defaultdict(list)

    for row in results:
        if row["average_precision"] is not None:
            groups[row["variant"]].append(
                row["average_precision"]
            )

        if row["sweep_auprc"] is not None:
            sweep_groups[row["variant"]].append(
                row["sweep_auprc"]
            )

    print("\nAUPRC summary by variant:")
    print("-" * 60)

    for variant in sorted(groups):
        ap_values = groups[variant]
        sweep_values = sweep_groups[variant]

        mean_ap = sum(ap_values) / len(ap_values)

        mean_sweep = (
            sum(sweep_values) / len(sweep_values)
            if sweep_values
            else None
        )

        sweep_text = (
            f"{mean_sweep:.4f}"
            if mean_sweep is not None
            else "undefined"
        )

        print(
            f"{variant}: "
            f"AP={mean_ap:.4f}, "
            f"11-point AUPRC={sweep_text}"
        )

    print(f"\nStreams evaluated: {len(results)}")
    print(f"Threshold rows: {len(sweep_results)}")
    print(f"Saved results to: {OUT_DIR}")


if __name__ == "__main__":
    main()
