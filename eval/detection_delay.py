
"""Detection delay using clean-calibrated confidence thresholds."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.prepare_mnistc import CORRUPTIONS
from eval.detection import WINDOW_SIZE, load_predictions, window_means


PRED_DIR = Path("results/preds/conditions")
OUT_DIR = Path("results/detection_delay")
CORRUPTION_START = 10000


def evaluate_delay(clean, corrupted):
    """Measure detection delay for one clean-to-corruption stream."""

    if not np.array_equal(clean["ids"], corrupted["ids"]):
        raise ValueError("Clean/corrupted sample ID mismatch")

    if not np.array_equal(clean["labels"], corrupted["labels"]):
        raise ValueError("Clean/corrupted label mismatch")

    # Calibrate threshold from clean-only windows.
    clean_window_confidence = window_means(clean["confidence"])
    tau = float(np.percentile(clean_window_confidence, 1))

    # Construct the complete clean-then-corruption stream.
    stream_confidence = np.concatenate(
        [clean["confidence"], corrupted["confidence"]]
    )

    mean_confidence = window_means(stream_confidence)

    # Alarm only when mean confidence is strictly below tau.
    alarms = mean_confidence < tau

    # Windows use half-open [start, end) indices.
    starts = np.arange(len(mean_confidence))
    ends = starts + WINDOW_SIZE

    # Search for the first alarming window ending after onset.
    post_onset = ends > CORRUPTION_START
    alarm_indices = np.flatnonzero(alarms & post_onset)

    if len(alarm_indices) == 0:
        detection_delay = None
        detected = False
    else:
        first_alarm_index = int(alarm_indices[0])
        detection_delay = int(
            ends[first_alarm_index] - CORRUPTION_START
        )
        detected = True

    # Detection rate uses only fully corrupted windows.
    fully_corrupted = starts >= CORRUPTION_START

    corrupted_alarms = alarms[fully_corrupted]

    detection_rate = float(np.mean(corrupted_alarms))

    return {
        "tau": tau,
        "detected": detected,
        "delay_samples": detection_delay,
        "detection_rate": detection_rate,
        "clean_windows": len(clean_window_confidence),
        "fully_corrupted_windows": int(
            np.sum(fully_corrupted)
        ),
    }


def save_csv(path, rows):
    """Save dictionaries as CSV rows."""
    if not rows:
        raise ValueError(f"No results to save: {path}")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def read_severity_map(path):
    """Read an agreed corruption-to-severity mapping."""
    if path is None:
        return {}

    mapping = {}

    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if not {"corruption", "severity"}.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "Severity CSV requires corruption,severity columns"
            )

        for row in reader:
            corruption = row["corruption"].strip()
            severity = row["severity"].strip().lower()

            if corruption in mapping:
                raise ValueError(
                    f"Duplicate corruption: {corruption}"
                )

            if severity not in {"mild", "moderate", "severe"}:
                raise ValueError(
                    f"Invalid severity for {corruption}: {severity}"
                )

            mapping[corruption] = severity

    if set(mapping) != set(CORRUPTIONS):
        raise ValueError(
            "Severity mapping must contain exactly "
            "the 15 protocol corruptions"
        )

    return mapping


def summarize(rows):
    """Summarize detection coverage and delay by variant/severity."""
    groups = defaultdict(list)

    for row in rows:
        groups[(row["variant"], "overall")].append(row)

        if row["severity"] != "unassigned":
            groups[
                (row["variant"], row["severity"])
            ].append(row)

    summaries = []

    for (variant, severity), group in sorted(groups.items()):
        delays = [
            row["delay_samples"]
            for row in group
            if row["delay_samples"] is not None
        ]

        detected_count = len(delays)

        summaries.append({
            "variant": variant,
            "severity": severity,
            "streams": len(group),
            "detected_streams": detected_count,
            "undetected_streams": len(group) - detected_count,
            "detection_coverage": (
                detected_count / len(group)
            ),
            # Median excludes undetected streams.
            "median_delay_samples": (
                float(np.median(delays))
                if delays
                else None
            ),
            "mean_detection_rate": float(
                np.mean([
                    row["detection_rate"]
                    for row in group
                ])
            ),
        })

    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate clean-calibrated detection delay"
    )

    parser.add_argument(
        "--severity-map",
        type=Path,
        default=None,
        help="Optional CSV with corruption,severity columns",
    )

    args = parser.parse_args()

    severity_map = read_severity_map(args.severity_map)

    clean_files = sorted(PRED_DIR.glob("*_clean*.npz"))

    if not clean_files:
        raise FileNotFoundError(
            f"No clean prediction files found in {PRED_DIR}"
        )

    rows = []

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

        model_id = f"{method}_seed{seed}{suffix}"
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
                    f"Missing prediction: {corrupted_path}"
                )

            corrupted = load_predictions(corrupted_path)

            outcome = evaluate_delay(clean, corrupted)

            rows.append({
                "model_id": model_id,
                "method": method,
                "variant": variant,
                "seed": int(seed),
                "corruption": corruption,
                "severity": severity_map.get(
                    corruption, "unassigned"
                ),
                **outcome,
            })

        print(f"Completed: {model_id}")

    summaries = summarize(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    save_csv(
        OUT_DIR / "delay_per_stream.csv",
        rows,
    )

    save_csv(
        OUT_DIR / "delay_summary.csv",
        summaries,
    )

    print("\nDetection delay summary:")
    print("-" * 75)

    for row in summaries:
        median = row["median_delay_samples"]

        median_text = (
            f"{median:.1f}"
            if median is not None
            else "not detected"
        )

        print(
            f"{row['variant']} | "
            f"{row['severity']} | "
            f"median delay={median_text} samples | "
            f"coverage={row['detection_coverage']:.1%} | "
            f"mean detection rate={row['mean_detection_rate']:.1%}"
        )

    print(f"\nStreams evaluated: {len(rows)}")
    print(f"Summary rows: {len(summaries)}")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
