"""Deployment cost of each method: size and CPU time.

    python -m eval.cost_table
    python -m eval.cost_table --threads 1 --repeats 5
    python -m eval.cost_table --config configs/small_cnn.yaml \\
        --variants base mc_dropout_passes2 deep_ensemble_members2 qute \\
        --out results/small_cnn/tables/cost_table.csv          # exploratory small-CNN rows

Pre-registered extension (2), RUNLOG 2026-11-11. For each deployed variant:

    deployed_params   parameters needed at inference
                        BASE, MC Dropout: the CNN (dropout has no parameters)
                        Deep ensemble:    K x the CNN
                        QUTE:             base CNN minus its original classifier,
                                          plus the K output blocks (early exits
                                          are training-only and dropped)
    fp32_kib          deployed_params x 4 bytes, in KiB
    forward_passes    full network passes per image: BASE 1, MC-K K, DE-K K, QUTE 1
    ms_per_1000       CPU milliseconds per 1,000 images, batch 256, including the
                      softmax and averaging each method deploys. Median of
                      --repeats timed runs after warm-up, fixed thread count

Timing method
-------------
- Inputs are a fixed random tensor of --n-images MNIST-shaped images (default
  10,000, the test set size), so data loading and disk are not timed.
- Cost depends only on the architecture, not on the trained weights, so models
  are built from the config with seed-11 initialization; no checkpoints needed.
  The three seeds of a variant have identical cost.
- Every variant is warmed up first, then the repeats are interleaved (round 1:
  all variants, round 2: all variants, ...) so slow drift on the machine (other
  apps, thermal throttling) affects all variants alike rather than one.
- torch.inference_mode, one process, torch.set_num_threads(--threads).
- Numbers are relative within one machine and one run. Batch-256 throughput is
  not the batch-1 latency a microcontroller would see; report the ratios, and
  state the CPU and thread count with the table.

qute_no_transfer has the same deployed architecture as QUTE, so it is not timed
separately.

Writes results/tables/cost_table.csv (or --out) and prints a markdown table.
Nothing is trained and no prediction files are written.
"""

import argparse
import csv
import datetime
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch

from baselines.base import build_base_model, build_cnn_model, load_protocol
from models import enable_mc_dropout
from models.interface import _git_commit
from qute.train import build_qute_from_protocol

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "tables" / "cost_table.csv"
VARIANTS = (
    "base",
    "mc_dropout_passes2",
    "mc_dropout_passes20",
    "deep_ensemble_members2",
    "deep_ensemble_members5",
    "qute",
)
SEED = 11
INPUT_SHAPE = (1, 28, 28)


def _count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _k_from(variant: str, prefix: str) -> int:
    return int(variant[len(prefix):])


def build_variant(protocol: dict, variant: str, seed: int = SEED) -> Dict:
    """Return {"predict": fn(images) -> probs [B, C], "params": int, "passes": int}.

    predict runs exactly what the method deploys, softmax and averaging included.
    """
    if variant == "base":
        model = build_base_model(protocol, seed).eval()

        def predict(x):
            return torch.softmax(model(x), dim=1)

        return {"predict": predict, "params": _count(model), "passes": 1}

    if variant.startswith("mc_dropout_passes"):
        passes = _k_from(variant, "mc_dropout_passes")
        rate = float(protocol["baselines"]["mc_dropout"]["rate"])
        model = build_cnn_model(protocol, seed, dropout_rate=rate)
        enable_mc_dropout(model)

        def predict(x):
            return torch.stack([torch.softmax(model(x), dim=1) for _ in range(passes)], dim=1).mean(dim=1)

        return {"predict": predict, "params": _count(model), "passes": passes}

    if variant.startswith("deep_ensemble_members"):
        k = _k_from(variant, "deep_ensemble_members")
        members = [build_base_model(protocol, seed * 100 + i).eval() for i in range(1, k + 1)]

        def predict(x):
            return torch.stack([torch.softmax(m(x), dim=1) for m in members], dim=1).mean(dim=1)

        return {"predict": predict, "params": sum(_count(m) for m in members), "passes": k}

    if variant == "qute":
        model = build_qute_from_protocol(protocol, seed).eval()

        def predict(x):
            return model(x)  # already the mean of the K output blocks' softmaxes

        return {"predict": predict, "params": int(model.deployed_parameters()), "passes": 1}

    raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")


def _run_once(predict: Callable, batches: Sequence[torch.Tensor]) -> float:
    start = time.perf_counter()
    for batch in batches:
        predict(batch)
    return time.perf_counter() - start


def time_variants(
    built: Dict[str, Dict],
    n_images: int,
    batch_size: int,
    warmup_batches: int,
    repeats: int,
) -> Dict[str, List[float]]:
    """ms per 1,000 images for each repeat, per variant (interleaved rounds)."""
    generator = torch.Generator().manual_seed(0)
    images = torch.randn((n_images, *INPUT_SHAPE), generator=generator)
    batches = list(images.split(batch_size))
    results = {name: [] for name in built}
    with torch.inference_mode():
        for name, entry in built.items():
            for batch in batches[:warmup_batches]:
                entry["predict"](batch)
        for _ in range(repeats):
            for name, entry in built.items():
                seconds = _run_once(entry["predict"], batches)
                results[name].append(seconds * 1000.0 * 1000.0 / n_images)
    return results


def cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def set_threads(threads: int) -> None:
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass  # can only be set once per process, before any parallel work


def build_rows(protocol: dict, variants: Sequence[str], timings: Dict[str, List[float]],
               built: Dict[str, Dict], settings: dict) -> List[dict]:
    base_median = statistics.median(timings["base"]) if "base" in timings else None
    rows = []
    for name in variants:
        times = timings[name]
        median = statistics.median(times)
        params = built[name]["params"]
        rows.append({
            "variant": name,
            "config_version": protocol["config_version"],
            "channels": "-".join(str(c) for c in protocol["model"]["channels"]),
            "deployed_params": params,
            "fp32_kib": round(params * 4 / 1024, 1),
            "forward_passes": built[name]["passes"],
            "ms_per_1000_median": round(median, 2),
            "ms_per_1000_min": round(min(times), 2),
            "ms_per_1000_max": round(max(times), 2),
            "time_vs_base": round(median / base_median, 2) if base_median else "",
            **settings,
        })
    return rows


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: List[dict]) -> str:
    lines = [
        "| Variant | Deployed params | FP32 (KiB) | Forward passes | CPU ms / 1,000 images | vs BASE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['deployed_params']:,} | {r['fp32_kib']} | {r['forward_passes']} "
            f"| {r['ms_per_1000_median']} ({r['ms_per_1000_min']}-{r['ms_per_1000_max']}) | {r['time_vs_base']} |"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="protocol yaml (default configs/base.yaml)")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--n-images", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=None, help="default: training.batch_size (256)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    set_threads(args.threads)
    protocol = load_protocol(args.config)
    batch_size = args.batch_size or int(protocol["training"]["batch_size"])

    built = {name: build_variant(protocol, name) for name in args.variants}
    print(f"config v{protocol['config_version']}, channels {protocol['model']['channels']}, "
          f"batch {batch_size}, threads {torch.get_num_threads()}, repeats {args.repeats}, "
          f"{args.n_images} images")
    timings = time_variants(built, args.n_images, batch_size, args.warmup_batches, args.repeats)

    settings = {
        "batch_size": batch_size,
        "threads": torch.get_num_threads(),
        "repeats": args.repeats,
        "warmup_batches": args.warmup_batches,
        "n_images": args.n_images,
        "torch": torch.__version__,
        "cpu": cpu_name(),
        "git_commit": _git_commit(),
        "date": datetime.date.today().isoformat(),
    }
    rows = build_rows(protocol, args.variants, timings, built, settings)
    write_csv(rows, args.out)
    print()
    print(markdown(rows))
    print(f"\nCPU: {settings['cpu']}, torch {settings['torch']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
