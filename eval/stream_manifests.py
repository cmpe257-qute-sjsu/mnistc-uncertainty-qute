
"""Step 14: Deterministic clean-to-corruption stream manifests.

Manifests describe evaluation order without loading images
or running model inference.
"""

import json
from pathlib import Path

from data.prepare_mnistc import CORRUPTIONS


CLEAN_COUNT = 10_000
CORRUPTION_COUNT = 10_000


def build_stream_manifests():
    """Create 15 streams: 10k clean followed by 10k corrupted."""

    manifests = {}

    for corruption in CORRUPTIONS:
        manifests[corruption] = {
            "stream_name": f"clean_to_{corruption}",
            "total_samples": CLEAN_COUNT + CORRUPTION_COUNT,
            "segments": [
                {
                    "condition": "clean",
                    "start": 0,
                    "stop_exclusive": CLEAN_COUNT,
                    "sample_id_start": 0,
                    "sample_id_stop_exclusive": CLEAN_COUNT,
                },
                {
                    "condition": corruption,
                    "start": CLEAN_COUNT,
                    "stop_exclusive": CLEAN_COUNT + CORRUPTION_COUNT,
                    "sample_id_start": 0,
                    "sample_id_stop_exclusive": CORRUPTION_COUNT,
                },
            ],
        }

    assert len(manifests) == 15
    return manifests


def save_stream_manifests(
    output_path="data/manifests/evaluation_streams.json",
):
    """Save the evaluation stream definitions as JSON."""

    manifests = build_stream_manifests()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(manifests, f, indent=2)

    print(f"Saved: {path}")
    return manifests


if __name__ == "__main__":
    streams = save_stream_manifests()

    for name, stream in streams.items():
        assert stream["total_samples"] == 20_000
        print(name, stream["total_samples"])

    print("\nSTEP 14 STREAM MANIFESTS SUCCESSFUL")
