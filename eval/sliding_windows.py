
"""Step 15: Deterministic sliding-window definitions."""

WINDOW_SIZE = 100
WINDOW_STRIDE = 1


def make_windows(
    total_samples,
    window_size=WINDOW_SIZE,
    stride=WINDOW_STRIDE,
):
    """Return (start, stop_exclusive) for all full windows."""
    if total_samples < 0:
        raise ValueError("total_samples must be nonnegative")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    if total_samples < window_size:
        return []

    return [
        (start, start + window_size)
        for start in range(
            0,
            total_samples - window_size + 1,
            stride,
        )
    ]


def windows_from_manifest(stream_manifest):
    """Construct windows using a Step 14 stream manifest."""
    return make_windows(stream_manifest["total_samples"])


if __name__ == "__main__":
    from eval.stream_manifests import build_stream_manifests

    manifests = build_stream_manifests()

    for name, manifest in manifests.items():
        windows = windows_from_manifest(manifest)
        assert len(windows) == 19_901
        print(name, len(windows))

    print("\nSTEP 15 SLIDING WINDOWS SUCCESSFUL")
