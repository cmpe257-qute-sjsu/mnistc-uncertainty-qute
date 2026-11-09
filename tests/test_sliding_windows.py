
from eval.sliding_windows import make_windows, windows_from_manifest
from eval.stream_manifests import build_stream_manifests


def test_standard_stream():
    windows = make_windows(20_000)

    assert len(windows) == 19_901
    assert windows[0] == (0, 100)
    assert windows[-1] == (19_900, 20_000)


def test_window_stride():
    assert make_windows(105, stride=2) == [
        (0, 100),
        (2, 102),
        (4, 104),
    ]


def test_short_stream():
    assert make_windows(99) == []
    assert make_windows(100) == [(0, 100)]


def test_invalid_arguments():
    import pytest

    with pytest.raises(ValueError):
        make_windows(-1)

    with pytest.raises(ValueError):
        make_windows(100, window_size=0)

    with pytest.raises(ValueError):
        make_windows(100, stride=0)


def test_manifests():
    manifests = build_stream_manifests()

    assert len(manifests) == 15

    for manifest in manifests.values():
        windows = windows_from_manifest(manifest)

        assert len(windows) == 19_901
        assert windows[0] == (0, 100)
        assert windows[-1] == (19_900, 20_000)
