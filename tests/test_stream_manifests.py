
"""Tests for Step 14 clean-to-corruption manifests."""

from eval.stream_manifests import build_stream_manifests
from data.prepare_mnistc import CORRUPTIONS


def test_fifteen_streams():
    streams = build_stream_manifests()

    assert len(streams) == 15
    assert set(streams) == set(CORRUPTIONS)


def test_clean_then_corrupted_order():
    streams = build_stream_manifests()

    for corruption, stream in streams.items():
        clean, corrupted = stream["segments"]

        assert stream["total_samples"] == 20_000

        assert clean["condition"] == "clean"
        assert clean["start"] == 0
        assert clean["stop_exclusive"] == 10_000

        assert corrupted["condition"] == corruption
        assert corrupted["start"] == 10_000
        assert corrupted["stop_exclusive"] == 20_000


def test_condition_aware_sample_ids():
    streams = build_stream_manifests()

    for stream in streams.values():
        clean, corrupted = stream["segments"]

        for segment in (clean, corrupted):
            assert segment["sample_id_start"] == 0
            assert segment["sample_id_stop_exclusive"] == 10_000


def test_manifest_json(tmp_path):
    from eval.stream_manifests import save_stream_manifests

    output = tmp_path / "evaluation_streams.json"
    streams = save_stream_manifests(output)

    assert output.exists()
    assert len(streams) == 15
