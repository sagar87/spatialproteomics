"""
Snapshot tests for the threshold method.
These tests compare the current implementation against snapshots generated
from the old implementation. To regenerate snapshots, run:

    python tests/scripts/generate_threshold_snapshots.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spatialproteomics.constants import Layers

SNAPSHOTS = json.loads((Path(__file__).parent / ".." / "snapshots" / "threshold_snapshots.json").read_text())


# === Helpers ===


def make_key(channels, kwargs):
    ch_str = channels if isinstance(channels, str) else "__".join(channels)
    return f"{ch_str}__{json.dumps(kwargs, sort_keys=True)}"


def assert_snapshot(arr, key):
    snap = SNAPSHOTS[key]
    assert (
        list(arr.shape) == snap["shape"]
    ), f"Shape mismatch for '{key}': expected {snap['shape']}, got {list(arr.shape)}"
    np.testing.assert_allclose(arr.mean(), snap["mean"], rtol=1e-5, err_msg=f"Mean mismatch for '{key}'")
    np.testing.assert_allclose(arr.min(), snap["min"], rtol=1e-5, err_msg=f"Min mismatch for '{key}'")
    np.testing.assert_allclose(arr.max(), snap["max"], rtol=1e-5, err_msg=f"Max mismatch for '{key}'")
    actual_hash = hashlib.md5(arr.tobytes()).hexdigest()
    assert (
        actual_hash == snap["array_hash"]
    ), f"Pixel-level mismatch for '{key}': array contents differ despite matching summary stats"


# === Single channel ===


@pytest.mark.parametrize(
    "channel,kwargs",
    [
        ("CD8", dict(intensity=10)),
        ("CD8", dict(intensity=10, shift=False)),
        ("CD8", dict(quantile=0.9)),
        ("CD8", dict(quantile=0.9, shift=False)),
        ("CD4", dict(intensity=10)),
        ("CD4", dict(intensity=10, shift=False)),
        ("CD4", dict(quantile=0.9)),
        ("CD4", dict(quantile=0.9, shift=False)),
    ],
)
def test_threshold_snapshot_single_channel(ds_image, channel, kwargs):
    arr = ds_image.pp[channel].pp.threshold(**kwargs)[Layers.IMAGE].values
    assert_snapshot(arr, make_key(channel, kwargs))


# === Multiple channels ===


@pytest.mark.parametrize(
    "channels,kwargs",
    [
        (["CD8", "CD4"], dict(intensity=10)),
        (["CD8", "CD4"], dict(intensity=10, shift=False)),
        (["CD8", "CD4"], dict(quantile=0.9)),
        (["CD8", "CD4"], dict(quantile=0.9, shift=False)),
        (["CD8", "CD4"], dict(intensity=[10, 20])),
        (["CD8", "CD4"], dict(intensity=[10, 20], shift=False)),
        (["CD8", "CD4"], dict(quantile=[0.9, 0.95])),
        (["CD8", "CD4"], dict(quantile=[0.9, 0.95], shift=False)),
    ],
)
def test_threshold_snapshot_multi_channel(ds_image, channels, kwargs):
    result = ds_image.pp[channels].pp.threshold(**kwargs)
    for ch in channels:
        arr = result.pp[ch][Layers.IMAGE].values
        assert_snapshot(arr, make_key(ch, kwargs))


# === Selected channels (threshold on full ds, only some channels affected) ===


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(intensity=10, channels="CD8"),
        dict(intensity=10, channels="CD8", shift=False),
        dict(quantile=0.9, channels="CD8"),
        dict(quantile=0.9, channels="CD8", shift=False),
    ],
)
def test_threshold_snapshot_selected_channel(ds_image, kwargs):
    arr = ds_image.pp.threshold(**kwargs).pp[kwargs["channels"]][Layers.IMAGE].values
    assert_snapshot(arr, make_key("full_ds", kwargs))


# === Untouched channels are not affected ===


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(intensity=10, channels="CD8"),
        dict(intensity=10, channels="CD8", shift=False),
        dict(quantile=0.9, channels="CD8"),
        dict(quantile=0.9, channels="CD8", shift=False),
    ],
)
def test_threshold_snapshot_unaffected_channel_unchanged(ds_image, kwargs):
    result = ds_image.pp.threshold(**kwargs)
    np.testing.assert_array_equal(
        result.pp["CD4"][Layers.IMAGE].values,
        ds_image.pp["CD4"][Layers.IMAGE].values,
        err_msg=f"CD4 was modified when only CD8 was thresholded (kwargs: {kwargs})",
    )
