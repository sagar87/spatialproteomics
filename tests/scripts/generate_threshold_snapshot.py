"""
One-time script to generate ground-truth snapshots from the current (old)
threshold implementation. Run this before switching to the new implementation:

    python tests/scripts/generate_threshold_snapshots.py

Output is saved to tests/snapshots/threshold_snapshots.json.
Commit the resulting JSON to version control.
"""

import hashlib
import json
import os

import xarray as xr

from spatialproteomics.constants import Layers

# === Load the same dataset the tests use ===
TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "..", "test_files")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "snapshots", "threshold_snapshots.json")

ds_image = xr.load_dataset(os.path.join(TEST_FILES_DIR, "ds_image.zarr"), engine="zarr")

# === Define all configurations to snapshot ===
single_configs = [
    ("CD8", dict(intensity=10)),
    ("CD8", dict(intensity=10, shift=False)),
    ("CD8", dict(quantile=0.9)),
    ("CD8", dict(quantile=0.9, shift=False)),
    ("CD4", dict(intensity=10)),
    ("CD4", dict(intensity=10, shift=False)),
    ("CD4", dict(quantile=0.9)),
    ("CD4", dict(quantile=0.9, shift=False)),
]

multi_configs = [
    (["CD8", "CD4"], dict(intensity=10)),
    (["CD8", "CD4"], dict(intensity=10, shift=False)),
    (["CD8", "CD4"], dict(quantile=0.9)),
    (["CD8", "CD4"], dict(quantile=0.9, shift=False)),
    (["CD8", "CD4"], dict(intensity=[10, 20])),
    (["CD8", "CD4"], dict(intensity=[10, 20], shift=False)),
    (["CD8", "CD4"], dict(quantile=[0.9, 0.95])),
    (["CD8", "CD4"], dict(quantile=[0.9, 0.95], shift=False)),
]

selected_channel_configs = [
    dict(intensity=10, channels="CD8"),
    dict(intensity=10, channels="CD8", shift=False),
    dict(quantile=0.9, channels="CD8"),
    dict(quantile=0.9, channels="CD8", shift=False),
]


def make_key(channels, kwargs):
    ch_str = channels if isinstance(channels, str) else "__".join(channels)
    return f"{ch_str}__{json.dumps(kwargs, sort_keys=True)}"


def snapshot_array(arr):
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "shape": list(arr.shape),
        "array_hash": hashlib.md5(arr.tobytes()).hexdigest(),
    }


# === Generate snapshots ===
snapshots = {}

for channels, kwargs in single_configs:
    print(f"Snapshotting single: {channels}, {kwargs}")
    arr = ds_image.pp[channels].pp.threshold(**kwargs)[Layers.IMAGE].values
    key = make_key(channels, kwargs)
    snapshots[key] = snapshot_array(arr)

for channels, kwargs in multi_configs:
    print(f"Snapshotting multi: {channels}, {kwargs}")
    result = ds_image.pp[channels].pp.threshold(**kwargs)
    for ch in channels:
        arr = result.pp[ch][Layers.IMAGE].values
        key = make_key(ch, kwargs)
        snapshots[key] = snapshot_array(arr)

for kwargs in selected_channel_configs:
    print(f"Snapshotting selected channel: {kwargs}")
    arr = ds_image.pp.threshold(**kwargs).pp[kwargs["channels"]][Layers.IMAGE].values
    key = make_key("full_ds", kwargs)
    snapshots[key] = snapshot_array(arr)

# === Save ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(snapshots, f, indent=2)

print(f"\nDone. Snapshots saved to {OUTPUT_PATH}")
print(f"Total configurations snapshotted: {len(snapshots)}")
