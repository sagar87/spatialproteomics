import numpy as np
import pytest
import xarray as xr

from spatial_data.constants import Dims, Features, Layers
from spatial_data.pp.transforms import _normalize


def test_add_obs_centroids(dataset):
    dataset.pp.add_properties()

    assert Layers.OBS in dataset
    assert Dims.FEATURES in dataset.coords
    assert "centroid-0" in dataset[Layers.OBS].coords[Dims.FEATURES]
    assert "centroid-1" in dataset[Layers.OBS].coords[Dims.FEATURES]


def test_add_obs_append_table(dataset):
    dataset.pp.add_properties("area")

    assert Layers.OBS in dataset
    assert Dims.FEATURES in dataset.coords
    assert "centroid-1" in dataset[Layers.OBS].coords[Dims.FEATURES]
    assert "centroid-0" in dataset[Layers.OBS].coords[Dims.FEATURES]
    assert "area" not in dataset[Layers.OBS].coords[Dims.FEATURES]

    dataset = dataset.pp.add_properties("area")

    assert Layers.OBS in dataset
    assert Dims.FEATURES in dataset.coords
    assert "centroid-1" in dataset[Layers.OBS].coords[Dims.FEATURES]
    assert "centroid-0" in dataset[Layers.OBS].coords[Dims.FEATURES]
    assert "area" in dataset[Layers.OBS].coords[Dims.FEATURES]


def test_add_obs_returns_xarray(dataset):
    da = dataset.pp.add_properties("area", return_xarray=True)

    assert isinstance(da, xr.DataArray)
