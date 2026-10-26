import numpy as np
import xarray as xr

from spatialproteomics.constants import Dims, Layers


def test_add_obs_centroids(ds_segmentation):
    ds_segmentation.pp.add_observations()

    assert Layers.OBS in ds_segmentation
    assert Dims.FEATURES in ds_segmentation.coords
    assert "centroid-0" in ds_segmentation[Layers.OBS].coords[Dims.FEATURES]
    assert "centroid-1" in ds_segmentation[Layers.OBS].coords[Dims.FEATURES]


def test_add_obs_append_table(ds_segmentation):
    ds_segmentation.pp.add_observations("area")

    assert Layers.OBS in ds_segmentation
    assert Dims.FEATURES in ds_segmentation.coords
    assert "centroid-1" in ds_segmentation[Layers.OBS].coords[Dims.FEATURES]
    assert "centroid-0" in ds_segmentation[Layers.OBS].coords[Dims.FEATURES]
    assert "area" not in ds_segmentation[Layers.OBS].coords[Dims.FEATURES]

    dataset = ds_segmentation.pp.add_observations("area")

    assert Layers.OBS in dataset
    assert Dims.FEATURES in dataset.coords
    assert "centroid-1" in dataset[Layers.OBS].coords[Dims.FEATURES]
    assert "centroid-0" in dataset[Layers.OBS].coords[Dims.FEATURES]
    assert "area" in dataset[Layers.OBS].coords[Dims.FEATURES]


def test_add_obs_returns_xarray(ds_segmentation):
    da = ds_segmentation.pp.add_observations("area", return_xarray=True)
    assert isinstance(da, xr.DataArray)


def test_add_obs_with_mismatching_obs_keeps_segmentation(ds_segmentation):
    # obs that do not match the number of cells in the segmentation mask are replaced
    ds = ds_segmentation.isel({Dims.CELLS: slice(0, 10)})
    assert ds.sizes[Dims.CELLS] != len(np.unique(ds[Layers.SEGMENTATION].values)) - 1

    dataset = ds.pp.add_observations("area")

    assert Layers.SEGMENTATION in dataset
    assert Layers.IMAGE in dataset
    assert Layers.OBS in dataset
    assert dataset.sizes[Dims.CELLS] == ds_segmentation.sizes[Dims.CELLS]
    assert "area" in dataset.coords[Dims.FEATURES]
    assert not np.isnan(dataset[Layers.OBS].values).any()
