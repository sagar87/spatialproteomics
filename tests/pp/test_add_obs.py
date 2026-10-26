import numpy as np
import xarray as xr

from spatialproteomics.constants import Dims, Features, Layers


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


def _scale_coords(ds):
    # coordinates that are not plain pixel indices, e.g. after reading from spatialdata with a Scale transform
    return ds.assign_coords(
        {
            Dims.X: 0.5 * np.arange(ds.sizes[Dims.X]) + 10,
            Dims.Y: 0.5 * np.arange(ds.sizes[Dims.Y]) + 20,
        }
    )


def test_add_obs_centroids_with_scaled_coordinates(ds_segmentation):
    ds_pixel = ds_segmentation.pp.drop_observations(["centroid-0", "centroid-1"])
    ds_pixel = ds_pixel.assign_coords(
        {Dims.X: np.arange(ds_pixel.sizes[Dims.X]), Dims.Y: np.arange(ds_pixel.sizes[Dims.Y])}
    )
    pixel_centroids = ds_pixel.pp.add_observations()

    ds = _scale_coords(ds_segmentation.pp.drop_observations(["centroid-0", "centroid-1"])).pp.add_observations()
    x = ds[Layers.OBS].sel(features=Features.X).values
    y = ds[Layers.OBS].sel(features=Features.Y).values

    # the centroids lie within the coordinates
    assert x.min() >= ds.coords[Dims.X].values.min() and x.max() <= ds.coords[Dims.X].values.max()
    assert y.min() >= ds.coords[Dims.Y].values.min() and y.max() <= ds.coords[Dims.Y].values.max()
    # and are the pixel centroids mapped onto the coordinates
    assert np.allclose(x, 0.5 * pixel_centroids[Layers.OBS].sel(features=Features.X).values + 10)
    assert np.allclose(y, 0.5 * pixel_centroids[Layers.OBS].sel(features=Features.Y).values + 20)


def test_get_bbox_with_scaled_coordinates(ds_segmentation):
    ds = ds_segmentation.pp.drop_observations(["centroid-0", "centroid-1"]).pp.add_observations()
    ds_scaled = _scale_coords(ds_segmentation.pp.drop_observations(["centroid-0", "centroid-1"])).pp.add_observations()

    # the same region in pixel and in scaled coordinates
    cropped = ds.pp[1620:1660, 2120:2160]
    cropped_scaled = ds_scaled.pp[20:40, 30:50]

    assert np.array_equal(cropped.coords[Dims.CELLS].values, cropped_scaled.coords[Dims.CELLS].values)


def test_add_obs_from_dataframe_does_not_modify_accessor(ds_segmentation):
    # xarray caches the accessor of a dataset, so the method must not replace the accessor's object
    import pandas as pd

    ds = ds_segmentation.drop_dims(["cells", "features"])
    num_cells = len(ds_segmentation.coords["cells"])
    df = pd.DataFrame({"feature": range(num_cells)})
    ds.pp.add_obs_from_dataframe(df)

    assert ds.pp._obj is ds
    assert Layers.OBS not in ds.pp._obj
