import numpy as np
import pytest
import xarray as xr

from spatialproteomics.constants import Dims, Layers


def create_segmentation_to_merge(ds, key="_segmentation_to_merge"):
    merge_array = np.zeros(ds[Layers.SEGMENTATION].shape)
    merge_array = np.stack((merge_array, merge_array))
    merge_array[0, 50:70, 50:70] = 1
    merge_array[1, 60:90, 30:90] = 1

    x = ds.coords[Dims.X]
    y = ds.coords[Dims.Y]

    da = xr.DataArray(
        merge_array,
        coords=[["DAPI", "CD4"], x, y],
        dims=[Dims.CHANNELS, Dims.X, Dims.Y],
        name=key,
    ).astype(int)

    ds = xr.merge([ds.pp[["DAPI", "CD4"]], da])

    return ds


def test_merge_segmentation(ds_segmentation):
    # normal case, should work
    key = "_segmentation_to_merge"
    ds = create_segmentation_to_merge(ds_segmentation, key=key)

    # just seeing if the method runs through without error
    # we cannot compare the number of cells here, because the merged segmentation does not get added to the dataset
    ds.pp.merge_segmentation(layer_key=key)
    ds.pp.merge_segmentation(layer_key=key, labels=["test_label_1", "test_label_2"])


def test_merge_segmentation_wrong_key(ds_segmentation):
    ds = create_segmentation_to_merge(ds_segmentation)
    with pytest.raises(
        AssertionError,
        match="The key '_wrong_key' does not exist in the object.",
    ):
        ds.pp.merge_segmentation(layer_key="_wrong_key")


def test_merge_segmentation_key_already_exists(ds_segmentation):
    key = "_segmentation_to_merge"
    ds = create_segmentation_to_merge(ds_segmentation, key=key)
    with pytest.raises(
        AssertionError,
        match=f"The key '{key}' already exists in the object.",
    ):
        ds.pp.merge_segmentation(layer_key=key, key_added=key)


def test_merge_segmentation_2d(ds_segmentation):
    # merging a 2D segmentation  mask should not be allowed
    with pytest.raises(
        ValueError,
        match="The segmentation mask '_segmentation' must be 3D",
    ):
        ds_segmentation.pp.merge_segmentation(layer_key=Layers.SEGMENTATION)


def test_merge_segmentation_from_multiple_layers(ds_segmentation):
    # just seeing if the method runs through without error
    # we cannot compare the number of cells here, because the merged segmentation does not get added to the dataset
    ds_segmentation.pp.merge_segmentation(layer_key=["_segmentation", "_segmentation"])
    ds_segmentation.pp.merge_segmentation(
        layer_key=["_segmentation", "_segmentation"], labels=["test_label_1", "test_label_2"]
    )


def test_merge_segmentation_from_multiple_layers_incompatible_shapes(ds_segmentation):
    # if the user passes one 2D and one 3D segmentation, it should raise an error
    key = "_segmentation_to_merge"
    ds = create_segmentation_to_merge(ds_segmentation, key=key)
    with pytest.raises(
        ValueError,
        match="Segmentation mask '_segmentation_to_merge' must be 2D.",
    ):
        ds.pp.merge_segmentation(layer_key=["_segmentation", key])
