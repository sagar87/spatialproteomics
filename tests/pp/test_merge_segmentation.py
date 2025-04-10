import numpy as np
import pytest
import xarray as xr

from spatialproteomics.constants import Dims, Layers


# TODO: this file also has not yet been edited to reflect the changes in the test suite
def create_segmentation_to_merge(dataset_full, key="_segmentation_to_merge"):
    merge_array = np.zeros(dataset_full[Layers.SEGMENTATION].shape)
    merge_array = np.stack((merge_array, merge_array))
    merge_array[0, 200:300, 200:300] = 1
    merge_array[1, 350:400, 350:400] = 1

    da = xr.DataArray(
        merge_array,
        coords=[["Hoechst", "CD4"], range(500), range(500)],
        dims=[Dims.CHANNELS, Dims.X, Dims.Y],
        name=key,
    ).astype(int)

    ds = xr.merge([dataset_full.pp[["Hoechst", "CD4"]], da])

    return ds


def test_merge_segmentation(dataset_full):
    # normal case, should work
    key = "_segmentation_to_merge"
    ds = create_segmentation_to_merge(dataset_full, key=key)

    # just seeing if the method runs through without error
    # we cannot compare the number of cells here, because the merged segmentation does not get added to the dataset
    ds.pp.merge_segmentation(layer_key=key)
    ds.pp.merge_segmentation(layer_key=key, labels=["test_label_1", "test_label_2"])


def test_merge_segmentation_wrong_key(dataset_full):
    ds = create_segmentation_to_merge(dataset_full)
    with pytest.raises(
        AssertionError,
        match="The key _wrong_key does not exist in the object.",
    ):
        ds.pp.merge_segmentation(layer_key="_wrong_key")


def test_merge_segmentation_key_already_exists(dataset_full):
    key = "_segmentation_to_merge"
    ds = create_segmentation_to_merge(dataset_full, key=key)
    with pytest.raises(
        AssertionError,
        match=f"The key {key} already exists in the object.",
    ):
        ds.pp.merge_segmentation(layer_key=key, key_added=key)


def test_merge_segmentation_2d(dataset_full):
    # merging a 2D segmentation  mask should not have any effect
    merged = dataset_full.pp.merge_segmentation(layer_key=Layers.SEGMENTATION)
    assert np.all(merged["_merged_segmentation"].values == dataset_full[Layers.SEGMENTATION].values)
