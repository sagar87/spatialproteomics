import numpy as np
import pytest

from spatialproteomics.constants import Dims, Layers

# test that there are less cells after
# test that cells are labeled consecutively
# test that the mask is binary


def test_mask_cells(ds_image):
    # x and y shape of the dataset
    x, y = ds_image.sizes[Dims.X], ds_image.sizes[Dims.Y]
    # creating a dummy mask
    labels = np.ones((x, y))
    labels[0:200, 0:200] = 0

    ds = ds_image.pp.add_layer(labels).pp.mask_cells()

    # check that there are less cells after masking
    assert len(ds.coords[Dims.CELLS]) < len(ds_image.coords[Dims.CELLS])

    # check that cells are labeled consecutively
    assert np.all(np.diff(ds.coords[Dims.CELLS]) == 1)


def test_mask_cells_no_mask(ds_image):
    # x and y shape of the dataset
    x, y = ds_image.sizes[Dims.X], ds_image.sizes[Dims.Y]
    # creating a dummy mask
    labels = np.ones((x, y))

    ds = ds_image.pp.add_layer(labels).pp.mask_cells()

    # check that there are the same cells before and after masking
    assert len(ds.coords[Dims.CELLS]) == len(ds_image.coords[Dims.CELLS])
    # check that the labels also stayed the same
    assert np.all(ds.coords[Dims.CELLS] == ds_image.coords[Dims.CELLS])


def test_mask_cells_non_binary_mask(ds_image):
    # x and y shape of the dataset
    x, y = ds_image.sizes[Dims.X], ds_image.sizes[Dims.Y]
    # creating a dummy mask
    labels = np.ones((x, y))
    labels[0:200, 0:200] = 0
    labels[200:400, 200:400] = 2

    with pytest.raises(
        AssertionError,
        match="The mask must only contain zeroes and ones.",
    ):
        ds_image.pp.add_layer(labels).pp.mask_cells()


def test_mask_cells_wrong_key(ds_image):
    with pytest.raises(
        AssertionError,
        match=f"The key {Layers.MASK} does not exist in the object.",
    ):
        ds_image.pp.mask_cells()
