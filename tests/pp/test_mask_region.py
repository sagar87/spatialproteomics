import numpy as np
import pytest

from spatialproteomics.constants import Dims, Layers


def test_mask_region(ds_image):
    # x and y shape of the dataset
    x, y = ds_image.sizes[Dims.X], ds_image.sizes[Dims.Y]
    # creating a dummy mask
    labels = np.ones((x, y))
    labels[0:200, 0:200] = 0

    ds = ds_image.pp.add_layer(labels).pp.mask_region()

    # check that there are more pixels that are 0 after masking
    assert np.sum(ds[Layers.IMAGE]) < np.sum(ds_image[Layers.IMAGE])

    # check that the masked region is 0
    assert np.all(ds[Layers.IMAGE][:, 0:200, 0:200] == 0)

    # check that the unmasked region is the same
    assert np.all(ds[Layers.IMAGE][:, 200:, 200:] == ds_image[Layers.IMAGE][:, 200:, 200:])


def test_mask_region_no_mask(ds_image):
    # x and y shape of the dataset
    x, y = ds_image.sizes[Dims.X], ds_image.sizes[Dims.Y]
    # creating a dummy mask
    labels = np.ones((x, y))

    ds = ds_image.pp.add_layer(labels).pp.mask_region()

    # check that there are more pixels that are 0 after masking
    assert np.all(ds[Layers.IMAGE] == ds_image[Layers.IMAGE])


def test_mask_region_new_key(ds_image):
    # x and y shape of the dataset
    x, y = ds_image.sizes[Dims.X], ds_image.sizes[Dims.Y]
    # creating a dummy mask
    labels = np.ones((x, y))
    labels[0:200, 0:200] = 0

    ds = ds_image.pp.add_layer(labels).pp.mask_region(key_added="_masked_image")

    # check that the image is the same
    assert np.all(ds[Layers.IMAGE] == ds_image[Layers.IMAGE])

    # check that the masked layer is different
    assert np.sum(ds["_masked_image"]) < np.sum(ds_image[Layers.IMAGE])


def test_mask_region_non_binary_mask(ds_image):
    # x and y shape of the dataset
    x, y = ds_image.sizes[Dims.X], ds_image.sizes[Dims.Y]
    # creating a dummy mask
    labels = np.ones((x, y))
    labels[0:50, 0:50] = 0
    labels[50:100, 50:100] = 2

    with pytest.raises(
        AssertionError,
        match="The mask must only contain zeroes and ones.",
    ):
        ds_image.pp.add_layer(labels).pp.mask_region()


def test_mask_region_wrong_key(ds_image):
    with pytest.raises(
        AssertionError,
        match=f"The key {Layers.MASK} does not exist in the object.",
    ):
        ds_image.pp.mask_region()
