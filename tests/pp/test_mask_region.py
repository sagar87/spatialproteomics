import numpy as np
import pytest

from spatialproteomics.constants import Dims, Layers


def test_mask_region(dataset):
    # x and y shape of the dataset
    x, y = dataset.sizes[Dims.X], dataset.sizes[Dims.Y]
    # creating a dummy mask
    labels = np.ones((x, y))
    labels[0:200, 0:200] = 0

    ds = dataset.pp.add_layer(labels).pp.mask_region()

    # check that there are more pixels that are 0 after masking
    assert np.sum(ds[Layers.IMAGE]) < np.sum(dataset[Layers.IMAGE])

    # check that the masked region is 0
    assert np.all(ds[Layers.IMAGE][:, 0:200, 0:200] == 0)

    # check that the unmasked region is the same
    assert np.all(ds[Layers.IMAGE][:, 200:, 200:] == dataset[Layers.IMAGE][:, 200:, 200:])


def test_mask_region_no_mask(dataset):
    # x and y shape of the dataset
    x, y = dataset.sizes[Dims.X], dataset.sizes[Dims.Y]
    # creating a dummy mask
    labels = np.ones((x, y))

    ds = dataset.pp.add_layer(labels).pp.mask_region()

    # check that there are more pixels that are 0 after masking
    assert np.all(ds[Layers.IMAGE] == dataset[Layers.IMAGE])


def test_mask_region_new_key(dataset):
    # x and y shape of the dataset
    x, y = dataset.sizes[Dims.X], dataset.sizes[Dims.Y]
    # creating a dummy mask
    labels = np.ones((x, y))
    labels[0:200, 0:200] = 0

    ds = dataset.pp.add_layer(labels).pp.mask_region(key_added="_masked_image")

    # check that the image is the same
    assert np.all(ds[Layers.IMAGE] == dataset[Layers.IMAGE])

    # check that the masked layer is different
    assert np.sum(ds["_masked_image"]) < np.sum(dataset[Layers.IMAGE])


def test_mask_region_non_binary_mask(dataset):
    # x and y shape of the dataset
    x, y = dataset.sizes[Dims.X], dataset.sizes[Dims.Y]
    # creating a dummy mask
    labels = np.ones((x, y))
    labels[0:200, 0:200] = 0
    labels[200:400, 200:400] = 2

    with pytest.raises(
        AssertionError,
        match="The mask must only contain zeroes and ones.",
    ):
        dataset.pp.add_layer(labels).pp.mask_region()


def test_mask_region_wrong_key(dataset):
    with pytest.raises(
        AssertionError,
        match=f"The key {Layers.MASK} does not exist in the object.",
    ):
        dataset.pp.mask_region()
