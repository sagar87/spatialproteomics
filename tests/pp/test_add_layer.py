import numpy as np
import pytest

from spatialproteomics.constants import Dims, Layers


def test_add_layer_2d(ds_image):
    # x and y shape of the dataset
    x, y = ds_image.sizes[Dims.X], ds_image.sizes[Dims.Y]
    # creating a dummy mask
    labels = np.ones((x, y))

    ds = ds_image.pp.add_layer(labels, key_added=Layers.MASK)
    assert Layers.MASK in ds


def test_add_layer_3d(ds_image):
    # x and y shape of the dataset
    channels, x, y = ds_image.sizes[Dims.CHANNELS], ds_image.sizes[Dims.X], ds_image.sizes[Dims.Y]

    # creating a dummy mask
    labels = np.ones((channels, x, y))
    ds = ds_image.pp.add_layer(labels, key_added=Layers.MASK)
    assert Layers.MASK in ds


def test_add_layer_wrong_dims(ds_image):
    # x and y shape of the dataset
    x, y = ds_image.sizes[Dims.X], ds_image.sizes[Dims.Y]

    # creating a dummy mask
    labels = np.ones((x, y, 3, 3))
    labels[500:1000, 700:1200] = 0

    with pytest.raises(
        AssertionError,
        match="The array to add mask must 2 or 3-dimensional.",
    ):
        ds_image.pp.add_layer(labels, key_added=Layers.MASK)


def test_add_layer_wrong_shape(ds_image):
    # creating a dummy mask
    labels = np.ones((3000, 2000))

    with pytest.raises(
        AssertionError,
        match="The shape of array does not match that of the image.",
    ):
        ds_image.pp.add_layer(labels, key_added=Layers.MASK)


def test_add_layer_key_exists(ds_image):
    # x and y shape of the dataset
    x, y = ds_image.sizes[Dims.X], ds_image.sizes[Dims.Y]

    # creating a dummy mask
    labels = np.ones((x, y))
    labels[500:1000, 700:1200] = 0

    with pytest.raises(
        AssertionError,
        match=f"Layer {Layers.MASK} already exists.",
    ):
        ds_image.pp.add_layer(labels, key_added=Layers.MASK).pp.add_layer(labels, key_added=Layers.MASK)
