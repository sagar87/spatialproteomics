import numpy as np
import pytest

from spatialproteomics.constants import Dims, Layers


def test_add_layer(dataset):
    # x and y shape of the dataset
    x, y = dataset.sizes[Dims.X], dataset.sizes[Dims.Y]
    # creating a dummy mask
    labels = np.ones((x, y))

    ds = dataset.pp.add_layer(labels, key_added=Layers.MASK)
    assert Layers.MASK in ds


def test_add_layer_wrong_dims(dataset):
    # x and y shape of the dataset
    x, y = dataset.sizes[Dims.X], dataset.sizes[Dims.Y]

    # creating a dummy mask
    labels = np.ones((x, y, 3))
    labels[500:1000, 700:1200] = 0

    with pytest.raises(
        AssertionError,
        match="The array to add mask must 2 dimensional.",
    ):
        dataset.pp.add_layer(labels, key_added=Layers.MASK)


def test_add_layer_wrong_shape(dataset):
    # creating a dummy mask
    labels = np.ones((3000, 2000))

    with pytest.raises(
        AssertionError,
        match="The shape of array does not match that of the image.",
    ):
        dataset.pp.add_layer(labels, key_added=Layers.MASK)


def test_add_layer_key_exists(dataset):
    # x and y shape of the dataset
    x, y = dataset.sizes[Dims.X], dataset.sizes[Dims.Y]

    # creating a dummy mask
    labels = np.ones((x, y))
    labels[500:1000, 700:1200] = 0

    with pytest.raises(
        AssertionError,
        match=f"Layer {Layers.MASK} already exists.",
    ):
        dataset.pp.add_layer(labels, key_added=Layers.MASK).pp.add_layer(labels, key_added=Layers.MASK)
