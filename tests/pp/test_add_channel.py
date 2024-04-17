import numpy as np
import pytest

from spatial_data.constants import Layers


def test_add_channel_existing_channel(dataset):
    new_channel_array = dataset[Layers.IMAGE].values[0]

    # trying to add a channel that already exists
    with pytest.raises(AssertionError, match="Can't add a channel that already exists."):
        dataset.pp.add_channel(channels="Hoechst", array=new_channel_array)


def test_add_channel_wrong_shape(dataset):
    new_channel_array = dataset[Layers.IMAGE].values[0, :50, :]

    # trying to add a channel with wrong shape
    with pytest.raises(AssertionError, match="Dimensions of the original image and the input array do not match."):
        dataset.pp.add_channel(channels="new_channel", array=new_channel_array)


def test_add_channel_wrong_dimensions(dataset):
    new_channel_array = np.random.rand(1, 50, 50, 50)
    # trying to add a 4D array of channels
    with pytest.raises(AssertionError, match="Added channels must be 2D or 3D arrays."):
        dataset.pp.add_channel(channels="new_channel", array=new_channel_array)


def test_add_channel_wrong_dtype(dataset):
    new_channel_array = dataset[Layers.IMAGE].values[0]
    # trying to add a channel with wrong dtype
    with pytest.raises(AssertionError, match="Added channels must be numpy arrays."):
        dataset.pp.add_channel(channels="new_channel", array=list(new_channel_array))


def test_add_channel_too_many_dims(dataset):
    new_channel_array = dataset[Layers.IMAGE].values[0]
    # duplicate the array to have a shape that is inconsistent with the number of channels
    new_channel_array = np.stack([new_channel_array, new_channel_array], axis=2)

    # trying to add something where the number of channels and the array shape are inconsistent
    with pytest.raises(AssertionError, match="The length of channels must match the number of channels in array"):
        dataset.pp.add_channel(channels="new_channel", array=new_channel_array)


def test_add_channel_too_many_markers(dataset):
    new_channel_array = dataset[Layers.IMAGE].values[0]

    # trying to add something where the number of channels and the array shape are inconsistent
    with pytest.raises(AssertionError, match="The length of channels must match the number of channels in array"):
        dataset.pp.add_channel(channels=["new_channel_1", "new_channel_2"], array=new_channel_array)
