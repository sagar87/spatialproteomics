import numpy as np
import pytest

from spatialproteomics.constants import Layers


def test_add_channel_existing_channel(ds_image):
    new_channel_array = ds_image[Layers.IMAGE].values[0]

    # trying to add a channel that already exists
    with pytest.raises(AssertionError, match="Can't add a channel that already exists."):
        ds_image.pp.add_channel(channels="DAPI", array=new_channel_array)


def test_add_channel_wrong_shape(ds_image):
    new_channel_array = ds_image[Layers.IMAGE].values[0, :50, :]

    # trying to add a channel with wrong shape
    with pytest.raises(AssertionError, match="Dimensions of the original image and the input array do not match."):
        ds_image.pp.add_channel(channels="new_channel", array=new_channel_array)


def test_add_channel_wrong_dimensions(ds_image):
    new_channel_array = np.random.rand(1, 50, 50, 50)
    # trying to add a 4D array of channels
    with pytest.raises(AssertionError, match="Added channels must be 2D or 3D arrays."):
        ds_image.pp.add_channel(channels="new_channel", array=new_channel_array)


def test_add_channel_wrong_dtype(ds_image):
    new_channel_array = ds_image[Layers.IMAGE].values[0]
    # trying to add a channel with wrong dtype
    with pytest.raises(AssertionError, match="Added channels must be numpy arrays."):
        ds_image.pp.add_channel(channels="new_channel", array=list(new_channel_array))


def test_add_channel_too_many_dims(ds_image):
    new_channel_array = ds_image[Layers.IMAGE].values[0]
    # duplicate the array to have a shape that is inconsistent with the number of channels
    new_channel_array = np.stack([new_channel_array, new_channel_array], axis=2)

    # trying to add something where the number of channels and the array shape are inconsistent
    with pytest.raises(AssertionError, match="The length of channels must match the number of channels in array"):
        ds_image.pp.add_channel(channels="new_channel", array=new_channel_array)


def test_add_channel_too_many_markers(ds_image):
    new_channel_array = ds_image[Layers.IMAGE].values[0]

    # trying to add something where the number of channels and the array shape are inconsistent
    with pytest.raises(AssertionError, match="The length of channels must match the number of channels in array"):
        ds_image.pp.add_channel(channels=["new_channel_1", "new_channel_2"], array=new_channel_array)
