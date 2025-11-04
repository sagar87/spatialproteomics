import numpy as np
import pytest

from spatialproteomics.constants import Dims, Features, Layers


def test_image_slicing_two_coordinates(ds_segmentation):
    sub = ds_segmentation.pp[1600:1650, 2100:2150]

    assert Layers.IMAGE in sub
    assert Layers.SEGMENTATION in sub

    assert ~np.all(sub[Layers.OBS].loc[:, Features.X] > 1650)
    assert ~np.all(sub[Layers.OBS].loc[:, Features.Y] > 2150)


def test_image_slicing_out_of_bounds(ds_segmentation):
    with pytest.raises(
        ValueError,
        match="X_slice is out of bounds. You are trying to access coordinates 5000:6000 but the image has coordinates from 1600 to 1700.",
    ):
        ds_segmentation.pp[5000:6000, 2100:2150]


def test_image_slicing_invalid_coords(ds_segmentation):
    with pytest.raises(ValueError, match="is invalid: start"):
        ds_segmentation.pp[2000:1000, 2100:2150]


def test_image_slicing_two_implicit_coordinate(ds_segmentation):
    sub = ds_segmentation.pp[:1650, :2150]

    assert Layers.IMAGE in sub
    assert Layers.SEGMENTATION in sub

    assert ~np.all(sub[Layers.OBS].loc[:, Features.X] > 1650)
    assert ~np.all(sub[Layers.OBS].loc[:, Features.Y] > 2150)


def test_image_slicing_channels_with_str(ds_segmentation):
    sub = ds_segmentation.pp["DAPI", :1650, :2150]

    assert Layers.IMAGE in sub
    assert "DAPI" in sub[Layers.IMAGE].coords[Dims.CHANNELS]
    assert "CD4" not in sub[Layers.IMAGE].coords[Dims.CHANNELS]
    assert Layers.SEGMENTATION in sub

    assert ~np.all(sub[Layers.OBS].loc[:, Features.X] > 1650)
    assert ~np.all(sub[Layers.OBS].loc[:, Features.Y] > 2150)


def test_image_slicing_channels_with_list(ds_segmentation):
    sub = ds_segmentation.pp[["DAPI", "CD4"], :1650, :2150]

    assert Layers.IMAGE in sub
    assert "DAPI" in sub[Layers.IMAGE].coords[Dims.CHANNELS]
    assert "CD4" in sub[Layers.IMAGE].coords[Dims.CHANNELS]
    assert Layers.SEGMENTATION in sub

    assert ~np.all(sub[Layers.OBS].loc[:, Features.X] > 1650)
    assert ~np.all(sub[Layers.OBS].loc[:, Features.Y] > 2150)


def test_image_slicing_false_channel_type(ds_segmentation):
    with pytest.raises(AssertionError, match="First index must index channel coordinates."):
        ds_segmentation.pp[4, :1650, :2150]


def test_image_slicing_one_channel_coordinate_str(ds_segmentation):
    sub = ds_segmentation.pp["DAPI"]

    assert Layers.IMAGE in sub
    assert "DAPI" in sub[Layers.IMAGE].coords[Dims.CHANNELS]
    assert "CD4" not in sub[Layers.IMAGE].coords[Dims.CHANNELS]
    assert Layers.SEGMENTATION in sub


def test_image_slicing_one_channel_coordinate_list(ds_segmentation):
    sub = ds_segmentation.pp[["DAPI", "CD4"]]

    assert Layers.IMAGE in sub
    assert "DAPI" in sub[Layers.IMAGE].coords[Dims.CHANNELS]
    assert "CD4" in sub[Layers.IMAGE].coords[Dims.CHANNELS]
    assert Layers.SEGMENTATION in sub


def test_image_slicing_dict_keys(ds_segmentation):
    sub = ds_segmentation.pp[{"DAPI": "dummy1", "CD4": "dummy2"}.keys()]

    assert Layers.IMAGE in sub
    assert "DAPI" in sub[Layers.IMAGE].coords[Dims.CHANNELS]
    assert "CD4" in sub[Layers.IMAGE].coords[Dims.CHANNELS]
    assert Layers.SEGMENTATION in sub


def test_image_slicing_dict_values(ds_segmentation):
    sub = ds_segmentation.pp[{"dummy1": "DAPI", "dummy2": "CD4"}.values()]

    assert Layers.IMAGE in sub
    assert "DAPI" in sub[Layers.IMAGE].coords[Dims.CHANNELS]
    assert "CD4" in sub[Layers.IMAGE].coords[Dims.CHANNELS]
    assert Layers.SEGMENTATION in sub


def test_image_slicing_wrong_input(ds_segmentation):
    with pytest.raises(TypeError, match="Invalid input. To subselect, you can input a string, slice, list, or tuple."):
        ds_segmentation.pp[True]


def test_image_slicing_inconsistent_type(ds_segmentation):
    with pytest.raises(TypeError, match="Invalid input. Found non-string elements in the list."):
        ds_segmentation.pp[["DAPI", 3]]
