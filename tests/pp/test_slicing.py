import numpy as np
import pytest

from spatialproteomics.constants import Dims, Features, Layers


def test_image_slicing_two_coordinates(ds_image):
    sub = ds_image.pp[0:50, 0:50]

    assert Layers.IMAGE in sub
    assert Layers.SEGMENTATION in sub

    assert ~np.all(sub[Layers.OBS].loc[:, Features.X] > 50)
    assert ~np.all(sub[Layers.OBS].loc[:, Features.Y] > 50)


def test_image_slicing_two_implicit_coordinate(ds_image):
    sub = ds_image.pp[:50, :50]

    assert Layers.IMAGE in sub
    assert Layers.SEGMENTATION in sub

    assert ~np.all(sub[Layers.OBS].loc[:, Features.X] > 50)
    assert ~np.all(sub[Layers.OBS].loc[:, Features.Y] > 50)


def test_image_slicing_channels_with_str(ds_image):
    sub = ds_image.pp["DAPI", :50, :50]

    assert Layers.IMAGE in sub
    assert "DAPI" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" not in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub

    assert ~np.all(sub[Layers.OBS].loc[:, Features.X] > 50)
    assert ~np.all(sub[Layers.OBS].loc[:, Features.Y] > 50)


def test_image_slicing_channels_with_list(ds_image):
    sub = ds_image.pp[["DAPI", "CD4"], :50, :50]

    assert Layers.IMAGE in sub
    assert "DAPI" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub

    assert ~np.all(sub[Layers.OBS].loc[:, Features.X] > 50)
    assert ~np.all(sub[Layers.OBS].loc[:, Features.Y] > 50)


def test_image_slicing_false_channel_type(ds_image):

    with pytest.raises(AssertionError, match="First index must index channel coordinates."):
        ds_image.pp[4, :50, :50]


def test_image_slicing_one_channel_coordinate_str(ds_image):
    sub = ds_image.pp["DAPI"]

    assert Layers.IMAGE in sub
    assert "DAPI" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" not in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub


def test_image_slicing_one_channel_coordinate_list(ds_image):
    sub = ds_image.pp[["DAPI", "CD4"]]

    assert Layers.IMAGE in sub
    assert "DAPI" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub


def test_image_slicing_dict_keys(ds_image):
    sub = ds_image.pp[{"DAPI": "dummy1", "CD4": "dummy2"}.keys()]

    assert Layers.IMAGE in sub
    assert "DAPI" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub


def test_image_slicing_dict_values(ds_image):
    sub = ds_image.pp[{"dummy1": "DAPI", "dummy2": "CD4"}.values()]

    assert Layers.IMAGE in sub
    assert "DAPI" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub


def test_image_slicing_wrong_input(ds_image):
    with pytest.raises(TypeError, match="Invalid input. To subselect, you can input a string, slice, list, or tuple."):
        ds_image.pp[True]


def test_image_slicing_inconsistent_type(ds_image):
    with pytest.raises(TypeError, match="Invalid input. Found non-string elements in the list."):
        ds_image.pp[["DAPI", 3]]
