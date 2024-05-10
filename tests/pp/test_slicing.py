import numpy as np
import pytest

from spatialproteomics.constants import Dims, Features, Layers


def test_image_slicing_two_coordinates(dataset):
    sub = dataset.pp[0:50, 0:50]

    assert Layers.IMAGE in sub
    assert Layers.SEGMENTATION in sub

    assert ~np.all(sub[Layers.OBS].loc[:, Features.X] > 50)
    assert ~np.all(sub[Layers.OBS].loc[:, Features.Y] > 50)


def test_image_slicing_two_implicit_coordinate(dataset):
    sub = dataset.pp[:50, :50]

    assert Layers.IMAGE in sub
    assert Layers.SEGMENTATION in sub

    assert ~np.all(sub[Layers.OBS].loc[:, Features.X] > 50)
    assert ~np.all(sub[Layers.OBS].loc[:, Features.Y] > 50)


def test_image_slicing_channels_with_str(dataset_full):
    sub = dataset_full.pp["Hoechst", :50, :50]

    assert Layers.IMAGE in sub
    assert "Hoechst" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" not in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub

    assert ~np.all(sub[Layers.OBS].loc[:, Features.X] > 50)
    assert ~np.all(sub[Layers.OBS].loc[:, Features.Y] > 50)


def test_image_slicing_channels_with_list(dataset_full):
    sub = dataset_full.pp[["Hoechst", "CD4"], :50, :50]

    assert Layers.IMAGE in sub
    assert "Hoechst" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub

    assert ~np.all(sub[Layers.OBS].loc[:, Features.X] > 50)
    assert ~np.all(sub[Layers.OBS].loc[:, Features.Y] > 50)


def test_image_slicing_false_channel_type(dataset_full):

    with pytest.raises(AssertionError, match="First index must index channel coordinates."):
        dataset_full.pp[4, :50, :50]


def test_image_slicing_one_channel_coordinate_str(dataset_full):
    sub = dataset_full.pp["Hoechst"]

    assert Layers.IMAGE in sub
    assert "Hoechst" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" not in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub


def test_image_slicing_one_channel_coordinate_list(dataset_full):
    sub = dataset_full.pp[["Hoechst", "CD4"]]

    assert Layers.IMAGE in sub
    assert "Hoechst" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub


def test_image_slicing_dict_keys(dataset_full):
    sub = dataset_full.pp[{"Hoechst": "dummy1", "CD4": "dummy2"}.keys()]

    assert Layers.IMAGE in sub
    assert "Hoechst" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub


def test_image_slicing_dict_values(dataset_full):
    sub = dataset_full.pp[{"dummy1": "Hoechst", "dummy2": "CD4"}.values()]

    assert Layers.IMAGE in sub
    assert "Hoechst" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub
