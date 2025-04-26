import pytest
import xarray as xr

from spatialproteomics.constants import Layers
from spatialproteomics.container import load_image_data, read_from_spatialdata


def test_load_data_proper_five_channel_input(data_dic):
    dataset = load_image_data(data_dic["image"], ["DAPI", "PAX5", "CD3", "CD4", "CD8"])

    assert type(dataset) is xr.Dataset
    assert Layers.IMAGE in dataset
    assert Layers.SEGMENTATION not in dataset
    assert Layers.LA_PROPERTIES not in dataset

    dataset = load_image_data(
        data_dic["image"],
        ["DAPI", "PAX5", "CD3", "CD4", "CD8"],
        segmentation=data_dic["segmentation"],
    )

    assert type(dataset) is xr.Dataset
    assert Layers.IMAGE in dataset
    assert Layers.SEGMENTATION in dataset
    assert Layers.LA_PROPERTIES not in dataset


def test_load_data_proper_input_one_channel_input(data_dic):
    # load_data handles 2 dimensional data_dic
    dataset = load_image_data(
        data_dic["image"][0],
        "DAPI",
        segmentation=data_dic["segmentation"],
    )

    assert type(dataset) is xr.Dataset
    assert Layers.IMAGE in dataset
    assert Layers.SEGMENTATION in dataset
    assert Layers.OBS in dataset


def test_load_data_assertions(data_dic):
    with pytest.raises(AssertionError, match="Length of channel_coords must match"):
        load_image_data(
            data_dic["image"],
            ["DAPI", "PAX5", "CD3", "CD4"],
        )


def test_load_data_wrong_inputs_segmentation_mask_dim_error(data_dic):
    with pytest.raises(AssertionError, match="The shape of segmentation mask"):
        load_image_data(
            data_dic["image"],
            ["DAPI", "PAX5", "CD3", "CD4", "CD8"],
            segmentation=data_dic["segmentation"][:50, :50],
        )


def test_read_from_spatialdata(ds_neighborhoods_spatialdata):
    read_from_spatialdata(ds_neighborhoods_spatialdata)
