import pytest
import xarray as xr

from spatialproteomics.constants import Layers
from spatialproteomics.container import load_image_data


def test_load_data_proper_five_channel_input(data_dic):
    dataset = load_image_data(data_dic["input"], ["Hoechst", "CD4", "CD8", "FOXP3", "BCL6"])

    assert type(dataset) is xr.Dataset
    assert Layers.IMAGE in dataset
    assert Layers.SEGMENTATION not in dataset
    assert Layers.LABELS not in dataset

    dataset = load_image_data(
        data_dic["input"],
        ["Hoechst", "CD4", "CD8", "FOXP3", "BCL6"],
        segmentation=data_dic["segmentation"],
    )

    assert type(dataset) is xr.Dataset
    assert Layers.IMAGE in dataset
    assert Layers.SEGMENTATION in dataset
    assert Layers.LABELS not in dataset


def test_load_data_proper_input_one_channel_input(data_dic):
    # load_data handles 2 dimensional data_dic
    dataset = load_image_data(
        data_dic["input"][0],
        "Hoechst",
        data_dic["segmentation"],
    )

    assert type(dataset) is xr.Dataset
    assert Layers.IMAGE in dataset
    assert Layers.SEGMENTATION in dataset
    assert Layers.OBS in dataset


def test_load_data_assertions(data_dic):
    with pytest.raises(AssertionError, match="Length of channel_coords must match"):
        load_image_data(
            data_dic["input"],
            [
                "Hoechst",
                "CD4",
                "CD8",
                "FOXP3",
            ],
        )


def test_load_data_wrong_inputs_segmentation_mask_dim_error(data_dic):
    with pytest.raises(AssertionError, match="The shape of segmentation mask"):
        load_image_data(
            data_dic["input"],
            ["Hoechst", "CD4", "CD8", "FOXP3", "BCL6"],
            data_dic["segmentation"][:300, :300],
        )
