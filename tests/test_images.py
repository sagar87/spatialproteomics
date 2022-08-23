import numpy as np
import pytest
import xarray as xr

from spatial_data.constants import Dims, Layers
from spatial_data.image import normalize


def test_image_slicing_two_coordinates(dataset):
    sub = dataset.im[0:50, 0:50]

    assert Layers.IMAGE in sub
    assert Layers.SEGMENTATION in sub
    assert Layers.COORDINATES in sub
    # assert Layers.DATA in sub

    assert ~np.all(sub[Layers.COORDINATES].loc[:, "x"] > 50)
    assert ~np.all(sub[Layers.COORDINATES].loc[:, "y"] > 50)
    # assert np.all(
    #     sub[Layers.DATA].coords["cell_idx"]
    #     == sub[Layers.COORDINATES].coords["cell_idx"]
    # )


def test_image_slicing_two_implicit_coordinate(dataset):
    sub = dataset.im[:50, :50]

    assert Layers.IMAGE in sub
    assert Layers.SEGMENTATION in sub
    assert Layers.COORDINATES in sub
    # assert Layers.DATA in sub

    assert ~np.all(sub[Layers.COORDINATES].loc[:, "x"] > 50)
    assert ~np.all(sub[Layers.COORDINATES].loc[:, "y"] > 50)
    # assert np.all(
    #     sub[Layers.DATA].coords["cell_idx"]
    #     == sub[Layers.COORDINATES].coords["cell_idx"]
    # )


def test_image_slicing_channels_with_str(dataset_full):
    sub = dataset_full.im["Hoechst", :50, :50]

    # import pdb; pdb.set_trace()

    assert Layers.IMAGE in sub
    assert "Hoechst" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" not in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub
    assert Layers.COORDINATES in sub
    # assert Layers.DATA in sub

    assert ~np.all(sub[Layers.COORDINATES].loc[:, "x"] > 50)
    assert ~np.all(sub[Layers.COORDINATES].loc[:, "y"] > 50)
    # assert np.all(
    #     sub[Layers.DATA].coords["cell_idx"]
    #     == sub[Layers.COORDINATES].coords["cell_idx"]
    # )


def test_image_slicing_channels_with_list(dataset_full):
    sub = dataset_full.im[["Hoechst", "CD4"], :50, :50]

    assert Layers.IMAGE in sub
    assert "Hoechst" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub
    assert Layers.COORDINATES in sub
    # assert Layers.DATA in sub

    assert ~np.all(sub[Layers.COORDINATES].loc[:, "x"] > 50)
    assert ~np.all(sub[Layers.COORDINATES].loc[:, "y"] > 50)
    # assert np.all(
    #     sub[Layers.DATA].coords["cell_idx"]
    #     == sub[Layers.COORDINATES].coords["cell_idx"]
    # )


def test_image_slicing_false_channel_type(dataset_full):

    with pytest.raises(
        AssertionError, match="First index must index channel coordinates."
    ):
        dataset_full.im[4, :50, :50]


def test_image_slicing_one_channel_coordinate_str(dataset_full):
    sub = dataset_full.im["Hoechst"]

    assert Layers.IMAGE in sub
    assert "Hoechst" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" not in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub
    assert Layers.COORDINATES in sub
    # assert Layers.DATA in sub


def test_image_slicing_one_channel_coordinate_list(dataset_full):
    sub = dataset_full.im[["Hoechst", "CD4"]]

    assert Layers.IMAGE in sub
    assert "Hoechst" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert "CD4" in sub[Layers.IMAGE].coords[Dims.IMAGE[0]]
    assert Layers.SEGMENTATION in sub
    assert Layers.COORDINATES in sub
    # assert Layers.DATA in sub


def test_image_normalize(dataset_full):
    normalized_image = normalize(dataset_full[Layers.IMAGE])

    assert type(normalized_image) is xr.DataArray
    assert normalized_image.shape[0] == 5
