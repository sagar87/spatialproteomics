import pytest

import spatialproteomics as sp
from spatialproteomics.constants import SDLayers


def test_cellpose_segmentation_already_exists(ds_segmentation_spatialdata):
    with pytest.raises(
        AssertionError,
        match="Key segmentation already exists in spatial data object",
    ):
        sp.tl.cellpose(ds_segmentation_spatialdata, key_added=SDLayers.SEGMENTATION)


def test_cellpose_joint_segmentation_multichannel(ds_image_spatialdata):
    with pytest.raises(
        AssertionError,
        match="Joint segmentation requires exactly two channels.",
    ):
        sp.tl.cellpose(ds_image_spatialdata, key_added="_dummy_key", channel_settings=[1, 2])
