import pytest

from spatialproteomics.constants import Layers


def test_cellpose_segmentation_already_exists(ds_segmentation):
    with pytest.raises(
        KeyError,
        match=f'The key "{Layers.SEGMENTATION}" already exists.',
    ):
        ds_segmentation.tl.cellpose(key_added=Layers.SEGMENTATION)


def test_cellpose_joint_segmentation_multichannel(ds_image):
    with pytest.raises(
        AssertionError,
        match="Joint segmentation requires exactly two channels.",
    ):
        ds_image.tl.cellpose(key_added="_dummy_key", channel_settings=[1, 2])
