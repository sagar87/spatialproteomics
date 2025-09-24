import pytest

from spatialproteomics.constants import Layers


def test_cellpose_segmentation(ds_image):
    sd_segmented = ds_image.tl.cellpose(channel="DAPI", gpu=False)
    assert Layers.SEGMENTATION not in ds_image
    assert Layers.SEGMENTATION in sd_segmented


def test_cellpose_segmentation_already_exists(ds_segmentation):
    with pytest.raises(
        KeyError,
        match=f'The key "{Layers.SEGMENTATION}" already exists.',
    ):
        ds_segmentation.tl.cellpose(key_added=Layers.SEGMENTATION, gpu=False)
