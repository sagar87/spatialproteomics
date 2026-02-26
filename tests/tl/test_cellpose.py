import pytest

from spatialproteomics.constants import Layers


def test_cellpose_segmentation_already_exists(ds_segmentation):
    with pytest.raises(
        KeyError,
        match=f'The key "{Layers.SEGMENTATION}" already exists.',
    ):
        ds_segmentation.tl.cellpose(key_added=Layers.SEGMENTATION, gpu=False)
