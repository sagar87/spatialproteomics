import pytest

import spatialproteomics as sp
from spatialproteomics.constants import SDLayers


def test_instanseg_segmentation_already_exists(ds_segmentation_spatialdata):
    with pytest.raises(
        AssertionError,
        match="Key segmentation already exists in spatial data object",
    ):
        sp.tl.instanseg(ds_segmentation_spatialdata, key_added=SDLayers.SEGMENTATION)
