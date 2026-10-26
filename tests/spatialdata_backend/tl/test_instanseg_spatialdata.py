import pytest

import spatialproteomics as sp
from spatialproteomics.constants import SDLayers


def test_instanseg_segmentation_already_exists(ds_segmentation_spatialdata):
    with pytest.raises(
        AssertionError,
        match="Key segmentation already exists in spatial data object",
    ):
        sp.tl.instanseg(ds_segmentation_spatialdata, key_added=SDLayers.SEGMENTATION)


def test_instanseg(ds_image_spatialdata):
    pytest.importorskip("instanseg")
    sdata = sp.tl.instanseg(
        ds_image_spatialdata, pixel_size=0.5, target="all_outputs", device="cpu", verbosity=0, copy=True
    )

    assert SDLayers.SEGMENTATION in sdata.labels
    assert f"{SDLayers.SEGMENTATION}_nuclei" in sdata.labels
    # the original object is not modified when copy=True
    assert SDLayers.SEGMENTATION not in ds_image_spatialdata.labels
