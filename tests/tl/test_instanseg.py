import numpy as np
import pytest

from spatialproteomics.constants import Layers
from spatialproteomics.tl.utils import _instanseg_keys


def test_instanseg_segmentation_already_exists(ds_segmentation):
    with pytest.raises(
        KeyError,
        match=f'The key "{Layers.SEGMENTATION}" already exists.',
    ):
        ds_segmentation.tl.instanseg(key_added=Layers.SEGMENTATION)


def test_instanseg_keys():
    mask = np.zeros((2, 2))
    assert _instanseg_keys({"cells": mask}, "_seg") == {"cells": "_seg"}
    assert _instanseg_keys({"nuclei": mask}, "_seg") == {"nuclei": "_seg"}
    assert _instanseg_keys({"nuclei": mask, "cells": mask}, "_seg") == {"nuclei": "_seg_nuclei", "cells": "_seg"}


def test_instanseg(ds_image):
    pytest.importorskip("instanseg")
    ds = ds_image.tl.instanseg(pixel_size=0.5, device="cpu", verbosity=0)

    assert Layers.SEGMENTATION in ds
    assert Layers.OBS in ds
    assert ds[Layers.SEGMENTATION].dims == ("y", "x")
    assert ds[Layers.SEGMENTATION].values.max() > 0
    # the input is not modified
    assert Layers.SEGMENTATION not in ds_image


def test_instanseg_all_outputs(ds_image):
    pytest.importorskip("instanseg")
    ds = ds_image.tl.instanseg(channel=["DAPI", "CD4"], pixel_size=0.5, target="all_outputs", device="cpu", verbosity=0)

    assert Layers.SEGMENTATION in ds
    assert f"{Layers.SEGMENTATION}_nuclei" in ds
    assert ds[f"{Layers.SEGMENTATION}_nuclei"].dims == ("y", "x")


def test_instanseg_custom_key_tiled(ds_image):
    pytest.importorskip("instanseg")
    ds = ds_image.tl.instanseg(
        channel="DAPI",
        key_added="_nuclei",
        target="nuclei",
        pixel_size=0.5,
        method="medium",
        tile_size=64,
        device="cpu",
        verbosity=0,
    )

    assert "_nuclei" in ds
    assert Layers.SEGMENTATION not in ds
    assert ds["_nuclei"].shape == (ds_image.sizes["y"], ds_image.sizes["x"])


def test_instanseg_invalid_target(ds_image):
    pytest.importorskip("instanseg")
    with pytest.raises(ValueError, match="Target must be"):
        ds_image.tl.instanseg(target="membranes")
