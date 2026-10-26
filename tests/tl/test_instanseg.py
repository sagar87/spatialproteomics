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


def test_instanseg_invalid_target(ds_image):
    pytest.importorskip("instanseg")
    with pytest.raises(ValueError, match="Target must be"):
        ds_image.tl.instanseg(target="membranes")
