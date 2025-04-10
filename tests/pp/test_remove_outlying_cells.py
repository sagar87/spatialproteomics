import numpy as np
import pytest

from spatialproteomics.constants import Layers


def test_remove_outlying_cells(ds_image):
    num_cells_segmentation = np.unique(ds_image[Layers.SEGMENTATION].values).shape[0] - 1
    ds_filtered = ds_image.pp.remove_outlying_cells()
    num_cells_segmentation_filtered = np.unique(ds_filtered[Layers.SEGMENTATION].values).shape[0] - 1
    assert num_cells_segmentation_filtered <= num_cells_segmentation


def test_remove_outlying_cells_wrong_threshold(ds_image):
    with pytest.raises(ValueError, match="Dilation size and threshold must be positive integers."):
        ds_image.pp.remove_outlying_cells(threshold=0)


def test_remove_outlying_cells_wrong_dilation_size(ds_image):
    with pytest.raises(ValueError, match="Dilation size and threshold must be positive integers."):
        ds_image.pp.remove_outlying_cells(dilation_size=0)
