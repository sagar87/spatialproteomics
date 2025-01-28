import numpy as np
import pytest

from spatialproteomics.constants import Layers


def test_remove_outlying_cells(dataset):
    num_cells_segmentation = np.unique(dataset[Layers.SEGMENTATION].values).shape[0] - 1
    ds_filtered = dataset.pp.remove_outlying_cells()
    num_cells_segmentation_filtered = np.unique(ds_filtered[Layers.SEGMENTATION].values).shape[0] - 1
    assert num_cells_segmentation_filtered <= num_cells_segmentation


def test_remove_outlying_cells_wrong_threshold(dataset):
    with pytest.raises(ValueError, match="Dilation size and threshold must be positive integers."):
        dataset.pp.remove_outlying_cells(threshold=0)


def test_remove_outlying_cells_wrong_dilation_size(dataset):
    with pytest.raises(ValueError, match="Dilation size and threshold must be positive integers."):
        dataset.pp.remove_outlying_cells(dilation_size=0)
