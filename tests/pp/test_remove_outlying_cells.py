import numpy as np
import pytest

from spatialproteomics.constants import Layers


def test_remove_outlying_cells(ds_segmentation):
    num_cells_segmentation = np.unique(ds_segmentation[Layers.SEGMENTATION].values).shape[0] - 1
    ds_filtered = ds_segmentation.pp.remove_outlying_cells()
    num_cells_segmentation_filtered = np.unique(ds_filtered[Layers.SEGMENTATION].values).shape[0] - 1
    assert num_cells_segmentation_filtered <= num_cells_segmentation
    # check that cell IDs are changed when reindex is True (the default)
    assert set(np.unique(ds_filtered[Layers.SEGMENTATION].values)) - {0} == set(
        range(1, num_cells_segmentation_filtered + 1)
    )


def test_remove_outlying_cells_no_reindex(ds_segmentation):
    num_cells_segmentation = np.unique(ds_segmentation[Layers.SEGMENTATION].values).shape[0] - 1
    ds_filtered = ds_segmentation.pp.remove_outlying_cells(reindex=False)
    num_cells_segmentation_filtered = np.unique(ds_filtered[Layers.SEGMENTATION].values).shape[0] - 1
    assert num_cells_segmentation_filtered <= num_cells_segmentation
    # check that cell IDs are not changed when reindex is False
    assert set(np.unique(ds_filtered[Layers.SEGMENTATION].values)) - {0} == set(
        np.unique(ds_segmentation[Layers.SEGMENTATION].values)
    ) - {0}


def test_remove_outlying_cells_wrong_threshold(ds_segmentation):
    with pytest.raises(ValueError, match="Dilation size and threshold must be positive integers."):
        ds_segmentation.pp.remove_outlying_cells(threshold=0)


def test_remove_outlying_cells_wrong_dilation_size(ds_segmentation):
    with pytest.raises(ValueError, match="Dilation size and threshold must be positive integers."):
        ds_segmentation.pp.remove_outlying_cells(dilation_size=0)
