import numpy as np

from spatialproteomics.constants import Dims, Layers
from spatialproteomics.pp.utils import _relabel_cells


def test_relabel_cells_without_background():
    segmentation = np.array([[1, 1], [2, 2]])
    relabeled, mapping = _relabel_cells(segmentation)

    np.testing.assert_array_equal(relabeled, segmentation)
    assert mapping[1] == 1 and mapping[2] == 2


def test_relabel_cells_non_consecutive():
    segmentation = np.array([[0, 5, 5], [7, 0, 12], [12, 12, 0]], dtype=np.uint16)
    relabeled, mapping = _relabel_cells(segmentation)

    np.testing.assert_array_equal(relabeled, [[0, 1, 1], [2, 0, 3], [3, 3, 0]])
    assert relabeled.dtype == segmentation.dtype
    assert mapping == {0: 0, 5: 1, 7: 2, 12: 3}


def test_add_segmentation_reindex_without_background(ds_image):
    # a mask without background, where the smallest cell id is not 1
    segmentation = np.full((ds_image.sizes[Dims.Y], ds_image.sizes[Dims.X]), 3)
    segmentation[:, 50:] = 8

    ds = ds_image.pp.add_segmentation(segmentation, reindex=True)

    assert set(np.unique(ds[Layers.SEGMENTATION].values)) == {1, 2}
    np.testing.assert_array_equal(ds.coords[Dims.CELLS].values, [1, 2])
