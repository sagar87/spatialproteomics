import numpy as np

from spatialproteomics.pl.utils import _label_segmentation_mask


def test_label_segmentation_mask(test_segmentation):
    seg_mask = test_segmentation
    relabel_dict = {1: [1, 2, 3], 2: [5, 7], 3: [8, 9]}

    res = _label_segmentation_mask(seg_mask, relabel_dict)

    assert np.all(res[np.where(seg_mask == 1)] == 1)
    assert np.all(res[np.where(seg_mask == 2)] == 1)
    assert np.all(res[np.where(seg_mask == 3)] == 1)

    assert np.all(res[np.where(seg_mask == 5)] == 2)
    assert np.all(res[np.where(seg_mask == 7)] == 2)

    assert np.all(res[np.where(seg_mask == 8)] == 3)
    assert np.all(res[np.where(seg_mask == 9)] == 3)

    relabel_dict = {1: [1, 2, 3], 2: [5], 3: [8]}
    res = _label_segmentation_mask(seg_mask, relabel_dict)

    assert np.all(res[np.where(seg_mask == 1)] == 1)
    assert np.all(res[np.where(seg_mask == 2)] == 1)
    assert np.all(res[np.where(seg_mask == 3)] == 1)

    assert np.all(res[np.where(seg_mask == 5)] == 2)
    assert np.all(res[np.where(seg_mask == 7)] == 0)

    assert np.all(res[np.where(seg_mask == 8)] == 3)
    assert np.all(res[np.where(seg_mask == 9)] == 0)

    relabel_dict = {1: [1, 2, 3]}
    res = _label_segmentation_mask(seg_mask, relabel_dict)

    assert np.all(res[np.where(seg_mask == 1)] == 1)
    assert np.all(res[np.where(seg_mask == 2)] == 1)
    assert np.all(res[np.where(seg_mask == 3)] == 1)

    assert np.all(res[np.where(seg_mask == 5)] == 0)
    assert np.all(res[np.where(seg_mask == 7)] == 0)

    assert np.all(res[np.where(seg_mask == 8)] == 0)
    assert np.all(res[np.where(seg_mask == 9)] == 0)
