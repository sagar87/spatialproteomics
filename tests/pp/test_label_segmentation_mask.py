import numpy as np

from spatial_data.pp.utils import _label_segmentation_mask, _remove_unlabeled_cells


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


def test_remove_unlabeled_cells(test_segmentation):
    seg_mask = test_segmentation
    # no cells should be removed
    cells = np.array([1, 2, 3, 5, 7, 8, 9])
    res = _remove_unlabeled_cells(seg_mask.copy(), cells)

    assert 1 in res
    assert 2 in res
    assert 3 in res
    assert 5 in res
    assert 7 in res
    assert 8 in res
    assert 9 in res

    # cells 5 and 7 should be removed
    cells = np.array([1, 2, 3, 8, 9])
    res = _remove_unlabeled_cells(seg_mask.copy(), cells)

    assert 1 in res
    assert 2 in res
    assert 3 in res
    assert 5 not in res
    assert 7 not in res
    assert 8 in res
    assert 9 in res

    # all cells should be removed
    cells = np.array([])
    res = _remove_unlabeled_cells(seg_mask.copy(), cells)

    assert np.all(res == 0)
