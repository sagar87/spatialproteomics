import numpy as np

from spatial_data.pp.utils import _remove_unlabeled_cells


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


def test_quantification(dataset):

    pass
    # cell_idx = np.unique(dataset[Layers.SEGMENTATION])
    # test_cell = np.random.choice(cell_idx[cell_idx > 0])

    # x = (
    #     dataset[Layers.IMAGE]
    #     .loc["Hoechst"]
    #     .values[dataset[Layers.SEGMENTATION] == test_cell]
    #     .sum()
    # )
    # y = dataset[Layers.DATA].loc[test_cell, "Hoechst"].values

    # assert x == y
