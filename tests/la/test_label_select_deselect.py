import numpy as np
import pytest

from spatial_data.constants import Dims


def test_label_get_item_correct_inputs(dataset_labeled):
    # test that all cell types are present
    assert np.all(dataset_labeled.coords[Dims.LABELS].values == np.arange(0, 12))
    # indexing via integer
    assert np.all(dataset_labeled.la[1].coords[Dims.LABELS].values == np.array([1]))
    # indexing via list of integers
    assert np.all(dataset_labeled.la[[1]].coords[Dims.LABELS].values == np.array([1]))
    assert np.all(dataset_labeled.la[[1, 2]].coords[Dims.LABELS].values == np.array([1, 2]))
    assert np.all(dataset_labeled.la[[1, 3]].coords[Dims.LABELS].values == np.array([1, 3]))
    # indexing via slice
    assert np.all(dataset_labeled.la[1:3].coords[Dims.LABELS].values == np.array([1, 2]))
    assert np.all(dataset_labeled.la[:4].coords[Dims.LABELS].values == np.array([1, 2, 3]))
    assert np.all(dataset_labeled.la[9:].coords[Dims.LABELS].values == np.array([9, 10, 11]))
    # indexing via string
    assert np.all(dataset_labeled.la["Cell type 1"].coords[Dims.LABELS].values == np.array([1]))
    assert np.all(dataset_labeled.la["Cell type 5"].coords[Dims.LABELS].values == np.array([5]))
    # indexing via List[str]
    assert np.all(dataset_labeled.la[["Cell type 1"]].coords[Dims.LABELS].values == np.array([1]))
    assert np.all(dataset_labeled.la[["Cell type 1", "Cell type 5"]].coords[Dims.LABELS].values == np.array([1, 5]))


def test_label_get_item_wrong_inputs(dataset_labeled):
    # test that all cell types are present
    with pytest.raises(TypeError, match="Label indices must be valid integers"):
        dataset_labeled.la[1.5]

    # index with list that contains a float
    with pytest.raises(TypeError, match="Label indices must be valid integers"):
        dataset_labeled.la[[1.5, 3]]

    with pytest.raises(ValueError, match="Label type"):
        dataset_labeled.la["Cell type 13"]


def test_label_deselect(dataset_labeled):
    # test that all cell types are present
    assert np.all(dataset_labeled.coords[Dims.LABELS].values == np.arange(0, 12))
    # deselect a single label
    ds = dataset_labeled.la.deselect(1)
    assert np.all(ds.coords[Dims.LABELS].values == np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    # deselect multiple labels
    ds = dataset_labeled.la.deselect([1, 3, 4])
    assert np.all(ds.coords[Dims.LABELS].values == np.array([0, 2, 5, 6, 7, 8, 9, 10, 11]))
    # deselect a label by name
    ds = dataset_labeled.la.deselect("Cell type 1")
    assert np.all(ds.coords[Dims.LABELS].values == np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    # deselect multiple labels by name
    ds = dataset_labeled.la.deselect(["Cell type 1", "Cell type 3", "Cell type 4"])
    assert np.all(ds.coords[Dims.LABELS].values == np.array([0, 2, 5, 6, 7, 8, 9, 10, 11]))
    # indexing via slice
    ds = dataset_labeled.la.deselect(slice(1, 4))
    assert np.all(ds.coords[Dims.LABELS].values == np.array([0, 4, 5, 6, 7, 8, 9, 10, 11]))


def test_label_deselect_wrong_input(dataset_labeled):
    # test that all cell types are present
    with pytest.raises(AssertionError, match="Label must be provided as slices, lists, tuple or int."):
        dataset_labeled.la.deselect(1.5)

    # index with list that contains a float
    with pytest.raises(AssertionError, match="All label indices must be integers or strings."):
        dataset_labeled.la.deselect([1.5, 3])

    with pytest.raises(ValueError, match="Label type"):
        dataset_labeled.la.deselect("Cell type 13")
