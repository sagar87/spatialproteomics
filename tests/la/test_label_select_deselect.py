import numpy as np
import pytest

from spatialproteomics.constants import Dims


def test_label_get_item_correct_inputs(ds_labels):
    # test that all cell types are present
    assert np.all(ds_labels.coords[Dims.LABELS].values == np.arange(1, 5))
    # indexing via integer
    assert np.all(ds_labels.la[1].coords[Dims.LABELS].values == np.array([1]))
    # indexing via list of integers
    assert np.all(ds_labels.la[[1]].coords[Dims.LABELS].values == np.array([1]))
    assert np.all(ds_labels.la[[1, 2]].coords[Dims.LABELS].values == np.array([1, 2]))
    assert np.all(ds_labels.la[[1, 3]].coords[Dims.LABELS].values == np.array([1, 3]))
    # indexing via slice
    assert np.all(ds_labels.la[1:3].coords[Dims.LABELS].values == np.array([1, 2]))
    assert np.all(ds_labels.la[:4].coords[Dims.LABELS].values == np.array([1, 2, 3]))
    assert np.all(ds_labels.la[2:].coords[Dims.LABELS].values == np.array([2, 3]))
    # indexing via string
    assert np.all(ds_labels.la["B"].coords[Dims.LABELS].values == np.array([1]))
    assert np.all(ds_labels.la["T"].coords[Dims.LABELS].values == np.array([2]))
    # indexing via List[str]
    assert np.all(ds_labels.la[["B"]].coords[Dims.LABELS].values == np.array([1]))
    assert np.all(ds_labels.la[["B", "T"]].coords[Dims.LABELS].values == np.array([1, 2]))


def test_label_get_item_wrong_inputs(ds_labels):
    # test that all cell types are present
    with pytest.raises(TypeError, match="Label indices must be valid integers"):
        ds_labels.la[1.5]

    # index with list that contains a float
    with pytest.raises(TypeError, match="Label indices must be valid integers"):
        ds_labels.la[[1.5, 3]]

    with pytest.raises(ValueError, match="Label type"):
        ds_labels.la["Cell type 13"]


def test_label_deselect(ds_labels):
    # test that all cell types are present
    assert np.all(ds_labels.coords[Dims.LABELS].values == np.arange(1, 5))
    # deselect a single label
    ds = ds_labels.la.deselect(1)
    assert np.all(ds.coords[Dims.LABELS].values == np.array([2, 3, 4]))
    # deselect multiple labels
    ds = ds_labels.la.deselect([1, 3])
    assert np.all(ds.coords[Dims.LABELS].values == np.array([2, 4]))
    # deselect a label by name
    ds = ds_labels.la.deselect("B")
    assert np.all(ds.coords[Dims.LABELS].values == np.array([2, 3, 4]))
    # deselect multiple labels by name
    ds = ds_labels.la.deselect(["B", "T"])
    assert np.all(ds.coords[Dims.LABELS].values == np.array([3, 4]))
    # indexing via slice
    ds = ds_labels.la.deselect(slice(1, 3))
    assert np.all(ds.coords[Dims.LABELS].values == np.array([3, 4]))


def test_label_deselect_wrong_input(ds_labels):
    # test that all cell types are present
    with pytest.raises(AssertionError, match="Label must be provided as slices, lists, tuple or int."):
        ds_labels.la.deselect(1.5)

    # index with list that contains a float
    with pytest.raises(AssertionError, match="All label indices must be integers or strings."):
        ds_labels.la.deselect([1.5, 3])

    with pytest.raises(ValueError, match="Label type"):
        ds_labels.la.deselect("Cell type 13")
