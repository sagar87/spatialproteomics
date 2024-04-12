import numpy as np
import pytest

from spatial_data.constants import Dims


def test_label_get_item_correct_inputs(dataset_labeled):
    # test that all cell types are present
    assert np.all(dataset_labeled.coords[Dims.LABELS].values == np.arange(1, 13))
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
