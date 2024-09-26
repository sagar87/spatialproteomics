import numpy as np
import pytest

from spatialproteomics.constants import Dims


def test_neighborhood_get_item_correct_inputs(dataset_neighborhoods):
    # test that all neighborhoods are present (indices start from 1 if they are provided as strings)
    assert np.all(dataset_neighborhoods.coords[Dims.NEIGHBORHOODS].values == np.arange(1, 6))
    # indexing via integer
    assert np.all(dataset_neighborhoods.nh[1].coords[Dims.NEIGHBORHOODS].values == np.array([1]))
    # indexing via list of integers
    assert np.all(dataset_neighborhoods.nh[[1]].coords[Dims.NEIGHBORHOODS].values == np.array([1]))
    assert np.all(dataset_neighborhoods.nh[[1, 2]].coords[Dims.NEIGHBORHOODS].values == np.array([1, 2]))
    assert np.all(dataset_neighborhoods.nh[[1, 3]].coords[Dims.NEIGHBORHOODS].values == np.array([1, 3]))
    # indexing via slice
    assert np.all(dataset_neighborhoods.nh[1:3].coords[Dims.NEIGHBORHOODS].values == np.array([1, 2]))
    assert np.all(dataset_neighborhoods.nh[:4].coords[Dims.NEIGHBORHOODS].values == np.array([1, 2, 3]))
    assert np.all(dataset_neighborhoods.nh[3:].coords[Dims.NEIGHBORHOODS].values == np.array([3, 4]))
    # indexing via string
    assert np.all(dataset_neighborhoods.nh["Neighborhood 1"].coords[Dims.NEIGHBORHOODS].values == np.array([1]))
    assert np.all(dataset_neighborhoods.nh["Neighborhood 4"].coords[Dims.NEIGHBORHOODS].values == np.array([4]))
    # indexing via List[str]
    assert np.all(dataset_neighborhoods.nh[["Neighborhood 1"]].coords[Dims.NEIGHBORHOODS].values == np.array([1]))
    assert np.all(
        dataset_neighborhoods.nh[["Neighborhood 1", "Neighborhood 5"]].coords[Dims.NEIGHBORHOODS].values
        == np.array([1, 5])
    )


def test_neighborhood_get_item_wrong_inputs(dataset_neighborhoods):
    # test that all cell types are present
    with pytest.raises(TypeError, match="Neighborhood indices must be valid integers"):
        dataset_neighborhoods.nh[1.5]

    # index with list that contains a float
    with pytest.raises(TypeError, match="Neighborhood indices must be valid integers"):
        dataset_neighborhoods.nh[[1.5, 3]]

    with pytest.raises(ValueError, match="Neighborhood type"):
        dataset_neighborhoods.nh["Neighborhood 13"]


def test_neighborhood_deselect(dataset_neighborhoods):
    # test that all cell types are present
    assert np.all(dataset_neighborhoods.coords[Dims.NEIGHBORHOODS].values == np.arange(1, 6))
    # deselect a single label
    ds = dataset_neighborhoods.nh.deselect(1)
    assert np.all(ds.coords[Dims.NEIGHBORHOODS].values == np.array([2, 3, 4, 5]))
    # deselect multiple labels
    ds = dataset_neighborhoods.nh.deselect([1, 3, 4])
    assert np.all(ds.coords[Dims.NEIGHBORHOODS].values == np.array([2, 5]))
    # deselect a label by name
    ds = dataset_neighborhoods.nh.deselect("Neighborhood 1")
    assert np.all(ds.coords[Dims.NEIGHBORHOODS].values == np.array([2, 3, 4, 5]))
    # deselect multiple labels by name
    ds = dataset_neighborhoods.nh.deselect(["Neighborhood 1", "Neighborhood 3", "Neighborhood 4"])
    assert np.all(ds.coords[Dims.NEIGHBORHOODS].values == np.array([2, 5]))
    # indexing via slice
    ds = dataset_neighborhoods.nh.deselect(slice(1, 4))
    assert np.all(ds.coords[Dims.NEIGHBORHOODS].values == np.array([4, 5]))


def test_neighborhood_deselect_wrong_input(dataset_neighborhoods):
    # test that all cell types are present
    with pytest.raises(AssertionError, match="Neighborhood must be provided as slices, lists, tuple or int."):
        dataset_neighborhoods.nh.deselect(1.5)

    # index with list that contains a float
    with pytest.raises(AssertionError, match="All neighborhood indices must be integers or strings."):
        dataset_neighborhoods.nh.deselect([1.5, 3])

    with pytest.raises(ValueError, match="Neighborhood Neighborhood 13 not found."):
        dataset_neighborhoods.nh.deselect("Neighborhood 13")
