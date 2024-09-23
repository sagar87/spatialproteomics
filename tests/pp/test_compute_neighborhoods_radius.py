import pytest


def test_compute_neighborhoods_radius(dataset_labeled):
    dataset_labeled.pp.compute_neighborhoods_radius()


def test_compute_neighborhoods_radius_no_labels(dataset):
    with pytest.raises(AssertionError, match="No cell type labels found in the object."):
        dataset.pp.compute_neighborhoods_radius()


def test_compute_neighborhoods_radius_negative_radius(dataset):
    with pytest.raises(AssertionError, match="Radius must be greater than 0."):
        dataset.pp.compute_neighborhoods_radius(radius=-1)
