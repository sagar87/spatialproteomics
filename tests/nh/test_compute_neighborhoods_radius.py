import pytest


def test_compute_neighborhoods_radius(dataset_labeled):
    dataset_labeled.nh.compute_neighborhoods_radius()


def test_compute_neighborhoods_radius_no_labels(dataset):
    with pytest.raises(AssertionError, match="No cell type labels found in the object."):
        dataset.nh.compute_neighborhoods_radius()


def test_compute_neighborhoods_radius_negative_radius(dataset):
    with pytest.raises(AssertionError, match="Radius must be greater than 0."):
        dataset.nh.compute_neighborhoods_radius(radius=-1)
