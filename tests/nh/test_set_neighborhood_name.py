import pytest


def test_set_neighborhood_name(dataset_neighborhoods):
    dataset_neighborhoods.nh.set_neighborhood_name("Neighborhood 1", "Dummy Neighborhood")
    dataset_neighborhoods.nh.set_neighborhood_name(3, "Dummy Neighborhood 2")


def test_set_neighborhood_name_already_exists(dataset_neighborhoods):
    with pytest.raises(AssertionError, match="Neighborhood name Neighborhood 2 already exists."):
        dataset_neighborhoods.nh.set_neighborhood_name("Neighborhood 1", "Neighborhood 2")


def test_set_neighborhood_name_not_found(dataset_neighborhoods):
    with pytest.raises(AssertionError, match="Neighborhood Neighborhood NA not found."):
        dataset_neighborhoods.nh.set_neighborhood_name("Neighborhood NA", "Dummy Neighborhood")
