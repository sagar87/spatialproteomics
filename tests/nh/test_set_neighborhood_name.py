import pytest


def test_set_neighborhood_name(ds_neighborhoods):
    # single values, either as string or as integer
    ds_neighborhoods.nh.set_neighborhood_name("Neighborhood 1", "Dummy Neighborhood")
    ds_neighborhoods.nh.set_neighborhood_name(3, "Dummy Neighborhood 2")

    # lists
    ds_neighborhoods.nh.set_neighborhood_name(
        ["Neighborhood 1", "Neighborhood 2"], ["Dummy Neighborhood 3", "Dummy Neighborhood 4"]
    )
    ds_neighborhoods.nh.set_neighborhood_name([3, 4], ["Dummy Neighborhood 3", "Dummy Neighborhood 4"])

    # dict keys and values
    tmp_dict = {"Neighborhood 1": "Dummy Neighborhood 5", "Neighborhood 2": "Dummy Neighborhood 6"}
    ds_neighborhoods.nh.set_neighborhood_name(tmp_dict.keys(), tmp_dict.values())
    tmp_dict = {3: "Dummy Neighborhood 7", 4: "Dummy Neighborhood 8"}
    ds_neighborhoods.nh.set_neighborhood_name(tmp_dict.keys(), tmp_dict.values())


def test_set_neighborhood_name_different_length(ds_neighborhoods):
    with pytest.raises(AssertionError, match="Mismatch in lengths"):
        ds_neighborhoods.nh.set_neighborhood_name(["Neighborhood 1", "Neighborhood 2"], ["Dummy Neighborhood 3"])


def test_set_neighborhood_name_already_exists(ds_neighborhoods):
    with pytest.raises(AssertionError, match="already exist in the data object."):
        ds_neighborhoods.nh.set_neighborhood_name("Neighborhood 1", "Neighborhood 2")


def test_set_neighborhood_name_not_found(ds_neighborhoods):
    # string
    with pytest.raises(AssertionError, match="not found. Existing neighborhoods"):
        ds_neighborhoods.nh.set_neighborhood_name("Neighborhood NA", "Dummy Neighborhood")

    # integer
    with pytest.raises(AssertionError, match="not found. Existing neighborhoods"):
        ds_neighborhoods.nh.set_neighborhood_name(10, "Dummy Neighborhood")


def test_set_neighborhood_name_mixed_inputs(ds_neighborhoods):
    with pytest.raises(
        AssertionError, match="Neighborhoods must be provided as either strings or integers, but not mixed."
    ):
        ds_neighborhoods.nh.set_neighborhood_name(
            [3, "Neighborhood 1"], ["Dummy Neighborhood 3", "Dummy Neighborhood 4"]
        )


def test_set_neighborhood_name_int_name(ds_neighborhoods):
    with pytest.raises(AssertionError, match="Names must be provided as strings."):
        ds_neighborhoods.nh.set_neighborhood_name(3, 4)


def test_set_neighborhood_name_duplicate_names(ds_neighborhoods):
    with pytest.raises(AssertionError, match="Names must be unique."):
        ds_neighborhoods.nh.set_neighborhood_name(
            ["Neighborhood 1", "Neighborhood 2"], ["Dummy Neighborhood 3", "Dummy Neighborhood 3"]
        )
