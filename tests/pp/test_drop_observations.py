import pytest


def test_drop_observations_single(ds_labels):
    ds = ds_labels.pp.drop_observations("CD4_binarized")
    assert "CD4_binarized" not in ds.coords["features"].values
    assert "CD8_binarized" in ds.coords["features"].values
    assert "centroid-0" in ds.coords["features"].values
    assert "centroid-1" in ds.coords["features"].values
    # ensuring that nothing happened in-place
    assert "CD4_binarized" in ds_labels.coords["features"].values


def test_drop_observations_multi(ds_labels):
    ds = ds_labels.pp.drop_observations(["centroid-0", "centroid-1"])
    assert "CD4_binarized" in ds.coords["features"].values
    assert "CD8_binarized" in ds.coords["features"].values
    assert "centroid-0" not in ds.coords["features"].values
    assert "centroid-1" not in ds.coords["features"].values
    # ensuring that nothing happened in-place
    assert "centroid-0" in ds_labels.coords["features"].values
    assert "centroid-1" in ds_labels.coords["features"].values


def test_drop_observations_no_obs(ds_image):
    with pytest.raises(AssertionError, match="Coordinate dummy_key not found in the object."):
        ds_image.pp.drop_observations("CD4_binarized", key="dummy_key")


def test_drop_observations_wrong_key(ds_labels):
    with pytest.raises(AssertionError, match="Property dummy_property not found in the object."):
        ds_labels.pp.drop_observations("dummy_property")
