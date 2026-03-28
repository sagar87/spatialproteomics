import numpy as np
import pytest


def test_find_bright_cells(ds_labels):
    ds = ds_labels.pp.find_bright_cells(frac_channels=0.5, quantile=0.9)
    assert "is_bright" in ds.pp.get_layer_as_df().columns
    # check that the values are either 0 or 1
    assert set(ds.pp.get_layer_as_df()["is_bright"].values).issubset({0, 1})


def test_find_bright_cells_no_bright_cells(ds_labels):
    ds = ds_labels.pp.find_bright_cells(frac_channels=1.0, quantile=1.0)
    assert "is_bright" in ds.pp.get_layer_as_df().columns
    # check that all cells are labeled as not bright
    assert np.all(ds.pp.get_layer_as_df()["is_bright"].values == 0)


def test_find_bright_cells_all_bright_cells(ds_labels):
    ds = ds_labels.pp.find_bright_cells(frac_channels=0.0, quantile=0.0)
    assert "is_bright" in ds.pp.get_layer_as_df().columns
    # check that all cells are labeled as bright
    assert np.all(ds.pp.get_layer_as_df()["is_bright"].values == 1)


def test_find_bright_cells_nonexistent_channel(ds_labels):
    with pytest.raises(AssertionError, match="The following channels were not found"):
        ds_labels.pp.find_bright_cells(channels=["dummy_channel"])


def test_find_bright_cells_no_quantification(ds_image):
    with pytest.raises(ValueError, match="No intensity matrix found at layer"):
        ds_image.pp.find_bright_cells()


def test_find_bright_cells_invalid_frac_channels(ds_labels):
    with pytest.raises(AssertionError, match="frac_channels must be between 0 and 1."):
        ds_labels.pp.find_bright_cells(frac_channels=-0.1)
    with pytest.raises(AssertionError, match="frac_channels must be between 0 and 1."):
        ds_labels.pp.find_bright_cells(frac_channels=1.5)


def test_find_bright_cells_invalid_quantile(ds_labels):
    with pytest.raises(AssertionError, match="quantile must be between 0 and 1."):
        ds_labels.pp.find_bright_cells(quantile=-0.1)
    with pytest.raises(AssertionError, match="quantile must be between 0 and 1."):
        ds_labels.pp.find_bright_cells(quantile=1.5)
