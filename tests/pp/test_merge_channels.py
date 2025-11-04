import pytest

from spatialproteomics.constants import Dims


def test_merge_channels(ds_image):
    for normalize in [True, False]:
        for method in ["sum", "mean", "max"]:
            ds = ds_image.pp.merge_channels(["DAPI", "CD4"], key_added="DAPI_CD4", normalize=normalize, method=method)
            assert (
                "DAPI_CD4" in ds.coords[Dims.CHANNELS]
            ), f"Channel 'DAPI_CD4' not found in channels after merging with method={method} and normalize={normalize}."


def test_merge_channels_invalid_key_added(ds_image):
    with pytest.raises(AssertionError, match="already exists in the object"):
        ds_image.pp.merge_channels(["DAPI", "CD4"], key_added="DAPI")


def test_merge_channels_nonexistent_channel(ds_image):
    with pytest.raises(KeyError, match="not all values found in index"):
        ds_image.pp.merge_channels(["DAPI", "Nonexistent_Channel"], key_added="DAPI_Nonexistent")


def test_merge_channels_empty_channel_list(ds_image):
    with pytest.raises(AssertionError, match="At least two channels must be provided to merge"):
        ds_image.pp.merge_channels([], key_added="Empty_Channel")


def test_merge_channels_invalid_method(ds_image):
    with pytest.raises(ValueError, match="Unknown merging method"):
        ds_image.pp.merge_channels(["DAPI", "CD4"], key_added="DAPI_CD4", method="invalid_method")
