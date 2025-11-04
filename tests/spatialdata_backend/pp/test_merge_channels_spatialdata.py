import pytest

import spatialproteomics as sp


def test_merge_channels(ds_image_spatialdata):
    for normalize in [True, False]:
        for method in ["sum", "mean", "max"]:
            sdata_merged = sp.pp.merge_channels(
                ds_image_spatialdata,
                ["DAPI", "CD4"],
                key_added="DAPI_CD4",
                normalize=normalize,
                method=method,
                copy=True,
            )
            assert (
                "DAPI_CD4" in sdata_merged.images["image"].coords["c"].values
            ), f"Channel 'DAPI_CD4' not found in channels after merging with method={method} and normalize={normalize}."


def test_merge_channels_invalid_key_added(ds_image_spatialdata):
    with pytest.raises(AssertionError, match="already exists in the object"):
        sp.pp.merge_channels(ds_image_spatialdata, ["DAPI", "CD4"], key_added="DAPI", copy=True)


def test_merge_channels_nonexistent_channel(ds_image_spatialdata):
    with pytest.raises(KeyError, match="not all values found in index"):
        sp.pp.merge_channels(
            ds_image_spatialdata, ["DAPI", "Nonexistent_Channel"], key_added="DAPI_Nonexistent", copy=True
        )


def test_merge_channels_empty_channel_list(ds_image_spatialdata):
    with pytest.raises(AssertionError, match="At least two channels must be provided to merge"):
        sp.pp.merge_channels(ds_image_spatialdata, [], key_added="Empty_Channel", copy=True)


def test_merge_channels_invalid_method(ds_image_spatialdata):
    with pytest.raises(ValueError, match="Unknown merging method"):
        sp.pp.merge_channels(
            ds_image_spatialdata, ["DAPI", "CD4"], key_added="DAPI_CD4", method="invalid_method", copy=True
        )
