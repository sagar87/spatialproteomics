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


def test_merge_channels_keep_xy_range(ds_image):
    ds = ds_image.pp.merge_channels(["DAPI", "CD4"], key_added="DAPI_CD4")
    assert len(ds.y) == len(ds_image.y)
    assert len(ds.x) == len(ds_image.x)


def test_merge_channels_asymmetric_dimensions(ds_image):
    ds = ds_image.pp[1600:1650, 2100:2134]
    original_y_dim = len(ds.coords[Dims.Y])
    original_x_dim = len(ds.coords[Dims.X])

    # This should succeed without coordinate validation errors
    result = ds.pp.merge_channels(["DAPI", "CD4"], key_added="merged", method="max")

    # Verify the merged channel was added
    assert "merged" in result.coords[Dims.CHANNELS].values

    # Verify dimensions are preserved correctly (bug would swap these)
    assert len(result.coords[Dims.Y]) == original_y_dim, f"Y dimension should be {original_y_dim}"
    assert len(result.coords[Dims.X]) == original_x_dim, f"X dimension should be {original_x_dim}"

    assert result.coords[Dims.Y].values.min() == ds.coords[Dims.Y].values.min(), "min Y coordinates should be preserved"
    assert result.coords[Dims.X].values.min() == ds.coords[Dims.X].values.min(), "min X coordinates should be preserved"
    assert result.coords[Dims.Y].values.max() == ds.coords[Dims.Y].values.max(), "max Y coordinates should be preserved"
    assert result.coords[Dims.X].values.max() == ds.coords[Dims.X].values.max(), "max X coordinates should be preserved"
