import numpy as np
import pytest

import spatialproteomics as sp
from spatialproteomics.constants import SDLayers


def test_threshold_all_channels(ds_image_spatialdata):
    # === with shift ===
    d1 = sp.pp.threshold(ds_image_spatialdata, intensity=10, copy=True)
    d2 = sp.pp.threshold(ds_image_spatialdata, quantile=0.9, copy=True)

    assert d1[SDLayers.IMAGE].values.max() == ds_image_spatialdata[SDLayers.IMAGE].values.max() - 10
    assert d1[SDLayers.IMAGE].values.min() == 0
    assert d2[SDLayers.IMAGE].values.min() == 0

    d1 = sp.pp.threshold(ds_image_spatialdata, intensity=[10, 20, 10, 20, 10], copy=True)
    d2 = sp.pp.threshold(ds_image_spatialdata, quantile=[0.9, 0.95, 0.9, 0.95, 0.9], copy=True)

    assert d1[SDLayers.IMAGE].values.max() == ds_image_spatialdata[SDLayers.IMAGE].values.max() - 10
    assert d1[SDLayers.IMAGE].values.min() == 0
    assert d2[SDLayers.IMAGE].values.min() == 0

    # === without shift ===
    d1 = sp.pp.threshold(ds_image_spatialdata, intensity=10, shift=False, copy=True)
    d2 = sp.pp.threshold(ds_image_spatialdata, quantile=0.9, shift=False, copy=True)

    # check that there are no pixels with values between 0 and 10 in the resulting image
    assert d1[SDLayers.IMAGE].values[d1[SDLayers.IMAGE].values > 0].min() >= 10

    d1 = sp.pp.threshold(ds_image_spatialdata, intensity=[10, 20, 10, 20, 10], shift=False, copy=True)
    d2 = sp.pp.threshold(ds_image_spatialdata, quantile=[0.9, 0.95, 0.9, 0.95, 0.9], shift=False, copy=True)

    # check that there are no pixels with values between 0 and 10 in the resulting image
    assert d1[SDLayers.IMAGE].values[d1[SDLayers.IMAGE].values > 0].min() >= 10


def test_threshold_too_high_intensity(ds_image_spatialdata):
    with pytest.raises(
        AssertionError,
        match="Intensity values must be smaller than the maximum intensity.",
    ):
        sp.pp.threshold(ds_image_spatialdata, intensity=100000)


def test_threshold_too_low_intensity(ds_image_spatialdata):
    with pytest.raises(
        AssertionError,
        match="Intensity values must be positive.",
    ):
        sp.pp.threshold(ds_image_spatialdata, intensity=-1)


def test_threshold_too_high_quantile(ds_image_spatialdata):
    with pytest.raises(
        AssertionError,
        match="Quantile values must be between 0 and 1.",
    ):
        sp.pp.threshold(ds_image_spatialdata, quantile=1.1)


def test_threshold_too_low_quantile(ds_image_spatialdata):
    with pytest.raises(
        AssertionError,
        match="Quantile values must be between 0 and 1.",
    ):
        sp.pp.threshold(ds_image_spatialdata, quantile=-0.1)


def test_threshold_no_threshold(ds_image_spatialdata):
    with pytest.raises(
        ValueError,
        match="Please provide a quantile or absolute intensity cut off.",
    ):
        sp.pp.threshold(ds_image_spatialdata)


def test_threshold_wrong_number_of_intensity_thresholds(ds_image_spatialdata):
    with pytest.raises(
        AssertionError,
        match="Intensity threshold must be a single value or a list of values with the same length as the number of channels.",
    ):
        sp.pp.threshold(ds_image_spatialdata, intensity=[0.8, 0.9])


def test_threshold_wrong_number_of_quantile_thresholds(ds_image_spatialdata):
    with pytest.raises(
        AssertionError,
        match="Quantile threshold must be a single value or a list of values with the same length as the number of channels.",
    ):
        sp.pp.threshold(ds_image_spatialdata, quantile=[0.8, 0.9])


def test_threshold_on_selected_channels(ds_image_spatialdata):
    cd8_img = ds_image_spatialdata.images["image"].sel({"c": "CD8"}).values
    # === with shift ===
    quantile_value = np.quantile(cd8_img, 0.9)

    d1 = sp.threshold(ds_image_spatialdata, intensity=10, channels="CD8", copy=True)
    assert ds_image_spatialdata[SDLayers.IMAGE].shape[0] == d1[SDLayers.IMAGE].shape[0]
    assert d1.images[SDLayers.IMAGE].sel({"c": "CD8"}).values.max() == cd8_img.max() - 10
    assert d1[SDLayers.IMAGE].values.min() == 0

    d1 = sp.pp.threshold(ds_image_spatialdata, quantile=0.9, channels="CD8", copy=True)
    assert ds_image_spatialdata[SDLayers.IMAGE].shape[0] == d1[SDLayers.IMAGE].shape[0]
    assert d1.images[SDLayers.IMAGE].sel({"c": "CD8"}).values.max() == cd8_img.max() - quantile_value
    assert d1[SDLayers.IMAGE].values.min() == 0

    with pytest.raises(
        AssertionError,
        match="The number of channels must match the number of intensity values.",
    ):
        sp.pp.threshold(ds_image_spatialdata, intensity=10, channels=["CD8", "CD4"], copy=True)

    with pytest.raises(
        AssertionError,
        match="The number of channels must match the number of quantile values.",
    ):
        d1 = sp.pp.threshold(ds_image_spatialdata, quantile=0.9, channels=["CD8", "CD4"], copy=True)

    # === without shift ===
    d1 = sp.pp.threshold(ds_image_spatialdata, intensity=10, channels="CD8", shift=False, copy=True)
    assert ds_image_spatialdata[SDLayers.IMAGE].shape[0] == d1[SDLayers.IMAGE].shape[0]
    assert (
        d1.images[SDLayers.IMAGE].sel({"c": "CD8"}).values[d1.images[SDLayers.IMAGE].sel({"c": "CD8"}).values > 0].min()
        >= 10
    )

    d1 = sp.pp.threshold(ds_image_spatialdata, quantile=0.9, channels="CD8", shift=False, copy=True)
    assert ds_image_spatialdata[SDLayers.IMAGE].shape[0] == d1[SDLayers.IMAGE].shape[0]
    assert (
        d1.images[SDLayers.IMAGE].sel({"c": "CD8"}).values[d1.images[SDLayers.IMAGE].sel({"c": "CD8"}).values > 0].min()
        >= quantile_value
    )

    with pytest.raises(
        AssertionError,
        match="The number of channels must match the number of intensity values.",
    ):
        sp.pp.threshold(ds_image_spatialdata, intensity=10, channels=["CD8", "CD4"], shift=False, copy=True)

    with pytest.raises(
        AssertionError,
        match="The number of channels must match the number of quantile values.",
    ):
        sp.pp.threshold(ds_image_spatialdata, quantile=0.9, channels=["CD8", "CD4"], shift=False, copy=True)


def test_thresholds_nonexistent_channel(ds_image_spatialdata):
    with pytest.raises(
        AssertionError,
        match="The following channels are not present in the image layer",
    ):
        sp.pp.threshold(ds_image_spatialdata, intensity=10, channels="dummy_channel")
    with pytest.raises(
        AssertionError,
        match="The following channels are not present in the image layer",
    ):
        sp.pp.threshold(ds_image_spatialdata, quantile=0.9, channels="dummy_channel")
    with pytest.raises(
        AssertionError,
        match="The following channels are not present in the image layer",
    ):
        sp.pp.threshold(ds_image_spatialdata, intensity=10, channels=["dummy_channel", "CD4"])
    with pytest.raises(
        AssertionError,
        match="The following channels are not present in the image layer",
    ):
        sp.pp.threshold(ds_image_spatialdata, quantile=0.9, channels=["dummy_channel", "CD4"])
