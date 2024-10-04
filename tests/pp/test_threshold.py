import numpy as np
import pytest

from spatialproteomics.constants import Layers


def test_threshold_single_channel(dataset_full):
    # === with shift ===
    d1 = dataset_full.pp["CD8"].pp.threshold(intensity=10)
    assert d1[Layers.IMAGE].values.max() == dataset_full[Layers.IMAGE].values.max() - 10
    assert d1[Layers.IMAGE].values.min() == 0

    d1 = dataset_full.pp["CD8"].pp.threshold(quantile=0.9)
    quantile_value = np.quantile(dataset_full.pp["CD8"][Layers.IMAGE].values, 0.9)
    assert d1[Layers.IMAGE].values.max() == dataset_full[Layers.IMAGE].values.max() - quantile_value
    assert d1[Layers.IMAGE].values.min() == 0

    # === without shift ===
    d1 = dataset_full.pp["CD8"].pp.threshold(intensity=10, shift=False)
    # check that there are no pixels with values between 0 and 10 in the resulting image
    assert d1[Layers.IMAGE].values[d1[Layers.IMAGE].values > 0].min() >= 10

    d1 = dataset_full.pp["CD8"].pp.threshold(quantile=0.9, shift=False)
    # check that there are no pixels with values between 0 and the 90th percentile in the resulting image
    quantile_value = np.quantile(dataset_full.pp["CD8"][Layers.IMAGE].values, 0.9)
    assert d1[Layers.IMAGE].values[d1[Layers.IMAGE].values > 0].min() >= quantile_value


def test_threshold_multiple_channels(dataset_full):
    # === with shift ===
    d1 = dataset_full.pp[["CD8", "CD4"]].pp.threshold(intensity=10)
    d2 = dataset_full.pp[["CD8", "CD4"]].pp.threshold(quantile=0.9)
    quantile_value = np.quantile(dataset_full.pp["CD8"][Layers.IMAGE].values, 0.9)

    assert d1[Layers.IMAGE].values.max() == dataset_full[Layers.IMAGE].values.max() - 10
    assert d1[Layers.IMAGE].values.min() == 0
    assert d2[Layers.IMAGE].values.max() == dataset_full[Layers.IMAGE].values.max() - quantile_value
    assert d2[Layers.IMAGE].values.min() == 0

    d1 = dataset_full.pp[["CD8", "CD4"]].pp.threshold(intensity=[10, 20])
    d2 = dataset_full.pp[["CD8", "CD4"]].pp.threshold(quantile=[0.9, 0.95])

    assert d1.pp["CD8"][Layers.IMAGE].values.max() == dataset_full[Layers.IMAGE].values.max() - 10
    assert d1.pp["CD8"][Layers.IMAGE].values.min() == 0
    assert d2.pp["CD8"][Layers.IMAGE].values.max() == dataset_full[Layers.IMAGE].values.max() - quantile_value
    assert d2.pp["CD8"][Layers.IMAGE].values.min() == 0

    # === without shift ===
    d1 = dataset_full.pp[["CD8", "CD4"]].pp.threshold(intensity=10, shift=False)
    d2 = dataset_full.pp[["CD8", "CD4"]].pp.threshold(quantile=0.9, shift=False)

    # check that there are no pixels with values between 0 and 10 in the resulting image
    assert d1.pp["CD8"][Layers.IMAGE].values[d1.pp["CD8"][Layers.IMAGE].values > 0].min() >= 10
    # check that there are no pixels with values between 0 and the 90th percentile in the resulting image
    quantile_value = np.quantile(dataset_full.pp["CD8"][Layers.IMAGE].values, 0.9)
    assert d2.pp["CD8"][Layers.IMAGE].values[d2.pp["CD8"][Layers.IMAGE].values > 0].min() >= quantile_value

    d1 = dataset_full.pp[["CD8", "CD4"]].pp.threshold(intensity=[10, 20], shift=False)
    d2 = dataset_full.pp[["CD8", "CD4"]].pp.threshold(quantile=[0.9, 0.95], shift=False)

    # check that there are no pixels with values between 0 and 10 in the resulting image
    assert d1.pp["CD8"][Layers.IMAGE].values[d1.pp["CD8"][Layers.IMAGE].values > 0].min() >= 10
    # check that there are no pixels with values between 0 and the 90th percentile in the resulting image
    assert d2.pp["CD8"][Layers.IMAGE].values[d2.pp["CD8"][Layers.IMAGE].values > 0].min() >= quantile_value


def test_threshold_equivalence(dataset_full):
    for shift in [False, True]:
        d1 = dataset_full.pp["CD8"].pp.threshold(intensity=10, shift=shift)
        d2 = dataset_full.pp["CD8"].pp.threshold(intensity=[10], shift=shift)
        assert d1.equals(d2)

        d1 = dataset_full.pp["CD8"].pp.threshold(quantile=0.9, shift=shift)
        d2 = dataset_full.pp["CD8"].pp.threshold(quantile=[0.9], shift=shift)
        assert d1.equals(d2)

        d1 = dataset_full.pp["CD8"].pp.threshold(intensity=10, shift=shift)
        d2 = dataset_full.pp[["CD8", "CD4"]].pp.threshold(intensity=[10, 20], shift=shift)
        assert np.all(d1[Layers.IMAGE] == d2.pp["CD8"][Layers.IMAGE])

        d1 = dataset_full.pp["CD8"].pp.threshold(quantile=0.9, shift=shift)
        d2 = dataset_full.pp[["CD8", "CD4"]].pp.threshold(quantile=[0.9, 0.95], shift=shift)
        assert np.all(d1[Layers.IMAGE] == d2.pp["CD8"][Layers.IMAGE])


def test_threshold_too_high_intensity(dataset_full):
    with pytest.raises(
        AssertionError,
        match="Intensity values must be smaller than the maximum intensity.",
    ):
        dataset_full.pp["CD8"].pp.threshold(intensity=100000)


def test_threshold_too_low_intensity(dataset_full):
    with pytest.raises(
        AssertionError,
        match="Intensity values must be positive.",
    ):
        dataset_full.pp["CD8"].pp.threshold(intensity=-1)


def test_threshold_too_high_quantile(dataset_full):
    with pytest.raises(
        AssertionError,
        match="Quantile values must be between 0 and 1.",
    ):
        dataset_full.pp["CD8"].pp.threshold(quantile=1.1)


def test_threshold_too_low_quantile(dataset_full):
    with pytest.raises(
        AssertionError,
        match="Quantile values must be between 0 and 1.",
    ):
        dataset_full.pp["CD8"].pp.threshold(quantile=-0.1)


def test_threshold_no_threshold(dataset_full):
    with pytest.raises(
        ValueError,
        match="Please provide a quantile or absolute intensity cut off.",
    ):
        dataset_full.pp["CD8"].pp.threshold()


def test_threshold_wrong_number_of_intensity_thresholds(dataset_full):
    with pytest.raises(
        AssertionError,
        match="Intensity threshold must be a single value or a list of values with the same length as the number of channels.",
    ):
        dataset_full.pp["CD8"].pp.threshold(intensity=[0.8, 0.9])


def test_threshold_wrong_number_of_quantile_thresholds(dataset_full):
    with pytest.raises(
        AssertionError,
        match="Quantile threshold must be a single value or a list of values with the same length as the number of channels.",
    ):
        dataset_full.pp["CD8"].pp.threshold(quantile=[0.8, 0.9])


def test_threshold_on_selected_channels(dataset_full):
    # === with shift ===
    quantile_value = np.quantile(dataset_full.pp["CD8"][Layers.IMAGE].values, 0.9)

    d1 = dataset_full.pp.threshold(intensity=10, channels="CD8")
    assert dataset_full[Layers.IMAGE].shape[0] == d1[Layers.IMAGE].shape[0]
    assert d1.pp["CD8"][Layers.IMAGE].values.max() == dataset_full[Layers.IMAGE].values.max() - 10
    assert d1[Layers.IMAGE].values.min() == 0

    d1 = dataset_full.pp.threshold(quantile=0.9, channels="CD8")
    assert dataset_full[Layers.IMAGE].shape[0] == d1[Layers.IMAGE].shape[0]
    assert d1.pp["CD8"][Layers.IMAGE].values.max() == dataset_full[Layers.IMAGE].values.max() - quantile_value
    assert d1[Layers.IMAGE].values.min() == 0

    with pytest.raises(
        AssertionError,
        match="The number of channels must match the number of intensity values.",
    ):
        dataset_full.pp.threshold(intensity=10, channels=["CD8", "CD4"])

    with pytest.raises(
        AssertionError,
        match="The number of channels must match the number of quantile values.",
    ):
        d1 = dataset_full.pp.threshold(quantile=0.9, channels=["CD8", "CD4"])

    # === without shift ===
    d1 = dataset_full.pp.threshold(intensity=10, channels="CD8", shift=False)
    assert dataset_full[Layers.IMAGE].shape[0] == d1[Layers.IMAGE].shape[0]
    assert d1.pp["CD8"][Layers.IMAGE].values[d1.pp["CD8"][Layers.IMAGE].values > 0].min() >= 10

    d1 = dataset_full.pp.threshold(quantile=0.9, channels="CD8", shift=False)
    assert dataset_full[Layers.IMAGE].shape[0] == d1[Layers.IMAGE].shape[0]
    assert d1.pp["CD8"][Layers.IMAGE].values[d1.pp["CD8"][Layers.IMAGE].values > 0].min() >= quantile_value

    with pytest.raises(
        AssertionError,
        match="The number of channels must match the number of intensity values.",
    ):
        dataset_full.pp.threshold(intensity=10, channels=["CD8", "CD4"], shift=False)

    with pytest.raises(
        AssertionError,
        match="The number of channels must match the number of quantile values.",
    ):
        dataset_full.pp.threshold(quantile=0.9, channels=["CD8", "CD4"], shift=False)


def test_thresholds_nonexistent_channel(dataset_full):
    with pytest.raises(
        AssertionError,
        match="The following channels are not present in the image layer",
    ):
        dataset_full.pp.threshold(intensity=10, channels="dummy_channel")
    with pytest.raises(
        AssertionError,
        match="The following channels are not present in the image layer",
    ):
        dataset_full.pp.threshold(quantile=0.9, channels="dummy_channel")
    with pytest.raises(
        AssertionError,
        match="The following channels are not present in the image layer",
    ):
        dataset_full.pp.threshold(intensity=10, channels=["dummy_channel", "CD4"])
    with pytest.raises(
        AssertionError,
        match="The following channels are not present in the image layer",
    ):
        dataset_full.pp.threshold(quantile=0.9, channels=["dummy_channel", "CD4"])
