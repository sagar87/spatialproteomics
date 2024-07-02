import pytest


def test_threshold_single_channel(dataset_full):
    dataset_full.pp["CD8"].pp.threshold(intensity=10)
    dataset_full.pp["CD8"].pp.threshold(quantile=0.9)


def test_threshold_multiple_channels(dataset_full):
    dataset_full.pp[["CD8", "CD4"]].pp.threshold(intensity=10)
    dataset_full.pp[["CD8", "CD4"]].pp.threshold(quantile=0.9)

    dataset_full.pp[["CD8", "CD4"]].pp.threshold(intensity=[10, 20])
    dataset_full.pp[["CD8", "CD4"]].pp.threshold(quantile=[0.9, 0.95])


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
