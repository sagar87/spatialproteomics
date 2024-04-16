import pytest

from spatial_data.constants import Layers


def test_colorize(dataset):
    colorized = dataset.pl.colorize()

    # ensure a plot layer has been added
    assert Layers.PLOT in colorized


def test_colorize_plot_layer_exists(dataset):
    with pytest.raises(AssertionError, match="A plot layer already exists."):
        dataset.pl.colorize().pl.colorize()


def test_colorize_too_many_colors(dataset_full):
    # this should run, because we have more colors than selected channels
    dataset_full.pp[["CD4", "CD8"]].pl.colorize(colors=["#000000", "#e6194b", "#3cb44b", "#ffe119"])


def test_colorize_too_few_colors(dataset_full):
    with pytest.raises(
        AssertionError, match="Length of colors must at least be greater or equal the number of channels of the image."
    ):
        dataset_full.pp[["CD4", "CD8"]].pl.colorize(colors=["#ffe119"])


def test_colorize_corrupted_colors(dataset):
    with pytest.raises(ValueError, match="Invalid RGBA argument"):
        dataset.pl.colorize(colors=["not_a_color"])
