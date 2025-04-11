import pytest

from spatialproteomics.constants import Layers


def test_colorize(ds_image):
    colorized = ds_image.pl.colorize()

    # ensure a plot layer has been added
    assert Layers.PLOT in colorized


def test_colorize_plot_layer_exists(ds_image):
    with pytest.raises(AssertionError, match="A plot layer already exists."):
        ds_image.pl.colorize().pl.colorize()


def test_colorize_too_many_colors(ds_image):
    # this should run, because we have more colors than selected channels
    ds_image.pp[["CD4", "CD8"]].pl.colorize(colors=["#000000", "#e6194b", "#3cb44b", "#ffe119"])


def test_colorize_too_few_colors(ds_image):
    with pytest.raises(
        AssertionError, match="Length of colors must at least be greater or equal the number of channels of the image."
    ):
        ds_image.pp[["CD4", "CD8"]].pl.colorize(colors=["#ffe119"])


def test_colorize_corrupted_colors(ds_image):
    with pytest.raises(ValueError, match="Invalid RGBA argument"):
        ds_image.pl.colorize(colors=["red", "green", "blue", "not_a_color", "yellow"])
