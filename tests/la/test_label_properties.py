import pytest

from spatialproteomics.constants import Dims, Layers, Props


def test_add_label_property(dataset_labeled):
    ds = dataset_labeled.la.add_label_property(["black"] * 12, "alternative_color")
    # checking that the property got added correctly
    assert "alternative_color" in ds.coords[Dims.LA_PROPS].values
    # checking that previous properties are still there (subset)
    assert len(set(dataset_labeled.coords[Dims.LA_PROPS].values) - set(ds.coords[Dims.LA_PROPS].values)) == 0

    # making sure we can't add a property that already exists
    with pytest.raises(AssertionError, match="Property alternative_color already exists."):
        ds.la.add_label_property(["black"] * 12, "alternative_color")

    # making sure we can't add a property with the wrong length
    with pytest.raises(AssertionError, match="The length of the array must match the number of labels."):
        ds.la.add_label_property(["black"] * 11, "alternative_color_2")


def test_set_label_name(dataset_labeled):
    ds = dataset_labeled.la.set_label_name("Cell type 1", "Macrophage")
    cell_types = ds.la

    # ensuring ct1 is no longer in the label names
    assert "Cell type 1" not in cell_types
    # ensuring macrophage is in the label names
    assert "Macrophage" in cell_types

    # making sure we can't input a label that doesn't exist
    with pytest.raises(AssertionError, match="Cell type Cell type 999 not found."):
        ds.la.set_label_name("Cell type 999", "Macrophage")

    # making sure we can't assign a label name that already exists
    with pytest.raises(AssertionError, match="Label name Macrophage already exists."):
        ds.la.set_label_name("Cell type 2", "Macrophage")


def test_set_label_colors(dataset_labeled):
    ds = dataset_labeled.la.set_label_colors("Cell type 1", "black")
    colors = ds[Layers.LA_PROPERTIES].sel(la_props=Props.COLOR).values

    # ensuring black is now a color
    assert "black" in colors


def test_set_label_colors_multiple_colors(dataset_labeled):
    # adding multiple colors
    ds = dataset_labeled.la.set_label_colors(["Cell type 1", "Cell type 2"], ["black", "white"])
    colors = ds[Layers.LA_PROPERTIES].sel(la_props=Props.COLOR).values
    assert "black" in colors
    assert "white" in colors

    # testing what happens if the number of colors and cell types does not match
    with pytest.raises(AssertionError, match="The number of labels and colors must be the same."):
        dataset_labeled.la.set_label_colors(["Cell type 1", "Cell type 2"], ["black"])
