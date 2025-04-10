import pytest

from spatialproteomics.constants import Dims, Layers, Props


def test_add_label_property(ds_labels):
    n_labels = ds_labels.sizes[Dims.LABELS]
    ds = ds_labels.la.add_label_property(["black"] * n_labels, "alternative_color")

    # checking that the property got added correctly
    assert "alternative_color" in ds.coords[Dims.LA_PROPS].values
    # checking that previous properties are still there (subset)
    assert len(set(ds_labels.coords[Dims.LA_PROPS].values) - set(ds.coords[Dims.LA_PROPS].values)) == 0

    # making sure we can't add a property that already exists
    with pytest.raises(AssertionError, match="Property alternative_color already exists."):
        ds.la.add_label_property(["black"] * n_labels, "alternative_color")

    # making sure we can't add a property with the wrong length
    with pytest.raises(AssertionError, match="The length of the array must match the number of labels."):
        ds.la.add_label_property(["black"] * 10, "alternative_color_2")


def test_set_label_name(ds_labels):
    ds = ds_labels.la.set_label_name("T", "Macrophage")
    cell_types = ds.la

    # ensuring ct1 is no longer in the label names
    assert "T" not in cell_types
    # ensuring macrophage is in the label names
    assert "Macrophage" in cell_types

    # making sure we can't input a label that doesn't exist
    with pytest.raises(AssertionError, match="Cell type Cell type 999 not found."):
        ds.la.set_label_name("Cell type 999", "Macrophage")

    # making sure we can't assign a label name that already exists
    with pytest.raises(AssertionError, match="Label name Macrophage already exists."):
        ds.la.set_label_name("B", "Macrophage")


def test_set_label_colors(ds_labels):
    ds = ds_labels.la.set_label_colors("B", "black")
    colors = ds[Layers.LA_PROPERTIES].sel(la_props=Props.COLOR).values

    # ensuring black is now a color
    assert "black" in colors


def test_set_label_colors_multiple_colors(ds_labels):
    # adding multiple colors
    ds = ds_labels.la.set_label_colors(["B", "T"], ["black", "white"])
    colors = ds[Layers.LA_PROPERTIES].sel(la_props=Props.COLOR).values
    assert "black" in colors
    assert "white" in colors

    # testing what happens if the number of colors and cell types does not match
    with pytest.raises(AssertionError, match="The number of labels and colors must be the same."):
        ds_labels.la.set_label_colors(["B", "T"], ["black"])
