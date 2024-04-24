import pytest

from spatial_data.constants import Dims, Layers, Props


def test_add_label_property(dataset_labeled):
    ds = dataset_labeled.la.add_label_property(["black"] * 12, "alternative_color")
    # checking that the property got added correctly
    assert "alternative_color" in ds.coords[Dims.PROPS].values
    # checking that previous properties are still there (subset)
    assert len(set(dataset_labeled.coords[Dims.PROPS].values) - set(ds.coords[Dims.PROPS].values)) == 0

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
