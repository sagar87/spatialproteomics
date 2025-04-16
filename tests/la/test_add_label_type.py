import pytest

from spatialproteomics.constants import Dims, Layers


def test_add_label_type(ds_segmentation):
    ds = ds_segmentation.la.add_label_type("Cell type 1")

    assert Layers.LA_PROPERTIES in ds
    assert Layers.LA_PROPERTIES not in ds_segmentation
    assert Dims.LABELS in ds.coords
    assert Dims.LABELS not in ds_segmentation.coords

    assert ds[Layers.LA_PROPERTIES].sel({Dims.LA_PROPS: "_name", Dims.LABELS: 1}).values.item() == "Cell type 1"
    assert ds[Layers.LA_PROPERTIES].sel({Dims.LA_PROPS: "_color", Dims.LABELS: 1}).values.item() == "w"

    ds = ds.la.add_label_type("Cell type 2", color="k")

    assert Layers.LA_PROPERTIES in ds
    assert Layers.LA_PROPERTIES not in ds_segmentation
    assert Dims.LABELS in ds.coords
    assert Dims.LABELS not in ds_segmentation.coords

    assert ds[Layers.LA_PROPERTIES].sel({Dims.LA_PROPS: "_name", Dims.LABELS: 2}).values.item() == "Cell type 2"
    assert ds[Layers.LA_PROPERTIES].sel({Dims.LA_PROPS: "_color", Dims.LABELS: 2}).values.item() == "k"


def test_label_no_segmentation(ds_image):
    with pytest.raises(ValueError, match="No segmentation mask found."):
        ds_image.la.add_label_type("Cell type 1")


def test_add_duplicate_label_type(ds_segmentation):
    ds = ds_segmentation.la.add_label_type("Cell type 1")

    assert Layers.LA_PROPERTIES in ds
    assert Layers.LA_PROPERTIES not in ds_segmentation
    assert Dims.LABELS in ds.coords
    assert Dims.LABELS not in ds_segmentation.coords

    with pytest.raises(ValueError, match="Label type already exists."):
        ds.la.add_label_type("Cell type 1")
