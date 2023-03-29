import pytest

from spatial_data.constants import Dims, Layers


def test_add_label_type(dataset_full):
    ds = dataset_full.la.add_label_type("Cell type 1")

    assert "_labels" in ds
    assert "_labels" not in dataset_full
    assert Dims.LABELS in ds.coords
    assert Dims.LABELS not in dataset_full.coords

    assert ds[Layers.LABELS].sel({Dims.PROPS: "_name", Dims.LABELS: 1}).values.item() == "Cell type 1"
    assert ds[Layers.LABELS].sel({Dims.PROPS: "_color", Dims.LABELS: 1}).values.item() == "w"

    ds = ds.la.add_label_type("Cell type 2", color="k")

    assert "_labels" in ds
    assert "_labels" not in dataset_full
    assert Dims.LABELS in ds.coords
    assert Dims.LABELS not in dataset_full.coords

    assert ds[Layers.LABELS].sel({Dims.PROPS: "_name", Dims.LABELS: 2}).values.item() == "Cell type 2"
    assert ds[Layers.LABELS].sel({Dims.PROPS: "_color", Dims.LABELS: 2}).values.item() == "k"
    # import pdb; pdb.set_trace()


def test_label_no_segmentationt(dataset_segmentation):

    with pytest.raises(ValueError, match="No segmentation mask found."):
        dataset_segmentation.la.add_label_type("Cell type 1")


def test_add_duplicate_label_type(dataset_full):
    ds = dataset_full.la.add_label_type("Cell type 1")

    assert "_labels" in ds
    assert "_labels" not in dataset_full
    assert Dims.LABELS in ds.coords
    assert Dims.LABELS not in dataset_full.coords

    with pytest.raises(ValueError, match="Label type already exists."):
        ds.la.add_label_type("Cell type 1")
