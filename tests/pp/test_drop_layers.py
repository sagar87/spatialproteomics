import pytest

from spatialproteomics.constants import Layers


def test_drop_layers(dataset_labeled):
    # dropping one layer
    reduced = dataset_labeled.pp.drop_layers(Layers.PROPERTIES)

    # ensuring that the layer was dropped
    assert Layers.PROPERTIES not in reduced
    # ensuring that the props dimension was also dropped
    assert "props" not in reduced.dims

    # dropping multiple layers
    reduced = dataset_labeled.pp.drop_layers([Layers.PROPERTIES, Layers.SEGMENTATION])
    # ensuring that the props dimension was dropped
    assert "props" not in reduced.dims
    # ensuring that x and y were not dropped, since they are still needed for the image
    assert "x" in reduced.dims
    assert "y" in reduced.dims


def test_drop_layers_nonexistent_layer(dataset_labeled):
    with pytest.raises(
        AssertionError, match="Some layers that you are trying to remove are not in the image container."
    ):
        dataset_labeled.pp.drop_layers("nonexistent_layer")


def test_drop_layers_drop_and_keep(dataset_labeled):
    with pytest.raises(AssertionError, match="Please provide either layers or keep."):
        dataset_labeled.pp.drop_layers(layers=Layers.PROPERTIES, keep=Layers.PROPERTIES)


def test_drop_layers_no_input(dataset_labeled):
    with pytest.raises(AssertionError, match="Please provide either layers or keep."):
        dataset_labeled.pp.drop_layers()


def test_drop_layers_keep(dataset_labeled):
    # keeping one layer
    reduced = dataset_labeled.pp.drop_layers(keep=Layers.PROPERTIES)

    # ensuring that the layer was not dropped
    assert Layers.PROPERTIES in reduced
    # ensuring that all other layers were dropped
    assert Layers.SEGMENTATION not in reduced
    assert Layers.IMAGE not in reduced
    assert Layers.OBS not in reduced

    # keeping multiple layers
    reduced = dataset_labeled.pp.drop_layers(keep=[Layers.PROPERTIES, Layers.SEGMENTATION])
    # ensuring that the layers were not dropped
    assert Layers.PROPERTIES in reduced
    assert Layers.SEGMENTATION in reduced
    # ensuring that all other layers were dropped
    assert Layers.IMAGE not in reduced
    assert Layers.OBS not in reduced
