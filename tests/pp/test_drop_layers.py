import pytest

from spatialproteomics.constants import Layers


def test_drop_layers(ds_labels):
    # dropping one layer
    reduced = ds_labels.pp.drop_layers(Layers.LA_PROPERTIES)

    # ensuring that the layer was dropped
    assert Layers.LA_PROPERTIES not in reduced
    # ensuring that the props dimension was also dropped
    assert "props" not in reduced.dims

    # dropping multiple layers
    reduced = ds_labels.pp.drop_layers([Layers.LA_PROPERTIES, Layers.SEGMENTATION])
    # ensuring that the props dimension was dropped
    assert "props" not in reduced.dims
    # ensuring that x and y were not dropped, since they are still needed for the image
    assert "x" in reduced.dims
    assert "y" in reduced.dims


def test_drop_layers_nonexistent_layer(ds_labels):
    with pytest.raises(
        AssertionError, match="Some layers that you are trying to remove are not in the image container."
    ):
        ds_labels.pp.drop_layers("nonexistent_layer")


def test_drop_layers_drop_and_keep(ds_labels):
    with pytest.raises(AssertionError, match="Please provide either layers or keep."):
        ds_labels.pp.drop_layers(layers=Layers.LA_PROPERTIES, keep=Layers.LA_PROPERTIES)


def test_drop_layers_no_input(ds_labels):
    with pytest.raises(AssertionError, match="Please provide either layers or keep."):
        ds_labels.pp.drop_layers()


def test_drop_layers_keep(ds_labels):
    # keeping one layer
    reduced = ds_labels.pp.drop_layers(keep=Layers.LA_PROPERTIES)

    # ensuring that the layer was not dropped
    assert Layers.LA_PROPERTIES in reduced
    # ensuring that all other layers were dropped
    assert Layers.SEGMENTATION not in reduced
    assert Layers.IMAGE not in reduced
    assert Layers.OBS not in reduced

    # keeping multiple layers
    reduced = ds_labels.pp.drop_layers(keep=[Layers.LA_PROPERTIES, Layers.IMAGE])
    # ensuring that the layers were not dropped
    assert Layers.LA_PROPERTIES in reduced
    assert Layers.IMAGE in reduced
    # ensuring that all other layers were dropped
    assert Layers.SEGMENTATION not in reduced
    assert Layers.OBS not in reduced


def test_drop_layers_segmentation(ds_labels):
    # when dropping the segmentation, obs also should be dropped automatically
    reduced = ds_labels.pp.drop_layers(Layers.SEGMENTATION)
    assert Layers.SEGMENTATION not in reduced
    assert Layers.OBS not in reduced


def test_drop_layers_obs(ds_labels):
    # when dropping obs, the segmentation also should be dropped automatically
    reduced = ds_labels.pp.drop_layers(Layers.OBS)
    assert Layers.OBS not in reduced
    assert Layers.SEGMENTATION not in reduced
