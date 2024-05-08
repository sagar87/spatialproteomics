import pytest

from spatial_proteomics.constants import Layers


def test_drop_layers(dataset_labeled):
    # dropping one layer
    reduced = dataset_labeled.pp.drop_layers(Layers.LABELS)

    # ensuring that the layer was dropped
    assert Layers.LABELS not in reduced
    # ensuring that the props dimension was also dropped
    assert "props" not in reduced.dims

    # dropping multiple layers
    reduced = dataset_labeled.pp.drop_layers([Layers.LABELS, Layers.SEGMENTATION])
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
