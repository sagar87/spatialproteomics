import pytest

from spatialproteomics.constants import Layers


def test_show(dataset_labeled):
    shown = dataset_labeled.pl.show()

    # ensure a plot layer has been added
    assert Layers.PLOT in shown

    # checking that is also works when we render all layers (intensities, labels, and segmentations)
    dataset_labeled.pl.show(render_image=True, render_labels=True, render_segmentation=True)


def test_show_no_rendering_element_specified(dataset):
    with pytest.raises(AssertionError, match="No rendering element specified."):
        dataset.pl.show(render_image=False, render_labels=False, render_segmentation=False)
