import pytest

from spatial_proteomics.constants import Layers


def test_imshow(dataset_labeled):
    shown = dataset_labeled.pl.imshow()

    # ensure a plot layer has been added
    assert Layers.PLOT in shown

    # checking that is also works when we render all layers (intensities, labels, and segmentations)
    dataset_labeled.pl.imshow(render_intensities=True, render_labels=True, render_segmentation=True)


def test_imshow_no_rendering_element_specified(dataset):
    with pytest.raises(AssertionError, match="No rendering element specified."):
        dataset.pl.imshow(render_intensities=False, render_labels=False, render_segmentation=False)
