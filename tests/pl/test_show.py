import pytest

from spatialproteomics.constants import Layers


def test_show(ds_labels):
    shown = ds_labels.pl.show()

    # ensure a plot layer has been added
    assert Layers.PLOT in shown

    # checking that is also works when we render all layers (intensities, labels, and segmentations)
    ds_labels.pl.show(render_image=True, render_labels=True, render_segmentation=True)


def test_show_no_rendering_element_specified(ds_image):
    with pytest.raises(AssertionError, match="No rendering element specified."):
        ds_image.pl.show(render_image=False, render_labels=False, render_segmentation=False)
