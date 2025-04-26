import pytest

from spatialproteomics.constants import Layers


def test_show(ds_labels):
    shown = ds_labels.pl.show()
    # ensure a plot layer has been added
    assert Layers.PLOT in shown
    # checking that is also works when we render all layers (intensities, labels, and segmentations)
    ds_labels.pl.show(render_image=True, render_labels=True, render_segmentation=True)


def test_show_neighborhoods(ds_neighborhoods):
    shown = ds_neighborhoods.pl.show(render_image=False, render_neighborhoods=True)
    # ensure a plot layer has been added
    assert Layers.PLOT in shown


def test_render_obs(ds_neighborhoods):
    shown = ds_neighborhoods.pl.render_obs(feature="centroid-0", cmap="plasma").pl.imshow(legend_obs=True)
    # ensure a plot layer has been added
    assert Layers.PLOT in shown


def test_annotate(ds_labels):
    shown = ds_labels.pl.annotate().pl.show(render_labels=True, label_kwargs={"alpha": 0.0})
    # ensure a plot layer has been added
    assert Layers.PLOT in shown


def test_scatter_labels(ds_labels):
    ds_labels.pl.scatter_labels(legend=False)


def test_scatter(ds_neighborhoods):
    ds_neighborhoods.pl.scatter(feature="CD4_binarized", legend=False)


def test_autocrop(ds_image):
    shown = ds_image.pl.autocrop().pl.show()
    # ensure a plot layer has been added
    assert Layers.PLOT in shown


def test_add_box(ds_image):
    shown = ds_image.pl.add_box([1620, 1640], [2140, 2180]).pl.show()
    # ensure a plot layer has been added
    assert Layers.PLOT in shown


def test_show_no_rendering_element_specified(ds_image):
    with pytest.raises(AssertionError, match="No rendering element specified."):
        ds_image.pl.show(render_image=False, render_labels=False, render_segmentation=False)
