import numpy as np
import pytest

from spatialproteomics.constants import Dims, Features, Layers


@pytest.mark.parametrize("scale", [2, 0.5])
def test_rescale_non_square_image(ds_labels, scale):
    ds = ds_labels.pp[1600:1680, 2100:2160]
    rescaled = ds.pp.rescale(scale)

    assert rescaled.sizes[Dims.Y] == round(ds.sizes[Dims.Y] * scale)
    assert rescaled.sizes[Dims.X] == round(ds.sizes[Dims.X] * scale)
    assert rescaled[Layers.IMAGE].shape == (ds.sizes[Dims.CHANNELS], rescaled.sizes[Dims.Y], rescaled.sizes[Dims.X])
    assert rescaled[Layers.IMAGE].dtype == ds[Layers.IMAGE].dtype


def test_rescale_preserves_cell_ids(ds_labels):
    rescaled = ds_labels.pp.rescale(2)

    seg = rescaled[Layers.SEGMENTATION]
    assert seg.dtype == ds_labels[Layers.SEGMENTATION].dtype
    assert set(np.unique(seg.values)) == set(np.unique(ds_labels[Layers.SEGMENTATION].values))
    # labels and other cell layers are unaffected
    assert np.all(rescaled.coords[Dims.CELLS].values == ds_labels.coords[Dims.CELLS].values)
    assert np.all(
        rescaled[Layers.OBS].sel(features=Features.LABELS) == ds_labels[Layers.OBS].sel(features=Features.LABELS)
    )


def test_rescale_keeps_coordinate_frame(ds_labels):
    rescaled = ds_labels.pp.rescale(2)

    assert rescaled.coords[Dims.X].values[0] == ds_labels.coords[Dims.X].values[0]
    assert rescaled.coords[Dims.Y].values[0] == ds_labels.coords[Dims.Y].values[0]
    assert np.allclose(np.diff(rescaled.coords[Dims.X].values), 0.5)

    # centroids are recomputed and stay (roughly) at the same positions
    old = ds_labels[Layers.OBS].sel(features=[Features.Y, Features.X]).values
    new = rescaled[Layers.OBS].sel(features=[Features.Y, Features.X]).values
    assert np.allclose(old, new, atol=0.5)
    assert not np.allclose(old, new)


def test_rescale_recomputes_area(ds_labels):
    ds = ds_labels.pp.add_observations("area")
    rescaled = ds.pp.rescale(2)

    old_area = ds[Layers.OBS].sel(features="area").values
    new_area = rescaled[Layers.OBS].sel(features="area").values
    assert np.allclose(new_area, 4 * old_area)


def test_rescale_downscaling_removes_vanished_cells(ds_labels):
    rescaled = ds_labels.pp.rescale(0.1)

    present = np.unique(rescaled[Layers.SEGMENTATION].values)
    present = present[present > 0]
    assert np.array_equal(np.sort(rescaled.coords[Dims.CELLS].values), np.sort(present))
    assert rescaled[Layers.OBS].sizes[Dims.CELLS] == len(present)
    assert rescaled[Layers.INTENSITY].sizes[Dims.CELLS] == len(present)
