import numpy as np
import pytest
import xarray as xr

from spatialproteomics.constants import Dims, Layers


def test_add_segmentation(ds_image, ds_segmentation):
    segmentation = ds_segmentation[Layers.SEGMENTATION].values
    segmented = ds_image.pp.add_segmentation(segmentation)

    assert Layers.SEGMENTATION in segmented
    assert Layers.SEGMENTATION not in ds_image
    assert Dims.CELLS in segmented.coords
    assert Dims.CELLS not in ds_image.coords
    assert Layers.OBS in segmented


def test_add_segmentation_already_exists(ds_segmentation):
    segmentation = ds_segmentation[Layers.SEGMENTATION].values
    with pytest.raises(AssertionError, match=f'The key "{Layers.SEGMENTATION}" already exists in the object'):
        ds_segmentation.pp.add_segmentation(segmentation)


def test_add_segmentation_from_layer(ds_image, ds_segmentation):
    segmentation = ds_segmentation[Layers.SEGMENTATION].values
    da = xr.DataArray(
        segmentation,
        coords=[ds_image.coords[Dims.X].values, ds_image.coords[Dims.Y].values],
        dims=[Dims.X, Dims.Y],
        name="_segmentation_preliminary",
    ).astype(int)

    ds = xr.merge([ds_image, da])
    segmented = ds.pp.add_segmentation("_segmentation_preliminary")

    assert "_segmentation_preliminary" in segmented
    assert Layers.SEGMENTATION in segmented
    assert Layers.SEGMENTATION not in ds_image
    assert Dims.CELLS in segmented.coords
    assert Dims.CELLS not in ds_image.coords
    assert Layers.OBS in segmented


def test_add_segmentation_wrong_dims(ds_image, ds_segmentation):
    segmentation = ds_segmentation[Layers.SEGMENTATION].values
    with pytest.raises(AssertionError, match="The shape of segmentation mask"):
        ds_image.pp.add_segmentation(segmentation[:50, :50])


def test_add_segmentation_negative_values(ds_image, ds_segmentation):
    segmentation = ds_segmentation[Layers.SEGMENTATION].values.copy()
    segmentation[10, 10] = -1
    with pytest.raises(AssertionError, match="A segmentation mask may not contain negative numbers."):
        ds_image.pp.add_segmentation(segmentation)


def test_add_segmentation_reindex(ds_image, ds_segmentation):
    segmentation = ds_segmentation[Layers.SEGMENTATION].values.copy()
    segmentation[10, 10] = np.max(segmentation) + 2
    num_cells = len(np.unique(segmentation)) - 1  # -1 because of the background

    segmented = ds_image.pp.add_segmentation(segmentation, reindex=True)
    cell_labels = sorted(np.unique(segmented["_segmentation"].values))[1:]  # removing the background

    assert cell_labels == list(range(1, num_cells + 1))
    assert list(segmented.cells.values) == list(range(1, num_cells + 1))
