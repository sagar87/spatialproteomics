import numpy as np
import pytest

from spatialproteomics.constants import Dims, Layers


def test_filter_by_obs(ds_segmentation):
    filtered = ds_segmentation.pp.add_observations("area").pp.filter_by_obs("area", func=lambda x: (x > 50) & (x < 100))

    # obs are retained after filtering
    assert Layers.OBS in filtered
    # size is smaller than before filtering
    assert len(filtered[Layers.OBS]) < len(ds_segmentation[Layers.OBS])

    # coords are synchronized with the segmentation mask
    assert filtered.sizes[Dims.CELLS] == len(np.unique(filtered[Layers.SEGMENTATION].values)) - 1
    cell_ids_in_coords = set(filtered.coords[Dims.CELLS].values)
    cell_ids_in_segmentation = set(np.unique(filtered[Layers.SEGMENTATION].values)) - {0}
    assert cell_ids_in_coords == cell_ids_in_segmentation
    # cell IDs are not changed when reindex is False
    assert set(filtered[Layers.OBS][Dims.CELLS].values).issubset(set(ds_segmentation[Layers.OBS][Dims.CELLS].values))
    # cell IDs do not start at 1 when reindex is False (they technically could, but not in this example dataset)
    assert 1 not in set(filtered[Layers.OBS][Dims.CELLS].values)


def test_filter_by_obs_reindex(ds_segmentation):
    filtered = ds_segmentation.pp.add_observations("area").pp.filter_by_obs(
        "area", func=lambda x: (x > 50) & (x < 100), reindex=True
    )

    # obs are retained after filtering
    assert Layers.OBS in filtered
    # size is smaller than before filtering
    assert len(filtered[Layers.OBS]) < len(ds_segmentation[Layers.OBS])

    # coords are synchronized with the segmentation mask
    assert filtered.sizes[Dims.CELLS] == len(np.unique(filtered[Layers.SEGMENTATION].values)) - 1
    cell_ids_in_coords = set(filtered.coords[Dims.CELLS].values)
    cell_ids_in_segmentation = set(np.unique(filtered[Layers.SEGMENTATION].values)) - {0}
    assert cell_ids_in_coords == cell_ids_in_segmentation


def test_filter_by_obs_no_change(ds_segmentation):
    filtered = ds_segmentation.pp.add_observations("area").pp.filter_by_obs("area", func=lambda x: x > 0)
    # nothing happens when you filter by something that does not affect the cells
    assert np.all(filtered[Layers.OBS] == ds_segmentation[Layers.OBS])


def test_filter_by_obs_nonexistent_feature(ds_segmentation):
    with pytest.raises(
        AssertionError, match="Feature nonexistent_feature not found in obs. You can add it with pp.add_observations"
    ):
        # filtering by a nonexistent feature raises an error
        ds_segmentation.pp.filter_by_obs("nonexistent_feature", func=lambda x: x > 0)
