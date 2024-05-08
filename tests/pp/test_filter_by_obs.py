import numpy as np
import pytest

from spatialproteomics.constants import Dims, Layers


def test_filter_by_obs(dataset):
    filtered = dataset.pp.add_observations("area").pp.filter_by_obs("area", func=lambda x: (x > 50) & (x < 100))

    # obs are retained after filtering
    assert Layers.OBS in filtered
    # size is smaller than before filtering
    assert len(filtered[Layers.OBS]) < len(dataset[Layers.OBS])

    # coords are synchronized with the segmentation mask
    assert filtered.dims[Dims.CELLS] == len(np.unique(filtered[Layers.SEGMENTATION].values)) - 1


def test_filter_by_obs_no_change(dataset):
    filtered = dataset.pp.add_observations("area").pp.filter_by_obs("area", func=lambda x: x > 0)
    # nothing happens when you filter by something that does not affect the cells
    assert np.all(filtered[Layers.OBS] == dataset[Layers.OBS])


def test_filter_by_obs_nonexistent_feature(dataset):
    with pytest.raises(
        AssertionError, match="Feature nonexistent_feature not found in obs. You can add it with pp.add_observations"
    ):
        # filtering by a nonexistent feature raises an error
        dataset.pp.filter_by_obs("nonexistent_feature", func=lambda x: x > 0)
