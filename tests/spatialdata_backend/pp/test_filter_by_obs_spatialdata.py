import numpy as np
import pytest

import spatialproteomics as sp
from spatialproteomics.constants import SDLayers


def test_filter_by_obs(ds_labels_spatialdata):
    ds = sp.pp.add_observations(ds_labels_spatialdata, "area", copy=True)
    sp.pp.filter_by_obs(ds, "area", func=lambda x: (x > 50) & (x < 100))

    # table is retained after filtering
    assert SDLayers.TABLE in ds.tables.keys()
    # size is smaller than before filtering
    assert ds.tables[SDLayers.TABLE].obs.shape[0] < ds_labels_spatialdata.tables[SDLayers.TABLE].obs.shape[0]

    # coords are synchronized with the segmentation mask
    assert ds.tables[SDLayers.TABLE].obs.shape[0] == len(np.unique(ds.labels[SDLayers.SEGMENTATION].values)) - 1


def test_filter_by_obs_no_change(ds_labels_spatialdata):
    ds = sp.pp.add_observations(ds_labels_spatialdata, "area", copy=True)
    sp.pp.filter_by_obs(ds, "area", func=lambda x: x > 0)
    # nothing happens when you filter by something that does not affect the cells
    assert ds_labels_spatialdata.tables[SDLayers.TABLE].obs.shape[0] == ds.tables[SDLayers.TABLE].obs.shape[0]


def test_filter_by_obs_nonexistent_feature(ds_labels_spatialdata):
    with pytest.raises(
        AssertionError, match="Feature nonexistent_feature not found in obs. You can add it with pp.add_observations"
    ):
        # filtering by a nonexistent feature raises an error
        sp.pp.filter_by_obs(ds_labels_spatialdata, "nonexistent_feature", func=lambda x: x > 0)
