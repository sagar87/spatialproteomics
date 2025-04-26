import numpy as np

import spatialproteomics as sp
from spatialproteomics.constants import SDFeatures, SDLayers


def test_grow_cells(ds_segmentation_spatialdata):
    # checking that the number of cells is the same in the segmentation and the coordinates
    num_cells_segmentation = len(np.unique(ds_segmentation_spatialdata.labels[SDLayers.SEGMENTATION].values)) - 1

    grown = sp.pp.grow_cells(ds_segmentation_spatialdata, iterations=0, copy=True)
    num_cells_grown = len(np.unique(grown.labels[SDLayers.SEGMENTATION].values)) - 1

    # if we grow by 0, the values in obs should be the same as before
    assert num_cells_segmentation == num_cells_grown


def test_grow_cells_with_table(ds_labels_spatialdata):
    # checking that the number of cells is the same in the segmentation and the coordinates
    num_cells_segmentation = len(np.unique(ds_labels_spatialdata.labels[SDLayers.SEGMENTATION].values)) - 1
    num_cells_coords = ds_labels_spatialdata.tables[SDLayers.TABLE].obs.shape[0]
    original_obs = ds_labels_spatialdata.tables[SDLayers.TABLE].obs.copy()
    assert num_cells_segmentation == num_cells_coords

    # if we grow by 0, the values in obs should be the same as before
    grown = sp.pp.grow_cells(ds_labels_spatialdata, iterations=0, copy=True)
    # ensuring that the cell segmentation still has the same number of cells, both in the segmentation mask and the coordinates
    num_cells_segmentation_grown = len(np.unique(grown.labels[SDLayers.SEGMENTATION].values)) - 1
    num_cells_coords_grown = grown.tables[SDLayers.TABLE].obs.shape[0]
    assert num_cells_segmentation == num_cells_coords == num_cells_segmentation_grown == num_cells_coords_grown

    # ensuring that the obs are the same as before
    assert np.all(
        original_obs[[SDFeatures.ID, SDFeatures.REGION]]
        == grown.tables[SDLayers.TABLE].obs[[SDFeatures.ID, SDFeatures.REGION]]
    )

    # trying different growths
    for growth in [1, 2, 3, 4, 5, 10, 25, 50, 100]:
        grown = sp.pp.grow_cells(ds_labels_spatialdata, growth, copy=True)

        # ensuring that the cell segmentation still has the same number of cells, both in the segmentation mask and the coordinates
        num_cells_segmentation_grown = len(np.unique(grown.labels[SDLayers.SEGMENTATION].values))
        if 0 in grown.labels[SDLayers.SEGMENTATION].values:
            num_cells_segmentation_grown -= 1  # removing the background
        num_cells_coords_grown = grown.tables[SDLayers.TABLE].obs.shape[0]
        assert (
            num_cells_segmentation == num_cells_coords == num_cells_segmentation_grown == num_cells_coords_grown
        ), f"Fail for growth {growth}: previously {num_cells_segmentation} and {num_cells_coords}, now {num_cells_segmentation_grown} and {num_cells_coords_grown}"

        # ensuring that the obs have been updated (are not the same as before)
        assert np.all(
            original_obs[[SDFeatures.ID, SDFeatures.REGION]]
            == grown.tables[SDLayers.TABLE].obs[[SDFeatures.ID, SDFeatures.REGION]]
        )
