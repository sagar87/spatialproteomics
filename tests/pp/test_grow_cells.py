import numpy as np

from spatialproteomics.constants import Dims, Layers


def test_grow_cells(ds_segmentation):
    # checking that the number of cells is the same in the segmentation and the coordinates
    num_cells_segmentation = np.unique(ds_segmentation[Layers.SEGMENTATION].values).shape[0] - 1
    num_cells_coords = ds_segmentation.sizes[Dims.CELLS]
    original_obs = ds_segmentation[Layers.OBS].copy()
    assert num_cells_segmentation == num_cells_coords

    # if we grow by 0, the values in obs should be the same as before
    grown = ds_segmentation.pp.grow_cells(iterations=0)
    # ensuring that the cell segmentation still has the same number of cells, both in the segmentation mask and the coordinates
    num_cells_segmentation_grown = np.unique(grown[Layers.SEGMENTATION].values).shape[0] - 1
    num_cells_coords_grown = grown.sizes[Dims.CELLS]
    assert num_cells_segmentation == num_cells_coords == num_cells_segmentation_grown == num_cells_coords_grown
    # ensuring that the obs are the same as before
    assert np.all(original_obs == grown[Layers.OBS])

    # trying different growths
    for growth in [1, 2, 3, 4, 5, 10, 25, 50, 100]:
        grown = ds_segmentation.pp.grow_cells(growth)

        # ensuring that the cell segmentation still has the same number of cells, both in the segmentation mask and the coordinates
        num_cells_segmentation_grown = np.unique(grown[Layers.SEGMENTATION].values).shape[0]
        if 0 in grown[Layers.SEGMENTATION].values:
            num_cells_segmentation_grown -= 1  # removing the background
        num_cells_coords_grown = grown.sizes[Dims.CELLS]
        assert (
            num_cells_segmentation == num_cells_coords == num_cells_segmentation_grown == num_cells_coords_grown
        ), f"Fail for growth {growth}: previously {num_cells_segmentation} and {num_cells_coords}, now {num_cells_segmentation_grown} and {num_cells_coords_grown}"

        # ensuring that the obs have been updated (are not the same as before)
        assert not np.all(original_obs == grown[Layers.OBS])
