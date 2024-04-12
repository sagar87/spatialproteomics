import numpy as np
import pytest

from spatial_data.constants import Dims, Layers


def test_add_segmentation(data_dic, dataset_segmentation):
    segmented = dataset_segmentation.pp.add_segmentation(data_dic["segmentation"])

    assert "_segmentation" in segmented
    assert "_segmentation" not in dataset_segmentation
    assert Dims.CELLS in segmented.coords
    assert Dims.CELLS not in dataset_segmentation.coords
    assert Layers.OBS in segmented


def test_add_segmentation_wrong_dims(data_dic, dataset_segmentation):
    with pytest.raises(AssertionError, match="The shape of segmentation mask"):
        dataset_segmentation.pp.add_segmentation(data_dic["segmentation"][:300, :300])


def test_add_segmentation_negative_values(data_dic, dataset_segmentation):
    corrupted_segmentation = data_dic["segmentation"]
    corrupted_segmentation[10, 10] = -1
    with pytest.raises(AssertionError, match="A segmentation mask may not contain negative numbers."):
        dataset_segmentation.pp.add_segmentation(corrupted_segmentation)


def test_add_segmentation_relabel(data_dic, dataset_segmentation):
    noncontinuous_segmentation = data_dic["segmentation"]
    noncontinuous_segmentation[10, 10] = np.max(noncontinuous_segmentation) + 2
    num_cells = len(np.unique(noncontinuous_segmentation)) - 1  # -1 because of the background

    segmented = dataset_segmentation.pp.add_segmentation(noncontinuous_segmentation, relabel=True)
    cell_labels = sorted(np.unique(segmented["_segmentation"].values))[1:]  # removing the background

    assert cell_labels == list(range(1, num_cells + 1))
    assert list(segmented.cells.values) == list(range(1, num_cells + 1))
