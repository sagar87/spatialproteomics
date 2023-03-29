import pytest

from spatial_data.constants import Dims


def test_add_segmentation(data_dic, dataset_segmentation):
    segmented = dataset_segmentation.pp.add_segmentation(data_dic["segmentation"])

    assert "_segmentation" in segmented
    assert "_segmentation" not in dataset_segmentation
    assert Dims.CELLS in segmented.coords
    assert Dims.CELLS not in dataset_segmentation.coords


def test_add_segmentation_wrong_dims(data_dic, dataset_segmentation):
    with pytest.raises(AssertionError, match="The shape of segmentation mask"):
        dataset_segmentation.pp.add_segmentation(data_dic["segmentation"][:300, :300])


def test_add_segmentation_negative_values(data_dic, dataset_segmentation):
    corrupted_segmentation = data_dic["segmentation"]
    corrupted_segmentation[10, 10] = -1
    with pytest.raises(AssertionError, match="A segmentation mask may not contain negative numbers."):
        dataset_segmentation.pp.add_segmentation(corrupted_segmentation)
