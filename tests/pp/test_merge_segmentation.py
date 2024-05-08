import numpy as np
import pytest

from spatial_proteomics.constants import Dims, Layers


def test_merge_segmentation_2d(dataset):
    # test that merging either 2D or 3D images work (with and without labels)
    merge_array = np.zeros(dataset[Layers.SEGMENTATION].shape, dtype=int)
    merge_array[200:300, 200:300] = 1
    merged_1 = dataset.pp.merge_segmentation(merge_array)
    merged_2 = dataset.pp.merge_segmentation(merge_array, labels=["test_label"])

    # ensure that the dimensionality and the number of cells in the segmentation mask are synchronized
    assert merged_1.dims[Dims.CELLS] == len(np.unique(merged_1[Layers.SEGMENTATION].values)) - 1
    assert merged_2.dims[Dims.CELLS] == len(np.unique(merged_2[Layers.SEGMENTATION].values)) - 1

    # ensure that the number of cells afterwards is smaller than the number of cells in the original image
    assert merged_1.dims[Dims.CELLS] == merged_2.dims[Dims.CELLS]
    assert dataset.dims[Dims.CELLS] > merged_1.dims[Dims.CELLS]
    assert dataset.dims[Dims.CELLS] > merged_2.dims[Dims.CELLS]


def test_merge_segmentation_3d(dataset):
    merge_array = np.zeros(dataset[Layers.SEGMENTATION].shape, dtype=int)
    merge_array = np.stack((merge_array, merge_array))
    merge_array[0, 200:300, 200:300] = 1
    merge_array[1, 350:400, 350:400] = 1
    merged_1 = dataset.pp.merge_segmentation(merge_array)
    merged_2 = dataset.pp.merge_segmentation(merge_array, labels=["test_label_1", "test_label_2"])

    # ensure that the dimensionality and the number of cells in the segmentation mask are synchronized
    assert merged_1.dims[Dims.CELLS] == len(np.unique(merged_1[Layers.SEGMENTATION].values)) - 1
    assert merged_2.dims[Dims.CELLS] == len(np.unique(merged_2[Layers.SEGMENTATION].values)) - 1

    # ensure that the number of cells afterwards is smaller than the number of cells in the original image
    assert merged_1.dims[Dims.CELLS] == merged_2.dims[Dims.CELLS]
    assert dataset.dims[Dims.CELLS] > merged_1.dims[Dims.CELLS]
    assert dataset.dims[Dims.CELLS] > merged_2.dims[Dims.CELLS]


def test_merge_segmentation_wrong_dimensionality(dataset):
    # test array with wrong dimensionality
    merge_array = np.zeros((500, 500, 2, 2), dtype=int)
    with pytest.raises(AssertionError, match="The input array must be 2D"):
        dataset.pp.merge_segmentation(merge_array)


def test_merge_segmentation_wrong_shape(dataset):
    # test array with wrong shape
    merge_array = np.zeros((500, 300), dtype=int)
    with pytest.raises(
        AssertionError, match="The shape of the input array does not match the shape of the segmentation mask."
    ):
        dataset.pp.merge_segmentation(merge_array)


def test_merge_segmentation_wrong_dtype(dataset):
    # test array with wrong dtype
    merge_array = np.zeros(dataset[Layers.SEGMENTATION].shape).astype("float32")
    with pytest.raises(AssertionError, match="The input array must be of type int."):
        dataset.pp.merge_segmentation(merge_array)


def test_merge_segmentation_zeros(dataset):
    # test array full of zeros (shouldn't remove any cells)
    merge_array = np.zeros(dataset[Layers.SEGMENTATION].shape, dtype=int)
    merged = dataset.pp.merge_segmentation(merge_array)
    assert dataset.dims["cells"] == merged.dims["cells"]


def test_merge_segmentation_wrong_labels(dataset):
    # test different dimensionality and labels
    merge_array = np.zeros((500, 500, 2), dtype=int)
    with pytest.raises(
        AssertionError,
        match="The number of labels must match the number of arrays. You submitted 1 channels compared to 500 arrays.",
    ):
        dataset.pp.merge_segmentation(merge_array, labels=["test_label_1"])


def test_merge_segmentation_no_segmentation(dataset):
    # check that it does not work if we don't already have a segmentation mask
    merge_array = np.zeros((500, 500, 2), dtype=int)
    with pytest.raises(AssertionError, match="No segmentation mask found in the xarray object."):
        dataset.pp.drop_layers(Layers.SEGMENTATION).pp.merge_segmentation(merge_array)
