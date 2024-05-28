import numpy as np
import pytest
import xarray as xr

from spatialproteomics.constants import Dims, Layers

# === case: numpy array, a segmentation mask already exists ===
# def test_merge_segmentation_2d(dataset):
#     # test that merging either 2D or 3D images work (with and without labels)
#     merge_array = np.zeros(dataset[Layers.SEGMENTATION].shape, dtype=int)
#     merge_array[200:300, 200:300] = 1
#     merged_1 = dataset.pp.merge_segmentation(array=merge_array, key_base_segmentation=Layers.SEGMENTATION)
#     merged_2 = dataset.pp.merge_segmentation(
#         array=merge_array, labels=["test_label"], key_base_segmentation=Layers.SEGMENTATION
#     )

#     # ensure that the dimensionality and the number of cells in the segmentation mask are synchronized
#     assert merged_1.dims[Dims.CELLS] == len(np.unique(merged_1[Layers.SEGMENTATION].values)) - 1
#     assert merged_2.dims[Dims.CELLS] == len(np.unique(merged_2[Layers.SEGMENTATION].values)) - 1

#     # ensure that the number of cells afterwards is smaller than the number of cells in the original image
#     assert merged_1.dims[Dims.CELLS] == merged_2.dims[Dims.CELLS]
#     assert dataset.dims[Dims.CELLS] > merged_1.dims[Dims.CELLS]
#     assert dataset.dims[Dims.CELLS] > merged_2.dims[Dims.CELLS]


# def test_merge_segmentation_3d(dataset):
#     merge_array = np.zeros(dataset[Layers.SEGMENTATION].shape, dtype=int)
#     merge_array = np.stack((merge_array, merge_array))
#     merge_array[0, 200:300, 200:300] = 1
#     merge_array[1, 350:400, 350:400] = 1
#     merged_1 = dataset.pp.merge_segmentation(array=merge_array, key_base_segmentation=Layers.SEGMENTATION)
#     merged_2 = dataset.pp.merge_segmentation(
#         array=merge_array, labels=["test_label_1", "test_label_2"], key_base_segmentation=Layers.SEGMENTATION
#     )

#     # ensure that the dimensionality and the number of cells in the segmentation mask are synchronized
#     assert merged_1.dims[Dims.CELLS] == len(np.unique(merged_1[Layers.SEGMENTATION].values)) - 1
#     assert merged_2.dims[Dims.CELLS] == len(np.unique(merged_2[Layers.SEGMENTATION].values)) - 1

#     # ensure that the number of cells afterwards is smaller than the number of cells in the original image
#     assert merged_1.dims[Dims.CELLS] == merged_2.dims[Dims.CELLS]
#     assert dataset.dims[Dims.CELLS] > merged_1.dims[Dims.CELLS]
#     assert dataset.dims[Dims.CELLS] > merged_2.dims[Dims.CELLS]


# def test_merge_segmentation_wrong_dimensionality(dataset):
#     # test array with wrong dimensionality
#     merge_array = np.zeros((500, 500, 2, 2), dtype=int)
#     with pytest.raises(AssertionError, match="The input array must be 2D"):
#         dataset.pp.merge_segmentation(array=merge_array, key_base_segmentation=Layers.SEGMENTATION)


# def test_merge_segmentation_wrong_shape(dataset):
#     # test array with wrong shape
#     merge_array = np.zeros((500, 300), dtype=int)
#     with pytest.raises(
#         AssertionError, match="The shape of the input array does not match the shape of the segmentation mask."
#     ):
#         dataset.pp.merge_segmentation(array=merge_array, key_base_segmentation=Layers.SEGMENTATION)


# def test_merge_segmentation_wrong_dtype(dataset):
#     # test array with wrong dtype
#     merge_array = np.zeros(dataset[Layers.SEGMENTATION].shape).astype("float32")
#     with pytest.raises(AssertionError, match="The input array must be of type int."):
#         dataset.pp.merge_segmentation(array=merge_array, key_base_segmentation=Layers.SEGMENTATION)


# def test_merge_segmentation_zeros(dataset):
#     # test array full of zeros (shouldn't remove any cells)
#     merge_array = np.zeros(dataset[Layers.SEGMENTATION].shape, dtype=int)
#     merged = dataset.pp.merge_segmentation(array=merge_array, key_base_segmentation=Layers.SEGMENTATION)
#     assert dataset.dims["cells"] == merged.dims["cells"]


# def test_merge_segmentation_wrong_labels(dataset):
#     # test different dimensionality and labels
#     merge_array = np.zeros((500, 500, 2), dtype=int)
#     with pytest.raises(
#         AssertionError,
#         match="The number of labels must match the number of arrays. You submitted 1 channels compared to 500 arrays.",
#     ):
#         dataset.pp.merge_segmentation(
#             array=merge_array, labels=["test_label_1"], key_base_segmentation=Layers.SEGMENTATION
#         )


# def test_merge_segmentation_no_segmentation(dataset):
#     # check that it does not work if we don't already have a segmentation mask
#     merge_array = np.zeros((500, 500, 2), dtype=int)
#     with pytest.raises(AssertionError, match=f"The key {Layers.SEGMENTATION} does not exist in the xarray object."):
#         dataset.pp.drop_layers(Layers.SEGMENTATION).pp.merge_segmentation(
#             array=merge_array, key_base_segmentation=Layers.SEGMENTATION
#         )


# # === case: from numpy array, no segmentation mask exists ===
# def test_merge_segmentation_3d_without_base_segmentation(dataset):
#     merge_array = np.zeros(dataset[Layers.SEGMENTATION].shape, dtype=int)
#     merge_array = np.stack((merge_array, merge_array))
#     merge_array[0, 200:300, 200:300] = 1
#     merge_array[1, 350:400, 350:400] = 1
#     merged_1 = dataset.pp.drop_layers(Layers.SEGMENTATION).pp.merge_segmentation(array=merge_array)
#     merged_2 = dataset.pp.drop_layers(Layers.SEGMENTATION).pp.merge_segmentation(
#         array=merge_array, labels=["test_label_1", "test_label_2"]
#     )

#     # ensure that the dimensionality and the number of cells in the segmentation mask are synchronized
#     assert merged_1.dims[Dims.CELLS] == len(np.unique(merged_1[Layers.SEGMENTATION].values)) - 1
#     assert merged_2.dims[Dims.CELLS] == len(np.unique(merged_2[Layers.SEGMENTATION].values)) - 1

#     # ensure that the number of cells afterwards is smaller than the number of cells in the original image
#     assert merged_1.dims[Dims.CELLS] == merged_2.dims[Dims.CELLS]
#     assert dataset.dims[Dims.CELLS] > merged_1.dims[Dims.CELLS]
#     assert dataset.dims[Dims.CELLS] > merged_2.dims[Dims.CELLS]


# def test_merge_segmentation_2d_without_specifying_base_segmentation(dataset):
#     # trying to merge a single segmentation mask to nothing
#     merge_array = np.zeros(dataset[Layers.SEGMENTATION].shape, dtype=int)
#     merge_array[200:300, 200:300] = 1
#     with pytest.raises(
#         AssertionError,
#         match="If you want to merge a single segmentation mask to an existing one, please use the key_base_segmentation argument.",
#     ):
#         dataset.pp.merge_segmentation(array=merge_array)


# def test_merge_segmentation_2d_without_base_segmentation_wrong_key(dataset):
#     # trying to merge a single segmentation mask to a non-existing layer
#     merge_array = np.zeros(dataset[Layers.SEGMENTATION].shape, dtype=int)
#     merge_array[200:300, 200:300] = 1
#     with pytest.raises(AssertionError, match=f"The key {Layers.SEGMENTATION} does not exist in the xarray object."):
#         dataset.pp.drop_layers(Layers.SEGMENTATION).pp.merge_segmentation(
#             array=merge_array, key_base_segmentation=Layers.SEGMENTATION
#         )


# def test_merge_segmentation_3d_without_base_segmentation_wrong_key(dataset):
#     merge_array = np.zeros(dataset[Layers.SEGMENTATION].shape, dtype=int)
#     merge_array = np.stack((merge_array, merge_array))
#     merge_array[0, 200:300, 200:300] = 1
#     merge_array[1, 350:400, 350:400] = 1
#     with pytest.raises(AssertionError, match=f"The key {Layers.SEGMENTATION} does not exist in the xarray object."):
#         dataset.pp.drop_layers(Layers.SEGMENTATION).pp.merge_segmentation(
#             array=merge_array, key_base_segmentation=Layers.SEGMENTATION
#         )


# # === case: layer ===
# def test_merge_segmentation_from_layer_2d(dataset):
#     merge_array = np.zeros(dataset[Layers.SEGMENTATION].shape, dtype=int)
#     merge_array[200:300, 200:300] = 1

#     da = xr.DataArray(
#         merge_array, coords=[range(500), range(500)], dims=[Dims.X, Dims.Y], name="_segmentation_to_merge"
#     )

#     ds = xr.merge([dataset, da])

#     with pytest.raises(
#         AssertionError,
#         match="If you want to merge a single segmentation mask to an existing one, please use the key_base_segmentation argument.",
#     ):
#         ds.pp.merge_segmentation(from_key="_segmentation_to_merge")


# def test_merge_segmentation_from_layer_3d(dataset_full):
#     merge_array = np.zeros(dataset_full[Layers.SEGMENTATION].shape)
#     merge_array = np.stack((merge_array, merge_array))
#     merge_array[0, 200:300, 200:300] = 1
#     merge_array[1, 350:400, 350:400] = 1

#     da = xr.DataArray(
#         merge_array,
#         coords=[["Hoechst", "CD4"], range(500), range(500)],
#         dims=[Dims.CHANNELS, Dims.X, Dims.Y],
#         name="_segmentation_to_merge",
#     ).astype(int)

#     ds = xr.merge([dataset_full.pp[["Hoechst", "CD4"]].pp.drop_layers(Layers.SEGMENTATION), da])

#     merged_1 = ds.pp.merge_segmentation(from_key="_segmentation_to_merge")
#     merged_2 = ds.pp.merge_segmentation(from_key="_segmentation_to_merge", labels=["test_label_1", "test_label_2"])

#     # ensure that the dimensionality and the number of cells in the segmentation mask are synchronized
#     assert merged_1.dims[Dims.CELLS] == len(np.unique(merged_1[Layers.SEGMENTATION].values)) - 1
#     assert merged_2.dims[Dims.CELLS] == len(np.unique(merged_2[Layers.SEGMENTATION].values)) - 1

#     # ensure that the number of cells afterwards is smaller than the number of cells in the original image
#     assert merged_1.dims[Dims.CELLS] == merged_2.dims[Dims.CELLS]
#     assert dataset_full.dims[Dims.CELLS] > merged_1.dims[Dims.CELLS]
#     assert dataset_full.dims[Dims.CELLS] > merged_2.dims[Dims.CELLS]
