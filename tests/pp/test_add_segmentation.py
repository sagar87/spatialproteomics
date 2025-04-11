# import numpy as np
# import pytest
# import xarray as xr

# from spatialproteomics.constants import Dims, Layers


# # TODO: all of these need to be adjusted to the new test structure
# def test_add_segmentation(data_dic, dataset_segmentation):
#     segmented = dataset_segmentation.pp.add_segmentation(data_dic["segmentation"])

#     assert Layers.SEGMENTATION in segmented
#     assert Layers.SEGMENTATION not in dataset_segmentation
#     assert Dims.CELLS in segmented.coords
#     assert Dims.CELLS not in dataset_segmentation.coords
#     assert Layers.OBS in segmented


# def test_add_segmentation_from_layer(data_dic, dataset_segmentation):
#     da = xr.DataArray(
#         data_dic["segmentation"],
#         coords=[range(500), range(500)],
#         dims=[Dims.X, Dims.Y],
#         name="_segmentation_preliminary",
#     ).astype(int)

#     ds = xr.merge([dataset_segmentation, da])
#     segmented = ds.pp.add_segmentation("_segmentation_preliminary")

#     assert "_segmentation_preliminary" in segmented
#     assert Layers.SEGMENTATION in segmented
#     assert Layers.SEGMENTATION not in dataset_segmentation
#     assert Dims.CELLS in segmented.coords
#     assert Dims.CELLS not in dataset_segmentation.coords
#     assert Layers.OBS in segmented


# def test_add_segmentation_wrong_dims(data_dic, dataset_segmentation):
#     with pytest.raises(AssertionError, match="The shape of segmentation mask"):
#         dataset_segmentation.pp.add_segmentation(data_dic["segmentation"][:300, :300])


# def test_add_segmentation_negative_values(data_dic, dataset_segmentation):
#     corrupted_segmentation = data_dic["segmentation"]
#     corrupted_segmentation[10, 10] = -1
#     with pytest.raises(AssertionError, match="A segmentation mask may not contain negative numbers."):
#         dataset_segmentation.pp.add_segmentation(corrupted_segmentation)


# def test_add_segmentation_reindex(data_dic, dataset_segmentation):
#     noncontinuous_segmentation = data_dic["segmentation"]
#     noncontinuous_segmentation[10, 10] = np.max(noncontinuous_segmentation) + 2
#     num_cells = len(np.unique(noncontinuous_segmentation)) - 1  # -1 because of the background

#     segmented = dataset_segmentation.pp.add_segmentation(noncontinuous_segmentation, reindex=True)
#     cell_labels = sorted(np.unique(segmented["_segmentation"].values))[1:]  # removing the background

#     assert cell_labels == list(range(1, num_cells + 1))
#     assert list(segmented.cells.values) == list(range(1, num_cells + 1))
