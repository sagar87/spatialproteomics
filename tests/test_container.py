import pytest
import xarray as xr

from spatialproteomics.constants import Layers
from spatialproteomics.container import load_image_data, read_from_spatialdata


def test_load_data_proper_five_channel_input(data_dic):
    dataset = load_image_data(data_dic["image"], ["DAPI", "PAX5", "CD3", "CD4", "CD8"])

    assert type(dataset) is xr.Dataset
    assert Layers.IMAGE in dataset
    assert Layers.SEGMENTATION not in dataset
    assert Layers.LA_PROPERTIES not in dataset

    dataset = load_image_data(
        data_dic["image"],
        ["DAPI", "PAX5", "CD3", "CD4", "CD8"],
        segmentation=data_dic["segmentation"],
    )

    assert type(dataset) is xr.Dataset
    assert Layers.IMAGE in dataset
    assert Layers.SEGMENTATION in dataset
    assert Layers.LA_PROPERTIES not in dataset


def test_load_data_proper_input_one_channel_input(data_dic):
    # load_data handles 2 dimensional data_dic
    dataset = load_image_data(
        data_dic["image"][0],
        "DAPI",
        segmentation=data_dic["segmentation"],
    )

    assert type(dataset) is xr.Dataset
    assert Layers.IMAGE in dataset
    assert Layers.SEGMENTATION in dataset
    assert Layers.OBS in dataset


def test_load_data_assertions(data_dic):
    with pytest.raises(AssertionError, match="Length of channel_coords must match"):
        load_image_data(
            data_dic["image"],
            ["DAPI", "PAX5", "CD3", "CD4"],
        )


def test_load_data_wrong_inputs_segmentation_mask_dim_error(data_dic):
    with pytest.raises(AssertionError, match="The shape of segmentation mask"):
        load_image_data(
            data_dic["image"],
            ["DAPI", "PAX5", "CD3", "CD4", "CD8"],
            segmentation=data_dic["segmentation"][:50, :50],
        )


def test_read_from_spatialdata(ds_neighborhoods_spatialdata):
    read_from_spatialdata(ds_neighborhoods_spatialdata)
    read_from_spatialdata(ds_neighborhoods_spatialdata, consolidate_segmentation=True)


# === SPATIALDATA MULTISCALE ===
# these would be very good to activate, but there are currently some issues with spatialdata and datatree (conflicting versions)
# once they figure out the issues, we can enable these tests again

# def test_read_from_spatialdata_multiscale_no_consolidation(ds_spatialdata_multiscale):
#     # when the image shape does not match the segmentation shape, there should be an error if consolidate_segmentation is not set to True
#     with pytest.raises(AssertionError, match="Image shape"):
#         read_from_spatialdata(
#             ds_spatialdata_multiscale,
#             image_key="raw_image",
#             data_key="scale1/image",
#             segmentation_key="segmentation_mask",
#         )


# def test_read_from_spatialdata_multiscale(ds_spatialdata_multiscale):
#     # reading from scale 0
#     read_from_spatialdata(
#         ds_spatialdata_multiscale,
#         image_key="raw_image",
#         data_key="scale0/image",
#         segmentation_key="segmentation_mask",
#         consolidate_segmentation=True,
#         cell_id="cell_ID",
#     )

#     # reading from scale 1
#     read_from_spatialdata(
#         ds_spatialdata_multiscale,
#         image_key="raw_image",
#         data_key="scale1/image",
#         segmentation_key="segmentation_mask",
#         consolidate_segmentation=True,
#         cell_id="cell_ID",
#     )


# def test_read_from_spatialdata_multiscale_too_many_cells_in_adata(ds_spatialdata_multiscale):
#     ds = copy.deepcopy(ds_spatialdata_multiscale)

#     # creating an example file where the number of cells in the segmentation is not equal to the cells in the anndata table
#     segmentation = ds.labels["segmentation_mask"].values
#     segmentation = np.where(segmentation == 1957, 1957, 0)
#     ds.labels["segmentation_mask"] = sd.models.Labels2DModel.parse(segmentation, transformations=None, dims=("y", "x"))
#     adata = ds.tables["table"]

#     # check that the number of cells in the segmentation is not equal to the number of cells in the anndata table
#     assert len(np.unique(segmentation)) - 1 == 1
#     assert len(np.unique(segmentation)) - 1 != len(
#         np.unique(adata.obs["cell_ID"])
#     ), "Number of cells in segmentation and anndata table should not match for this test."

#     # converting to spatialproteomics
#     ds_spatprot = read_from_spatialdata(
#         ds,
#         image_key="raw_image",
#         data_key="scale1/image",
#         segmentation_key="segmentation_mask",
#         consolidate_segmentation=True,
#         cell_id="cell_ID",
#     )

#     assert ds_spatprot.coords["cells"].values == np.array([1957])


# def test_read_from_spatialdata_multiscale_too_many_cells_in_segmentation(ds_spatialdata_multiscale):
#     ds = copy.deepcopy(ds_spatialdata_multiscale)

#     # example where only one cell_ID is present in the adata object
#     adata = ds.tables["table"]

#     adata = adata[adata.obs["cell_ID"] == 1957].copy()
#     ds.tables["table"] = adata
#     segmentation = ds.labels["segmentation_mask"].values

#     # check that the number of cells in the segmentation is not equal to the number of cells in the anndata table
#     assert adata.obs.shape[0] == 1
#     assert len(np.unique(segmentation)) - 1 != len(
#         np.unique(adata.obs["cell_ID"])
#     ), "Number of cells in segmentation and anndata table should not match for this test."

#     # converting to spatialproteomics
#     ds_spatprot = read_from_spatialdata(
#         ds,
#         image_key="raw_image",
#         data_key="scale1/image",
#         segmentation_key="segmentation_mask",
#         consolidate_segmentation=True,
#         cell_id="cell_ID",
#     )

#     assert ds_spatprot.coords["cells"] == np.array([1957])


# def test_read_from_spatialdata_multiscale_zero_cells_remain(ds_spatialdata_multiscale):
#     ds = copy.deepcopy(ds_spatialdata_multiscale)

#     # creating an example file where the number of cells in the segmentation is not equal to the cells in the anndata table
#     segmentation = ds.labels["segmentation_mask"]
#     segmentation = np.zeros(segmentation.shape, dtype=segmentation.dtype)
#     ds.labels["segmentation_mask"] = sd.models.Labels2DModel.parse(segmentation, transformations=None, dims=("y", "x"))
#     adata = ds.tables["table"]

#     # check that the number of cells in the segmentation is not equal to the number of cells in the anndata table
#     assert len(np.unique(segmentation)) - 1 == 0
#     assert len(np.unique(segmentation)) - 1 != len(
#         np.unique(adata.obs["cell_ID"])
#     ), "Number of cells in segmentation and anndata table should not match for this test."

#     # converting to spatialproteomics
#     ds_spatprot = read_from_spatialdata(
#         ds,
#         image_key="raw_image",
#         data_key="scale1/image",
#         segmentation_key="segmentation_mask",
#         consolidate_segmentation=True,
#         cell_id="cell_ID",
#     )

#     assert len(ds_spatprot.coords["cells"]) == 0, "There should be no cells in the dataset after consolidation."
