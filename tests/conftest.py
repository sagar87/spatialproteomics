import os
from distutils import dir_util

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from skimage.io import imread

import spatialproteomics as sp
from spatialproteomics.container import load_image_data


@pytest.fixture(scope="session")
def data_dir(tmpdir_factory):
    test_dir = os.path.join(os.path.dirname(__file__), "test_files")
    tmp_dir = tmpdir_factory.getbasetemp()

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmp_dir))

    return tmp_dir


@pytest.fixture(scope="session", name="data_dic")
def load_files(data_dir):
    files = os.listdir(data_dir)
    files_loaded = {
        str(f).split("/")[-1].split(".")[0]: imread(os.path.join(str(data_dir), f)) for f in files if f.endswith("tiff")
    }

    files_loaded["labels"] = pd.read_csv(os.path.join(str(data_dir), files[files.index("labels.csv")]), index_col=0)
    files_loaded["neighborhoods"] = pd.read_csv(
        os.path.join(str(data_dir), files[files.index("neighborhoods.csv")]), index_col=0
    )
    files_loaded["neighborhoods_numeric"] = pd.read_csv(
        os.path.join(str(data_dir), files[files.index("neighborhoods_numeric.csv")]), index_col=0
    )
    files_loaded["zarr"] = xr.open_zarr(os.path.join(str(data_dir), files[files.index("test.zarr")]))
    files_loaded["zarr_binarized"] = xr.open_zarr(
        os.path.join(str(data_dir), files[files.index("test_binarized.zarr")])
    )

    return files_loaded


@pytest.fixture(scope="session", name="dataset")
def load_dataset(data_dic):
    dataset = load_image_data(
        data_dic["input"][0],
        "Hoechst",
        segmentation=data_dic["segmentation"],
    )
    return dataset


@pytest.fixture(scope="session", name="dataset_full")
def load_dataset_five_dim(data_dic):
    dataset = load_image_data(
        data_dic["input"],
        ["Hoechst", "CD4", "CD8", "FOXP3", "BCL6"],
        segmentation=data_dic["segmentation"],
    )
    return dataset


@pytest.fixture(scope="session", name="dataset_labeled")
def load_labeled_dataset(data_dic):
    dataset = load_image_data(
        data_dic["input"],
        ["Hoechst", "CD4", "CD8", "FOXP3", "BCL6"],
        segmentation=data_dic["segmentation"],
        labels=data_dic["labels"],
    )
    return dataset


@pytest.fixture(scope="session", name="dataset_labeled_multilevel")
def load_labeled_dataset_multilevel(data_dic):
    basic_subtype_dict = {
        "Cell type 1": {"subtypes": [{"name": "Treg", "markers": ["FOXP3+"]}]},
        "Cell type 2": {"subtypes": [{"name": "T_h", "markers": ["CD4+"]}, {"name": "T_tox", "markers": ["CD8+"]}]},
    }

    dataset = load_image_data(
        data_dic["input"],
        ["Hoechst", "CD4", "CD8", "FOXP3", "BCL6"],
        segmentation=data_dic["segmentation"],
        labels=data_dic["labels"],
    )

    # predicting subtypes
    binarization_dict = {"CD4": 0.5, "CD8": 0.6, "FOXP3": 0.5, "BCL6": 0.7}
    ds = (
        dataset.pp[["CD4", "CD8", "FOXP3", "BCL6"]]
        .pp.threshold(quantile=[0.9, 0.9, 0.9, 0.9])
        .pp.add_quantification(func=sp.percentage_positive, key_added="_percentage_positive")
        .la.threshold_labels(binarization_dict, layer_key="_percentage_positive")
    )
    ds = ds.la.predict_cell_subtypes(basic_subtype_dict)

    return ds


@pytest.fixture(scope="session", name="dataset_neighborhoods")
def load_dataset_neighborhoods(data_dic):
    dataset = load_image_data(
        data_dic["input"],
        ["Hoechst", "CD4", "CD8", "FOXP3", "BCL6"],
        segmentation=data_dic["segmentation"],
        labels=data_dic["labels"],
        neighborhood=data_dic["neighborhoods"],
    )
    return dataset


@pytest.fixture(scope="session", name="dataset_neighborhoods_numeric")
def load_dataset_neighborhoods_numeric(data_dic):
    dataset = load_image_data(
        data_dic["input"],
        ["Hoechst", "CD4", "CD8", "FOXP3", "BCL6"],
        segmentation=data_dic["segmentation"],
        labels=data_dic["labels"],
        neighborhood=data_dic["neighborhoods_numeric"],
    )
    return dataset


@pytest.fixture(scope="session", name="dataset_segmentation")
def load_dataset_segmentation(data_dic):
    dataset = load_image_data(
        data_dic["input"],
        ["Hoechst", "CD4", "CD8", "FOXP3", "BCL6"],
    )
    return dataset


@pytest.fixture(scope="session", name="full_zarr")
def load_full_zarr(data_dic):
    return data_dic["zarr"]


@pytest.fixture(scope="session", name="dataset_binarized")
def load_binarized_zarr(data_dic):
    return data_dic["zarr_binarized"]


@pytest.fixture(scope="session", name="test_segmentation")
def load_test_segmentation():
    seg_mask = np.zeros((10, 10))
    seg_mask[0, 0] = 1
    seg_mask[0, 1] = 1
    seg_mask[1, 0] = 1
    seg_mask[1, 1] = 1

    seg_mask[4, 2] = 2
    seg_mask[4, 3] = 2
    seg_mask[5, 2] = 2
    seg_mask[5, 3] = 2

    seg_mask[1, 5] = 3
    seg_mask[1, 6] = 3
    seg_mask[2, 5] = 3
    seg_mask[2, 6] = 3

    seg_mask[7, 1] = 5
    seg_mask[7, 2] = 5
    seg_mask[8, 1] = 5
    seg_mask[8, 2] = 5

    seg_mask[8, 3] = 7
    seg_mask[8, 4] = 7
    seg_mask[9, 3] = 7
    seg_mask[9, 4] = 7

    seg_mask[5, 5] = 8
    seg_mask[5, 6] = 8
    seg_mask[5, 7] = 8
    seg_mask[5, 5] = 8
    seg_mask[6, 6] = 8
    seg_mask[6, 7] = 8
    seg_mask[7, 5] = 8
    seg_mask[7, 6] = 8
    seg_mask[7, 7] = 8

    seg_mask[2, 9] = 9

    return seg_mask
