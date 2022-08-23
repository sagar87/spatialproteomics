import os
from distutils import dir_util

import pytest
from skimage.io import imread

from spatial_data.container import load_image_data


@pytest.fixture(scope="session")
def data_dir(tmpdir_factory):
    # img = compute_expensive_image()
    test_dir = os.path.join(os.path.dirname(__file__), "test_files")
    tmp_dir = tmpdir_factory.getbasetemp()

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmp_dir))

    return tmp_dir


@pytest.fixture(scope="session", name="data_dic")
def load_files(data_dir):

    files = os.listdir(data_dir)
    files = {
        str(f).split("/")[-1].split(".")[0]: imread(os.path.join(str(data_dir), f))
        for f in files
        if f.endswith("tiff")
    }
    return files


@pytest.fixture(scope="session", name="dataset")
def load_dataset(data_dic):

    dataset = load_image_data(
        data_dic["input"][0],
        "Hoechst",
        segmentaton_mask=data_dic["segmentation"],
    )
    return dataset


@pytest.fixture(scope="session", name="dataset_full")
def load_dataset_five_dim(data_dic):

    dataset = load_image_data(
        data_dic["input"],
        ["Hoechst", "CD4", "CD8", "FOXP3", "BCL6"],
        segmentaton_mask=data_dic["segmentation"],
    )
    return dataset
