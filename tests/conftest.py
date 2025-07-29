import os
from distutils import dir_util

import pytest
import spatialdata as sd
import xarray as xr
from skimage.io import imread


# === NEW ===
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

    # loading all zarr files into the data_dic
    # we actually load the datasets directly into memory instead of just opening the zarr lazily
    # this should speed up testing a bit
    files_loaded = {
        str(f).split("/")[-1].split(".")[0]: xr.load_dataset(os.path.join(str(data_dir), f), engine="zarr")
        for f in files
        if f.endswith("zarr") and f != "ds_spatialdata_multiscale.zarr"
    }

    # this is just the path, because we want to test the loading of the multiscale zarr
    files_loaded["ds_spatialdata_multiscale"] = sd.read_zarr(
        os.path.join(str(data_dir), "ds_spatialdata_multiscale.zarr")
    )

    # adding the tiff files to test the loading of images
    for f in files:
        if f.endswith("tiff"):
            files_loaded[str(f).split("/")[-1].split(".")[0]] = imread(os.path.join(str(data_dir), f))

    return files_loaded


@pytest.fixture(scope="session", name="ds_image")
def load_ds_image(data_dic):
    return data_dic["ds_image"]


@pytest.fixture(scope="session", name="ds_segmentation")
def load_ds_segmentation(data_dic):
    return data_dic["ds_segmentation"]


@pytest.fixture(scope="session", name="ds_labels")
def load_ds_labels(data_dic):
    return data_dic["ds_labels"]


@pytest.fixture(scope="session", name="ds_neighborhoods")
def load_ds_neighborhoods(data_dic):
    return data_dic["ds_neighborhoods"]


# === SPATIALDATA OBJECTS ===
@pytest.fixture(scope="session", name="ds_image_spatialdata")
def load_ds_image_spatialdata(data_dic):
    return data_dic["ds_image"].tl.convert_to_spatialdata()


@pytest.fixture(scope="session", name="ds_segmentation_spatialdata")
def load_ds_segmentation_spatialdata(data_dic):
    return data_dic["ds_segmentation"].tl.convert_to_spatialdata()


@pytest.fixture(scope="session", name="ds_labels_spatialdata")
def load_ds_labels_spatialdata(data_dic):
    return data_dic["ds_labels"].tl.convert_to_spatialdata()


@pytest.fixture(scope="session", name="ds_neighborhoods_spatialdata")
def load_ds_neighborhoods_spatialdata(data_dic):
    return data_dic["ds_neighborhoods"].tl.convert_to_spatialdata()


@pytest.fixture(scope="session", name="ds_spatialdata_multiscale")
def load_ds_spatialdata_multiscale(data_dic):
    return data_dic["ds_spatialdata_multiscale"]
