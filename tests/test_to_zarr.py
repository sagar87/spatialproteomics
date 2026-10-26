import os
import tempfile

import pytest
import xarray as xr

import spatialproteomics  # noqa: F401


def test_to_zarr_image(ds_image):
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "zarr_writing_test.zarr")
        try:
            ds_image.drop_encoding().to_zarr(output_path, consolidated=False)
            assert os.path.exists(output_path) and os.path.isdir(
                output_path
            ), f"Directory {output_path} was not created."
        finally:
            # cleanup is handled by tempfile automatically
            pass


def test_to_zarr_segmentation(ds_segmentation):
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "zarr_writing_test.zarr")
        try:
            ds_segmentation.drop_encoding().to_zarr(output_path, consolidated=False)
            assert os.path.exists(output_path) and os.path.isdir(
                output_path
            ), f"Directory {output_path} was not created."
        finally:
            # cleanup is handled by tempfile automatically
            pass


def test_to_zarr_labels(ds_labels):
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "zarr_writing_test.zarr")
        try:
            ds_labels.drop_encoding().to_zarr(output_path, consolidated=False)
            assert os.path.exists(output_path) and os.path.isdir(
                output_path
            ), f"Directory {output_path} was not created."
        finally:
            # cleanup is handled by tempfile automatically
            pass


def test_to_zarr_neighborhoods(ds_neighborhoods):
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "zarr_writing_test.zarr")
        try:
            ds_neighborhoods.drop_encoding().to_zarr(output_path, consolidated=False)
            assert os.path.exists(output_path) and os.path.isdir(
                output_path
            ), f"Directory {output_path} was not created."
        finally:
            # cleanup is handled by tempfile automatically
            pass


@pytest.mark.parametrize("name", ["ds_image", "ds_segmentation", "ds_labels", "ds_neighborhoods"])
def test_read_zarr_format_v2(name, request, tmp_path):
    # legacy data is stored in zarr format v2, which must still be readable
    ds = request.getfixturevalue(name)
    output_path = tmp_path / "zarr_v2.zarr"
    ds.drop_encoding().to_zarr(output_path, zarr_format=2, consolidated=False)
    assert (output_path / ".zgroup").exists()

    ds_v2 = xr.load_dataset(output_path, engine="zarr", consolidated=False)
    xr.testing.assert_identical(ds_v2, ds)


def test_zarr_format_v2_label_workflow(ds_labels, tmp_path):
    # strings in zarr format v2 are read as fixed-width strings, which must not truncate longer label names
    output_path = tmp_path / "zarr_v2.zarr"
    ds_labels.drop_encoding().to_zarr(output_path, zarr_format=2, consolidated=False)
    ds = xr.load_dataset(output_path, engine="zarr", consolidated=False)

    ds = ds.la.set_label_name("T", "Macrophage")
    assert "Macrophage" in ds.la
