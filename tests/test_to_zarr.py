import os
import tempfile


def test_to_zarr_image(ds_image):
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "zarr_writing_test.zarr")
        try:
            ds_image.drop_encoding().to_zarr(output_path)
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
            ds_segmentation.drop_encoding().to_zarr(output_path)
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
            ds_labels.drop_encoding().to_zarr(output_path)
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
            ds_neighborhoods.drop_encoding().to_zarr(output_path)
            assert os.path.exists(output_path) and os.path.isdir(
                output_path
            ), f"Directory {output_path} was not created."
        finally:
            # cleanup is handled by tempfile automatically
            pass
