import os
import tempfile


def test_to_zarr(dataset):
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "zarr_writing_test.zarr")
        try:
            dataset.drop_encoding().to_zarr(output_path)
            assert os.path.exists(output_path) and os.path.isdir(
                output_path
            ), f"Directory {output_path} was not created."
        finally:
            # cleanup is handled by tempfile automatically
            pass


def test_to_zarr_labeled(dataset_labeled):
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "zarr_writing_test.zarr")
        try:
            dataset_labeled.drop_encoding().to_zarr(output_path)
            assert os.path.exists(output_path) and os.path.isdir(
                output_path
            ), f"Directory {output_path} was not created."
        finally:
            # cleanup is handled by tempfile automatically
            pass


def test_to_zarr_neighborhoods(dataset_neighborhoods):
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "zarr_writing_test.zarr")
        try:
            dataset_neighborhoods.drop_encoding().to_zarr(output_path)
            assert os.path.exists(output_path) and os.path.isdir(
                output_path
            ), f"Directory {output_path} was not created."
        finally:
            # cleanup is handled by tempfile automatically
            pass


def test_to_zarr_neighborhoods_numeric(dataset_neighborhoods_numeric):
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "zarr_writing_test.zarr")
        try:
            dataset_neighborhoods_numeric.drop_encoding().to_zarr(output_path)
            assert os.path.exists(output_path) and os.path.isdir(
                output_path
            ), f"Directory {output_path} was not created."
        finally:
            # cleanup is handled by tempfile automatically
            pass


def test_to_zarr_segmentation(dataset_segmentation):
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "zarr_writing_test.zarr")
        try:
            dataset_segmentation.drop_encoding().to_zarr(output_path)
            assert os.path.exists(output_path) and os.path.isdir(
                output_path
            ), f"Directory {output_path} was not created."
        finally:
            # cleanup is handled by tempfile automatically
            pass


def test_to_zarr_binarized(dataset_binarized):
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "zarr_writing_test.zarr")
        try:
            dataset_binarized.drop_encoding().to_zarr(output_path)
            assert os.path.exists(output_path) and os.path.isdir(
                output_path
            ), f"Directory {output_path} was not created."
        finally:
            # cleanup is handled by tempfile automatically
            pass
