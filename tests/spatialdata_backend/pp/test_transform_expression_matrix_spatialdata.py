import numpy as np
import pytest

import spatialproteomics as sp
from spatialproteomics.constants import SDLayers


def get_expression_matrix(spatialdata_object):
    return spatialdata_object.tables[SDLayers.TABLE].to_df()


def test_invalid_mode(ds_labels_spatialdata):
    with pytest.raises(
        ValueError,
        match="Unknown transformation method: dummy",
    ):
        sp.pp.transform_expression_matrix(ds_labels_spatialdata, method="dummy")


def test_layer_not_available(ds_labels_spatialdata):
    with pytest.raises(
        AssertionError,
        match="Tables key dummy_layer not found in spatial data object",
    ):
        sp.pp.transform_expression_matrix(ds_labels_spatialdata, table_key="dummy_layer")


def test_clip(ds_labels_spatialdata):
    expression_matrix = get_expression_matrix(ds_labels_spatialdata)
    expression_matrix_transformed = get_expression_matrix(
        sp.pp.transform_expression_matrix(ds_labels_spatialdata, method="clip", copy=True)
    )
    assert np.all(expression_matrix.min(axis=0) <= expression_matrix_transformed.min(axis=0))
    assert np.all(expression_matrix.max(axis=0) >= expression_matrix_transformed.max(axis=0))


def test_minmax(ds_labels_spatialdata):
    expression_matrix = get_expression_matrix(
        sp.pp.transform_expression_matrix(ds_labels_spatialdata, method="minmax", copy=True)
    )
    assert np.all(expression_matrix.min(axis=0) == 0)
    assert np.all(expression_matrix.max(axis=0) == 1)


def test_arcsinh(ds_labels_spatialdata):
    expression_matrix = get_expression_matrix(ds_labels_spatialdata)
    expression_matrix_transformed = get_expression_matrix(
        sp.pp.transform_expression_matrix(ds_labels_spatialdata, method="arcsinh", copy=True)
    )
    assert np.all(expression_matrix.max(axis=0) >= expression_matrix_transformed.max(axis=0))


def test_zscore(ds_labels_spatialdata):
    expression_matrix = get_expression_matrix(
        sp.pp.transform_expression_matrix(ds_labels_spatialdata, method="zscore", copy=True)
    )
    assert np.all(round(expression_matrix.mean(axis=0), 2) == 0)
    # can be slightly off depending on the number of cells
    assert np.allclose(expression_matrix.std(axis=0), 1, rtol=0.01)


def test_double_zscore(ds_labels_spatialdata):
    # no tests for this, as the values could be anything
    # this just checks is the method runs in general
    sp.pp.transform_expression_matrix(ds_labels_spatialdata, method="double_zscore")
