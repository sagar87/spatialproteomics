import numpy as np
import pytest

from spatialproteomics import Layers


def test_invalid_mode(dataset_labeled):
    with pytest.raises(
        ValueError,
        match="Unknown transformation method: dummy",
    ):
        dataset_labeled.pp.add_quantification().pp.transform_expression_matrix(method="dummy")


def test_layer_not_available(dataset_labeled):
    with pytest.raises(
        AssertionError,
        match="No expression matrix found at layer dummy_layer",
    ):
        dataset_labeled.pp.add_quantification().pp.transform_expression_matrix(key="dummy_layer")


def test_clip(dataset_labeled):
    expression_matrix = dataset_labeled.pp.add_quantification().pp.get_layer_as_df(Layers.INTENSITY)
    expression_matrix_transformed = (
        dataset_labeled.pp.add_quantification()
        .pp.transform_expression_matrix(method="clip")
        .pp.get_layer_as_df(Layers.INTENSITY)
    )
    assert np.all(expression_matrix.min(axis=0) <= expression_matrix_transformed.min(axis=0))
    assert np.all(expression_matrix.max(axis=0) >= expression_matrix_transformed.max(axis=0))


def test_minmax(dataset_labeled):
    expression_matrix = (
        dataset_labeled.pp.add_quantification()
        .pp.transform_expression_matrix(method="minmax")
        .pp.get_layer_as_df(Layers.INTENSITY)
    )
    assert np.all(expression_matrix.min(axis=0) == 0)
    assert np.all(expression_matrix.max(axis=0) == 1)


def test_arcsinh(dataset_labeled):
    expression_matrix = dataset_labeled.pp.add_quantification().pp.get_layer_as_df(Layers.INTENSITY)
    expression_matrix_transformed = (
        dataset_labeled.pp.add_quantification()
        .pp.transform_expression_matrix(method="arcsinh")
        .pp.get_layer_as_df(Layers.INTENSITY)
    )
    assert np.all(expression_matrix.max(axis=0) >= expression_matrix_transformed.max(axis=0))


def test_zscore(dataset_labeled):
    expression_matrix = (
        dataset_labeled.pp.add_quantification()
        .pp.transform_expression_matrix(method="zscore")
        .pp.get_layer_as_df(Layers.INTENSITY)
    )
    assert np.all(round(expression_matrix.mean(axis=0), 2) == 0)
    assert np.all(round(expression_matrix.std(axis=0), 2) == 1)


def test_double_zscore(dataset_labeled):
    # no tests for this, as the values could be anything
    # this just checks is the method runs in general
    dataset_labeled.pp.add_quantification().pp.transform_expression_matrix(method="double_zscore").pp.get_layer_as_df(
        Layers.INTENSITY
    )
