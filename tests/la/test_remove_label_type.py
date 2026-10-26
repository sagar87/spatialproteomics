import numpy as np
import pytest

from spatialproteomics.constants import Dims, Features, Layers


def test_remove_label_type(ds_labels):
    ds = ds_labels.la.remove_label_type("B")

    assert Layers.LA_PROPERTIES in ds
    assert Dims.LABELS in ds.coords
    assert np.all(ds.coords[Dims.LABELS].values == np.array([2, 3, 4]))
    assert 0 in ds.pp.get_layer_as_df(celltypes_to_str=False)[Features.LABELS].values
    # making sure there was no in-place operation
    assert np.all(ds_labels.coords[Dims.LABELS].values == np.array([1, 2, 3, 4]))
    assert 0 not in ds_labels.coords[Dims.LABELS].values
    assert 0 not in ds_labels.pp.get_layer_as_df(celltypes_to_str=False)[Features.LABELS].values

    with pytest.raises(ValueError, match="Cell type B not found"):
        ds.la.remove_label_type("B")


@pytest.mark.parametrize("cell_types", [[1, 2], ["B", "T"], [1, "T"]])
def test_remove_multiple_label_types(ds_labels, cell_types):
    ds = ds_labels.la.remove_label_type(cell_types)

    assert np.all(ds.coords[Dims.LABELS].values == np.array([3, 4]))
    old_labels = ds_labels.pp.get_layer_as_df(celltypes_to_str=False)[Features.LABELS].values
    new_labels = ds.pp.get_layer_as_df(celltypes_to_str=False)[Features.LABELS].values
    # cells of the removed types are unlabeled, all other cells keep their label
    assert np.all(new_labels[np.isin(old_labels, [1, 2])] == 0)
    assert np.all(new_labels[~np.isin(old_labels, [1, 2])] == old_labels[~np.isin(old_labels, [1, 2])])
