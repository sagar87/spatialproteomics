import numpy as np
import pytest

from spatialproteomics.constants import Dims, Layers


def test_remove_label_type(dataset_labeled):
    ds = dataset_labeled.la.remove_label_type("Cell type 1")

    assert Layers.LA_PROPERTIES in ds
    assert Dims.LABELS in ds.coords
    assert np.all(ds.coords[Dims.LABELS].values == np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))

    with pytest.raises(ValueError, match="Cell type Cell type 1 not found"):
        ds.la.remove_label_type("Cell type 1")
