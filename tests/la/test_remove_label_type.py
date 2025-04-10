import numpy as np
import pytest

from spatialproteomics.constants import Dims, Layers


def test_remove_label_type(ds_labels):
    ds = ds_labels.la.remove_label_type("B")

    assert Layers.LA_PROPERTIES in ds
    assert Dims.LABELS in ds.coords
    assert np.all(ds.coords[Dims.LABELS].values == np.array([2, 3, 4]))

    with pytest.raises(ValueError, match="Cell type B not found"):
        ds.la.remove_label_type("B")
