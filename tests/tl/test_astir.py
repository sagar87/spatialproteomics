import importlib.util
import sys
import types

import numpy as np
import pandas as pd
import pytest

import spatialproteomics as sp
from spatialproteomics.constants import Dims, Layers


@pytest.fixture
def fake_astir(monkeypatch):
    """Replaces astir (and torch) with a stub that assigns cell types in order, so that the wiring can be tested."""

    class FakeAstir:
        def __init__(self, expression_df, marker_dict, dtype=None, random_seed=None):
            self.index = expression_df.index
            self.cell_types = list(marker_dict["cell_type"].keys())

        def fit_type(self, **kwargs):
            pass

        def get_celltypes(self, threshold=0):
            # astir returns the cell types indexed by the (stringified) cell names of the expression matrix
            types_ = [self.cell_types[i % len(self.cell_types)] for i in range(len(self.index))]
            return pd.DataFrame({"cell_type": types_}, index=self.index.astype(str))

    monkeypatch.setitem(sys.modules, "astir", types.SimpleNamespace(Astir=FakeAstir))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float64="float64"))
    return FakeAstir


MARKER_DICT = {"cell_type": {"B": ["PAX5"], "T": ["CD3"]}}


@pytest.mark.skipif(importlib.util.find_spec("spatialdata") is None, reason="spatialdata is not installed")
def test_astir_spatialdata_with_string_obs_names(ds_segmentation, fake_astir):
    sdata = ds_segmentation.tl.convert_to_spatialdata()
    sp.pp.add_quantification(sdata)
    adata = sdata.tables["table"]
    obs_names = adata.obs_names.copy()
    assert obs_names[0] == "Cell_1"

    sp.tl.astir(sdata, MARKER_DICT)

    adata = sdata.tables["table"]
    assert np.array_equal(adata.obs_names, obs_names)
    assert "cell_type" in adata.obs.columns
    assert "cell_id" not in adata.obs.columns
    expected = ["B" if i % 2 == 0 else "T" for i in range(adata.n_obs)]
    assert list(adata.obs["cell_type"]) == expected


def test_astir_xarray_with_int_cell_ids(ds_segmentation, fake_astir):
    ds = ds_segmentation.pp.add_quantification(func="intensity_mean")
    ds = ds.tl.astir(MARKER_DICT)

    assert Layers.LA_PROPERTIES in ds
    assert ds.sizes[Dims.CELLS] == ds_segmentation.sizes[Dims.CELLS]
