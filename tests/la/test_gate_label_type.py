# import pytest

import spatial_data as sd

# from spatial_data.constants import Dims, Layers


def test_gate_cell_type_adds_graph(dataset_full):

    ds = dataset_full.se.quantify(sd.arcsinh_mean_intensity)
    ds = ds.la.add_label_type("CT1")

    ds = ds.la.gate_label_type("CT1", "CD4", 1e5, "_intensity")

    assert "graph" in ds.attrs
    assert "channel" in ds.attrs
    assert "threshold" in ds.attrs
    assert "intensity_key" in ds.attrs
    assert "override" in ds.attrs
    assert "label_id" in ds.attrs
    assert "step" in ds.attrs
    assert "num_cells" in ds.attrs
