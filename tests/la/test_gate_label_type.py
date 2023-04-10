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

    assert ds.attrs["channel"][1] == "CD4"
    assert ds.attrs["threshold"][1] == 1e5
    assert ds.attrs["intensity_key"][1] == "_intensity"
    assert ds.attrs["override"][1] is False
    assert ds.attrs["label_name"][1] == "CT1"
    assert ds.attrs["label_id"][1] == 1
    assert ds.attrs["step"][1] == 1
    # assert ds.attrs["num_cells"][1] == 626

    # add more cell types
    ds = ds.la.add_label_type("CT2")
    ds = ds.la.add_label_type("CT3")

    ds = ds.la.gate_label_type("CT3", "CD8", 2e5, "_intensity")

    assert ds.attrs["channel"][3] == "CD8"
    assert ds.attrs["threshold"][3] == 2e5
    assert ds.attrs["intensity_key"][3] == "_intensity"
    assert ds.attrs["override"][3] is False
    assert ds.attrs["label_name"][3] == "CT3"
    assert ds.attrs["label_id"][3] == 3
    assert ds.attrs["step"][3] == 2
    # assert ds.attrs["num_cells"][3] == 194
