import numpy as np

import spatial_data as sd

def test_gate_cell_type_adds_graph(dataset_full):

    ds = dataset_full.pp.add_quantification(func=sd.arcsinh_mean_intensity)
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

    assert ds.attrs["channel"][1] == ["CD4"]
    assert ds.attrs["threshold"][1] == 1e5
    assert ds.attrs["intensity_key"][1] == "_intensity"
    assert ds.attrs["override"][1] is False
    assert ds.attrs["label_name"][1] == "CT1"
    assert ds.attrs["label_id"][1] == 1
    assert ds.attrs["step"][1] == 1

    # add more cell types
    ds = ds.la.add_label_type("CT2")
    ds = ds.la.add_label_type("CT3")
    
    ds = ds.la.gate_label_type("CT3", "CD8", 2e5, "_intensity")

    assert ds.attrs["channel"][3] == ["CD8"]
    assert ds.attrs["threshold"][3] == 2e5
    assert ds.attrs["intensity_key"][3] == "_intensity"
    assert ds.attrs["override"][3] is False
    assert ds.attrs["label_name"][3] == "CT3"
    assert ds.attrs["label_id"][3] == 3
    assert ds.attrs["step"][3] == 2


def test_remove_label_type(full_zarr):
    cell_type = "Vascular"
    channel = "CD31"
    intensity = 0.5
    intensity_channel = "_transformed"
    override = False
    parent = 0

    sdata = full_zarr.la.add_label_type(cell_type, "C0")
    sdata = sdata.la.gate_label_type(cell_type, channel, intensity, intensity_channel, override, parent)

    assert sdata.attrs["channel"][1] == [channel]
    assert sdata.attrs["threshold"][1] == intensity
    assert sdata.attrs["intensity_key"][1] == intensity_channel
    assert sdata.attrs["override"][1] is override
    assert sdata.attrs["label_name"][1] == cell_type
    assert sdata.attrs["label_id"][1] == 1
    assert sdata.attrs["step"][1] == 1

    vascular = sdata.la.reset_label_type(cell_type)

    assert 1 not in vascular.attrs["channel"].keys()
    assert 1 not in vascular.attrs["threshold"].keys()
    assert 1 not in vascular.attrs["intensity_key"].keys()
    assert 1 not in vascular.attrs["override"].keys()
    assert 1 not in vascular.attrs["label_name"].keys()
    assert 1 not in vascular.attrs["label_id"].keys()
    assert 1 not in vascular.attrs["step"].keys()

    # more complex scenario with parent cells
    cell_type = "T cell"
    channel = "CD3"
    intensity = 1.1
    intensity_channel = "_transformed"
    override = False
    parent = 0

    sdata = sdata.la.add_label_type(cell_type, "C0")
    sdata = sdata.la.gate_label_type(cell_type, channel, intensity, intensity_channel, override, parent)

    assert sdata.attrs["channel"][2] == [channel]
    assert sdata.attrs["threshold"][2] == intensity
    assert sdata.attrs["intensity_key"][2] == intensity_channel
    assert sdata.attrs["override"][2] is override
    assert sdata.attrs["label_name"][2] == cell_type
    assert sdata.attrs["label_id"][2] == 2
    assert sdata.attrs["step"][2] == 2

    num_gated_cells = len(sdata.attrs["gated_cells"][2])

    cell_type = "T cell CD8+"
    channel = "CD8"
    intensity = 1.5
    intensity_channel = "_transformed"
    override = False
    parent = "T cell"

    sdata = sdata.la.add_label_type(cell_type, "C1")
    sdata = sdata.la.gate_label_type(cell_type, channel, intensity, intensity_channel, override, parent)

    assert sdata.attrs["channel"][3] == [channel]
    assert sdata.attrs["threshold"][3] == intensity
    assert sdata.attrs["intensity_key"][3] == intensity_channel
    assert sdata.attrs["override"][3] is override
    assert sdata.attrs["label_name"][3] == cell_type
    assert sdata.attrs["label_id"][3] == 3
    assert sdata.attrs["step"][3] == 3

    # removing CD8+ T cells should revert them to CD3+ T cells
    sdata = sdata.la.reset_label_type(cell_type)

    assert 3 not in sdata.attrs["channel"].keys()
    assert 3 not in sdata.attrs["threshold"].keys()
    assert 3 not in sdata.attrs["intensity_key"].keys()
    assert 3 not in sdata.attrs["override"].keys()
    assert 3 not in sdata.attrs["label_name"].keys()
    assert 3 not in sdata.attrs["label_id"].keys()
    assert 3 not in sdata.attrs["step"].keys()
    # checks if the number of CD3+ T cells has been restored
    assert len(sdata.attrs["gated_cells"][2]) == num_gated_cells

    # add CD8+ T cells again
    sdata = sdata.la.gate_label_type(cell_type, channel, intensity, intensity_channel, override, parent)

    assert sdata.attrs["channel"][3] == [channel]
    assert sdata.attrs["threshold"][3] == intensity
    assert sdata.attrs["intensity_key"][3] == intensity_channel
    assert sdata.attrs["override"][3] is override
    assert sdata.attrs["label_name"][3] == cell_type
    assert sdata.attrs["label_id"][3] == 3
    assert sdata.attrs["step"][3] == 3

    # now remove CD3+ T cell (parent class of CD8+ T cells) 
    # should delete all T cells including CD8+ T cells
    sdata = sdata.la.reset_label_type("T cell")

    assert 3 not in sdata.attrs["channel"].keys()
    assert 3 not in sdata.attrs["threshold"].keys()
    assert 3 not in sdata.attrs["intensity_key"].keys()
    assert 3 not in sdata.attrs["override"].keys()
    assert 3 not in sdata.attrs["label_name"].keys()
    assert 3 not in sdata.attrs["label_id"].keys()
    assert 3 not in sdata.attrs["step"].keys()

    assert 2 not in sdata.attrs["channel"].keys()
    assert 2 not in sdata.attrs["threshold"].keys()
    assert 2 not in sdata.attrs["intensity_key"].keys()
    assert 2 not in sdata.attrs["override"].keys()
    assert 2 not in sdata.attrs["label_name"].keys()
    assert 2 not in sdata.attrs["label_id"].keys()
    assert 2 not in sdata.attrs["step"].keys()

    # more complex tree
    cell_type = "T cell"
    channel = "CD3"
    intensity = 1.1
    intensity_channel = "_transformed"
    override = False
    parent = 0

    sdata = sdata.la.gate_label_type(cell_type, channel, intensity, intensity_channel, override, parent)

    cell_type = "T cell CD8+"
    channel = "CD8"
    intensity = 1.5
    intensity_channel = "_transformed"
    override = False
    parent = "T cell"

    sdata = sdata.la.gate_label_type(cell_type, channel, intensity, intensity_channel, override, parent)

    cell_type = "T cell CD4+"
    channel = "CD4"
    intensity = 1.5
    intensity_channel = "_transformed"
    override = False
    parent = "T cell"

    sdata = sdata.la.add_label_type(cell_type, "C2")
    sdata = sdata.la.gate_label_type(cell_type, channel, intensity, intensity_channel, override, parent)

    cell_type = "T-reg"
    channel = "FOXP3"
    intensity = 0.3
    intensity_channel = "_transformed"
    override = False
    parent = "T cell CD4+"

    sdata = sdata.la.add_label_type(cell_type, "C4")
    sdata = sdata.la.gate_label_type(cell_type, channel, intensity, intensity_channel, override, parent)

    assert 0 in sdata.attrs["graph"].keys()
    assert 1 in sdata.attrs["graph"].keys()
    assert 2 in sdata.attrs["graph"].keys()
    assert 3 in sdata.attrs["graph"].keys()
    assert 4 in sdata.attrs["graph"].keys()
    assert 5 in sdata.attrs["graph"].keys()

    treg_removed = sdata.la.reset_label_type("T-reg")

    assert 0 in treg_removed.attrs["graph"].keys()
    assert 1 in treg_removed.attrs["graph"].keys()
    assert 2 in treg_removed.attrs["graph"].keys()
    assert 3 in treg_removed.attrs["graph"].keys()
    assert 4 in treg_removed.attrs["graph"].keys()
    assert 5 not in treg_removed.attrs["graph"].keys()
    assert treg_removed.attrs["num_cells"][4] == sdata.attrs["num_cells"][4] + sdata.attrs["num_cells"][5]
    assert np.all(np.isin(sdata.attrs["gated_cells"][5], treg_removed.attrs["gated_cells"][4]))

    cd4_removed = sdata.la.reset_label_type("T cell CD4+")

    assert 0 in cd4_removed.attrs["graph"].keys()
    assert 1 in cd4_removed.attrs["graph"].keys()
    assert 2 in cd4_removed.attrs["graph"].keys()
    assert 3 in cd4_removed.attrs["graph"].keys()

    assert 4 not in cd4_removed.attrs["graph"].keys()
    assert 5 not in cd4_removed.attrs["graph"].keys()

    assert (
        cd4_removed.attrs["num_cells"][2]
        == sdata.attrs["num_cells"][2] + sdata.attrs["num_cells"][4] + sdata.attrs["num_cells"][5]
    )

    assert np.all(np.isin(sdata.attrs["gated_cells"][5], cd4_removed.attrs["gated_cells"][2]))
    assert np.all(np.isin(sdata.attrs["gated_cells"][4], cd4_removed.attrs["gated_cells"][2]))
