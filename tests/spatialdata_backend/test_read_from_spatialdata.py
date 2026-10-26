import numpy as np

import spatialproteomics as sp


def test_read_from_spatialdata_unsorted_table(ds_segmentation):
    sdata = ds_segmentation.pp.add_quantification().tl.convert_to_spatialdata()
    sdata.tables["table"] = sdata.tables["table"][::-1].copy()  # reverse the row order
    obj = sp.read_from_spatialdata(sdata)
    obs = obj.pp.get_layer_as_df()
    np.testing.assert_array_equal(obs["id"].values, obj.coords["cells"].values)

def test_read_from_spatialdata_missing_table_key(ds_segmentation):
    # a table stored under another key is not used, the observations are computed from the segmentation instead
    sdata = ds_segmentation.pp.add_quantification().tl.convert_to_spatialdata()
    sdata.tables["other_table"] = sdata.tables["table"]
    del sdata.tables["table"]

    obj = sp.read_from_spatialdata(sdata, table_key="table")
    assert "_obs" in obj
    assert len(obj.coords["cells"]) == len(ds_segmentation.coords["cells"])
