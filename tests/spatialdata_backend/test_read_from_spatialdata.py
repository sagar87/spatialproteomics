import numpy as np

import spatialproteomics as sp


def test_read_from_spatialdata_unsorted_table(ds_segmentation):
    sdata = ds_segmentation.pp.add_quantification().tl.convert_to_spatialdata()
    sdata.tables["table"] = sdata.tables["table"][::-1].copy()  # reverse the row order
    obj = sp.read_from_spatialdata(sdata)
    obs = obj.pp.get_layer_as_df()
    np.testing.assert_array_equal(obs["id"].values, obj.coords["cells"].values)
