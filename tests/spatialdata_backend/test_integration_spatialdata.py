import copy as cp

import spatialproteomics as sp
from spatialproteomics.constants import SDLayers

marker_ct_dict = {
    "PAX5": "B",
    "CD3": "T",
}

threshold_dict = {"CD4": 0.5, "CD8": 0.5}

subtype_dict = {
    "T": {
        "subtypes": [
            {"name": "T_h", "markers": ["CD4+"]},
            {"name": "T_tox", "markers": ["CD8+"]},
        ]
    },
}


def test_integration(ds_segmentation_spatialdata):
    ds = cp.deepcopy(ds_segmentation_spatialdata)

    # cell type prediction
    sp.pp.add_quantification(ds)
    sp.la.predict_cell_types_argmax(ds, marker_ct_dict)

    # cell subtype prediction
    sp.pp.threshold(ds, quantile=[0.8, 0.5], channels=["CD4", "CD8"], key_added="image_thresholded")
    sp.pp.add_quantification(
        ds, func=sp.percentage_positive, image_key="image_thresholded", layer_key="percentage_positive"
    )
    sp.la.threshold_labels(ds, threshold_dict, layer_key="percentage_positive")
    sp.la.predict_cell_subtypes(ds, subtype_dict)

    assert SDLayers.IMAGE in ds.images.keys()
    assert SDLayers.SEGMENTATION in ds.labels.keys()
    assert SDLayers.TABLE in ds.tables.keys()
