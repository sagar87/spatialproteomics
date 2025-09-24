import spatialproteomics as sp

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


def test_integration(ds_segmentation):
    # cell type prediction
    ds = ds_segmentation.pp.add_quantification().la.predict_cell_types_argmax(marker_ct_dict, key="_intensity")

    # cell subtype prediction
    ds = ds.pp.threshold(quantile=[0.8, 0.5], channels=["CD4", "CD8"]).pp.add_quantification(
        func=sp.percentage_positive, key_added="_percentage_positive"
    )
    ds = ds.la.threshold_labels(threshold_dict, layer_key="_percentage_positive")
    ds = ds.la.predict_cell_subtypes(subtype_dict)

    # neighborhood prediction
    sp_dict = {"1": ds}
    image_container = sp.ImageContainer(sp_dict)
    sp_dict = image_container.compute_neighborhoods()
    sp_dict["1"] = sp_dict["1"].nh.set_neighborhood_name(
        [f"Neighborhood {x}" for x in range(5)], [f"Neighborhood {x + 1}" for x in range(5)]
    )
