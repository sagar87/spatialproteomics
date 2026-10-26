import numpy as np
import pytest

import spatialproteomics as sp
from spatialproteomics.constants import Dims, Features, Layers


def test_image_container(ds_labels):
    input_dict = {"id_1": ds_labels, "id_2": ds_labels}
    image_container = sp.ImageContainer(input_dict)
    assert image_container.objects == input_dict


def test_image_container_compute_neighborhoods(ds_labels):
    input_dict = {"id_1": ds_labels, "id_2": ds_labels}
    image_container = sp.ImageContainer(input_dict)
    output_dict = image_container.compute_neighborhoods()
    assert Layers.NEIGHBORHOODS in output_dict["id_1"]
    assert Layers.NH_PROPERTIES in output_dict["id_1"]
    assert Layers.NEIGHBORHOODS in output_dict["id_2"]
    assert Layers.NH_PROPERTIES in output_dict["id_2"]


def test_image_container_compute_neighborhoods_already_exist(ds_labels):
    input_dict = {"id_1": ds_labels, "id_2": ds_labels}
    image_container = sp.ImageContainer(input_dict)
    output_dict = image_container.compute_neighborhoods()
    image_container = sp.ImageContainer(output_dict)
    with pytest.raises(ValueError, match="Neighborhoods are already present in the objects"):
        image_container.compute_neighborhoods()

    # checking that it does work if we use the overwrite argument
    image_container.compute_neighborhoods(overwrite=True)


def test_image_container_get_neighborhood_composition_no_neighborhoods(ds_labels):
    input_dict = {"id_1": ds_labels, "id_2": ds_labels}
    image_container = sp.ImageContainer(input_dict)

    with pytest.raises(AssertionError, match="Neighborhoods have not been computed yet"):
        image_container.get_neighborhood_composition()


def test_image_container_get_neighborhood_composition(ds_labels):
    input_dict = {"id_1": ds_labels, "id_2": ds_labels}
    image_container = sp.ImageContainer(input_dict)
    output_dict = image_container.compute_neighborhoods()
    image_container = sp.ImageContainer(output_dict)
    neighborhood_composition = image_container.get_neighborhood_composition()
    assert neighborhood_composition.shape == (5, 4)  # 5 neighborhoods, 4 columns (id, neighborhood, cell_type, count)


def test_image_container_get_neighborhood_composition_custom_column(ds_labels):
    # neighborhoods stored in a custom obs column can be used when the container is created from existing objects
    image_container = sp.ImageContainer({"id_1": ds_labels, "id_2": ds_labels})
    output_dict = image_container.compute_neighborhoods()
    renamed = {}
    for key, obj in output_dict.items():
        neighborhoods = obj[Layers.OBS].sel({Dims.FEATURES: Features.NEIGHBORHOODS}).values
        renamed[key] = obj.pp.add_feature("my_neighborhoods", neighborhoods)

    expected = sp.ImageContainer(output_dict).get_neighborhood_composition(standardize=False)
    image_container = sp.ImageContainer(renamed)
    neighborhood_composition = image_container.get_neighborhood_composition(
        standardize=False, neighborhood_col="my_neighborhoods"
    )

    assert neighborhood_composition.shape == expected.shape
    assert np.allclose(neighborhood_composition.values, expected.values)
    # the stored neighborhood dataframe is not modified
    assert Features.NEIGHBORHOODS not in image_container.neighborhood_df.columns
