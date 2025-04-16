import pytest

import spatialproteomics as sp
from spatialproteomics.constants import Layers


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
