def test_convert_to_spatialdata_image(ds_image):
    ds_image.tl.convert_to_spatialdata()


def test_convert_to_spatialdata_segmentation(ds_segmentation):
    ds_segmentation.tl.convert_to_spatialdata()


def test_convert_to_spatialdata_labels(ds_labels):
    ds_labels.tl.convert_to_spatialdata()


def test_convert_to_spatialdata_neighborhoods(ds_neighborhoods):
    ds_neighborhoods.tl.convert_to_spatialdata()
