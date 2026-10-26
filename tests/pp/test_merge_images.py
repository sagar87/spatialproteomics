import numpy as np
import pytest

from spatialproteomics.pp.utils import merge


def _rgba(color, alpha_value=1.0, shape=(4, 4)):
    image = np.zeros((*shape, 4))
    image[:, :, :3] = color
    image[:, :, 3] = alpha_value
    return image


def test_merge_blend_single_image():
    image = _rgba([1.0, 0.0, 0.0])
    merged = merge([image], proj="blend")

    np.testing.assert_array_equal(merged, image)
    assert merged is not image


def test_merge_blend_does_not_modify_inputs():
    images = [_rgba([1.0, 0.0, 0.0]), _rgba([0.0, 1.0, 0.0]), _rgba([0.0, 0.0, 1.0])]
    originals = [image.copy() for image in images]

    merged = merge(images, proj="blend", alpha=0.5)

    for image, original in zip(images, originals):
        np.testing.assert_array_equal(image, original)
    # the last image is blended on top with alpha 0.5
    np.testing.assert_allclose(merged[0, 0, :3], [0.25, 0.25, 0.5])


def test_merge_unknown_projection():
    with pytest.raises(ValueError, match="Unknown projection method"):
        merge([_rgba([1.0, 0.0, 0.0])], proj="max")
