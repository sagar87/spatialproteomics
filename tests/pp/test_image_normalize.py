import xarray as xr

from spatialproteomics.constants import Layers
from spatialproteomics.pp.utils import _normalize


def test_image_normalize(ds_image):
    normalized_image = _normalize(ds_image[Layers.IMAGE])

    assert type(normalized_image) is xr.DataArray
    assert normalized_image.shape[0] == 5
