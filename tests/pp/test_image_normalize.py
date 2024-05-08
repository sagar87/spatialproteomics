import xarray as xr

from spatialproteomics.constants import Layers
from spatialproteomics.pp.utils import _normalize


def test_image_normalize(dataset_full):
    normalized_image = _normalize(dataset_full[Layers.IMAGE])

    assert type(normalized_image) is xr.DataArray
    assert normalized_image.shape[0] == 5
