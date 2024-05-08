import xarray as xr

from spatial_proteomics.constants import Layers
from spatial_proteomics.pp.utils import _normalize


def test_image_normalize(dataset_full):
    normalized_image = _normalize(dataset_full[Layers.IMAGE])

    assert type(normalized_image) is xr.DataArray
    assert normalized_image.shape[0] == 5
