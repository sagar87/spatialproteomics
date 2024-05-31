import numpy as np
import xarray as xr

from ..constants import Dims, Layers


def _get_channels(obj, key_added, channel):
    if key_added in obj:
        raise KeyError(f'The key "{key_added}" already exists. Please choose another key.')

    if key_added == Layers.SEGMENTATION:
        raise KeyError(f'The key "{Layers.SEGMENTATION}" is reserved, use pp.add_segmentation if necessary.')

    if channel is not None:
        channels = [channel]
    else:
        channels = obj.coords[Dims.CHANNELS].values.tolist()

    return channels


def _convert_masks_to_data_array(obj, all_masks, key_added):
    # if there is one channel, we can squeeze the mask tensor
    if len(all_masks) == 1:
        da = xr.DataArray(
            all_masks[0].squeeze(),
            coords=[obj.coords[Dims.Y], obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=key_added,
        )
    # if we segment on all of the channels, we need to add the channel dimension
    else:
        da = xr.DataArray(
            np.stack(all_masks, 0),
            coords=[
                obj.coords[Dims.CHANNELS],
                obj.coords[Dims.Y],
                obj.coords[Dims.X],
            ],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=key_added,
        )

    return da
