from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

from ..base_logger import logger
from ..constants import Dims, Layers
from ..pp.utils import _normalize


def _get_channels(obj, key_added, channel):
    if key_added in obj:
        raise KeyError(f'The key "{key_added}" already exists. Please choose another key.')

    if key_added == Layers.SEGMENTATION:
        raise KeyError(f'The key "{Layers.SEGMENTATION}" is reserved, use pp.add_segmentation if necessary.')

    if channel is not None:
        if isinstance(channel, list):
            channels = channel
        else:
            channels = [channel]
    else:
        channels = obj.coords[Dims.CHANNELS].values.tolist()

    return channels


def _convert_masks_to_data_array(obj, all_masks, key_added):
    # if there is one channel, we can squeeze the mask tensor
    if all_masks.shape[0] == 1:
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


def _cellpose(
    img: np.ndarray,
    diameter: float = 0,
    channel_settings: list = [0, 0],
    num_iterations: int = 2000,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    batch_size: int = 8,
    gpu: bool = True,
    model_type: str = "cyto3",
    postprocess_func: Callable = lambda x: x,
    **kwargs,
):
    from cellpose import models

    # checking that the input is 2D or 3D
    if img.ndim not in [2, 3]:
        raise ValueError(f"Input image must be 2D or 3D, got {img.ndim}.")

    # if the input is 2D, we add a channel dimension
    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    if channel_settings != [0, 0]:
        assert (
            img.shape[0] == 2
        ), f"Joint segmentation requires exactly two channels. You set channel_settings to {channel_settings}, but provided {img.shape[0]} channels in the object."

    model = models.Cellpose(gpu=gpu, model_type=model_type)

    all_masks = []
    # if the channels are [0, 0], independent segmentation is performed on all channels
    if channel_settings == [0, 0]:
        if img.shape[0] > 1:
            logger.warn(
                "Performing independent segmentation on all markers. If you want to perform joint segmentation, please set the channel_settings argument appropriately."
            )
        for ch in range(img.shape[0]):
            # get the image at the channel
            masks_pred, _, _, diams = model.eval(
                img[ch].squeeze(),
                diameter=diameter,
                channels=channel_settings,
                niter=num_iterations,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                batch_size=batch_size,
                **kwargs,
            )

            masks_pred = postprocess_func(masks_pred)
            all_masks.append(masks_pred)
    else:
        # if the channels are anything else, joint segmentation is attempted
        masks_pred, _, _, diams = model.eval(
            img.squeeze(),
            diameter=diameter,
            channels=channel_settings,
            niter=num_iterations,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            batch_size=batch_size,
            **kwargs,
        )

        masks_pred = postprocess_func(masks_pred)
        all_masks.append(masks_pred)

    return np.array(all_masks), diams


def _stardist(
    img: np.ndarray,
    scale: float = 3,
    n_tiles: int = 12,
    normalize: bool = True,
    predict_big: bool = False,
    postprocess_func: Callable = lambda x: x,
    **kwargs,
):
    # checking that the input is 2D or 3D
    if img.ndim not in [2, 3]:
        raise ValueError(f"Input image must be 2D or 3D, got {img.ndim}.")

    # if the input is 2D, we add a channel dimension
    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    from stardist.models import StarDist2D

    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    all_masks = []
    for ch in range(img.shape[0]):
        single_image = img[ch]
        if normalize:
            single_image = _normalize(single_image[np.newaxis, :, :])

        # Predict the label image (different methods for large or small images, see the StarDist documentation for more details)
        if predict_big:
            labels, _ = model.predict_instances_big(single_image.squeeze(), scale=scale, **kwargs)
        else:
            labels, _ = model.predict_instances(
                single_image.squeeze(), scale=scale, n_tiles=(n_tiles, n_tiles), **kwargs
            )

        labels = postprocess_func(labels)
        all_masks.append(labels)

    return np.array(all_masks)


def _mesmer(img: np.ndarray, postprocess_func: Callable = lambda x: x, **kwargs):
    # checking that the input is 2D or 3D
    if img.ndim not in [2, 3]:
        raise ValueError(f"Input image must be 2D or 3D, got {img.ndim}.")

    # if the input is 2D, we add a channel dimension
    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    from deepcell.applications import Mesmer

    app = Mesmer()

    # mesmer requires the data to be in shape batch_size (1), x, y, channels (2)
    img = np.expand_dims(np.transpose(img, (1, 2, 0)), 0)

    all_masks = app.predict(img, **kwargs)
    all_masks = postprocess_func(all_masks)

    return np.array(all_masks)


def _astir(
    expression_df: pd.DataFrame,
    marker_dict: dict,
    threshold: float = 0,
    seed: int = 42,
    learning_rate: float = 0.001,
    batch_size: float = 64,
    n_init: int = 5,
    n_init_epochs: int = 5,
    max_epochs: int = 500,
    cell_id_col: str = "cell_id",
    cell_type_col: str = "cell_type",
    **kwargs,
):
    import astir
    import torch

    model = astir.Astir(expression_df, marker_dict, dtype=torch.float64, random_seed=seed)
    model.fit_type(
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_init=n_init,
        n_init_epochs=n_init_epochs,
        max_epochs=max_epochs,
        **kwargs,
    )

    # getting the predictions
    assigned_cell_types = model.get_celltypes(threshold=threshold)
    # assign the index to its own column
    assigned_cell_types = assigned_cell_types.reset_index()
    # renaming the columns
    assigned_cell_types.columns = [cell_id_col, cell_type_col]
    # setting the cell dtype to int
    assigned_cell_types[cell_id_col] = assigned_cell_types[cell_id_col].astype(int)

    return assigned_cell_types
