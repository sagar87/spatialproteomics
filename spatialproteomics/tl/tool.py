import copy as cp
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from ..base_logger import logger
from ..constants import Dims, Features, Layers, Props, SDLayers
from ..sd.utils import _get_channels_spatialdata, _process_adata, _process_image
from .utils import (
    _astir,
    _cellpose,
    _compute_transformation,
    _convert_masks_to_data_array,
    _get_channels,
    _mesmer,
    _stardist,
)


# === SPATIALDATA ACCESSOR ===
def cellpose(
    sdata,
    channel: Optional[str] = None,
    image_key: str = SDLayers.IMAGE,
    key_added: str = SDLayers.SEGMENTATION,
    data_key: Optional[str] = None,
    copy: bool = False,
    **kwargs,
):
    """
    This function runs the cellpose segmentation algorithm on the provided image data.
    It extracts the image data from the spatialdata object, applies the cellpose algorithm,
    and adds the segmentation masks to the spatialdata object.
    The segmentation masks are stored in the labels attribute of the spatialdata object.
    The function also handles multiple channels by iterating over the channels and applying the segmentation algorithm to each channel separately.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
        channel (Optional[str]): The channel(s) to be used for segmentation. If None, all channels will be used.
        image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
        key_added (str, optional): The key under which the segmentation masks will be stored in the labels attribute of the spatialdata object. Defaults to segmentation.
        data_key (Optional[str], optional): The key for the image data in the spatialdata object. If None, the image_key will be used. Defaults to None.
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the cellpose algorithm.
    """
    import spatialdata
    from spatialdata.transformations import get_transformation, set_transformation

    if copy:
        sdata = cp.deepcopy(sdata)

    channels = _get_channels_spatialdata(channel)

    # assert that the format is correct and extract the image
    image = _process_image(sdata, channels=channels, image_key=image_key, key_added=key_added, data_key=data_key)

    # run cellpose
    segmentation_masks, _ = _cellpose(image, **kwargs)

    # get transformations
    transformation = get_transformation(sdata[image_key])

    # add the segmentation masks to the spatial data object
    if segmentation_masks.shape[0] > 1:
        for i, channel in enumerate(channels):
            sdata.labels[f"{key_added}_{channel}"] = spatialdata.models.Labels2DModel.parse(
                segmentation_masks[i], transformations=None, dims=("y", "x")
            )
            set_transformation(sdata.labels[f"{key_added}_{channel}"], transformation)
    else:
        sdata.labels[key_added] = spatialdata.models.Labels2DModel.parse(
            segmentation_masks[0], transformations=None, dims=("y", "x")
        )
        set_transformation(sdata.labels[key_added], transformation)

    if copy:
        return sdata


def stardist(
    sdata,
    channel: Optional[str] = None,
    image_key: str = SDLayers.IMAGE,
    key_added: str = SDLayers.SEGMENTATION,
    data_key: Optional[str] = None,
    copy: bool = False,
    **kwargs,
):
    """
    This function runs the stardist segmentation algorithm on the provided image data.
    It extracts the image data from the spatialdata object, applies the stardist algorithm,
    and adds the segmentation masks to the spatialdata object.
    The segmentation masks are stored in the labels attribute of the spatialdata object.
    The function also handles multiple channels by iterating over the channels and applying the segmentation algorithm to each channel separately.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
        channel (Optional[str]): The channel(s) to be used for segmentation. If None, all channels will be used.
        image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
        key_added (str, optional): The key under which the segmentation masks will be stored in the labels attribute of the spatialdata object. Defaults to segmentation.
        data_key (Optional[str], optional): The key for the image data in the spatialdata object. If None, the image_key will be used. Defaults to None.
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the stardist algorithm.
    """
    import spatialdata
    from spatialdata.transformations import get_transformation, set_transformation

    if copy:
        sdata = cp.deepcopy(sdata)

    channels = _get_channels_spatialdata(channel)

    # assert that the format is correct and extract the image
    image = _process_image(sdata, channels=channels, image_key=image_key, key_added=key_added, data_key=data_key)

    # run stardist
    segmentation_masks = _stardist(image, **kwargs)

    # get transformations
    transformation = get_transformation(sdata[image_key])

    # add the segmentation masks to the spatial data object
    if segmentation_masks.shape[0] > 1:
        for i, channel in enumerate(channels):
            sdata.labels[f"{key_added}_{channel}"] = spatialdata.models.Labels2DModel.parse(
                segmentation_masks[i], transformations=None, dims=("y", "x")
            )
            set_transformation(sdata.labels[f"{key_added}_{channel}"], transformation)
    else:
        sdata.labels[key_added] = spatialdata.models.Labels2DModel.parse(
            segmentation_masks[0], transformations=None, dims=("y", "x")
        )
        set_transformation(sdata.labels[key_added], transformation)

    if copy:
        return sdata


def mesmer(
    sdata,
    channel: Optional[str] = None,
    image_key: str = SDLayers.IMAGE,
    key_added: str = SDLayers.SEGMENTATION,
    data_key: Optional[str] = None,
    copy: bool = False,
    **kwargs,
):
    """
    This function runs the mesmer segmentation algorithm on the provided image data.
    It extracts the image data from the spatialdata object, applies the mesmer algorithm,
    and adds the segmentation masks to the spatialdata object.
    The segmentation masks are stored in the labels attribute of the spatialdata object.
    The first channel is assumed to be nuclear and the second one membraneous.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
        channel (Optional[str]): The channel(s) to be used for segmentation.
        image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
        key_added (str, optional): The key under which the segmentation masks will be stored in the labels attribute of the spatialdata object. Defaults to segmentation.
        data_key (Optional[str], optional): The key for the image data in the spatialdata object. If None, the image_key will be used. Defaults to None.
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the mesmer algorithm.
    """
    import spatialdata
    from spatialdata.transformations import get_transformation, set_transformation

    if copy:
        sdata = cp.deepcopy(sdata)

    channels = _get_channels_spatialdata(channel)

    assert (
        len(channels) == 2
    ), "Mesmer only supports two channel segmentation. Please ensure that the first channel is nuclear and the second one is membraneous."

    # assert that the format is correct and extract the image
    image = _process_image(sdata, channels=channels, image_key=image_key, key_added=key_added, data_key=data_key)

    # run mesmer
    segmentation_masks = _mesmer(image, **kwargs)

    # get transformations
    transformation = get_transformation(sdata[image_key])

    # add the segmentation masks to the spatial data object
    sdata.labels[key_added] = spatialdata.models.Labels2DModel.parse(
        segmentation_masks[0].squeeze(), transformations=None, dims=("y", "x")
    )
    set_transformation(sdata.labels[key_added], transformation)

    if copy:
        return sdata


def astir(
    sdata,
    marker_dict: dict,
    table_key=SDLayers.TABLE,
    threshold: float = 0,
    seed: int = 42,
    learning_rate: float = 0.001,
    batch_size: float = 64,
    n_init: int = 5,
    n_init_epochs: int = 5,
    max_epochs: int = 500,
    cell_id_col: str = "cell_id",
    cell_type_col: str = "cell_type",
    copy: bool = False,
    **kwargs,
):
    """
    This function applies the ASTIR algorithm to predict cell types based on the expression matrix.
    It extracts the expression matrix from the spatialdata object, applies the ASTIR algorithm,
    and adds the predicted cell types to the spatialdata object.
    The predicted cell types are stored in the obs attribute of the AnnData object in the tables attribute of the spatialdata object.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the expression matrix.
        marker_dict (dict): A dictionary containing the marker genes for each cell type.
        table_key (str, optional): The key under which the expression matrix is stored in the tables attribute of the spatialdata object. Defaults to "table".
        threshold (float, optional): The threshold value to be used for the ASTIR algorithm. Defaults to 0.
        seed (int, optional): The random seed to be used for the ASTIR algorithm. Defaults to 42.
        learning_rate (float, optional): The learning rate to be used for the ASTIR algorithm. Defaults to 0.001.
        batch_size (float, optional): The batch size to be used for the ASTIR algorithm. Defaults to 64.
        n_init (int, optional): The number of initializations to be used for the ASTIR algorithm. Defaults to 5.
        n_init_epochs (int, optional): The number of initial epochs to be used for the ASTIR algorithm. Defaults to 5.
        max_epochs (int, optional): The maximum number of epochs to be used for the ASTIR algorithm. Defaults to 500.
        cell_id_col (str, optional): The name of the column containing the cell IDs in the expression matrix. Defaults to "cell_id".
        cell_type_col (str, optional): The name of the column containing the cell types in the expression matrix. Defaults to "cell_type".
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the ASTIR algorithm.
    """
    if copy:
        sdata = cp.deepcopy(sdata)

    adata = _process_adata(sdata, table_key=table_key)
    expression_df = adata.to_df()

    assigned_cell_types = _astir(
        expression_df=expression_df,
        marker_dict=marker_dict,
        threshold=threshold,
        seed=seed,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_init=n_init,
        n_init_epochs=n_init_epochs,
        max_epochs=max_epochs,
        cell_id_col=cell_id_col,
        cell_type_col=cell_type_col,
        **kwargs,
    )

    # merging the resulting dataframe to the adata object
    df = pd.DataFrame(adata.obs)
    df = df.merge(assigned_cell_types, left_on="id", right_on=cell_id_col, how="left")
    adata.obs = df.drop(columns=cell_id_col)

    if copy:
        return sdata


# === SPATIALPROTEOMICS ACCESSOR ===
@xr.register_dataset_accessor("tl")
class ToolAccessor:
    """The tool accessor enables the application of external tools such as StarDist or Astir."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def cellpose(
        self,
        channel: Optional[str] = None,
        key_added: str = Layers.SEGMENTATION,
        diameter: float = None,
        channel_settings: list = None,
        num_iterations: int = 2000,
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        batch_size: int = 8,
        gpu: bool = True,
        model_type: str = "cyto3",  # cellpose < 4.0
        pretrained_model: str = "cpsam",  # cellpose 4.0
        postprocess_func: Callable = lambda x: x,
        return_diameters: bool = False,
        **kwargs,
    ):
        """
        Segment cells using Cellpose. Adds a layer to the spatialproteomics object
        with dimension (X, Y) or (C, X, Y) dependent on whether channel argument
        is specified or not.

        Parameters
        ----------
        channel : str, optional
            Channel to use for segmentation. If None, all channels are used for independent segmentation.
        key_added : str, optional
            Key to assign to the segmentation results.
        diameter : float, optional
            Expected cell diameter in pixels.
        channel_settings : List[int], optional
            Channels for Cellpose to use for segmentation. If [0, 0], independent segmentation is performed on all channels. If it is anything else (e. g. [1, 2]), joint segmentation is attempted.
        num_iterations : int, optional
            Maximum number of iterations for segmentation.
        cellprob_threshold : float, optional
            Threshold for cell probability.
        flow_threshold : float, optional
            Threshold for flow.
        batch_size : int, optional
            Batch size for segmentation.
        gpu : bool, optional
            Whether to use GPU for segmentation.
        model_type : str, optional
            Type of Cellpose model to use.
        pretrained_model : str, optional
            Pretrained model to use for Cellpose (4+).
        postprocess_func : Callable, optional
            Function to apply to the segmentation masks after prediction.
        return_diameters : bool, optional
            Whether to return the cell diameters.

        Returns
        -------
        xr.Dataset
            Dataset containing original data and segmentation mask.

        Notes
        -----
        This method requires the 'cellpose' package to be installed.
        """
        channels = _get_channels(self._obj, key_added, channel)

        all_masks, diams = _cellpose(
            self._obj.pp[channels]._image.values,
            diameter=diameter,
            channel_settings=channel_settings,
            num_iterations=num_iterations,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            batch_size=batch_size,
            gpu=gpu,
            model_type=model_type,
            pretrained_model=pretrained_model,
            postprocess_func=postprocess_func,
            **kwargs,
        )

        if all_masks.shape[0] == 1:
            all_masks = all_masks[0].squeeze()
        # if we segment on all of the channels, we need to add the channel dimension
        else:
            all_masks = np.stack(all_masks, 0)

        # if no segmentation exists yet, and no key_added was specified, we make this one the default
        if key_added == Layers.SEGMENTATION:
            if return_diameters:
                return self._obj.pp.add_segmentation(all_masks), diams
            return self._obj.pp.add_segmentation(all_masks)

        da = _convert_masks_to_data_array(self._obj, all_masks, key_added)

        if return_diameters:
            return xr.merge([self._obj, da], join="outer", compat="no_conflicts"), diams

        return xr.merge([self._obj, da], join="outer", compat="no_conflicts")

    def stardist(
        self,
        channel: Optional[str] = None,
        key_added: str = Layers.SEGMENTATION,
        scale: float = 3,
        n_tiles: int = 12,
        normalize: bool = True,
        predict_big: bool = False,
        postprocess_func: Callable = lambda x: x,
        **kwargs,
    ) -> xr.Dataset:
        """
        Apply StarDist algorithm to perform instance segmentation on the nuclear image.

        Parameters
        ----------
        channel : str, optional
            Channel to use for segmentation. If None, all channels are used.
        key_added : str, optional
            Key to write the segmentation results to.
        scale : float, optional
            Scaling factor for the StarDist model (default is 3).
        n_tiles : int, optional
            Number of tiles to split the image into for prediction (default is 12).
        normalize : bool, optional
            Flag indicating whether to normalize the nuclear image (default is True).
        nuclear_channel : str, optional
            Name of the nuclear channel in the image (default is "DAPI").
        predict_big : bool, optional
            Flag indicating whether to use the 'predict_instances_big' method for large images (default is False).
        postprocess_func : Callable, optional
            Function to apply to the segmentation masks after prediction (default is lambda x: x).
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the StarDist prediction method.

        Returns
        -------
        obj : xr.Dataset
            Xarray dataset containing the segmentation mask and centroids.

        Raises
        ------
        ValueError
            If the object already contains a segmentation mask.

        """
        channels = _get_channels(self._obj, key_added, channel)

        all_masks = _stardist(
            self._obj.pp[channels]._image.values,
            scale=scale,
            n_tiles=n_tiles,
            normalize=n_tiles,
            predict_big=predict_big,
            postprocess_func=postprocess_func,
            **kwargs,
        )

        if all_masks.shape[0] == 1:
            all_masks = all_masks[0].squeeze()
        # if we segment on all of the channels, we need to add the channel dimension
        else:
            all_masks = np.stack(all_masks, 0)

        # if no segmentation exists yet, and no key_added was specified, we make this one the default
        if key_added == Layers.SEGMENTATION:
            return self._obj.pp.add_segmentation(all_masks)

        da = _convert_masks_to_data_array(self._obj, all_masks, key_added)

        return xr.merge([self._obj, da], join="outer", compat="no_conflicts")

    def mesmer(
        self,
        key_added: str = Layers.SEGMENTATION,
        channel: Optional[List] = None,
        postprocess_func: Callable = lambda x: x,
        **kwargs,
    ):
        """
        Segment cells using Mesmer. Adds a layer to the spatialproteomics object
        with dimension (C, X, Y). Assumes C is two and has the order (nuclear, membrane).

        Parameters
        ----------
        key_added : str, optional
            Key to assign to the segmentation results.
        channel : List, optional
            Channel to use for segmentation. If None, all channels are used.
        postprocess_func : Callable, optional
            Function to apply to the segmentation masks after prediction.

        Returns
        -------
        xr.Dataset
            Dataset containing original data and segmentation mask.

        Notes
        -----
        This method requires the 'mesmer' package to be installed.
        """
        channels = _get_channels(self._obj, key_added, channel)

        assert (
            len(channels) == 2
        ), "Mesmer only supports two channels for segmentation. If two channels are provided, the first channel is assumed to be the nuclear channel and the second channel is assumed to be the membrane channel. You can set the channels using the 'channel' argument."

        all_masks = _mesmer(self._obj.pp[channels]._image.values, postprocess_func=postprocess_func, **kwargs)

        if all_masks.shape[0] == 1:
            all_masks = all_masks[0].squeeze()
        # if we segment on all of the channels, we need to add the channel dimension
        else:
            all_masks = np.stack(all_masks, 0)

        # if no segmentation exists yet, and no key_added was specified, we make this one the default
        if key_added == Layers.SEGMENTATION:
            return self._obj.pp.add_segmentation(all_masks)

        da = _convert_masks_to_data_array(self._obj, all_masks, key_added)

        return xr.merge([self._obj, da], join="outer", compat="no_conflicts")

    def astir(
        self,
        marker_dict: dict,
        key: str = Layers.INTENSITY,
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
        """
        This method predicts cell types from an expression matrix using the Astir algorithm.

        Parameters
        ----------
        marker_dict : dict
            Dictionary mapping markers to cell types. Can also include cell states. Example: {"cell_type": {'B': ['PAX5'], 'T': ['CD3'], 'Myeloid': ['CD11b']}}
        key : str, optional
            Layer to use as expression matrix.
        threshold : float, optional
            Certainty threshold for astir to assign a cell type. Defaults to 0.
        seed : int, optional
            Random seed. Defaults to 42.
        learning_rate : float, optional
            Learning rate. Defaults to 0.001.
        batch_size : float, optional
            Batch size. Defaults to 64.
        n_init : int, optional
            Number of initializations. Defaults to 5.
        n_init_epochs : int, optional
            Number of epochs for each initialization. Defaults to 5.
        max_epochs : int, optional
            Maximum number of epochs. Defaults to 500.
        cell_id_col : str, optional
            Column name for cell IDs. Defaults to "cell_id".
        cell_type_col : str, optional
            Column name for cell types. Defaults to "cell_type".

        Raises
        ------
        ValueError
            If no expression matrix was present or the image is not of type uint8.

        Returns
        -------
        DataArray
            A DataArray with the assigned cell types.
        """
        # check if there is an expression matrix
        if key not in self._obj:
            raise ValueError(
                f"No expression matrix with key {key} found in the object. Make sure to call pp.quantify first."
            )

        # raise an error if the image is of anything other than uint8
        if self._obj[Layers.IMAGE].dtype != "uint8":
            logger.warning(
                "The image is not of type uint8, which is required for astir to work properly. Use pp.convert_to_8bit() to convert the image to uint8. If you applied operations such as filtering, you may ignore this warning."
            )

        # warn the user if the input dict has the wrong format
        if "cell_type" not in marker_dict.keys():
            logger.warning(
                "Did not find 'cell_type' key in the marker_dict. Your dictionary should have the following structure: {'cell_type': {'B': ['PAX5'], 'T': ['CD3'], 'Myeloid': ['CD11b']}}."
            )

        # converting the xarray to a pandas dataframe to keep track of channel names and indices after running astir
        expression_df = self._obj.pp.get_layer_as_df(key)

        # running astir
        assigned_cell_types = _astir(
            expression_df=expression_df,
            marker_dict=marker_dict,
            threshold=threshold,
            seed=seed,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_init=n_init,
            n_init_epochs=n_init_epochs,
            max_epochs=max_epochs,
            cell_id_col=cell_id_col,
            cell_type_col=cell_type_col,
        )

        # adding the labels to the xarray object
        return self._obj.la.add_labels_from_dataframe(
            assigned_cell_types, cell_col=cell_id_col, label_col=cell_type_col
        )

    def convert_to_anndata(
        self,
        expression_matrix_key: str = Layers.INTENSITY,
        obs_key: str = Layers.OBS,
        additional_layers: Optional[dict] = None,
        additional_uns: Optional[dict] = None,
    ):
        """
        Convert the spatialproteomics object to an anndata.AnnData object. The resulting AnnData object does not store the original image or segmentation mask.

        Parameters
        ----------
        expression_matrix_key : str, optional
            The key of the expression matrix in the spatialproteomics object. Default is '_intensity'.
        obs_key : str, optional
            The key of the observation data in the spatialproteomics object. Default is '_obs'.
        additional_layers : dict, optional
            Additional layers to include in the anndata.AnnData object. The keys are the names of the layers and the values are the corresponding keys in the spatialproteomics object.
        additional_uns : dict, optional
            Additional uns data to include in the anndata.AnnData object. The keys are the names of the uns data and the values are the corresponding keys in the spatialproteomics object.

        Returns
        -------
        adata : anndata.AnnData
            The converted anndata.AnnData object.

        Raises
        ------
        AssertionError
            If the expression matrix key or additional layers are not found in the spatialproteomics object.

        Notes
        -----
        - The expression matrix is extracted from the spatialproteomics object using the provided expression matrix key.
        - If additional layers are specified, they are extracted from the spatialproteomics object and added to the anndata.AnnData object.
        - If obs_key is present in the spatialproteomics object, it is used to create the obs DataFrame of the anndata.AnnData object.
        - If additional_uns is specified, the corresponding uns data is extracted from the spatialproteomics object and added to the anndata.AnnData object.
        """
        import anndata

        # if there is no expression matrix, we return an empty anndata object
        if expression_matrix_key not in self._obj:
            return anndata.AnnData()

        expression_matrix = self._obj[expression_matrix_key].values
        adata = anndata.AnnData(expression_matrix)
        if additional_layers:
            for key, layer in additional_layers.items():
                # checking that the additional layer is present in the object
                assert layer in self._obj, f"Layer {layer} not found in the object."
                adata.layers[key] = self._obj[layer].values
        adata.var_names = self._obj.coords[Dims.CHANNELS].values

        if obs_key in self._obj:
            adata.obs = self._obj.pp.get_layer_as_df(obs_key, idx_to_str=True)

            # converting the labels into categorical
            if Dims.LABELS in self._obj.dims:
                adata.obs[Features.LABELS] = adata.obs[Features.LABELS].astype("category")

            # if we have labels and colors for them, we add them to the anndata object
            if Dims.LABELS in self._obj.dims and Layers.LA_PROPERTIES in self._obj:
                properties = self._obj.pp.get_layer_as_df(Layers.LA_PROPERTIES)
                if Props.COLOR in properties.columns:
                    # putting it into the anndata object
                    adata.uns[f"{Features.LABELS}_colors"] = properties[Props.COLOR].values
                # if the labels are in there, we want to use the order of them for the categories
                if Props.NAME in properties.columns:
                    adata.obs[Features.LABELS] = adata.obs[Features.LABELS].cat.set_categories(
                        properties[Props.NAME].values, ordered=True
                    )

            # to be compatible with squidpy out of the box, a spatial key is added to obsm if possible
            if Features.X in adata.obs and Features.Y in adata.obs:
                adata.obs[Features.X] = adata.obs[Features.X].astype(float)
                adata.obs[Features.Y] = adata.obs[Features.Y].astype(float)
                adata.obsm["spatial"] = np.array(adata.obs[[Features.X, Features.Y]])

        if additional_uns:
            for key, layer in additional_uns.items():
                # checking that the additional layer is present in the object
                assert layer in self._obj, f"Layer {layer} not found in the object."
                adata.uns[key] = self._obj.pp.get_layer_as_df(layer)

        return adata

    def convert_to_spatialdata(
        self, image_key: str = Layers.IMAGE, segmentation_key: str = Layers.SEGMENTATION, **kwargs
    ):
        """
        Convert the spatialproteomics object to a spatialdata object.

        Parameters:
            image_key (str): The key of the image data in the object. Defaults to Layers.IMAGE.
            segmentation_key (str): The key of the segmentation data in the object. Defaults to Layers.SEGMENTATION.
            **kwargs: Additional keyword arguments to be passed to the convert_to_anndata method.

        Returns:
            spatial_data_object (spatialdata.SpatialData): The converted spatialdata object.
        """
        import spatialdata
        from spatialdata.transformations import set_transformation

        store_segmentation = False
        store_adata = False

        markers = self._obj.coords[Dims.CHANNELS].values
        image = spatialdata.models.Image2DModel.parse(
            self._obj[image_key].values, transformations=None, dims=("c", "x", "y"), c_coords=markers
        )
        if Dims.CELLS in self._obj.coords:
            cells = self._obj.coords[Dims.CELLS].values
            segmentation = spatialdata.models.Labels2DModel.parse(
                self._obj[segmentation_key].values, transformations=None, dims=("x", "y")
            )
            store_segmentation = True

        adata = self._obj.tl.convert_to_anndata(**kwargs)

        if adata.X is not None:
            # the anndata object within the spatialdata object requires some additional slots, which are created here
            adata.uns["spatialdata_attrs"] = {
                "region": SDLayers.SEGMENTATION,
                "region_key": "region",
                "instance_key": "id",
            }

            if adata.obs is not None:
                adata.obs["id"] = cells
                adata.obs["region"] = pd.Series([SDLayers.SEGMENTATION] * adata.n_obs, index=adata.obs.index).astype(
                    pd.api.types.CategoricalDtype(categories=[SDLayers.SEGMENTATION])
                )
            else:
                obs_df = pd.DataFrame(
                    {
                        "id": cells,
                        "region": pd.Series([SDLayers.SEGMENTATION] * len(cells)).astype(
                            pd.api.types.CategoricalDtype(categories=[SDLayers.SEGMENTATION])
                        ),
                    }
                )
                adata.obs = obs_df
            # anndata insists that the obs_names are strings, and will throw a warning if they are not
            # to be consistent with their examples, we add the "Cell_" prefix here
            adata.obs_names = [f"Cell_{x}" for x in cells]

            # transforming the index to string
            adata.obs.index = [str(x) for x in adata.obs.index]
            store_adata = True

        # Your known crop origin in the global coordinate space
        transformation = _compute_transformation(self._obj.coords["x"].values, self._obj.coords["y"].values)

        # storing only the info that is present in the object (image/segmentation/anndata)
        # we can only have an anndata if we also have a segmentation mask
        if store_segmentation:
            if store_adata:
                spatial_data_object = spatialdata.SpatialData(
                    images={SDLayers.IMAGE: image},
                    labels={SDLayers.SEGMENTATION: segmentation},
                    tables={SDLayers.TABLE: adata},
                )
            else:
                spatial_data_object = spatialdata.SpatialData(
                    images={SDLayers.IMAGE: image}, labels={SDLayers.SEGMENTATION: segmentation}
                )

            set_transformation(
                spatial_data_object.labels[SDLayers.SEGMENTATION], transformation, to_coordinate_system="global"
            )
        else:
            spatial_data_object = spatialdata.SpatialData(images={SDLayers.IMAGE: image})

        set_transformation(spatial_data_object.images[SDLayers.IMAGE], transformation, to_coordinate_system="global")

        return spatial_data_object
