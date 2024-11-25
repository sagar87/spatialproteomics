from typing import Callable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from ..base_logger import logger
from ..constants import Dims, Features, Layers, Props
from ..pp.utils import _normalize
from .utils import _convert_masks_to_data_array, _get_channels


@xr.register_dataset_accessor("tl")
class ToolAccessor:
    """The tool accessor enables the application of external tools such as StarDist or Astir."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def cellpose(
        self,
        channel: Optional[str] = None,
        key_added: Optional[str] = "_cellpose_segmentation",
        diameter: float = 0,
        channel_settings: list = [0, 0],
        num_iterations: int = 2000,
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        batch_size: int = 8,
        gpu: bool = True,
        model_type: str = "cyto3",
        postprocess_func: Callable = lambda x: x,
        return_diameters: bool = False,
    ):
        """
        Segment cells using Cellpose. Adds a layer to the spatialproteomics object
        with dimension (X, Y) or (C, X, Y) dependent on whether channel argument
        is specified or not.

        Parameters
        ----------
        channel : str, optional
            Channel to use for segmentation. If None, all channels are used.
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

        # checking that if segmentation should be performed jointly, there are exactly two channels
        if channel_settings != [0, 0]:
            assert (
                len(channels) == 2
            ), f"Joint segmentation requires exactly two channels. You set channel_settings to {channel_settings}, but provided {len(channels)} channels in the object."

        from cellpose import models

        model = models.Cellpose(gpu=gpu, model_type=model_type)

        all_masks = []
        # if the channels are [0, 0], independent segmentation is performed on all channels
        if channel_settings == [0, 0]:
            if len(channels) > 1:
                logger.warn(
                    "Performing independent segmentation on all markers. If you want to perform joint segmentation, please set the channel_settings argument appropriately."
                )
            for ch in channels:
                masks_pred, _, _, diams = model.eval(
                    self._obj.pp[ch]._image.values.squeeze(),
                    diameter=diameter,
                    channels=channel_settings,
                    niter=num_iterations,
                    cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold,
                    batch_size=batch_size,
                )

                masks_pred = postprocess_func(masks_pred)
                all_masks.append(masks_pred)
        else:
            # if the channels are anything else, joint segmentation is attempted
            masks_pred, _, _, diams = model.eval(
                self._obj._image.values.squeeze(),
                diameter=diameter,
                channels=channel_settings,
                niter=num_iterations,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                batch_size=batch_size,
            )

            masks_pred = postprocess_func(masks_pred)
            all_masks.append(masks_pred)

        da = _convert_masks_to_data_array(self._obj, all_masks, key_added)

        if return_diameters:
            return xr.merge([self._obj, da]), diams

        return xr.merge([self._obj, da])

    def stardist(
        self,
        channel: Optional[str] = None,
        key_added: Optional[str] = "_stardist_segmentation",
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

        from stardist.models import StarDist2D

        model = StarDist2D.from_pretrained("2D_versatile_fluo")

        all_masks = []
        for ch in channels:
            img = self._obj.pp[ch]._image.values
            if normalize:
                img = _normalize(img)

            # Predict the label image (different methods for large or small images, see the StarDist documentation for more details)
            if predict_big:
                labels, _ = model.predict_instances_big(img.squeeze(), scale=scale, **kwargs)
            else:
                labels, _ = model.predict_instances(img.squeeze(), scale=scale, n_tiles=(n_tiles, n_tiles), **kwargs)

            labels = postprocess_func(labels)
            all_masks.append(labels)

        da = _convert_masks_to_data_array(self._obj, all_masks, key_added)

        return xr.merge([self._obj, da])

    def mesmer(
        self,
        key_added: Optional[str] = "_mesmer_segmentation",
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
        channels = _get_channels(self._obj, key_added, None)

        assert (
            len(channels) == 2
        ), "Mesmer only supports two channels for segmentation. If two channels are provided, the first channel is assumed to be the nuclear channel and the second channel is assumed to be the membrane channel."

        from deepcell.applications import Mesmer

        app = Mesmer()
        # at this point, the shape of image is (channels (2), x, y)
        image = self._obj.pp[channels][Layers.IMAGE].values
        # mesmer requires the data to be in shape batch_size (1), x, y, channels (2)
        image = np.expand_dims(np.transpose(image, (1, 2, 0)), 0)

        all_masks = app.predict(image, **kwargs)
        all_masks = postprocess_func(all_masks)

        da = _convert_masks_to_data_array(self._obj, all_masks, key_added)

        return xr.merge([self._obj, da])

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
        import astir
        import torch

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

        Notes:
        ------
        - The expression matrix is extracted from the spatialproteomics object using the provided expression matrix key.
        - If additional layers are specified, they are extracted from the spatialproteomics object and added to the anndata.AnnData object.
        - If obs_key is present in the spatialproteomics object, it is used to create the obs DataFrame of the anndata.AnnData object.
        - If additional_uns is specified, the corresponding uns data is extracted from the spatialproteomics object and added to the anndata.AnnData object.
        """
        import anndata

        # checking that the expression matrix key is present in the object
        assert (
            expression_matrix_key in self._obj
        ), f"Expression matrix key {expression_matrix_key} not found in the object. Set the expression matrix key with the expression_matrix_key argument."

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

            # if we have labels and colors for them, we add them to the anndata object
            if Dims.LABELS in self._obj.dims and Layers.LA_PROPERTIES in self._obj:
                properties = self._obj.pp.get_layer_as_df(Layers.LA_PROPERTIES)
                if Props.COLOR in properties.columns:
                    # putting it into the anndata object
                    adata.uns[f"{Features.LABELS}_colors"] = list(properties[Props.COLOR].values)

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

        markers = self._obj.coords[Dims.CHANNELS].values
        cells = self._obj.coords[Dims.CELLS].values
        image = spatialdata.models.Image2DModel.parse(
            self._obj[image_key].values, transformations=None, dims=("c", "x", "y"), c_coords=markers
        )
        segmentation = spatialdata.models.Labels2DModel.parse(
            self._obj[segmentation_key].values, transformations=None, dims=("x", "y")
        )

        adata = self._obj.tl.convert_to_anndata(**kwargs)

        # the anndata object within the spatialdata object requires some additional slots, which are created here
        adata.uns["spatialdata_attrs"] = {"region": "segmentation", "region_key": "region", "instance_key": "id"}

        obs_df = pd.DataFrame(
            {
                "id": cells,
                "region": pd.Series(["segmentation"] * len(cells)).astype(
                    pd.api.types.CategoricalDtype(categories=["segmentation"])
                ),
            }
        )
        adata.obs = obs_df

        # transforming the index to string
        adata.obs.index = [str(x) for x in adata.obs.index]

        spatial_data_object = spatialdata.SpatialData(
            images={"image": image}, labels={"segmentation": segmentation}, table=adata
        )

        return spatial_data_object
