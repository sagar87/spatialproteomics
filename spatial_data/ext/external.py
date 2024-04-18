from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from ..constants import Dims, Layers


@xr.register_dataset_accessor("ext")
class ExternalAccessor:
    """The external accessor enables the application of external tools such as StarDist or Astir"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def cellpose(
        self,
        key_added: Optional[str] = "_cellpose_segmentation",
        diameter: int = 0,
        channel_settings: list = [0, 0],
        num_iterations: int = 2000,
        gpu: bool = True,
        model_type: str = "cyto3",
    ):
        """
        Segment cells using Cellpose.

        Parameters
        ----------
        key_added : str, optional
            Key to assign to the segmentation results.
        diameter : int, optional
            Expected cell diameter in pixels.
        channel_settings : List[int], optional
            Channels for Cellpose to use for segmentation.
        num_iterations : int, optional
            Maximum number of iterations for segmentation.
        gpu : bool, optional
            Whether to use GPU for segmentation.
        model_type : str, optional
            Type of Cellpose model to use.

        Returns
        -------
        xr.Dataset
            Dataset containing original data and segmentation mask.

        Notes
        -----
        This method requires the 'cellpose' package to be installed.
        """

        from cellpose import models

        model = models.Cellpose(gpu=gpu, model_type=model_type)

        all_masks = []
        for channel in self._obj.coords[Dims.CHANNELS]:
            masks_pred, _, _, _ = model.eval(
                self._obj.pp[channel.item()]._image.values.squeeze(),
                diameter=diameter,
                channels=channel_settings,
                niter=num_iterations,
            )
            all_masks.append(masks_pred)

        if len(all_masks) == 1:
            mask_tensor = np.expand_dims(all_masks[0], 0)
        else:
            mask_tensor = np.stack(all_masks, 0)

        da = xr.DataArray(
            mask_tensor,
            coords=[
                self._obj.coords[Dims.CHANNELS],
                self._obj.coords[Dims.Y],
                self._obj.coords[Dims.X],
            ],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=key_added,
        )

        return xr.merge([self._obj, da])

    def cellpose_denoise(
        self,
        key_added: List[str] = ["_cellpose_denoise", "_cellpose_denoise_segmentation"],
        diameter: int = 0,
        channel_settings: list = [0, 0],
        gpu: bool = True,
        model_type: str = "cyto3",
        restore_type: str = "denoise_cyto3",
        **kwargs,
    ):
        """
        Segment cells using Cellpose.

        Parameters
        ----------
        key_added : str, optional
            Key to assign to the segmentation results.
        diameter : int, optional
            Expected cell diameter in pixels.
        channel_settings : List[int], optional
            Channels for Cellpose to use for segmentation.
        num_iterations : int, optional
            Maximum number of iterations for segmentation.
        gpu : bool, optional
            Whether to use GPU for segmentation.
        model_type : str, optional
            Type of Cellpose model to use.

        Returns
        -------
        xr.Dataset
            Dataset containing original data and segmentation mask.

        Notes
        -----
        This method requires the 'cellpose' package to be installed.
        """

        from cellpose import denoise

        model = denoise.CellposeDenoiseModel(gpu=gpu, model_type=model_type, restore_type=restore_type)

        all_masks = []
        all_imags = []
        for channel in self._obj.coords[Dims.CHANNELS]:
            masks, flows, styles, imgs_dn = model.eval(
                self._obj.pp[channel.item()]._image.values.squeeze(),
                diameter=diameter,
                channels=channel_settings,
                **kwargs,
            )
            all_masks.append(masks)
            all_imags.append(imgs_dn)

        if len(all_masks) == 1:
            mask_tensor = np.expand_dims(all_masks[0], 0)
            img_tensor = np.expand_dims(all_imags[0], 0)
        else:
            mask_tensor = np.stack(all_masks, 0)
            img_tensor = np.stack(all_imags, 0)

        da = xr.DataArray(
            mask_tensor,
            coords=[
                self._obj.coords[Dims.CHANNELS],
                self._obj.coords[Dims.Y],
                self._obj.coords[Dims.X],
            ],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=key_added[1],
        )
        db = xr.DataArray(
            img_tensor,
            coords=[
                self._obj.coords[Dims.CHANNELS],
                self._obj.coords[Dims.Y],
                self._obj.coords[Dims.X],
            ],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=key_added[0],
        )
        return xr.merge([self._obj, da, db])

    def stardist(
        self,
        scale: float = 3,
        n_tiles: int = 12,
        normalize: bool = True,
        nuclear_channel: str = "DAPI",
        predict_big: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Apply StarDist algorithm to perform instance segmentation on the nuclear image.

        Parameters:
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
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the StarDist prediction method.

        Returns:
        -------
        obj : xr.Dataset
            Xarray dataset containing the segmentation mask and centroids.

        Raises:
        ------
        ValueError
            If the object already contains a segmentation mask.

        """
        import csbdeep.utils
        from stardist.models import StarDist2D

        if Layers.SEGMENTATION in self._obj:
            raise ValueError("The object already contains a segmentation mask. StarDist will not be executed.")

        # getting the nuclear image
        nuclear_img = self._obj.pp[nuclear_channel].to_array().values.squeeze()

        # normalizing the image
        if normalize:
            nuclear_img = csbdeep.utils.normalize(nuclear_img)

        # Load the StarDist model
        model = StarDist2D.from_pretrained("2D_versatile_fluo")

        # Predict the label image (different methods for large or small images, see the StarDist documentation for more details)
        if predict_big:
            labels, _ = model.predict_instances_big(nuclear_img, scale=scale, **kwargs)
        else:
            labels, _ = model.predict_instances(nuclear_img, scale=scale, n_tiles=(n_tiles, n_tiles), **kwargs)

        # Adding the segmentation mask  and centroids to the xarray dataset
        return self._obj.pp.add_segmentation(labels).pp.add_observations()

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
            raise ValueError(
                "The image is not of type uint8, which is required for astir to work properly. Use the dtype argument in add_quantification() to convert the image to uint8."
            )

        # converting the xarray to a pandas dataframe to keep track of channel names and indices after running astir
        expression_df = pd.DataFrame(self._obj[key].values, columns=self._obj.coords[Dims.CHANNELS].values)
        expression_df.index = self._obj.coords[Dims.CELLS].values

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
        return self._obj.pp.add_labels(assigned_cell_types, cell_col=cell_id_col, label_col=cell_type_col)
