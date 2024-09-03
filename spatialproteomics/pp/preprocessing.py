from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import skimage
import xarray as xr
from scipy.stats import norm, zscore
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels

from ..base_logger import logger
from ..constants import COLORS, Attrs, Dims, Features, Labels, Layers, Props
from ..la.utils import _format_labels
from .utils import (
    _get_disconnected_cell,
    _merge_segmentation,
    _normalize,
    _relabel_cells,
    _remove_unlabeled_cells,
)


@xr.register_dataset_accessor("pp")
class PreprocessingAccessor:
    """The image accessor enables fast indexing and preprocessing of the spatialproteomics object."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __getitem__(self, indices) -> xr.Dataset:
        """
        Fast subsetting the image container. The following examples show how
        the user can subset the image container:

        Subset the image container using x and y coordinates:
        >>> ds.pp[0:50, 0:50]

        Subset the image container using x and y coordinates and channels:
        >>> ds.pp['Hoechst', 0:50, 0:50]

        Subset the image container using channels:
        >>> ds.pp['Hoechst']

        Multiple channels can be selected by passing a list of channels:
        >>> ds.pp[['Hoechst', 'CD4']]

        Parameters
        ----------
        indices : str, slice, list, tuple
            The indices to subset the image container.

        Returns
        -------
        xarray.Dataset
            The subsetted image container.
        """
        # checking if the user provided dict_values or dict_keys and turns them into a list if that is the case
        if type(indices) is {}.keys().__class__ or type(indices) is {}.values().__class__:
            indices = list(indices)

        if type(indices) is str:
            c_slice = [indices]
            x_slice = slice(None)
            y_slice = slice(None)
        elif type(indices) is slice:
            c_slice = slice(None)
            x_slice = indices
            y_slice = slice(None)
        elif type(indices) is list:
            all_str = all([type(s) is str for s in indices])

            if all_str:
                c_slice = indices
                x_slice = slice(None)
                y_slice = slice(None)
            else:
                raise TypeError(f"Invalid input. Found non-string elements in the list. Input list: {indices}")

        elif type(indices) is tuple:
            all_str = all([type(s) is str for s in indices])

            if all_str:
                c_slice = [*indices]
                x_slice = slice(None)
                y_slice = slice(None)

            if len(indices) == 2:
                if (type(indices[0]) is slice) & (type(indices[1]) is slice):
                    c_slice = slice(None)
                    x_slice = indices[0]
                    y_slice = indices[1]
                elif (type(indices[0]) is str) & (type(indices[1]) is slice):
                    # Handles arguments in form of im['Hoechst', 500:1000]
                    c_slice = [indices[0]]
                    x_slice = indices[1]
                    y_slice = slice(None)
                elif (type(indices[0]) is list) & (type(indices[1]) is slice):
                    c_slice = indices[0]
                    x_slice = indices[1]
                    y_slice = slice(None)
                else:
                    raise AssertionError("Some error in handling the input arguments")

            elif len(indices) == 3:
                if type(indices[0]) is str:
                    c_slice = [indices[0]]
                elif type(indices[0]) is list:
                    c_slice = indices[0]
                else:
                    raise AssertionError("First index must index channel coordinates.")

                if (type(indices[1]) is slice) & (type(indices[2]) is slice):
                    x_slice = indices[1]
                    y_slice = indices[2]
        else:
            raise TypeError(
                f"Invalid input. To subselect, you can input a string, slice, list, or tuple. You provided {type(indices)}"
            )

        ds = self._obj.pp.get_channels(c_slice)

        return ds.pp.get_bbox(x_slice, y_slice)

    def get_bbox(self, x_slice: slice, y_slice: slice) -> xr.Dataset:
        """
        Returns the bounds of the image container.

        Parameters
        ----------
        x_slice : slice
            The slice representing the x-coordinates for the bounding box.
        y_slice : slice
            The slice representing the y-coordinates for the bounding box.

        Returns:
        --------
        xarray.Dataset
            The updated image container.
        """

        # get the dimensionality of the image
        xdim = self._obj.coords[Dims.X]
        ydim = self._obj.coords[Dims.Y]

        # set the start and stop indices
        x_start = xdim[0] if x_slice.start is None else x_slice.start
        y_start = ydim[0] if y_slice.start is None else y_slice.start
        x_stop = xdim[-1] if x_slice.stop is None else x_slice.stop
        y_stop = ydim[-1] if y_slice.stop is None else y_slice.stop

        # set up query
        query = {
            Dims.X: x_slice,
            Dims.Y: y_slice,
        }

        # handle case when there are cells in the image
        if Dims.CELLS in self._obj.sizes:
            coords = self._obj[Layers.OBS]
            cells = (
                (coords.loc[:, Features.X] >= x_start)
                & (coords.loc[:, Features.X] <= x_stop)
                & (coords.loc[:, Features.Y] >= y_start)
                & (coords.loc[:, Features.Y] <= y_stop)
            ).values

            # finalise query
            query[Dims.CELLS] = cells

        return self._obj.sel(query)

    def get_channels(self, channels: Union[List[str], str]) -> xr.Dataset:
        """
        Retrieve the specified channels from the dataset.

        Parameters:
        channels (Union[List[str], str]): The channels to retrieve. Can be a single channel name or a list of channel names.

        Returns:
        xr.Dataset: The dataset containing the specified channels.
        """
        if isinstance(channels, str):
            channels = [channels]

        # build query
        query = {Dims.CHANNELS: channels}

        return self._obj.sel(query)

    def add_channel(self, channels: Union[str, list], array: np.ndarray) -> xr.Dataset:
        """
        Adds channel(s) to an existing image container.

        Parameters
        ----------
        channels : Union[str, list]
            The name of the channel or a list of channel names to be added.
        array : np.ndarray
            The numpy array representing the channel(s) to be added.

        Returns
        -------
        xarray.Dataset
            The updated image container with added channel(s).
        """
        assert type(array) is np.ndarray, "Added channels must be numpy arrays."
        assert array.ndim in [2, 3], "Added channels must be 2D or 3D arrays."

        if array.ndim == 2:
            array = np.expand_dims(array, 0)

        if type(channels) is str:
            channels = [channels]

        assert (
            set(channels).intersection(set(self._obj.coords[Dims.CHANNELS].values)) == set()
        ), "Can't add a channel that already exists."

        self_channels, self_x_dim, self_y_dim = self._obj[Layers.IMAGE].shape
        other_channels, other_x_dim, other_y_dim = array.shape

        assert (
            len(channels) == other_channels
        ), "The length of channels must match the number of channels in array (DxMxN)."
        assert (self_x_dim == other_x_dim) & (
            self_y_dim == other_y_dim
        ), "Dimensions of the original image and the input array do not match."

        da = xr.DataArray(
            array,
            coords=[channels, range(other_x_dim), range(other_y_dim)],
            dims=Dims.IMAGE,
            name=Layers.IMAGE,
        )

        return xr.merge([self._obj, da])

    def add_segmentation(
        self,
        segmentation: Union[str, np.ndarray] = None,
        reindex: bool = True,
        keep_labels: bool = True,
    ) -> xr.Dataset:
        """
        Adds a segmentation mask (_segmentation) field to the xarray dataset.

        Parameters
        ----------
        segmentation : str or np.ndarray
            A segmentation mask, i.e., a np.ndarray with image.shape = (x, y),
            that indicates the location of each cell, or a layer key.
        mask_growth : int
            The number of pixels by which the segmentation mask should be grown.
        reindex : bool
            If true the segmentation mask is relabeled to have continuous numbers from 1 to n.
        keep_labels : bool
            When using cellpose on multiple channels, you may already get some initial celltype annotations from those.
            If you want to keep those annotations, set this to True. Default is True.

        Returns:
        --------
        xr.Dataset
            The amended xarray.
        """
        # flag indicating if the segmentation mask is provided as a layer key or as a numpy array
        from_layer = None
        if isinstance(segmentation, str):
            if segmentation not in self._obj:
                raise KeyError(f'The key "{segmentation}" does not exist.')

            from_layer = segmentation
            segmentation = self._obj[segmentation].values.squeeze()

        assert segmentation.ndim == 2, "A segmentation mask must 2 dimensional."
        assert ~np.any(segmentation < 0), "A segmentation mask may not contain negative numbers."

        y_dim, x_dim = segmentation.shape

        assert (x_dim == self._obj.sizes[Dims.X]) & (
            y_dim == self._obj.sizes[Dims.Y]
        ), "The shape of segmentation mask does not match that of the image."

        # checking if there are any disconnected cells in the input
        # handle_disconnected_cells(segmentation, mode=handle_disconnected)
        segmentation = segmentation.copy()

        if reindex:
            segmentation, reindex_dict = _relabel_cells(segmentation)

        # crete a data array with the segmentation mask
        da = xr.DataArray(
            segmentation,
            coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=Layers.SEGMENTATION,
        )

        # add cell coordinates
        obj = self._obj.copy()
        obj.coords[Dims.CELLS] = np.unique(segmentation[segmentation > 0]).astype(int)

        if keep_labels and from_layer is not None:
            # checking that the segmentation has labels in the attrs
            if Attrs.LABEL_SEGMENTATION in self._obj[from_layer].attrs:
                # this is a dict that maps from cell_id to a label (e. g. {1: 'CD68', 2: 'DAPI'})
                labels = self._obj[from_layer].attrs[Attrs.LABEL_SEGMENTATION]
                # if reindex was called, we first need to propagate the mapping to the labels before we can add them
                if reindex:
                    labels = {reindex_dict[k]: v for k, v in labels.items()}
                obj = obj.pp.add_labels(labels)

        return xr.merge([obj, da]).pp.add_observations()

    def add_layer(
        self,
        array: np.ndarray,
        key_added: str = Layers.MASK,
    ) -> xr.Dataset:
        """
        array : np.ndarray
            The array representing the segmentation mask or layer to be added.
        key_added : str, optional
            The name of the added layer in the xarray dataset. Default is '_mask'.
        Returns
        -------
            The amended xarray dataset.
        Raises
        ------
        AssertionError
            If the array is not 2-dimensional or its shape does not match the image shape.
        Notes
        -----
        This method adds a layer to the xarray dataset, where the layer has the same shape as the image field.
        The array should be a 2-dimensional numpy array representing the segmentation mask or layer to be added.
        The layer is created as a DataArray with the same coordinates and dimensions as the image field.
        The name of the added layer in the xarray dataset can be specified using the `key_added` parameter.
        The amended xarray dataset is returned after merging the original dataset with the new layer.
        """
        # checking that the layer does not exist yet
        assert key_added not in self._obj, f"Layer {key_added} already exists."
        assert array.ndim == 2, "The array to add mask must 2 dimensional."
        y_dim, x_dim = array.shape
        assert (x_dim == self._obj.sizes[Dims.X]) & (
            y_dim == self._obj.sizes[Dims.Y]
        ), f"The shape of array does not match that of the image. Image has shape ({self._obj.sizes[Dims.Y]}, {self._obj.sizes[Dims.X]}), array has shape {array.shape}."

        # crete a data array with the segmentation mask
        da = xr.DataArray(
            array,
            coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=key_added,
        )

        obj = self._obj.copy()
        return xr.merge([obj, da]).pp.add_observations()

    def add_observations(
        self,
        properties: Union[str, list, tuple] = ("label", "centroid"),
        layer_key: str = Layers.SEGMENTATION,
        return_xarray: bool = False,
    ) -> xr.Dataset:
        """
        Adds properties derived from the segmentation mask to the image container.

        Parameters
        ----------
        properties : Union[str, list, tuple]
            A list of properties to be added to the image container. See
            skimage.measure.regionprops_table for a list of available properties.
        layer_key : str
            The key of the layer that contains the segmentation mask.
        return_xarray : bool
            If true, the function returns an xarray.DataArray with the properties
            instead of adding them to the image container.

        Returns
        -------
        xr.DataSet
            The amended image container.
        """
        if layer_key not in self._obj:
            raise ValueError(
                f"No segmentation mask found at layer {layer_key}. You can specify which layer to use with the layer_key parameter."
            )

        if type(properties) is str:
            properties = [properties]

        if "label" not in properties:
            properties = ["label", *properties]

        table = regionprops_table(self._obj[layer_key].values, properties=properties)

        label = table.pop("label")
        data = []
        cols = []

        for k, v in table.items():
            if Dims.FEATURES in self._obj.coords:
                if k in self._obj.coords[Dims.FEATURES] and not return_xarray:
                    continue
            # when looking at centroids, it could happen that the image has been cropped before
            # in this case, the x and y coordinates do not necessarily start at 0
            # to accommodate for this, we add the x and y coordinates to the centroids
            if k == Features.X:
                v += self._obj.coords[Dims.X].values[0]
            if k == Features.Y:
                v += self._obj.coords[Dims.Y].values[0]
            cols.append(k)
            data.append(v)

        if len(data) == 0:
            return self._obj

        da = xr.DataArray(
            np.stack(data, -1),
            coords=[label, cols],
            dims=[Dims.CELLS, Dims.FEATURES],
            name=Layers.OBS,
        )

        if return_xarray:
            return da

        obj = self._obj.copy()

        # if there are already observations, concatenate them
        if Layers.OBS in obj:
            # checking if the new number of cells matches with the old one
            # if it does not match, we need to update the cell dimension, i. e. remove all old _obs
            if len(label) != len(obj.coords[Dims.CELLS]):
                logger.warning(
                    "Found _obs with different number of cells in the image container. Removing all old _obs for continuity."
                )
                obj = obj.drop_layers(Layers.OBSERVATIONS)
            else:
                logger.info("Found _obs in image container. Concatenating.")
                da = xr.concat(
                    [obj[Layers.OBS].copy(), da],
                    dim=Dims.FEATURES,
                )

        return xr.merge([obj, da])

    def add_feature(self, feature_name: str, feature_values: Union[list, np.ndarray]):
        """
        Adds a feature to the image container.

        Parameters
        ----------
        feature_name : str
            The name of the feature to be added.
        feature_values :
            The values of the feature to be added.

        Returns
        -------
        xr.Dataset
            The updated image container with the added feature.
        """
        # checking if the feature already exists
        assert feature_name not in self._obj.coords[Dims.FEATURES].values, f"Feature {feature_name} already exists."

        # checking if feature_values is a list or a numpy array
        assert type(feature_values) in [list, np.ndarray], "Feature values must be a list or a numpy array."

        # if feature_values is a list, we convert it to a numpy array
        if type(feature_values) is list:
            feature_values = np.array(feature_values)

        # collapsing the feature_values to a 1D array
        feature_values = feature_values.flatten()

        # checking if the length of the feature_values matches the number of cells
        assert len(feature_values) == len(
            self._obj.coords[Dims.CELLS]
        ), "Length of feature values must match the number of cells."

        # adding a new dimension to obtain a 2D array as required by xarray
        feature_values = np.expand_dims(feature_values, 1)

        # create a data array with the feature
        da = xr.DataArray(
            feature_values,
            coords=[self._obj.coords[Dims.CELLS], [feature_name]],
            dims=[Dims.CELLS, Dims.FEATURES],
            name=Layers.OBS,
        )

        da = xr.concat(
            [self._obj[Layers.OBS].copy(), da],
            dim=Dims.FEATURES,
        )

        return xr.merge([self._obj, da])

    def add_obs_from_dataframe(self, df: pd.DataFrame) -> xr.Dataset:
        """
        Adds an observation table to the image container. Columns of the
        dataframe have to match the feature coordinates of the image
        container, and the index of the dataframe has to match the cell coordinates
        of the image container.

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe with the observation values.

        Returns
        -------
        xr.DataSet
            The amended image container.
        """
        if Dims.CELLS not in self._obj.coords:
            self._obj = self._obj.pp.add_observations()

        # pulls out the cell and feature coordinates from the image container
        cells = self._obj.coords[Dims.CELLS].values

        # ensuring that the shape of the data frame fits the number of cells in the segmentation
        assert len(cells) == len(
            df.index
        ), "Number of cells in the image container does not match the number of cells in the dataframe."

        # create a data array from the dataframe
        da = xr.DataArray(
            df,
            coords=[cells, df.columns],
            dims=[Dims.CELLS, Dims.FEATURES],
            name=Layers.OBS,
        )

        return xr.merge([self._obj, da])

    def add_quantification(
        self,
        func: Union[str, Callable] = "intensity_mean",
        key_added: str = Layers.INTENSITY,
        return_xarray=False,
    ) -> xr.Dataset:
        """
        Quantify channel intensities over the segmentation mask.

        Parameters
        ----------
        func : Callable or str, optional
            The function used for quantification. Can either be a string to specify a function from skimage.measure.regionprops_table or a custom function. Default is 'intensity_mean'.
        key_added : str, optional
            The key under which the quantification data will be stored in the image container. Default is Layers.INTENSITY.
        return_xarray : bool, optional
            If True, the function returns an xarray.DataArray with the quantification data instead of adding it to the image container.

        Returns
        -------
        xr.Dataset or xr.DataArray
            The updated image container with added quantification data or the quantification data as a separate xarray.DataArray.
        """
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("No segmentation mask found.")

        assert (
            key_added not in self._obj
        ), f"Found {key_added} in image container. Please add a different key or remove the previous quantification."

        if Dims.CELLS not in self._obj.coords:
            logger.warning("No cell coordinates found. Adding _obs table.")
            self._obj = self._obj.pp.add_observations()

        measurements = []
        all_channels = self._obj.coords[Dims.CHANNELS].values.tolist()

        segmentation = self._obj[Layers.SEGMENTATION].values

        image = np.rollaxis(self._obj[Layers.IMAGE].values, 0, 3)

        # Check if the input is a string (referring to a default skimage property)
        if isinstance(func, str):
            # Use regionprops to get the available property names
            try:
                props = regionprops_table(segmentation, intensity_image=image, properties=["label", func])
            except AttributeError:
                raise AttributeError(
                    f"Invalid regionprop: {func}. Please provide a valid property or a custom function. Check skimage.measure.regionprops_table for available properties."
                )

            cell_idx = props.pop("label")
            for k in sorted(props.keys(), key=lambda x: int(x.split("-")[-1])):
                if k.startswith(func):
                    measurements.append(props[k])
        # If the input is a callable (function)
        elif callable(func):
            props = regionprops_table(segmentation, intensity_image=image, extra_properties=(func,))
            cell_idx = props.pop("label")

            for k in sorted(props.keys(), key=lambda x: int(x.split("-")[-1])):
                if k.startswith(func.__name__):
                    measurements.append(props[k])
        else:
            raise ValueError(
                "The func parameter should be either a string for default skimage properties or a callable function."
            )

        da = xr.DataArray(
            np.stack(measurements, -1),
            coords=[cell_idx, all_channels],
            dims=[Dims.CELLS, Dims.CHANNELS],
            name=key_added,
        )

        if return_xarray:
            return da

        return xr.merge([self._obj, da])

    def add_quantification_from_dataframe(self, df: pd.DataFrame, key_added: str = Layers.INTENSITY) -> xr.Dataset:
        """
        Adds an observation table to the image container. Columns of the
        dataframe have to match the channel coordinates of the image
        container, and the index of the dataframe has to match the cell coordinates
        of the image container.

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe with the quantification values.
        key_added : str, optional
            The key under which the quantification data will be added to the image container.

        Returns
        -------
        xr.DataSet
            The amended image container.
        """
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("No segmentation mask found. A segmentation mask is required to add quantification.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("The input must be a pandas DataFrame.")

        # pulls out the cell and channel coordinates from the image container
        cells = self._obj.coords[Dims.CELLS].values
        channels = self._obj.coords[Dims.CHANNELS].values

        # ensuring that all cells and channels are actually in the dataframe
        assert np.all([c in df.index for c in cells]), "Cells in the image container are not in the dataframe."
        assert np.all([c in df.columns for c in channels]), "Channels in the image container are not in the dataframe."

        # create a data array from the dataframe
        da = xr.DataArray(
            df.loc[cells, channels].values,
            coords=[cells, channels],
            dims=[Dims.CELLS, Dims.CHANNELS],
            name=key_added,
        )

        return xr.merge([self._obj, da])

    def add_properties(
        self, array: Union[np.ndarray, list], prop: str = Features.LABELS, return_xarray: bool = False
    ) -> xr.Dataset:
        """
        Adds properties to the image container.

        Parameters
        ----------
        array : Union[np.ndarray, list]
            An array or list of properties to be added to the image container.
        prop : str, optional
            The name of the property. Default is Features.LABELS.
        return_xarray : bool, optional
            If True, the function returns an xarray.DataArray with the properties instead of adding them to the image container.

        Returns
        -------
        xr.Dataset or xr.DataArray
            The updated image container with added properties or the properties as a separate xarray.DataArray.
        """
        unique_labels = np.unique(self._obj[Layers.OBS].sel({Dims.FEATURES: Features.LABELS}))

        if type(array) is list:
            array = np.array(array)

        if prop == Features.LABELS:
            unique_labels = np.unique(_format_labels(array))

        da = xr.DataArray(
            array.reshape(-1, 1),
            coords=[unique_labels.astype(int), [prop]],
            dims=[Dims.LABELS, Dims.PROPS],
            name=Layers.PROPERTIES,
        )

        if return_xarray:
            return da

        if Layers.PROPERTIES in self._obj:
            da = xr.concat(
                [self._obj[Layers.PROPERTIES], da],
                dim=Dims.PROPS,
            )

        return xr.merge([da, self._obj])

    def add_labels(self, labels: Union[dict, None] = None) -> xr.Dataset:
        """
        Add labels from a mapping (cell -> label) to the spatialproteomics object.

        Parameters:
        -----------
        labels : Union[dict, None], optional
            A dictionary containing cell labels as keys and corresponding labels as values.
            If None, a default labeling will be added. Default is None.

        Returns:
        --------
        xr.Dataset
            The spatialproteomics object with added labels.

        Notes:
        ------
        This method converts the input dictionary into a pandas DataFrame and then adds the labels to the object
        using the `pp.add_labels_from_dataframe` method.
        """
        # converting the dict into a df
        if labels is not None:
            labels = pd.DataFrame(labels.items(), columns=["cell", "label"])

        # adding the labels to the object
        return self._obj.pp.add_labels_from_dataframe(labels)

    def add_labels_from_dataframe(
        self,
        df: Union[pd.DataFrame, None] = None,
        cell_col: str = "cell",
        label_col: str = "label",
        colors: Union[list, None] = None,
        names: Union[list, None] = None,
    ) -> xr.Dataset:
        """
        Adds labels to the image container.

        Parameters
        ----------
        df : Union[pd.DataFrame, None], optional
            A dataframe with the cell and label information. If None, a default labeling will be applied.
        cell_col : str, optional
            The name of the column in the dataframe representing cell coordinates. Default is "cell".
        label_col : str, optional
            The name of the column in the dataframe representing cell labels. Default is "label".
        colors : Union[list, None], optional
            A list of colors corresponding to the cell labels. If None, random colors will be assigned. Default is None.
        names : Union[list, None], optional
            A list of names corresponding to the cell labels. If None, default names will be assigned. Default is None.

        Returns
        -------
        xr.Dataset
            The updated image container with added labels.
        """
        # check if properties are already present
        assert (
            Layers.PROPERTIES not in self._obj
        ), "Already found label properties in the object. Please remove them with pp.drop_layers('_properties') first."

        if df is None:
            cells = self._obj.coords[Dims.CELLS].values
            labels = np.ones(len(cells))
            formated_labels = np.ones(len(cells))
            unique_labels = np.unique(formated_labels)
        else:
            sub = df.loc[:, [cell_col, label_col]].dropna()
            cells = sub.loc[:, cell_col].to_numpy().squeeze()
            labels = sub.loc[:, label_col].to_numpy().squeeze()

            if np.all([isinstance(i, str) for i in labels]):
                unique_labels = np.unique(labels)

                # if zeroes are present in the labels, this means that there are unlabeled cells
                # these should have a value of 0
                # otherwise, we reindex the labels so they start at 1
                if Labels.UNLABELED in unique_labels:
                    # push unlabeled to the front of the list
                    unique_labels = np.concatenate(
                        ([Labels.UNLABELED], unique_labels[unique_labels != Labels.UNLABELED])
                    )
                    label_to_num = dict(zip(unique_labels, range(len(unique_labels))))
                else:
                    label_to_num = dict(zip(unique_labels, range(1, len(unique_labels) + 1)))

                labels = np.array([label_to_num[label] for label in labels])
                names = [k for k, v in sorted(label_to_num.items(), key=lambda x: x[1])]

            assert ~np.all(labels < 0), "Labels must be >= 0."

            formated_labels = _format_labels(labels)
            unique_labels = np.unique(formated_labels)

        da = xr.DataArray(
            np.stack([formated_labels], -1),
            coords=[cells, [Features.LABELS]],
            dims=[Dims.CELLS, Dims.FEATURES],
            name=Layers.OBS,
        )

        da = da.where(
            da.coords[Dims.CELLS].isin(
                self._obj.coords[Dims.CELLS],
            ),
            drop=True,
        )

        obj = self._obj.copy()
        obj = xr.merge([obj.sel(cells=da.cells), da])

        if colors is not None:
            assert len(colors) == len(unique_labels), "Colors has the same."
        else:
            colors = np.random.choice(COLORS, size=len(unique_labels), replace=False)

        obj = obj.pp.add_properties(colors, Props.COLOR)

        if names is not None:
            assert len(names) == len(unique_labels), "Names has the same."
        else:
            # if there is a 0 in unique labels, we need to add an unlabeled category
            if 0 in unique_labels:
                names = [Labels.UNLABELED, *[f"Cell type {i+1}" for i in range(len(unique_labels) - 1)]]
            else:
                names = [f"Cell type {i+1}" for i in range(len(unique_labels))]

        obj = obj.pp.add_properties(names, Props.NAME)

        return xr.merge([obj.sel(cells=da.cells), da])

    def drop_layers(
        self, layers: Optional[Union[str, list]] = None, keep: Optional[Union[str, list]] = None
    ) -> xr.Dataset:
        """
        Drops layers from the image container. Can either drop all layers specified in layers or drop all layers but the ones specified in keep.

        Parameters
        ----------
        layers : Union[str, list]
            The name of the layer or a list of layer names to be dropped.
        keep : Union[str, list]
            The name of the layer or a list of layer names to be kept.

        Returns
        -------
        xr.Dataset
            The updated image container with dropped layers.
        """
        # checking that either layers or keep is provided
        assert layers is not None or keep is not None, "Please provide either layers or keep."
        assert not (layers is not None and keep is not None), "Please provide either layers or keep."

        if type(layers) is str:
            layers = [layers]

        if type(keep) is str:
            keep = [keep]

        # if keep is provided, we drop all layers that are not in keep
        if keep is not None:
            layers = [str(x) for x in self._obj.data_vars if str(x) not in keep]

            # if the user wants to keep obs but not segmentation or vice versa, we throw a warning that this is not possible
            if Layers.SEGMENTATION in keep and Layers.OBS not in keep:
                logger.warning("Cannot drop segmentation and keep observations. Removing both.")
            if Layers.OBS in keep and Layers.SEGMENTATION not in keep:
                logger.warning("Cannot drop observations and keep segmentation. Removing both.")

        # if the segmentation layer is dropped, we also need to drop the obs and vice versa
        # this helps to ensure that the segmentation and obs always stay in sync
        if Layers.SEGMENTATION in layers and Layers.OBS in self._obj.data_vars:
            layers.append(Layers.OBS)
        if Layers.OBS in layers and Layers.SEGMENTATION in self._obj.data_vars:
            layers.append(Layers.SEGMENTATION)

        assert all(
            [layer in self._obj.data_vars for layer in layers]
        ), f"Some layers that you are trying to remove are not in the image container. Available layers are: {', '.join(self._obj.data_vars)}. Layers requested to drop: {layers}."

        obj = self._obj.drop_vars(layers)

        # iterating through the remaining layers to get the dims that should be kept
        dims_to_keep = []
        for layer in obj.data_vars:
            dims_to_keep.extend(obj[layer].dims)

        # removing all dims that are not in dims_to_keep
        for dim in obj.dims:
            if dim not in dims_to_keep:
                obj = obj.drop_dims(dim)

        return obj

    def threshold(self, quantile: float = None, intensity: int = None, key_added: Optional[str] = None):
        """
        Apply thresholding to the image layer of the object.

        Parameters:
        - quantile (float): The quantile value used for thresholding. If provided, the pixels below this quantile will be set to 0.
        - intensity (int): The absolute intensity value used for thresholding. If provided, the pixels below this intensity will be set to 0.
        - key_added (Optional[str]): The name of the new image layer after thresholding. If not provided, the original image layer will be replaced.

        Returns:
        - xr.Dataset: The object with the thresholding applied to the image layer.

        Raises:
        - ValueError: If both quantile and intensity are None or if both quantile and intensity are provided.
        """
        if (quantile is None and intensity is None) or (quantile is not None and intensity is not None):
            raise ValueError("Please provide a quantile or absolute intensity cut off.")

        if Layers.PLOT in self._obj:
            logger.warning(
                "Please only call pl.colorize() after any preprocessing. Otherwise, the image will not be displayed correctly."
            )

        # Pull out the image from its corresponding field (by default "_image")
        image_layer = self._obj[Layers.IMAGE]

        if quantile is not None:
            if isinstance(quantile, (float, int)):
                quantile = np.array([quantile])
            if isinstance(quantile, list):
                quantile = np.array(quantile)

            assert (
                len(quantile) == 1 or len(quantile) == image_layer.coords[Dims.CHANNELS].size
            ), "Quantile threshold must be a single value or a list of values with the same length as the number of channels."

            assert np.all(quantile >= 0) and np.all(quantile <= 1), "Quantile values must be between 0 and 1."

            # calculate quantile
            lower = np.quantile(image_layer.values.reshape(image_layer.values.shape[0], -1), quantile, axis=1)
            filtered = (image_layer - np.expand_dims(np.diag(lower) if lower.ndim > 1 else lower, (1, 2))).clip(min=0)

        if intensity is not None:
            if isinstance(intensity, (float, int)):
                intensity = np.array([intensity])
            if isinstance(intensity, list):
                intensity = np.array(intensity)

            assert (
                len(intensity) == 1 or len(intensity) == image_layer.coords[Dims.CHANNELS].size
            ), "Intensity threshold must be a single value or a list of values with the same length as the number of channels."

            assert np.all(intensity >= 0), "Intensity values must be positive."
            assert np.all(
                intensity <= np.max(image_layer.values)
            ), "Intensity values must be smaller than the maximum intensity."

            # calculate intensity
            filtered = (image_layer - intensity.reshape(-1, 1, 1)).clip(min=0)

        obj = self._obj.copy()

        if key_added is None:
            obj = obj.drop_vars(Layers.IMAGE)

        filtered = xr.DataArray(
            filtered,
            coords=image_layer.coords,
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=Layers.IMAGE if key_added is None else key_added,
        )
        return xr.merge([obj, filtered])

    def apply(self, func: Callable, key: str = Layers.IMAGE, key_added: str = Layers.IMAGE, **kwargs):
        """
        Apply a function to each channel independently.

        Parameters
        ----------
        func : Callable
            The function to apply to the layer.
        key : str
            The key of the layer to apply the function to. Default is Layers.IMAGE.
        key_added : str
            The key under which the updated layer will be stored. Default is Layers.IMAGE (i. e. the original image will be overwritten).
        **kwargs : dict, optional
            Additional keyword arguments to pass to the function.

        Returns
        -------
        xr.Dataset
            The updated image container with the applied function.
        """
        # checking if the key is in the object
        assert key in self._obj, f"Key {key} not found in the image container."

        obj = self._obj.copy()
        layer = obj[key].copy()

        # Apply the function independently across all channels
        # initially, I tried to vectorize this using xr.apply_ufunc(), but the results were spurious, esp. when applying a median filter
        processed_layers = []
        for channel in layer.coords[Dims.CHANNELS].values:
            channel_data = layer.sel({Dims.CHANNELS: channel})
            processed_channel_data = func(channel_data, **kwargs)
            processed_layers.append(processed_channel_data)

        # Stack the processed layers back into a single numpy array
        processed_layer = np.stack(processed_layers, 0)

        # adding the modified layer to the object
        obj[key_added] = xr.DataArray(
            processed_layer,
            coords=[self._obj.coords[Dims.CHANNELS], self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=key_added,
        )

        return obj

    def normalize(self):
        """
        Performs a percentile normalization on each channel.

        Returns
        -------
        xr.Dataset
            The image container with the normalized image stored in Layers.PLOT.
        """
        image_layer = self._obj[Layers.IMAGE]
        normed = xr.DataArray(
            _normalize(image_layer.values),
            coords=image_layer.coords,
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=Layers.PLOT,
        )

        return xr.merge([self._obj, normed])

    def downsample(self, rate: int):
        """
        Downsamples the image and segmentation mask in the object by a given rate.

        Parameters:
        - rate (int): The downsampling rate. Only every `rate`-th pixel will be kept.

        Returns:
        - xr.Dataset: The downsampled object containing the updated image and segmentation mask.

        Raises:
        - AssertionError: If no image layer is found in the object.
        """
        # checking if the object contains an image layer
        assert Layers.IMAGE in self._obj, "No image layer found in the object."

        image_layer = self._obj[Layers.IMAGE]

        x = self._obj.x.values[::rate]
        y = self._obj.y.values[::rate]
        c = self._obj.channels.values
        img = image_layer.values[:, ::rate, ::rate]
        new_img = xr.DataArray(img, coords=[c, y, x], dims=[Dims.CHANNELS, Dims.Y, Dims.X], name=Layers.IMAGE)
        obj = self._obj.drop(Layers.IMAGE)

        if Layers.SEGMENTATION in self._obj:
            # if a segmentation mask is present in the object
            seg_layer = self._obj[Layers.SEGMENTATION]
            new_seg = xr.DataArray(
                seg_layer.values[::rate, ::rate], coords=[y, x], dims=[Dims.Y, Dims.X], name=Layers.SEGMENTATION
            )
            obj = obj.drop(Layers.SEGMENTATION)

            obj = obj.drop_dims([Dims.Y, Dims.X])

            return xr.merge([obj, new_img, new_seg])
        else:
            # if no segmentation is present in the object
            obj = obj.drop_dims([Dims.Y, Dims.X])
            return xr.merge([obj, new_img])

    def rescale(self, scale: int):
        """
        Rescales the image and segmentation mask in the object by a given scale.

        Parameters:
        - scale (int): The scale factor by which to rescale the image and segmentation mask.

        Returns:
        - xr.Dataset: The rescaled object containing the updated image and segmentation mask.

        Raises:
        - AssertionError: If no image layer is found in the object.
        - AssertionError: If no segmentation mask is found in the object.
        """
        # checking if the object contains an image layer
        assert Layers.IMAGE in self._obj, "No image layer found in the object."
        # checking if the object contains a segmentation mask
        assert Layers.SEGMENTATION in self._obj, "No segmentation mask found in the object."

        image_layer = self._obj[Layers.IMAGE]
        img = skimage.transform.rescale(image_layer.values, scale=scale, channel_axis=0)
        x = np.array(range(img.shape[1]))
        y = np.array(range(img.shape[2]))
        c = self._obj.channels.values
        new_img = xr.DataArray(img, coords=[c, y, x], dims=[Dims.CHANNELS, Dims.Y, Dims.X], name=Layers.IMAGE)
        obj = self._obj.drop(Layers.IMAGE)

        if Layers.SEGMENTATION in self._obj:
            seg_layer = self._obj[Layers.SEGMENTATION]
            seg = skimage.transform.rescale(seg_layer.values, scale=scale)
            new_seg = xr.DataArray(seg, coords=[y, x], dims=[Dims.Y, Dims.X], name=Layers.SEGMENTATION)
            obj = obj.drop(Layers.SEGMENTATION)

        obj = obj.drop_dims([Dims.Y, Dims.X])

        return xr.merge([obj, new_img, new_seg])

    def filter_by_obs(self, col: str, func: Callable, segmentation_key: str = Layers.SEGMENTATION):
        """
        Filter the object by observations based on a given feature and filtering function.

        Parameters:
            col (str): The name of the feature to filter by.
            func (Callable): A filtering function that takes in the values of the feature and returns a boolean array.
            segmentation_key (str): The key of the segmentation mask in the object. Default is Layers.SEGMENTATION.

        Returns:
            xr.Dataset: The filtered object with the selected cells and updated segmentation mask.

        Raises:
            AssertionError: If the feature does not exist in the object's observations.

        Notes:
            - This method filters the object by selecting only the cells that satisfy the filtering condition.
            - It also updates the segmentation mask to remove cells that are not selected and relabels the remaining cells.

        Example:
            To filter the object by the feature "area" and keep only the cells with an area greater than 70px:
            >>> obj = obj.pp.add_observations('area').pp.filter_by_obs('area', lambda x: x > 70)
        """
        # checking if the feature exists in obs
        assert (
            col in self._obj.coords[Dims.FEATURES].values
        ), f"Feature {col} not found in obs. You can add it with pp.add_observations()."

        assert (
            segmentation_key in self._obj
        ), f"Segmentation mask with key {segmentation_key} not found in the object. You can specify the key with the segmentation_key parameter."

        cells = self._obj[Layers.OBS].sel({Dims.FEATURES: col}).values.copy()
        cells_bool = func(cells)
        cells_sel = self._obj.coords[Dims.CELLS][cells_bool].values

        # selecting only the cells that are in cells_sel
        obj = self._obj.sel({Dims.CELLS: cells_sel})

        # synchronizing the segmentation mask with the selected cells
        segmentation = obj[segmentation_key].values
        # setting all cells that are not in cells to 0
        segmentation = _remove_unlabeled_cells(segmentation, cells_sel)
        # relabeling cells in the segmentation mask so the IDs go from 1 to n again
        segmentation, relabel_dict = _relabel_cells(segmentation)
        # updating the cell coords of the object
        obj.coords[Dims.CELLS] = [relabel_dict[cell] for cell in obj.coords["cells"].values]

        # creating a data array with the segmentation mask, so that we can merge it to the original
        da = xr.DataArray(
            segmentation,
            coords=[obj.coords[Dims.Y], obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=segmentation_key,
        )

        # removing the old segmentation
        obj = obj.drop_vars(segmentation_key)

        # adding the new filtered and relabeled segmentation
        return xr.merge([obj, da])

    def grow_cells(self, iterations: int = 2, handle_disconnected: str = "ignore") -> xr.Dataset:
        """
        Grows the segmentation masks by expanding the labels in the object.

        Parameters:
        - iterations (int): The number of iterations to grow the segmentation masks. Default is 2.
        - handle_disconnected (str): The mode to handle disconnected segmentation masks. Options are "ignore", "remove", or "fill". Default is "ignore".

        Raises:
        - ValueError: If the object does not contain a segmentation mask.

        Returns:
        - obj (xarray.Dataset): The object with the grown segmentation masks and updated observations.
        """

        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("The object does not contain a segmentation mask.")

        # getting the segmentation mask
        segmentation = self._obj[Layers.SEGMENTATION].values

        # growing segmentation masks
        masks_grown = expand_labels(segmentation, iterations)

        # checking if there are any disconnected segmentation masks
        # handle_disconnected_cells(masks_grown, mode=handle_disconnected)

        # assigning the grown masks to the object
        da = xr.DataArray(
            masks_grown,
            coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=Layers.SEGMENTATION,
        )

        # replacing the old segmentation mask with the new one
        obj = self._obj.drop_vars(Layers.SEGMENTATION)
        obj = xr.merge([obj, da])

        # after segmentation masks were grown, the obs features (e. g. centroids and areas) need to be updated
        # if anything other than the default obs were present, a warning is shown, as they will be removed

        # getting all of the obs features
        obs_features = sorted(list(self._obj.coords[Dims.FEATURES].values))
        if obs_features != [Features.Y, Features.X]:
            logger.warning(
                "Mask growing requires recalculation of the observations. All features other than the centroids will be removed and should be recalculated with pp.add_observations()."
            )
        # removing the original obs and features from the object
        obj = obj.drop_vars(Layers.OBS)
        obj = obj.drop_dims(Dims.FEATURES)

        # adding the default obs back to the object
        return obj.pp.add_observations()

    def merge_segmentation(
        self,
        layer_key: str,
        key_added: str = "_merged_segmentation",
        labels: Optional[Union[str, List[str]]] = None,
        threshold: float = 0.8,
    ):
        """
        Merge segmentation masks.
        This can be done in two ways: either by merging a multi-dimensional array from the object directly, or by adding a numpy array.
        You can either just merge a multi-dimensional array, or merge to an existing 1D mask (e. g. a precomputed DAPI segmentation).

        Parameters:
            array (np.ndarray): The array containing the segmentation masks to be merged. It can be 2D or 3D.
            from_key (str): The key of the segmentation mask in the xarray object to be merged.
            labels (Optional[Union[str, List[str]]]): Optional. The labels corresponding to each segmentation mask in the array.
                If provided, the number of labels must match the number of arrays.
            threshold (float): Optional. The threshold value for merging cells. Default is 1.0.
            handle_disconnected (str): Optional. The method to handle disconnected cells. Default is "relabel".
            key_base_segmentation (str): Optional. The key of the base segmentation mask in the xarray object to merge to.
            key_added (str): Optional. The key under which the merged segmentation mask will be stored in the xarray object. Default is "_segmentation".

        Returns:
            obj (xarray.Dataset): The xarray object with the merged segmentation mask.

        Raises:
            AssertionError: If no segmentation mask is found in the xarray object.
            AssertionError: If the input array is not 2D or 3D.
            AssertionError: If the input array is not of type int.
            AssertionError: If the shape of the input array does not match the shape of the segmentation mask.

        Notes:
            - If the input array is 2D, it will be expanded to 3D.
            - If labels are provided, they need to match the number of arrays.
            - The merging process starts with merging the biggest cells first, then the smaller ones.
            - Disconnected cells in the input are handled based on the specified method.
        """

        # checking if the keys exist
        assert layer_key in self._obj, f"The key {layer_key} does not exist in the object."
        assert key_added not in self._obj, f"The key {key_added} already exists in the object."

        # merge big cells first, then small cells
        channels = self._obj.coords[Dims.CHANNELS].values.tolist()
        segmentation = self._obj.pp[channels[0]][layer_key].values

        # iterating through the array to merge the segmentation masks
        for i in range(1, len(channels)):
            if labels is not None:
                label_1, label_2 = labels[i - 1], labels[i]
            else:
                label_1, label_2 = channels[i - 1], channels[i]

            segmentation, final_mapping = _merge_segmentation(
                segmentation.squeeze(),
                self._obj.pp[channels[i]][layer_key].values.squeeze(),
                label1=label_1,
                label2=label_2,
                threshold=threshold,
            )

            if i == 1:
                # in the first iteration, we simply take the mapping we get from _merge_segmentation
                mapping = final_mapping
            else:
                # note the use of get here. If the cell already exists, we keep the original label, otherwise we use the new one
                mapping = {k: mapping.get(k, v) for k, v in final_mapping.items()}

        # if a segmentation mask already exists in the object, we merge to it
        obj = self._obj.copy()

        # assigning the new segmentation to the object
        da = xr.DataArray(
            segmentation,
            coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=key_added,
            attrs={Attrs.LABEL_SEGMENTATION: mapping},
        )

        return xr.merge([obj, da])

    def get_layer_as_df(
        self, layer: str = Layers.OBS, celltypes_to_str: bool = True, idx_to_str: bool = False
    ) -> pd.DataFrame:
        """
        Returns the specified layer as a pandas DataFrame.

        Parameters:
            layer (str): The name of the layer to retrieve. Defaults to Layers.OBS.
            celltypes_to_str (bool): Whether to convert celltype labels to strings. Defaults to True.
            idx_to_str (bool): Whether to convert the index to strings. Defaults to False.

        Returns:
            pandas.DataFrame: The layer data as a DataFrame.
        """
        data_array = self._obj[layer]

        dims = data_array.dims
        coords = data_array.coords
        c1, c2 = coords[dims[0]].values, coords[dims[1]].values
        df = pd.DataFrame(data_array.values, index=c1, columns=c2)

        # special case: when exporting obs, we can convert celltypes to strings
        if celltypes_to_str and layer == Layers.OBS and Features.LABELS in df.columns:
            label_dict = self._obj.la._label_to_dict(Props.NAME)
            df[Features.LABELS] = df[Features.LABELS].apply(lambda x: label_dict[x])

        if idx_to_str:
            df.index = df.index.astype(str)

        return df

    def get_disconnected_cell(self) -> int:
        """
        Returns the first disconnected cell from the segmentation layer.

        Returns:
            ndarray: The first disconnected cell from the segmentation layer.
        """
        return _get_disconnected_cell(self._obj[Layers.SEGMENTATION])

    def transform_expression_matrix(
        self,
        method: str = "arcsinh",
        key: str = Layers.INTENSITY,
        key_added: str = Layers.INTENSITY,
        cofactor: float = 5.0,
        min_percentile: float = 1.0,
        max_percentile: float = 99.0,
    ):
        """
        Transforms the expression matrix based on the specified mode.

        Parameters:
            method (str): The transformation method. Available options are "arcsinh", "zscore", "minmax", "double_zscore", and "clip".
            key (str): The key of the expression matrix in the object.
            key_added (str): The key to assign to the transformed matrix in the object.
            cofactor (float): The cofactor to use for the "arcsinh" transformation.
            min_percentile (float): The minimum percentile value to use for the "clip" transformation.
            max_percentile (float): The maximum percentile value to use for the "clip" transformation.

        Returns:
            xr.Dataset: The object with the transformed matrix added.

        Raises:
            ValueError: If an unknown transformation mode is specified.
            AssertionError: If no expression matrix is found at the specified layer.
        """
        # checking if there is an expression matrix in the object
        assert key in self._obj, f"No expression matrix found at layer {key}."

        # getting the expression matrix from the object
        expression_matrix = self._obj[key].values

        # applying the appropriate transform
        if method == "arcsinh":
            transformed_matrix = np.arcsinh(expression_matrix / cofactor)
        elif method == "zscore":
            # z-scoring along each channel
            transformed_matrix = zscore(expression_matrix, axis=0)
        elif method == "minmax":
            # applying min max scaling, so that the lowest value is 0 and the highest is 1
            transformed_matrix = (expression_matrix - np.min(expression_matrix, axis=0)) / (
                np.max(expression_matrix, axis=0) - np.min(expression_matrix, axis=0)
            )
        elif method == "double_zscore":
            # z-scoring along each channel
            transformed_matrix = zscore(expression_matrix, axis=0)
            # z-scoring along each cell
            transformed_matrix = zscore(transformed_matrix, axis=1)
            # turning the z-scores into probabilities using the cumulative density function
            transformed_matrix = norm.cdf(transformed_matrix)
            # taking the negative log of the inverse probability to amplify positive values
            transformed_matrix = -np.log(1 - transformed_matrix)
        elif method == "clip":
            min_value, max_value = np.percentile(expression_matrix, [min_percentile, max_percentile])
            transformed_matrix = np.clip(expression_matrix, min_value, max_value)
        else:
            raise ValueError(f"Unknown transformation method: {method}")

        # creating a new data array with the transformed matrix
        da = xr.DataArray(
            transformed_matrix,
            coords=[self._obj.coords[Dims.CELLS], self._obj.coords[Dims.CHANNELS]],
            dims=[Dims.CELLS, Dims.CHANNELS],
            name=key_added,
        )

        obj = self._obj.copy()
        # removing the old expression matrix from the object
        if key == key_added:
            obj = obj.drop_vars(key)

        # adding the transformed matrix to the object
        return xr.merge([obj, da])

    def mask_region(self, key: str = Layers.MASK, image_key=Layers.IMAGE, key_added=Layers.IMAGE) -> xr.Dataset:
        """
        Mask a region in the image.

        Parameters:
            key (str): The key of the region to mask.
            image_key (str): The key of the image layer in the object. Default is Layers.IMAGE.
            key_added (str): The key to assign to the masked image in the object. Default is Layers.IMAGE, which overwrites the original image.

        Returns:
            xr.Dataset: The object with the masked region in the image.
        """
        # checking if the keys exist
        assert key in self._obj, f"The key {key} does not exist in the object."
        assert image_key in self._obj, f"The key {image_key} does not exist in the object."

        # getting the region to mask
        mask = self._obj[key].values
        image = self._obj[image_key].values

        # checking that the mask only contains zeroes and ones
        assert np.all(np.isin(mask, [0, 1])), "The mask must only contain zeroes and ones."

        # masking the region in the image (so that only pixels with a one remain)
        masked_image = mask * image

        # removing the old image from the object
        if image_key == key_added:
            obj = self._obj.drop_vars(image_key)
        else:
            obj = self._obj.copy()

        # assigning the masked image to the object
        da = xr.DataArray(
            masked_image,
            coords=[self._obj.coords[Dims.CHANNELS], self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=key_added,
        )

        return xr.merge([obj, da])

    def mask_cells(self, mask_key: str = Layers.MASK, segmentation_key=Layers.SEGMENTATION) -> xr.Dataset:
        """
        Mask cells in the segmentation mask.

        Parameters:
            mask_key (str): The key of the mask to use for masking.
            segmentation_key (str): The key of the segmentation mask in the object. Default is Layers.SEGMENTATION.

        Returns:
            xr.Dataset: The object with the masked cells in the segmentation mask.
        """
        # checking if the keys exist
        assert mask_key in self._obj, f"The key {mask_key} does not exist in the object."
        assert segmentation_key in self._obj, f"The key {segmentation_key} does not exist in the object."

        # getting the mask and segmentation mask
        mask = self._obj[mask_key].values
        segmentation = self._obj[segmentation_key].values

        # checking that the mask only contains zeroes and ones
        assert np.all(np.isin(mask, [0, 1])), "The mask must only contain zeroes and ones."

        # getting all of the cells that overlap with the region where the mask is 0
        cells_to_remove = np.unique(segmentation[mask == 0])

        # removing the cells from the segmentation mask
        cells_sel = np.array(sorted(set(self._obj.coords[Dims.CELLS].values) - set(cells_to_remove)))

        # selecting only the cells that are in cells_sel
        obj = self._obj.sel({Dims.CELLS: cells_sel})

        # synchronizing the segmentation mask with the selected cells
        segmentation = obj[segmentation_key].values
        # setting all cells that are not in cells to 0
        segmentation = _remove_unlabeled_cells(segmentation, cells_sel)
        # relabeling cells in the segmentation mask so the IDs go from 1 to n again
        segmentation, relabel_dict = _relabel_cells(segmentation)
        # updating the cell coords of the object
        obj.coords[Dims.CELLS] = [relabel_dict[cell] for cell in obj.coords["cells"].values]

        # creating a data array with the segmentation mask, so that we can merge it to the original
        da = xr.DataArray(
            segmentation,
            coords=[obj.coords[Dims.Y], obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=segmentation_key,
        )

        # removing the old segmentation
        obj = obj.drop_vars(segmentation_key)

        # adding the new filtered and relabeled segmentation
        return xr.merge([obj, da])
