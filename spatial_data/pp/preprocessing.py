from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import medfilt2d, wiener
from skimage.filters.rank import maximum, mean, median, minimum
from skimage.measure import regionprops_table
from skimage.morphology import disk
from skimage.restoration import unsupervised_wiener
from skimage.segmentation import expand_labels

from ..base_logger import logger
from ..constants import COLORS, Dims, Features, Layers, Props
from ..la.label import _format_labels
from .intensity import sum_intensity
from .utils import (
    _autocrop,
    _merge_segmentation,
    _normalize,
    _relabel_cells,
    _remove_unlabeled_cells,
)


@xr.register_dataset_accessor("pp")
class PreprocessingAccessor:
    """The image accessor enables fast indexing and preprocesses image.data"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __getitem__(self, indices) -> xr.Dataset:
        """Fast subsetting the image container. The following examples show how
        the user can subset the image container:

        Subset the image container using x and y coordinates:
        >> ds.pp[0:50, 0:50]

        Subset the image container using x and y coordinates and channels:
        >> ds.pp['Hoechst', 0:50, 0:50]

        Subset the image container using channels:
        >> ds.pp['Hoechst']

        Multiple channels can be selected by passing a list of channels:
        >> ds.pp[['Hoechst', 'CD4']]

        Parameters:
        -----------
        indices: str, slice, list, tuple
            The indices to subset the image container.
        Returns:
        --------
        xarray.Dataset
            The subsetted image container.
        """
        # argument handling
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
            # num_cells = self._obj.sizes[Dims.CELLS]

            coords = self._obj[Layers.OBS]
            cells = (
                (coords.loc[:, Features.X] >= x_start)
                & (coords.loc[:, Features.X] <= x_stop)
                & (coords.loc[:, Features.Y] >= y_start)
                & (coords.loc[:, Features.Y] <= y_stop)
            ).values
            # calculates the number of cells that were dropped due setting the bounding box
            # lost_cells = num_cells - sum(cells)

            # if lost_cells > 0:
            # logger.warning(f"Dropped {lost_cells} cells.")

            # finalise query
            query[Dims.CELLS] = cells

        return self._obj.sel(query)

    def get_channels(self, channels: Union[List[str], str]) -> xr.Dataset:
        """
        Returns a single channel as a numpy array.

        Parameters
        ----------
        channels: Union[str, list]
            The name of the channel or a list of channel names.

        Returns
        -------
        xarray.Dataset
            The selected channels as a new image container.
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
        # im = xr.concat([self._obj[Layers.IMAGE], da], dim=Dims.IMAGE[0])

        return xr.merge([self._obj, da])

    def add_segmentation(
        self, segmentation: np.ndarray, mask_growth: int = 0, relabel: bool = True, copy: bool = True
    ) -> xr.Dataset:
        """
        Adds a segmentation mask (_segmentation) field to the xarray dataset.

        Parameters
        ----------
        segmentation : np.ndarray
            A segmentation mask, i.e., a np.ndarray with image.shape = (x, y),
            that indicates the location of each cell.
        mask_growth : int
            The number of pixels by which the segmentation mask should be grown.
        relabel : bool
            If true the segmentation mask is relabeled to have continuous numbers from 1 to n.
        copy : bool
            If true the segmentation mask is copied.

        Returns:
        --------
        xr.Dataset
            The amended xarray.
        """

        assert ~np.any(segmentation < 0), "A segmentation mask may not contain negative numbers."

        y_dim, x_dim = segmentation.shape

        assert (x_dim == self._obj.sizes[Dims.X]) & (
            y_dim == self._obj.sizes[Dims.Y]
        ), "The shape of segmentation mask does not match that of the image."

        if copy:
            segmentation = segmentation.copy()

        if relabel:
            segmentation, _ = _relabel_cells(segmentation)

        if mask_growth > 0:
            segmentation = expand_labels(segmentation, mask_growth)

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

        return xr.merge([obj, da]).pp.add_observations()

    def add_observations(
        self,
        properties: Union[str, list, tuple] = ("label", "centroid"),
        return_xarray: bool = False,
    ) -> xr.Dataset:
        """
        Adds properties derived from the mask to the image container.

        Parameters
        ----------
        properties : Union[str, list, tuple]
            A list of properties to be added to the image container. See
            skimage.measure.regionprops_table for a list of available properties.
        return_xarray : bool
            If true, the function returns an xarray.DataArray with the properties
            instead of adding them to the image container.

        Returns
        -------
        xr.DataSet
            The amended image container.
        """
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("No segmentation mask found.")

        if type(properties) is str:
            properties = [properties]

        if "label" not in properties:
            properties = ["label", *properties]

        table = regionprops_table(self._obj[Layers.SEGMENTATION].values, properties=properties)

        label = table.pop("label")
        data = []
        cols = []

        for k, v in table.items():
            if Dims.FEATURES in self._obj.coords:
                if k in self._obj.coords[Dims.FEATURES] and not return_xarray:
                    logger.warning(f"Found {k} in _obs. Skipping.")
                    continue
            cols.append(k)
            data.append(v)

        if len(data) == 0:
            logger.warning("Warning: No properties were added.")
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

    def add_quantification(
        self,
        func=sum_intensity,
        key_added: str = Layers.INTENSITY,
        return_xarray=False,
    ) -> xr.Dataset:
        """
        Quantify channel intensities over the segmentation mask.

        Parameters
        ----------
        func : Callable, optional
            The function used for quantification. Default is sum_intensity.
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
        props = regionprops_table(segmentation, intensity_image=image, extra_properties=(func,))
        cell_idx = props.pop("label")
        for k in sorted(props.keys(), key=lambda x: int(x.split("-")[-1])):
            if k.startswith(func.__name__):
                measurements.append(props[k])

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
            name=Layers.LABELS,
        )

        if return_xarray:
            return da

        if Layers.LABELS in self._obj:
            da = xr.concat(
                [self._obj[Layers.LABELS], da],
                dim=Dims.PROPS,
            )

        return xr.merge([da, self._obj])

    def add_labels(
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
                label_to_num = dict(zip(unique_labels, range(1, len(unique_labels) + 1)))
                # num_to_label = {v: k for k, v in label_to_num.items()}
                labels = np.array([label_to_num[label] for label in labels])
                names = [k for k, v in sorted(label_to_num.items(), key=lambda x: x[1])]

            assert ~np.all(labels < 0), "Labels must be >= 0."

            formated_labels = _format_labels(labels)
            unique_labels = np.unique(formated_labels)

        if np.all(formated_labels == labels):
            da = xr.DataArray(
                np.stack([formated_labels, labels], -1),
                coords=[cells, [Features.LABELS, Features.ORIGINAL_LABELS]],
                dims=[Dims.CELLS, Dims.FEATURES],
                name=Layers.OBS,
            )
        else:
            da = xr.DataArray(
                np.stack([formated_labels, labels], -1),
                coords=[
                    cells,
                    [
                        Features.LABELS,
                        Features.ORIGINAL_LABELS,
                    ],
                ],
                dims=[Dims.CELLS, Dims.FEATURES],
                name=Layers.OBS,
            )

        da = da.where(
            da.coords[Dims.CELLS].isin(
                self._obj.coords[Dims.CELLS],
            ),
            drop=True,
        )

        self._obj = xr.merge([self._obj.sel(cells=da.cells), da])

        if colors is not None:
            assert len(colors) == len(unique_labels), "Colors has the same."
        else:
            colors = np.random.choice(COLORS, size=len(unique_labels), replace=False)

        self._obj = self._obj.pp.add_properties(colors, Props.COLOR)

        if names is not None:
            assert len(names) == len(unique_labels), "Names has the same."
        else:
            names = [f"Cell type {i+1}" for i in range(len(unique_labels))]

        self._obj = self._obj.pp.add_properties(names, Props.NAME)
        self._obj[Layers.SEGMENTATION].values = _remove_unlabeled_cells(
            self._obj[Layers.SEGMENTATION].values, self._obj.coords[Dims.CELLS].values
        )

        return xr.merge([self._obj.sel(cells=da.cells), da])

    def drop_layers(self, layers: Union[str, list]) -> xr.Dataset:
        """
        Drops layers from the image container.

        Parameters
        ----------
        layers : Union[str, list]
            The name of the layer or a list of layer names to be dropped.

        Returns
        -------
        xr.Dataset
            The updated image container with dropped layers.
        """
        if type(layers) is str:
            layers = [layers]

        assert all(
            [layer in self._obj.data_vars for layer in layers]
        ), f"Some layers that you are trying to remove are not in the image container. Available layers are: {', '.join(self._obj.data_vars)}."

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

    def restore(self, method="wiener", **kwargs):
        """
        Restores the image using a specified method.

        Parameters
        ----------
        method : str, optional
            The method used for image restoration. Options are "wiener", "unsupervised_wiener", or "threshold". Default is "wiener".
        **kwargs : dict, optional
            Additional keyword arguments specific to the chosen method.

        Returns
        -------
        xr.Dataset
            The restored image container.
        """
        image_layer = self._obj[Layers.IMAGE]

        obj = self._obj.drop(Layers.IMAGE)

        if method == "wiener":
            restored = wiener(image_layer.values)
        elif method == "unsupervised_wiener":
            psf = np.ones((5, 5)) / 25
            restored, _ = unsupervised_wiener(image_layer.values.squeeze(), psf)
            restored = np.expand_dims(restored, 0)
        elif method == "threshold":
            value = kwargs.get("value", 128)
            rev_func = kwargs.get("rev_func", lambda x: x)
            restored = np.zeros_like(image_layer)
            idx = np.where(image_layer > rev_func(value))
            restored[idx] = image_layer[idx]
        elif method == "median":
            selem = kwargs.get("selem", disk(radius=1))
            restored = median(image_layer.values.squeeze(), footprint=selem)
        elif method == "mean":
            selem = kwargs.get("selem", disk(radius=1))
            restored = mean(image_layer.values.squeeze(), footprint=selem)
        elif method == "minimum":
            selem = kwargs.get("selem", disk(radius=1))
            restored = minimum(image_layer.values.squeeze(), footprint=selem)
        elif method == "maximum":
            selem = kwargs.get("selem", disk(radius=1))
            restored = maximum(image_layer.values.squeeze(), footprint=selem)
        elif method == "medfilt2d":
            kernel_size = kwargs.get("kernel_size", 3)

            if image_layer.values.ndim == 3:
                restore_array = []
                for i in range(image_layer.values.shape[0]):
                    restore_array.append(medfilt2d(image_layer.values[i].squeeze(), kernel_size))
                restored = np.stack(restore_array, 0)
            else:
                restored = medfilt2d(image_layer.values.squeeze(), kernel_size)

        if restored.ndim == 2:
            restored = np.expand_dims(restored, 0)

        normed = xr.DataArray(
            restored,
            coords=image_layer.coords,
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=Layers.IMAGE,
        )
        return xr.merge([obj, normed])

    def filter(self, quantile: float = 0.99, key_added: Optional[str] = None):
        # Pull out the image from its corresponding field (by default "_image")
        image_layer = self._obj[Layers.IMAGE]
        if isinstance(quantile, list):
            quantile = np.array(quantile)
        # Calulate quat
        lower = np.quantile(image_layer.values.reshape(image_layer.values.shape[0], -1), quantile, axis=1)
        filtered = (image_layer - np.expand_dims(np.diag(lower) if lower.ndim > 1 else lower, (1, 2))).clip(min=0)

        if key_added is None:
            obj = self._obj.drop(Layers.IMAGE)

        filtered = xr.DataArray(
            filtered,
            coords=image_layer.coords,
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=Layers.IMAGE if key_added is None else key_added,
        )
        return xr.merge([obj, filtered])

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
        image_layer = self._obj[Layers.IMAGE]
        # image_data = image_layer.values[:, ::rate,::rate]

        x = self._obj.x.values[::rate]
        y = self._obj.y.values[::rate]
        c = self._obj.channels.values
        # import pdb;pdb.set_trace()
        img = image_layer.values[:, ::rate, ::rate]
        # import pdb;pdb.set_trace()
        new_img = xr.DataArray(img, coords=[c, y, x], dims=[Dims.CHANNELS, Dims.Y, Dims.X], name=Layers.IMAGE)
        # import pdb;pdb.set_trace()
        obj = self._obj.drop(Layers.IMAGE)
        # import pdb;pdb.set_trace()

        if Layers.SEGMENTATION in self._obj:
            seg_layer = self._obj[Layers.SEGMENTATION]
            # import pdb;pdb.set_trace()
            new_seg = xr.DataArray(
                seg_layer.values[::rate, ::rate], coords=[y, x], dims=[Dims.Y, Dims.X], name=Layers.SEGMENTATION
            )
            # import pdb;pdb.set_trace()
            obj = obj.drop(Layers.SEGMENTATION)

        obj = obj.drop_dims([Dims.Y, Dims.X])

        return xr.merge([obj, new_img, new_seg])

    def filter_by_obs(self, col: str, func: Callable):
        """Returns the list of cells with the labels from items."""
        # checking if the feature exists in obs
        assert (
            col in self._obj.coords[Dims.FEATURES].values
        ), f"Feature {col} not found in obs. You can add it with pp.add_observations()."

        cells = self._obj[Layers.OBS].sel({Dims.FEATURES: col}).values.copy()
        cells_bool = func(cells)
        cells_sel = self._obj.coords[Dims.CELLS][cells_bool].values

        # selecting only the cells that are in cells_sel
        obj = self._obj.sel({Dims.CELLS: cells_sel})

        # synchronizing the segmentation mask with the selected cells
        segmentation = obj[Layers.SEGMENTATION].values
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
            name=Layers.SEGMENTATION,
        )

        # removing the old segmentation
        obj = obj.drop_vars(Layers.SEGMENTATION)

        # adding the new filtered and relabeled segmentation
        return xr.merge([obj, da])

    def grow_cells(self, iterations: int = 2):
        """
        Grows the cells in the segmentation mask.
        """
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("The object does not contain a segmentation mask.")

        # getting the segmentation mask
        segmentation = self._obj[Layers.SEGMENTATION].values

        # growing segmentation masks
        masks_grown = expand_labels(segmentation, iterations)

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

    def autocrop(self, channel=None, downsample=10):
        slices = _autocrop(self._obj, channel=channel, downsample=downsample)
        return self._obj.pp[slices[0], slices[1]]

    def merge_segmentation(
        self, array: np.ndarray, labels: Optional[Union[str, List[str]]] = None, threshold: float = 1.0
    ):
        # array = all of the arrays that will iteratively be merged to the existing segmentation mask
        # ensuring that a segmentation mask already exists
        assert (
            Layers.SEGMENTATION in self._obj
        ), "No segmentation mask found in the xarray object. Please add one first using pp.add_segmentation() or ext.stardist()/ext.cellpose()."

        # checking that the array is 2D or 3D
        assert array.ndim in [
            2,
            3,
        ], "The input array must be 2D (if you want to merge one segmentation mask) or 3D (if you want to iteratively merge multiple segmentation masks)."

        # checking that the input type is int
        assert array.dtype == int, "The input array must be of type int."

        # if the array is 2D, it gets expanded to 3D
        if array.ndim == 2:
            array = np.expand_dims(array, 0)

        # if labels are provided, they need to match the number of arrays
        if labels is not None:
            assert (
                len(labels) == array.shape[0]
            ), f"The number of labels must match the number of arrays. You submitted {len(labels)} channels compared to {array.shape[0]} arrays."

        # ensuring that the second and third dimension of the array match the segmentation mask
        assert (array.shape[1] == self._obj.sizes[Dims.Y]) & (
            array.shape[2] == self._obj.sizes[Dims.X]
        ), "The shape of the input array does not match the shape of the segmentation mask."

        # merge big cells first, then small cells
        i = 0
        segmentation = array[i, :, :]

        # iterating through the array to merge the segmentation masks
        for i in range(1, array.shape[0]):
            if labels is not None:
                label_1, label_2 = labels[i - 1], labels[i]
            else:
                label_1, label_2 = i, i + 1

            segmentation, final_mapping = _merge_segmentation(
                segmentation, array[i, :, :], label1=label_1, label2=label_2, threshold=1.0
            )

            if i == 1:
                # in the first iteration, we simply take the mapping we get from _merge_segmentation
                mapping = final_mapping
            else:
                # note the use of get here. If the cell already exists, we keep the original label, otherwise we use the new one
                mapping = {k: mapping.get(k, v) for k, v in final_mapping.items()}

        # finally, we merge the smallest cells, which we assume to be in the segmentation mask already
        if labels is not None:
            label_1, label_2 = labels[i], "Other"
        else:
            label_1, label_2 = i, i + 1

        segmentation, final_mapping = _merge_segmentation(
            segmentation, self._obj[Layers.SEGMENTATION].values, label1=label_1, label2=label_2, threshold=1.0
        )
        # if there is only one array to merge, we can simply take the mapping obtained by _merge_segmentation as the final mapping
        if array.shape[0] == 1:
            mapping = final_mapping
        else:
            # note the use of get here. If the cell already exists, we keep the original label, otherwise we use the new one
            mapping = {k: mapping.get(k, v) for k, v in final_mapping.items()}

        # assigning the new segmentation to the object
        da = xr.DataArray(
            segmentation,
            coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=Layers.SEGMENTATION,
        )

        # replacing the old segmentation mask and obs with the new one
        obj = self._obj.pp.drop_layers(Layers.SEGMENTATION)
        if Layers.OBS in obj:
            obj = obj.pp.drop_layers(Layers.OBS)

        obj = xr.merge([obj, da])

        # after segmentation masks are altered, the obs features (e. g. centroids and areas) need to be updated
        # if anything other than the default obs were present, a warning is shown, as they will be removed

        # getting all of the obs features
        obs_features = sorted(list(self._obj.coords[Dims.FEATURES].values))
        if obs_features != [Features.Y, Features.X]:
            logger.warning(
                "Mask merging requires recalculation of the observations. All features other than the centroids will be removed and should be recalculated with pp.add_observations()."
            )

        # adding the default obs back to the object
        obj = obj.pp.add_observations()

        # if labels are provided, they are added to the object
        if labels is not None:
            label_df = pd.DataFrame({"cell": mapping.keys(), "label": mapping.values()})
            obj = obj.pp.add_labels(label_df)

        return obj
