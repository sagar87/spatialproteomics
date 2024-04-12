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
import re


from ..base_logger import logger
from ..constants import COLORS, Attrs, Dims, Features, Layers, Props
from ..la.label import _format_labels
from ..pl import _get_listed_colormap
from .intensity import sum_intensity
from .utils import (
    _colorize,
    _label_segmentation_mask,
    _normalize,
    _relabel_cells,
    _remove_segmentation_mask_labels,
    _remove_unlabeled_cells,
    _render_label,
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
        # print(indices)
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
        if Dims.CELLS in self._obj.dims:
            # num_cells = self._obj.dims[Dims.CELLS]

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
        assert type(array) is np.ndarray, "Added channel(s) must be numpy arrays"

        if array.ndim == 2:
            array = np.expand_dims(array, 0)

        if type(channels) is str:
            channels = [channels]

        self_channels, self_x_dim, self_y_dim = self._obj[Layers.IMAGE].shape
        other_channels, other_x_dim, other_y_dim = array.shape

        assert (
            len(channels) == other_channels
        ), "The length of channels must match the number of channels in array (DxMxN)."
        assert (self_x_dim == other_x_dim) & (self_y_dim == other_y_dim), "Dims do not match."

        da = xr.DataArray(
            array,
            coords=[channels, range(other_x_dim), range(other_y_dim)],
            dims=Dims.IMAGE,
            name=Layers.IMAGE,
        )
        # im = xr.concat([self._obj[Layers.IMAGE], da], dim=Dims.IMAGE[0])

        return xr.merge([self._obj, da])

    def add_segmentation(self, segmentation: np.ndarray, mask_growth: int = 0, relabel: bool = True, copy: bool = True) -> xr.Dataset:
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

        assert (x_dim == self._obj.dims[Dims.X]) & (
            y_dim == self._obj.dims[Dims.Y]
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

        # if there are already observations, concatenate them
        if Layers.OBS in self._obj:
            logger.info("Found _obs in image container. Concatenating.")
            da = xr.concat(
                [self._obj[Layers.OBS].copy(), da],
                dim=Dims.FEATURES,
            )

        return xr.merge([self._obj, da])

    def add_quantification(
        self,
        channels: Union[str, list] = "all",
        func=sum_intensity,
        remove_unlabeled=True,
        key_added: str = Layers.INTENSITY,
        return_xarray=False,
    ) -> xr.Dataset:
        """
        Quantify channel intensities over the segmentation mask.

        Parameters
        ----------
        channels : Union[str, list], optional
            The name of the channel or a list of channel names to be added. Default is "all".
        func : Callable, optional
            The function used for quantification. Default is sum_intensity.
        remove_unlabeled : bool, optional
            Whether to remove unlabeled cells. Default is True.
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

        if key_added in self._obj:
            logger.warning(f"Found {key_added} in image container. Please add a different key.")
            return self._obj

        if Dims.CELLS not in self._obj.coords:
            logger.warning("No cell coordinates found. Adding _obs table.")
            self._obj = self._obj.pp.add_observations()

        measurements = []
        all_channels = self._obj.coords[Dims.CHANNELS].values.tolist()

        segmentation = self._obj[Layers.SEGMENTATION].values
        segmentation = _remove_unlabeled_cells(segmentation, self._obj.coords[Dims.CELLS].values)

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

        # pulls out the cell and channel coordinates from the image container
        cells = self._obj.coords[Dims.CELLS].values
        channels = self._obj.coords[Dims.CHANNELS].values

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

        # import pdb;pdb.set_trace()
        return xr.merge([self._obj.sel(cells=da.cells), da])

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
        print(lower, lower.shape)
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

    def colorize(
        self,
        colors: List[str] = [
            "#e6194B",
            "#3cb44b",
            "#ffe119",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#42d4f4",
            "#f032e6",
            "#bfef45",
            "#fabed4",
            "#469990",
            "#dcbeff",
            "#9A6324",
            "#fffac8",
            "#800000",
            "#aaffc3",
            "#808000",
            "#ffd8b1",
            "#000075",
            "#a9a9a9",
        ],
        background: str = "black",
        normalize: bool = True,
        merge: bool = True,
    ) -> xr.Dataset:
        """
        Colorizes a stack of images.

        Parameters
        ----------
        colors : List[str], optional
            A list of strings that denote the color of each channel. Default is ["C0", "C1", "C2", "C3"].
        background : str, optional
            Background color of the colorized image. Default is "black".
        normalize : bool, optional
            Normalize the image prior to colorizing it. Default is True.
        merge : True, optional
            Merge the channel dimension. Default is True.

        Returns
        -------
        xr.Dataset
            The image container with the colorized image stored in Layers.PLOT.
        """
        if isinstance(colors, str):
            colors = [colors]

        image_layer = self._obj[Layers.IMAGE]
        colored = _colorize(
            image_layer.values,
            colors=colors,
            background=background,
            normalize=normalize,
        )
        da = xr.DataArray(
            colored,
            coords=[
                image_layer.coords[Dims.CHANNELS],
                image_layer.coords[Dims.Y],
                image_layer.coords[Dims.X],
                ["r", "g", "b", "a"],
            ],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X, Dims.RGBA],
            name=Layers.PLOT,
            attrs={Attrs.IMAGE_COLORS: {k.item(): v for k, v in zip(image_layer.coords[Dims.CHANNELS], colors)}},
        )

        if merge:
            da = da.sum(Dims.CHANNELS, keep_attrs=True)
            da.values[da.values > 1] = 1.0

        return xr.merge([self._obj, da])

    def render_segmentation(
        self,
        alpha: float = 0,
        alpha_boundary: float = 1,
        mode: str = "inner",
    ) -> xr.Dataset:
        """
        Render the segmentation layer of the data object.

        This method renders the segmentation layer of the data object and returns an updated data object
        with the rendered visualization. The rendered segmentation is represented in RGBA format.

        Parameters
        ----------
        alpha : float, optional
            The alpha value to control the transparency of the rendered segmentation. Default is 0.
        alpha_boundary : float, optional
            The alpha value for boundary pixels in the rendered segmentation. Default is 1.
        mode : str, optional
            The mode for rendering the segmentation: "inner" for internal region, "boundary" for boundary pixels.
            Default is "inner".

        Returns
        -------
        any
            The updated data object with the rendered segmentation as a new plot layer.

        Notes
        -----
        - The function extracts the segmentation layer and information about boundary pixels from the data object.
        - It applies the specified alpha values and mode to render the segmentation.
        - The rendered segmentation is represented in RGBA format and added as a new plot layer to the data object.
        """
        assert Layers.SEGMENTATION in self._obj, "Add Segmentation first."

        color_dict = {1: "white"}
        cmap = _get_listed_colormap(color_dict)
        segmentation = self._obj[Layers.SEGMENTATION].values
        segmentation = _remove_segmentation_mask_labels(segmentation, self._obj.coords[Dims.CELLS].values)
        # mask = _label_segmentation_mask(segmentation, cells_dict)

        if Layers.PLOT in self._obj:
            attrs = self._obj[Layers.PLOT].attrs
            rendered = _render_label(
                segmentation,
                cmap,
                self._obj[Layers.PLOT].values,
                alpha=alpha,
                alpha_boundary=alpha_boundary,
                mode=mode,
            )
            self._obj = self._obj.drop_vars(Layers.PLOT)
        else:
            attrs = {}
            rendered = _render_label(
                segmentation,
                cmap,
                alpha=alpha,
                alpha_boundary=alpha_boundary,
                mode=mode,
            )

        da = xr.DataArray(
            rendered,
            coords=[
                self._obj.coords[Dims.Y],
                self._obj.coords[Dims.X],
                ["r", "g", "b", "a"],
            ],
            dims=[Dims.Y, Dims.X, Dims.RGBA],
            name=Layers.PLOT,
            attrs=attrs,
        )
        return xr.merge([self._obj, da])

    def render_label(
        self, alpha: float = 0, alpha_boundary: float = 1, mode: str = "inner", override_color: Union[str, None] = None
    ) -> xr.Dataset:
        """
        Render the labeled cells in the data object.

        This method renders the labeled cells in the data object based on the label colors and segmentation.
        The rendered visualization is represented in RGBA format.

        Parameters
        ----------
        alpha : float, optional
            The alpha value to control the transparency of the rendered labels. Default is 0.
        alpha_boundary : float, optional
            The alpha value for boundary pixels in the rendered labels. Default is 1.
        mode : str, optional
            The mode for rendering the labels: "inner" for internal region, "boundary" for boundary pixels.
            Default is "inner".
        override_color : any, optional
            The color value to override the default label colors. Default is None.

        Returns
        -------
        any
            The updated data object with the rendered labeled cells as a new plot layer.

        Raises
        ------
        AssertionError
            If the data object does not contain label information. Use 'add_labels' function to add labels first.

        Notes
        -----
        - The function retrieves label colors from the data object and applies the specified alpha values and mode.
        - It renders the labeled cells based on the label colors and the segmentation layer.
        - The rendered visualization is represented in RGBA format and added as a new plot layer to the data object.
        - If 'override_color' is provided, all labels will be rendered using the specified color.
        """
        assert Layers.LABELS in self._obj, "Add labels via the add_labels function first."

        # TODO: Attribute class in constants.py
        color_dict = self._obj.la._label_to_dict(Props.COLOR, relabel=True)
        if override_color is not None:
            color_dict = {k: override_color for k in color_dict.keys()}

        cmap = _get_listed_colormap(color_dict)

        cells_dict = self._obj.la._cells_to_label(relabel=True)
        segmentation = self._obj[Layers.SEGMENTATION].values
        mask = _label_segmentation_mask(segmentation, cells_dict)

        if Layers.PLOT in self._obj:
            attrs = self._obj[Layers.PLOT].attrs
            rendered = _render_label(
                mask,
                cmap,
                self._obj[Layers.PLOT].values,
                alpha=alpha,
                alpha_boundary=alpha_boundary,
                mode=mode,
            )
            self._obj = self._obj.drop_vars(Layers.PLOT)
        else:
            attrs = {}
            rendered = _render_label(mask, cmap, alpha=alpha, alpha_boundary=alpha_boundary, mode=mode)

        da = xr.DataArray(
            rendered,
            coords=[
                self._obj.coords[Dims.Y],
                self._obj.coords[Dims.X],
                ["r", "g", "b", "a"],
            ],
            dims=[Dims.Y, Dims.X, Dims.RGBA],
            name=Layers.PLOT,
            attrs=attrs,
        )

        return xr.merge([self._obj, da])
