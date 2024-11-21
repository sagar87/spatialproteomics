from typing import List, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..base_logger import logger
from ..constants import Attrs, Dims, Features, Layers, Props
from .utils import (
    _autocrop,
    _colorize,
    _compute_erosion,
    _get_listed_colormap,
    _label_segmentation_mask,
    _render_labels,
    _render_neighborhoods,
    _render_obs,
    _render_segmentation,
)


@xr.register_dataset_accessor("pl")
class PlotAccessor:
    """Adds plotting functions to the image container."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _get_bounds(self):
        """
        Get the bounds of the object.

        Returns
        -------
        list
            A list containing the minimum and maximum values for the x and y coordinates.
        """
        xmin = self._obj.coords[Dims.X].values[0]
        ymin = self._obj.coords[Dims.Y].values[0]
        xmax = self._obj.coords[Dims.X].values[-1]
        ymax = self._obj.coords[Dims.Y].values[-1]

        return [xmin, xmax, ymin, ymax]

    def _create_channel_legend(self, **kwargs):
        """
        Create a legend for the channels in the plot layer. Used when rendering intensities.

        Returns:
            elements (list): A list of Patch objects representing the legend elements.
        """
        # checking if the plot layer exists
        assert Layers.PLOT in self._obj, "No plot layer found. Please call pl.colorize() first."
        # checking if the image colors attribute exists
        assert (
            Attrs.IMAGE_COLORS in self._obj[Layers.PLOT].attrs
        ), "No image colors found. Please call pl.colorize() first."

        color_dict = self._obj[Layers.PLOT].attrs[Attrs.IMAGE_COLORS]

        # removing unlabeled cells (label = 0)
        color_dict = {k: v for k, v in color_dict.items() if k != 0}

        elements = [Patch(facecolor=c, label=ch, **kwargs) for ch, c in color_dict.items()]
        return elements

    def _create_segmentation_legend(self):
        """
        Create a legend for the segmentation layers.

        Returns:
            elements (list): A list of Line2D objects representing the segmentation layers.
        """
        assert (
            Attrs.SEGMENTATION_COLORS in self._obj[Layers.PLOT].attrs
        ), "No segmentation colors found. Please specify colors when rendering multiple segmentation masks."

        color_dict = self._obj[Layers.PLOT].attrs[Attrs.SEGMENTATION_COLORS]

        elements = [
            Line2D(
                [0],
                [0],
                color=c,
                label=ch,
            )
            for ch, c in color_dict.items()
        ]

        return elements

    def _create_label_legend(self, order=None):
        """
        Create a legend for the cell type labels.

        Args:
            order (list, optional): A list specifying the order of the labels in the legend.
                                    If None, the labels will be sorted by default.

        Returns:
            elements (list): A list of Line2D objects representing the legend elements.
        """
        # getting colors and names for each cell type label
        color_dict = self._obj.la._label_to_dict(Props.COLOR)
        names_dict = self._obj.la._label_to_dict(Props.NAME)

        # removing unlabeled cells (label = 0)
        # also removing labels which are not present in the object
        present_labels = np.unique(self._obj[Layers.OBS].sel(features=Features.LABELS).values)
        color_dict = {k: v for k, v in color_dict.items() if k != 0 and k in present_labels}
        names_dict = {k: v for k, v in names_dict.items() if k != 0 and k in present_labels}

        # Apply custom ordering if provided, else sort by default
        if order is not None:
            # if all list items are strings, we want to convert them to the corresponding indices
            if all(isinstance(i, str) for i in order):
                label_dict = self._obj.la._label_to_dict(Props.NAME, reverse=True)
                order = [label_dict.get(i) for i in order]
            ordered_labels = [label for label in order if label in color_dict]
        else:
            ordered_labels = sorted(color_dict)

        elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=names_dict[i],
                markerfacecolor=color_dict[i],
                markersize=15,
            )
            for i in ordered_labels
            if i in self._obj.coords[Dims.LABELS]
        ]

        return elements

    def _create_neighborhood_legend(self, order=None):
        """
        Create a legend for the neighborhoods.

        Parameters:
            order (list, optional): A list specifying the order of the neighborhood indices.
                                    If None, the default sorted order of color_dict is used.

        Returns:
            elements (list): A list of Line2D objects representing the legend elements.
        """
        # Getting colors and names for each cell type label
        color_dict = self._obj.nh._neighborhood_to_dict(Props.COLOR)
        names_dict = self._obj.nh._neighborhood_to_dict(Props.NAME)

        # Use the provided order, or default to sorting the color_dict keys
        if order is None:
            order = sorted(color_dict)
        else:
            invalid_keys = set(order) - set(color_dict.keys())
            missing_keys = set(color_dict.keys()) - set(order)

            # if the order list contains strings, we want to convert them to the corresponding indices
            # in that case, we also want to convert the invalid and missing keys to the corresponding strings
            if all(isinstance(i, str) for i in order):
                # getting neighborhoods for which we have invalid or missing keys
                neighborhood_dict = self._obj.nh._neighborhood_to_dict(Props.NAME)
                invalid_keys = set(order) - set(neighborhood_dict.values())
                missing_keys = set(neighborhood_dict.values()) - set(order)

                # converting the strings into indices
                neighborhood_dict = self._obj.nh._neighborhood_to_dict(Props.NAME, reverse=True)
                order = [neighborhood_dict.get(i) for i in order]

            # ensuring that the elements in order are the same as the ones present in the object
            assert all(
                i in color_dict for i in order
            ), f"Some keys in the order list are not present in the object. Invalid keys: {invalid_keys}"
            # ensuring that there are no duplicates in order
            assert len(order) == len(set(order)), "The order list contains duplicates. Please remove them."
            # checking that all keys in the object are present in the order list
            assert all(
                i in order for i in color_dict
            ), f"Some keys in the object are not present in the order list. Missing keys: {missing_keys}"

        # Creating legend elements based on the ordered keys
        elements = [
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                label=names_dict[i],
                markerfacecolor=color_dict[i],
                markersize=15,
            )
            for i in order
        ]

        return elements

    def _create_obs_legend(
        self,
        ax,
        fraction=0.046,
        pad=0.04,
        shrink=1.0,
        aspect=20,
        location="right",
        cbar_label=True,
        vmin=None,
        vmax=None,
        **kwargs,
    ):
        """
        Create and adjust the observation colorbar.

        Parameters:
        - ax: The axis to place the colorbar.
        - fraction: Fraction of the original axes to use for the colorbar.
        - pad: Padding between the colorbar and the plot.
        - shrink: Fraction by which to shrink the colorbar.
        - aspect: Aspect ratio of the colorbar (length vs width).
        - location: Location of the colorbar ('right', 'left', 'top', 'bottom').
        - cbar_label: Whether to show the feature name as the colorbar label.
        - vmin: The minimum value for the colorbar.
        - vmax: The maximum value for the colorbar.

        Returns:
        - cbar: The created colorbar.
        """
        assert (
            Attrs.OBS_COLORS in self._obj[Layers.PLOT].attrs
        ), "No observation colors found. Please call pl.render_obs() first."

        obs_colors = self._obj[Layers.PLOT].attrs[Attrs.OBS_COLORS]
        feature = obs_colors["feature"]
        cmap = obs_colors["cmap"]
        if vmin is not None and vmax is not None:
            min_val = vmin
            max_val = vmax
        else:
            min_val = obs_colors["min"]
            max_val = obs_colors["max"]

        # Create the colorbar with dynamic positioning and size
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
        sm.set_array([])

        # Adjust the colorbar using the provided arguments
        if not cbar_label:
            feature = ""
        cbar = plt.colorbar(
            sm,
            ax=ax,
            label=feature,
            fraction=fraction,
            pad=pad,
            shrink=shrink,
            aspect=aspect,
            location=location,
            **kwargs,
        )

        return [cbar]

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
        layer_key: str = Layers.IMAGE,
    ) -> xr.Dataset:
        """
        Colorizes a stack of images.

        Parameters
        ----------
        colors : List[str], optional
            A list of strings that denote the color of each channel.
        background : str, optional
            Background color of the colorized image. Default is "black".
        normalize : bool, optional
            Normalize the image prior to colorizing it. Default is True.
        merge : True, optional
            Merge the channel dimension. Default is True.
        layer_key : str, optional
            The key of the layer containing the image. Default is '_image'.

        Returns
        -------
        xr.Dataset
            The image container with the colorized image stored in '_plot'.
        """
        # check if a plot already exists
        assert (
            Layers.PLOT not in self._obj
        ), "A plot layer already exists. If you want to plot the channel intensities and a segmentation mask, make sure to call pl.colorize() first, and then pl.render_segmentation() to render the segmentation on top of it. Alternatively, you can call pl.imshow(render_segmentation=True) to achieve the same effect."

        if isinstance(colors, str):
            colors = [colors]

        image_layer = self._obj[layer_key]
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

    def show(
        self,
        render_image: bool = True,
        render_segmentation: bool = False,
        render_labels: bool = False,
        render_neighborhoods: bool = False,
        ax=None,
        legend_image: bool = True,
        legend_segmentation: bool = True,
        legend_label: bool = True,
        legend_neighborhoods: bool = True,
        background: str = "black",
        downsample: int = 1,
        label_order: Optional[List] = None,
        neighborhood_order: Optional[List] = None,
        legend_kwargs: dict = {"framealpha": 1},
        segmentation_kwargs: dict = {},
        label_kwargs: dict = {},
        neighborhood_kwargs: dict = {},
    ) -> xr.Dataset:
        """
        Display an image with optional rendering elements. Can be used to render intensities, segmentation masks and labels, either individually or all at once.

        Parameters
        ----------
        render_image : bool
            Whether to render the image with channel intensities. Default is True.
        render_segmentation : bool
            Whether to render segmentation. Default is False.
        render_labels : bool
            Whether to render labels. Default is False.
        render_neighborhoods : bool
            Whether to render neighborhoods. Default is False.
        ax
            The matplotlib axes to plot on. If None, the current axes will be used.
        legend_image : bool
            Whether to show the channel legend. Default is True.
        legend_segmentation : bool
            Whether to show the segmentation legend (only becomes relevant when dealing with multiple segmentation layers, e. g. when using cellpose). Default is False.
        legend_label : bool
            Whether to show the label legend. Default is True.
        legend_neighborhoods : bool
            Whether to show the neighborhood legend. Default is True.
        background : str
            Background color of the image. Default is "black".
        downsample : int
            Downsample factor for the image. Default is 1 (no downsampling).
        label_order : list
            A list specifying the order of the label indices. Default is None.
        neighborhood_order : list
            A list specifying the order of the neighborhood indices. Default is None.
        legend_kwargs : dict
            Keyword arguments for configuring the legend. Default is {"framealpha": 1}.
        segmentation_kwargs : dict
            Keyword arguments for rendering the segmentation. Default is {}.
        label_kwargs : dict
            Keyword arguments for rendering the labels. Default is {}.
        neighborhood_kwargs : dict
            Keyword arguments for rendering the neighborhoods. Default is {}.

        Returns
        -------
        xr.Dataset
            The dataset object including the plot.

        Raises
        ------
        - AssertionError: If no rendering element is specified.

        Notes
        -----
        - This method displays an image with optional rendering elements such as intensities, labels, and segmentation.
        - The `render_intensities`, `render_labels`, and `render_segmentation` parameters control which rendering elements are displayed.
        - The `ax` parameter allows specifying a specific matplotlib axes to plot on. If None, the current axes will be used.
        - The `show_channel_legend` and `show_label_legend` parameters control whether to show the channel and label legends, respectively.
        - The `legend_kwargs`, `segmentation_kwargs`, and `label_kwargs` parameters allow configuring the appearance of the legend, segmentation, and labels, respectively.
        """
        # check that at least one rendering element is specified
        assert any(
            [render_image, render_labels, render_segmentation, render_neighborhoods]
        ), "No rendering element specified. Please set at least one of 'render_image', 'render_labels', 'render_segmentation', or 'render_neighborhoods' to True."

        # store a copy of the original object to avoid overwriting it
        obj = self._obj.copy()
        if Layers.PLOT not in self._obj:
            if render_image:
                # if there are more than 20 channels, only the first one is plotted
                if self._obj.sizes[Dims.CHANNELS] > 20:
                    logger.warning(
                        "More than 20 channels are present in the image. Plotting first channel only. You can subset the channels via pp.[['channel1', 'channel2', ...]] or specify your own color scheme by calling pl.colorize() before calling pl.show()."
                    )
                    channel = str(self._obj.coords[Dims.CHANNELS].values[0])
                    obj = self._obj.pp[channel].pl.colorize(colors=["white"], background=background)
                # if there are less than 20 channels, all are plotted
                else:
                    obj = self._obj.pl.colorize(background=background)
            else:
                # if no image is rendered, we need to add a plot layer with a background color
                rgba_background = mcolors.to_rgba(background)
                colored = np.ones((self._obj.sizes[Dims.Y], self._obj.sizes[Dims.X], 4)) * rgba_background

                da = xr.DataArray(
                    colored,
                    coords=[
                        self._obj.coords[Dims.Y],
                        self._obj.coords[Dims.X],
                        ["r", "g", "b", "a"],
                    ],
                    dims=[Dims.Y, Dims.X, Dims.RGBA],
                    name=Layers.PLOT,
                )

                obj = xr.merge([self._obj, da])
        else:
            # if a plot already exists, but the user tries to set a background color, we raise a warning
            if background != "black":
                logger.warning(
                    "The background color is set during the first color pass. If you called pl.colorize() before pl.show(), please set the background color there instead using pl.colorize(background='your_color')."
                )

        if render_neighborhoods:
            obj = obj.pl.render_neighborhoods(**neighborhood_kwargs)

        if render_labels:
            obj = obj.pl.render_labels(**label_kwargs)

        if render_segmentation:
            obj = obj.pl.render_segmentation(**segmentation_kwargs)
            # it should not be possible to show a segmentation legend if there is only one segmentation layer
            segmentation_shape = self._obj[segmentation_kwargs.get("layer_key", Layers.SEGMENTATION)].values.shape
            legend_segmentation = legend_segmentation and len(segmentation_shape) > 2

        legend_image = legend_image and render_image
        legend_segmentation = legend_segmentation and render_segmentation
        legend_label = legend_label and render_labels
        legend_neighborhoods = legend_neighborhoods and render_neighborhoods

        return obj.pl.imshow(
            legend_image=legend_image,
            legend_segmentation=legend_segmentation,
            legend_label=legend_label,
            legend_neighborhoods=legend_neighborhoods,
            label_order=label_order,
            neighborhood_order=neighborhood_order,
            ax=ax,
            downsample=downsample,
            legend_kwargs=legend_kwargs,
        )

    def annotate(
        self,
        variable: str = "cell",
        layer_key: str = Layers.OBS,
        highlight: list = [],
        text_kwargs: dict = {"color": "w", "fontsize": 12},
        highlight_kwargs: dict = {"color": "w", "fontsize": 16, "fontweight": "bold"},
        bbox: Union[List, None] = None,
        format_string: str = "",
        ax=None,
    ) -> xr.Dataset:
        """
        Annotates cells with their respective number on the plot.

        Parameters
        ----------
        variable : str, optional
            The feature in the observation table to be used for cell annotation. Default is "cell".
        layer_key : str, optional
            The key representing the layer in the data object. Default is Layers.OBS.
        highlight : list, optional
            A list containing cell IDs to be highlighted in the plot.
        text_kwargs : dict, optional
            Keyword arguments passed to matplotlib's text function for normal cell annotations.
        highlight_kwargs : dict, optional
            Similar to 'text_kwargs' but specifically for the cell IDs passed via 'highlight'.
        bbox : Union[List, None], optional
            A list containing bounding box coordinates [xmin, xmax, ymin, ymax] to annotate cells only within the box.
            Default is None, which annotates all cells.
        format_string : str, optional
            The format string used to format the cell annotation text. Default is "" (no formatting).
        ax : matplotlib.axes, optional
            The matplotlib axis to plot on. If not provided, the current axis will be used.

        Returns
        -------
        xr.Dataset
            The updated image container.

        Notes
        -----
        - The function annotates cells with their respective values from the selected feature.
        - You can highlight specific cells in the plot using the 'highlight' parameter.
        - Bounding box coordinates can be provided via 'bbox' to annotate cells only within the specified box.
        - 'format_string' can be used to format the cell annotation text (e.g., "{t:.2f}" for float formatting).
        """
        if ax is None:
            ax = plt.gca()

        if bbox is None:
            cells = self._obj.coords[Dims.CELLS]
        else:
            assert len(bbox) == 4, "The bbox-argument must specify [xmin, xmax, ymin, ymax]."
            sub = self._obj.im[bbox[0] : bbox[1], bbox[2] : bbox[3]]
            cells = sub.coords[Dims.CELLS]

        for cell in cells:
            x = self._obj[Layers.OBS].sel({Dims.CELLS: cell, Dims.FEATURES: Features.X}).values
            y = self._obj[Layers.OBS].sel({Dims.CELLS: cell, Dims.FEATURES: Features.Y}).values
            if variable != "cell":
                table = self._obj[layer_key]
                dims = table.sizes
                if len(dims) != 2:
                    raise ValueError("Layer does not have the dimension.")
                if Dims.CELLS not in dims:
                    raise ValueError("Layer does not have a cell dimension.")

                dim = [d for d in dims if d != Dims.CELLS][0]

                t = table.sel({Dims.CELLS: cell, dim: variable}).values
            else:
                t = cell.values

            if cell in highlight:
                ax.text(x, y, s=f"{t:{format_string}}", **highlight_kwargs)
            else:

                ax.text(x, y, s=f"{t:{format_string}}", **text_kwargs)

        return self._obj

    def render_segmentation(
        self,
        layer_key: str = Layers.SEGMENTATION,
        colors: List[str] = [
            "white",
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
        alpha: float = 0.0,
        alpha_boundary: float = 1.0,
        mode: str = "inner",
    ) -> xr.Dataset:
        """
        Renders the segmentation mask with optional alpha blending and boundary rendering.

        Parameters:
            layer_key (str, optional): The key of the layer containing the segmentation mask. Default is '_segmentation'.
            colors (List[str], optional): A list of colors to be used for rendering the segmentation mask. Default is ['white'].
            alpha (float, optional): The alpha value for blending the segmentation mask with the plot. Default is 0.0.
            alpha_boundary (float, optional): The alpha value for rendering the boundary of the segmentation mask. Default is 1.0.
            mode (str, optional): The mode for rendering the segmentation mask. Possible values are "inner" and "outer". Default is "inner".

        Returns:
            xr.Dataset: The modified xarray Dataset.

        Raises:
            AssertionError: If no segmentation layer is found in the object.

        Note:
            - The segmentation mask must be added to the object before calling this method.
            - The segmentation mask is expected to have a single channel with integer labels.
            - The rendered segmentation mask will be added as a new layer to the object.
        """
        assert (
            layer_key in self._obj
        ), f"Could not find segmentation layer with key {layer_key}. Please add a segmentation mask before calling this method."

        segmentation = self._obj[layer_key].values

        # the segmentation mask can either be 2D or 3D
        # if it is 2D, we transform it into 3D for compatibility with 3D segmentation masks (where multiple channels are present)
        if len(segmentation.shape) == 2:
            segmentation = np.expand_dims(segmentation, axis=0)
            channels = [1]
        else:
            channels = list(range(1, len(self._obj[layer_key].coords[Dims.CHANNELS]) + 1))

        # checking that there are as many colors as there are channels in the segmentation mask which is rendered
        assert (
            len(colors) >= segmentation.shape[0]
        ), f"Trying to render {segmentation.shape[0]} segmentation layers. Please provide custom colors using the colors argument."

        colors = colors[: len(channels)]

        if Layers.PLOT in self._obj:
            attrs = self._obj[Layers.PLOT].attrs
            rendered = _render_segmentation(
                segmentation,
                colors=colors,
                background="black",
                img=self._obj[Layers.PLOT].values,
                alpha=alpha,
                alpha_boundary=alpha_boundary,
                mode=mode,
            )
            self._obj = self._obj.drop_vars(Layers.PLOT)
        else:
            attrs = {}
            rendered = _render_segmentation(
                segmentation,
                colors=colors,
                background="black",
                alpha=alpha,
                alpha_boundary=alpha_boundary,
                mode=mode,
            )

        # adding segmentation colors to the attributes
        if Dims.CHANNELS in self._obj[layer_key].coords:
            attrs[Attrs.SEGMENTATION_COLORS] = {
                k.item(): v for k, v in zip(self._obj[layer_key].coords[Dims.CHANNELS], colors)
            }

        da = xr.DataArray(
            rendered,
            coords=[
                channels,
                self._obj.coords[Dims.Y],
                self._obj.coords[Dims.X],
                ["r", "g", "b", "a"],
            ],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X, Dims.RGBA],
            name=Layers.PLOT,
            attrs=attrs,
        )

        # merging the masks together into a 2D array
        da = da.sum(Dims.CHANNELS, keep_attrs=True)
        da.values[da.values > 1] = 1.0

        return xr.merge([self._obj, da])

    def render_labels(
        self,
        alpha: float = 1.0,
        alpha_boundary: float = 1.0,
        mode: str = "inner",
        override_color: Optional[str] = None,
    ) -> xr.Dataset:
        """
        Renders cell type labels on the plot.

        Parameters:
            alpha (float, optional): The transparency of the labels. Defaults to 1.0.
            alpha_boundary (float, optional): The transparency of the label boundaries. Defaults to 1.0.
            mode (str, optional): The mode of rendering. Can be "inner" or "outer". Defaults to "inner".
            override_color (str, optional): The color to override the label colors. Defaults to None.

        Returns:
            xr.Dataset: The modified dataset with rendered labels.

        Raises:
            AssertionError: If no labels are found in the object.

        Notes:
            - This method requires labels to be present in the object. Add labels first using `la.predict_cell_types_argmax()` or `tl.astir()`.
            - The `mode` parameter determines whether the labels are rendered inside or outside the label boundaries.
            - The `override_color` parameter can be used to override the label colors with a single color.
        """
        assert (
            Layers.LA_PROPERTIES in self._obj
        ), "No labels found in the object. Add labels first, for example by using la.predict_cell_types_argmax() or tl.astir()."

        color_dict = self._obj.la._label_to_dict(Props.COLOR, relabel=True)

        if override_color is not None:
            color_dict = {k: override_color for k in color_dict.keys()}

        cmap = _get_listed_colormap(color_dict)

        cells_dict = self._obj.la._cells_to_label(relabel=True)
        segmentation = self._obj[Layers.SEGMENTATION].values
        mask = _label_segmentation_mask(segmentation, cells_dict)

        if Layers.PLOT in self._obj:
            attrs = self._obj[Layers.PLOT].attrs
            rendered = _render_labels(
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
            rendered = _render_labels(mask, cmap, alpha=alpha, alpha_boundary=alpha_boundary, mode=mode)

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

    def render_neighborhoods(
        self,
        style: str = "neighborhoods",
        alpha: float = 1.0,
        alpha_boundary: float = 1.0,
        boundary_color: str = "dimgray",
        boundary_thickness: int = 3,
        dilation_strength: int = 40,
        erosion_strength: int = 35,
    ) -> xr.Dataset:
        """
        Render neighborhoods on the spatial data.

        Parameters
        ----------
        style : str, optional
            The style of rendering, either 'cells' or 'neighborhoods'. Default is 'neighborhoods'.
        alpha : float, optional
            The alpha transparency for the rendered image. Default is 1.0.
        alpha_boundary : float, optional
            The alpha transparency for the boundary of the rendered image. Default is 1.0.
        boundary_color : str, optional
            The color of the boundary lines. Default is 'dimgray'.
        boundary_thickness : int, optional
            The thickness of the boundary lines. Default is 3.
        dilation_strength : int, optional
            The strength of the dilation applied to the cells. Default is 40.
        erosion_strength : int, optional
            The strength of the erosion applied to the cells. Default is 35.

        Returns
        -------
        xr.Dataset
            The dataset with the rendered neighborhoods.

        Raises
        ------
        AssertionError
            If no neighborhoods are found in the object or if the style is not 'cells' or 'neighborhoods'.
        """

        assert (
            Layers.NH_PROPERTIES in self._obj
        ), "No neighborhoods found in the object. Add neighborhoods first, for example by using nh.compute_neighborhoods_radius()."
        assert style in ["cells", "neighborhoods"], "Style must be either 'cells' or 'neighborhoods'."
        assert dilation_strength > 0, "Dilation strength must be greater than 0."
        assert erosion_strength > 0, "Erosion strength must be greater than 0."
        assert (
            dilation_strength >= erosion_strength
        ), "Dilation strength must be greater than or equal to erosion strength."

        color_dict = self._obj.nh._neighborhood_to_dict(Props.COLOR, relabel=True)

        cmap = _get_listed_colormap(color_dict)
        rendered = None

        if style == "cells":
            cells_dict = self._obj.nh._cells_to_neighborhood(relabel=True)
            segmentation = self._obj[Layers.SEGMENTATION].values
            mask = _label_segmentation_mask(segmentation, cells_dict)

            if Layers.PLOT in self._obj:
                attrs = self._obj[Layers.PLOT].attrs
                rendered = _render_labels(
                    mask,
                    cmap,
                    self._obj[Layers.PLOT].values,
                    alpha=alpha,
                    alpha_boundary=alpha_boundary,
                    mode="inner",
                )
                self._obj = self._obj.drop_vars(Layers.PLOT)
            else:
                attrs = {}
                rendered = _render_labels(mask, cmap, alpha=alpha, alpha_boundary=alpha_boundary, mode="inner")
        elif style == "neighborhoods":
            cells_dict = self._obj.nh._cells_to_neighborhood(relabel=True)

            # step 1: apply a Voronoi tesselation to the neighborhoods
            segmentation = self._obj.pp.grow_cells(dilation_strength, suppress_warning=True)[Layers.SEGMENTATION].values
            eroded_mask = _compute_erosion(segmentation, erosion_strength=erosion_strength)

            # multiplying the segmentation with the eroded mask
            segmentation = segmentation * eroded_mask

            mask = _label_segmentation_mask(segmentation, cells_dict)

            if Layers.PLOT in self._obj:
                attrs = self._obj[Layers.PLOT].attrs
                rendered = _render_neighborhoods(
                    mask,
                    cmap,
                    self._obj[Layers.PLOT].values,
                    alpha=alpha,
                    alpha_boundary=alpha_boundary,
                    boundary_color=boundary_color,
                    boundary_thickness=boundary_thickness,
                )
                self._obj = self._obj.drop_vars(Layers.PLOT)
            else:
                attrs = {}
                rendered = _render_neighborhoods(
                    mask,
                    cmap,
                    alpha=alpha,
                    alpha_boundary=alpha_boundary,
                    boundary_color=boundary_color,
                    boundary_thickness=boundary_thickness,
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

    def render_obs(
        self,
        feature: str = None,
        cmap: str = "viridis",
        alpha: float = 1.0,
        alpha_boundary: float = 1.0,
        mode: str = "inner",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> xr.Dataset:
        """
        Render the observation layer with the specified feature and colormap.

        Parameters
        ----------
        feature : str, optional
            The feature to be rendered from the observation layer. Default is None.
        cmap : str, optional
            The colormap to be used for rendering. Default is "viridis".
        alpha : float, optional
            The alpha value for the rendered image. Default is 1.0.
        alpha_boundary : float, optional
            The alpha value for the boundaries in the rendered image. Default is 1.0.
        mode : str, optional
            The mode for rendering. Default is "inner".
        vmin : float, optional
            The minimum value for colormap normalization. Default is None.
        vmax : float, optional
            The maximum value for colormap normalization. Default is None.

        Returns
        -------
        xr.Dataset
            The dataset with the rendered observation layer.

        Raises
        ------
        AssertionError
            If the observation layer or segmentation layer is not found in the object.
            If the specified feature is not found in the observation layer.
        """
        assert (
            Layers.OBS in self._obj
        ), "No observation layer found in the object. Please add an observation layer first."
        assert (
            Layers.SEGMENTATION in self._obj
        ), "No segmentation layer found in the object. Please add a segmentation layer first."
        obs = self._obj[Layers.OBS]
        segmentation = self._obj[Layers.SEGMENTATION].values

        assert feature in obs.coords[Dims.FEATURES], f"Feature {feature} not found in the observation layer."

        # creating a continuous colormap
        cmap = plt.cm.get_cmap(cmap)
        feature_values = obs.sel(features=feature).values

        # mapping the feature values onto the segmentation mask (replacing the cell indices with the feature values)
        # we need to ensure the mapping between the feature values and the segmentation is correct
        cell_indices = self._obj.coords[Dims.CELLS]

        # creating a boolean mask for the background
        # the reason why this needs to be computed before the mapping is that there could be obs values equal to 0
        background_array = segmentation > 0

        # Ensure vmin and vmax are determined
        vmin = vmin if vmin is not None else feature_values.min()
        vmax = vmax if vmax is not None else feature_values.max()

        # dict that maps each cell ID to its feature value
        feature_mapping = {k.item(): v for k, v in zip(cell_indices, feature_values)}
        # setting the background to zero
        feature_mapping[0] = 0
        # mapping the feature values onto the segmentation mask
        segmentation = np.vectorize(lambda x: feature_mapping.get(x, 0), otypes=[np.float64])(segmentation)

        if Layers.PLOT in self._obj:
            attrs = self._obj[Layers.PLOT].attrs

            rendered = _render_obs(
                segmentation,
                cmap,
                self._obj[Layers.PLOT].values,
                background_array=background_array,
                alpha=alpha,
                alpha_boundary=alpha_boundary,
                mode=mode,
                vmin=vmin,
                vmax=vmax,
            )

            self._obj = self._obj.drop_vars(Layers.PLOT)
        else:
            attrs = {}
            rendered = _render_obs(
                segmentation,
                cmap,
                background_array=background_array,
                alpha=alpha,
                alpha_boundary=alpha_boundary,
                mode=mode,
                vmin=vmin,
                vmax=vmax,
            )

        # adding information for rendering the colorbar
        # in here, we need the name of the feature, the colormap and the min and max values of the feature
        attrs[Attrs.OBS_COLORS] = {
            "feature": feature,
            "cmap": cmap,
            "min": vmin,
            "max": vmax,
        }

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

    def imshow(
        self,
        legend_image: bool = False,
        legend_segmentation: bool = False,
        legend_label: bool = False,
        legend_neighborhoods: bool = False,
        legend_obs: bool = False,
        downsample: int = 1,
        label_order: Optional[List[str]] = None,
        neighborhood_order: Optional[List[str]] = None,
        legend_kwargs: dict = {"framealpha": 1},
        cbar_kwargs: dict = {},
        ax=None,
    ):
        """
        Plots the image after rendering certain layers.
        Meant to be used in conjunction with pl.colorize(), pl.render_segmentation, pl.render_label() or pl.render_neighborhoods().
        For a more high level wrapper, please refer to pl.show() instead.

        Parameters
        ----------
        legend_image : bool, optional
            Show the legend for the channels. Default is False.
        legend_segmentation : bool, optional
            Show the legend for the segmentation. Default is False.
        legend_label : bool, optional
            Show the label legend. Default is False.
        legend_neighborhoods : bool, optional
            Show the neighborhood legend. Default is False.
        legend_obs: bool, optional
            Show the observation colorbar. Default is False.
        downsample : int, optional
            Downsample factor for the image. Default is 1.
        legend_kwargs : dict, optional
            Additional keyword arguments for configuring the legend. Default is {"framealpha": 1}.
        cbar_kwargs : dict, optional
            Additional keyword arguments for configuring the colorbar. Default is {}.
        label_order: Optional[List[str]], optional
            The order in which the labels should be displayed. Default is None.
        neighborhood_order: Optional[List[str]], optional
            The order in which the neighborhoods should be displayed. Default is None.
        ax : matplotlib.axes, optional
            The matplotlib axis to plot on. If not provided, the current axis will be used.

        Returns
        -------
        xr.Dataset
            The updated image container.

        Notes
        -----
        - The function is used to plot images in conjunction with 'im.colorize' and 'la.render_label'.
        - The appearance of the plot and the inclusion of legends can be controlled using the respective parameters.
        """
        if Layers.PLOT not in self._obj:
            logger.warning("No plot defined yet. Plotting the first channel only. Please call pl.colorize() first.")
            channel = str(self._obj.coords[Dims.CHANNELS].values[0])
            self._obj = self._obj.pp[channel].pl.colorize(colors=["white"])

        if ax is None:
            ax = plt.gca()

        bounds = self._obj.pl._get_bounds()

        ax.imshow(
            self._obj[Layers.PLOT].values[::downsample, ::downsample],
            origin="lower",
            interpolation="none",
            extent=bounds,
        )

        legend = []

        if legend_image:
            legend += self._obj.pl._create_channel_legend()

        if legend_neighborhoods:
            # check that the neighborhood order is either a list of strings or a list of integers
            if neighborhood_order is not None:
                assert all(
                    [isinstance(i, (str, int)) for i in neighborhood_order]
                ), "The neighborhood order must be a list of strings or integers."
            legend += self._obj.pl._create_neighborhood_legend(order=neighborhood_order)

        if legend_segmentation:
            legend += self._obj.pl._create_segmentation_legend()

        if legend_label:
            legend += self._obj.pl._create_label_legend(order=label_order)

        if legend_obs:
            legend += self._obj.pl._create_obs_legend(ax=ax, **cbar_kwargs)

        if legend_image or legend_segmentation or legend_label or legend_neighborhoods:
            ax.legend(handles=legend, **legend_kwargs)

        return self._obj

    def scatter_labels(
        self,
        legend: bool = True,
        size: float = 1.0,
        alpha: float = 0.9,
        zorder: int = 10,
        render_edges: bool = False,
        render_self_edges: bool = False,
        ax=None,
        legend_kwargs: dict = {"framealpha": 1},
        scatter_kwargs: dict = {},
    ) -> xr.Dataset:
        """
        Scatter plot of labeled cells.

        Parameters
        ----------
        legend : bool, optional
            Whether to show the legend. Default is True.
        size : float, optional
            Size of the scatter markers. Default is 1.0.
        alpha : float, optional
            Transparency of the scatter markers. Default is 0.9.
        zorder : int, optional
            The z-order of the scatter markers. Default is 10.
        render_edges : bool, optional
            Whether to render the edges between cells within the same neighborhood. Default is False.
        render_self_edges : bool, optional
            Whether to render the edges between cells and themselves. Default is False.
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the scatter. If not provided, the current axes will be used.
        legend_kwargs : dict, optional
            Additional keyword arguments for configuring the legend. Default is {"framealpha": 1}.
        scatter_kwargs : dict, optional
            Additional keyword arguments for configuring the scatter plot.

        Returns
        -------
        xr.Dataset
            The modified spatialproteomics object.
        """
        if ax is None:
            ax = plt.gca()

        color_dict = self._obj.la._label_to_dict(Props.COLOR)
        label_dict = self._obj.la._cells_to_label()

        if not render_edges:
            for celltype in label_dict.keys():
                label_subset = self._obj.la[celltype]
                obs_layer = label_subset[Layers.OBS]
                x = obs_layer.loc[:, Features.X]
                y = obs_layer.loc[:, Features.Y]
                ax.scatter(
                    x.values, y.values, s=size, c=color_dict[celltype], alpha=alpha, zorder=zorder, **scatter_kwargs
                )
        else:
            # if we want to render the edges, we need to use the adjacency matrix and networkx
            assert (
                Layers.ADJACENCY_MATRIX in self._obj
            ), "No adjacency matrix found in the object. Please compute the adjacency matrix first by running either of the methods contained in the nh module (e. g. nh.compute_neighborhoods_radius())."

            try:
                import networkx as nx

                adjacency_matrix = self._obj[Layers.ADJACENCY_MATRIX].values

                # removing self-edges
                if not render_self_edges:
                    # this works in-place
                    np.fill_diagonal(adjacency_matrix, 0)

                G = nx.from_numpy_array(adjacency_matrix)
                spatial_df = self._obj.pp.get_layer_as_df(Layers.OBS)
                assert Features.X in spatial_df.columns, f"Feature {Features.X} not found in the observation layer."
                assert Features.Y in spatial_df.columns, f"Feature {Features.Y} not found in the observation layer."
                assert (
                    Features.LABELS in spatial_df.columns
                ), f"Feature {Features.LABELS} not found in the observation layer."
                spatial_df = spatial_df[[Features.X, Features.Y, Features.LABELS]].reset_index(drop=True)
                # Create node positions based on the centroid coordinates
                positions = {
                    i: (spatial_df.loc[i, Features.X], spatial_df.loc[i, Features.Y]) for i in range(len(spatial_df))
                }
                color_dict = self._obj.la._label_to_dict(Props.COLOR, keys_as_str=True)

                # Assign node colors based on the label
                node_colors = [color_dict[spatial_df.loc[i, Features.LABELS]] for i in range(len(spatial_df))]

                # Use networkx to draw the graph
                nx.draw(
                    G,
                    pos=positions,
                    node_color=node_colors,
                    with_labels=False,
                    node_size=size,
                    edge_color="gray",
                    ax=ax,
                    **scatter_kwargs,
                )
            except ImportError:
                raise ImportError("Please install networkx to render edges between cells.")

        xmin, xmax, ymin, ymax = self._obj.pl._get_bounds()
        ax.set_ylim([ymin, ymax])
        ax.set_xlim([xmin, xmax])

        ax.set_aspect("equal")  # Set equal aspect ratio for x and y axes

        if legend:
            legend = self._obj.pl._create_label_legend()
            ax.legend(handles=legend, **legend_kwargs).set_zorder(102)

        return self._obj

    def scatter(
        self,
        feature: str,
        palette: dict = None,
        legend: bool = True,
        layer_key: str = Layers.OBS,
        size: float = 1.0,
        alpha: float = 0.9,
        zorder: int = 10,
        ax=None,
        legend_kwargs: dict = {"framealpha": 1},
        scatter_kws: dict = {},
    ) -> xr.Dataset:
        """
        Create a scatter plot of some feature. At the moment, only categorical features are supported.

        Parameters
        ----------
        feature : str
            The feature to be plotted.
        palette : dict, optional
            A dictionary mapping feature values to colors. If not provided, a default palette will be used.
        legend : bool, optional
            Whether to show the legend. Default is True.
        layer_key : str, optional
            The key of the layer to be plotted. Default is Layers.OBS.
        size : float, optional
            The size of the scatter points. Default is 1.0.
        alpha : float, optional
            The transparency of the scatter points. Default is 0.9.
        zorder : int, optional
            The z-order of the scatter points. Default is 10.
        ax : object, optional
            The matplotlib axes object to plot on. If not provided, the current axes will be used.
        legend_kwargs : dict, optional
            Additional keyword arguments for configuring the legend. Default is {"framealpha": 1}.
        scatter_kws : dict, optional
            Additional keyword arguments for configuring the scatter plot. Default is {}.

        Returns
        -------
        xr.Dataset
            The original data object.

        Raises
        ------
        - AssertionError: If the layer_key is not found in the data object.
        - AssertionError: If the feature is not found in the specified layer.
        - AssertionError: If the X or Y coordinates are not found in the specified layer.
        - AssertionError: If the number of unique feature values is greater than 10 and no color_scheme is provided.
        - AssertionError: If not all unique feature values are present in the provided palette.

        """
        if ax is None:
            ax = plt.gca()

        # check if the layer exists
        assert layer_key in self._obj, f"Layer {layer_key} not found in the data object."

        layer = self._obj[layer_key]

        # check that the feature exists
        assert feature in layer.coords[Dims.FEATURES], f"Feature {feature} not found in the layer {layer_key}."

        # check if the layer contains X and Y coordinates
        assert Features.X in layer.coords[Dims.FEATURES], f"Feature {Features.X} not found in the layer {layer_key}."
        assert Features.Y in layer.coords[Dims.FEATURES], f"Feature {Features.Y} not found in the layer {layer_key}."

        x = layer.loc[:, Features.X]
        y = layer.loc[:, Features.Y]

        # if no palette is provided, we check if the data is categorical (i.e. has less than 10 unique values) or continuous
        # if it is categorical, we use a default palette
        unique_values = np.unique(layer.sel(features=feature))
        if palette is None and len(unique_values) <= 10:
            # default palette
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))  # Using tab10 colormap for 10 unique colors
            palette = {val: color for val, color in zip(unique_values, colors)}

        if palette:
            # check if all unique values are present in the palette
            assert set(np.unique(layer.sel(features=feature))) <= set(
                palette.keys()
            ), f"Not all values are present in the palette. Make sure the following keys are in your palette: {np.unique(layer.sel(features=feature))}."

            # Assigning colors based on feature values
            colors = [palette.get(val, "gray") for val in layer.sel(features=feature).values]

            ax.scatter(x.values, y.values, color=colors, s=size, alpha=alpha, zorder=zorder, **scatter_kws)

            if legend:
                # Creating legend labels based on unique feature values
                legend_handles = [
                    plt.Line2D([0], [0], marker="o", color="w", markersize=5, markerfacecolor=color, label=val)
                    for val, color in palette.items()
                ]
                ax.legend(handles=legend_handles, **legend_kwargs).set_zorder(102)
        else:
            # if there is no palette provided and none was inferred, we treat the data as continuous
            ax.scatter(
                x.values, y.values, c=layer.sel(features=feature), s=size, alpha=alpha, zorder=zorder, **scatter_kws
            )

            if legend:
                # adding colorbar
                cbar = plt.colorbar(ax.collections[0], ax=ax)
                cbar.set_label(feature)

        xmin, xmax, ymin, ymax = self._obj.pl._get_bounds()
        ax.set_ylim([ymin, ymax])
        ax.set_xlim([xmin, xmax])

        ax.set_aspect("equal")  # Set equal aspect ratio for x and y axes

        return self._obj

    def add_box(
        self,
        xlim: List[int],
        ylim: List[int],
        color: str = "w",
        linewidth: float = 2,
        ax=None,
    ):
        """
        Adds a box to the current plot.

        Parameters
        ----------
        xlim : List[int]
            The x-bounds of the box [xstart, xstop].
        ylim : List[int]
            The y-bounds of the box [ymin, ymax].
        color : str, optional
            The color of the box. Default is "w" (white).
        linewidth : float, optional
            The linewidth of the box. Default is 2.
        ax : matplotlib.axes, optional
            The matplotlib axis to plot on. If not provided, the current axis will be used.

        Returns
        -------
        xr.Dataset
            The updated spatialproteomics object.

        Notes
        -----
        - The function adds a rectangular box to the current plot with specified x and y bounds.
        - The box can be customized using the 'color' and 'linewidth' parameters.
        """

        if ax is None:
            ax = plt.gca()

        # unpack data
        xmin, xmax = xlim
        ymin, ymax = ylim

        ax.hlines(xmin=xmin, xmax=xmax, y=ymin, color=color, linewidth=linewidth)
        ax.hlines(xmin=xmin, xmax=xmax, y=ymax, color=color, linewidth=linewidth)
        ax.vlines(ymin=ymin, ymax=ymax, x=xmin, color=color, linewidth=linewidth)
        ax.vlines(ymin=ymin, ymax=ymax, x=xmax, color=color, linewidth=linewidth)

        return self._obj

    def autocrop(
        self, padding: int = 50, downsample: int = 10, key: str = Layers.IMAGE, channel: Optional[str] = None
    ) -> xr.Dataset:

        """
        Crop the image so that the background surrounding the tissue/TMA gets cropped away.

        Parameters
        ----------
        padding : int
            The padding to be added around the cropped image in pixels. Default is 50.
        downsample : int
            The downsampling factor for the image. Default is 10.
        key : str
            The key of the image to be cropped. Default is Layers.IMAGE.
        channel : str, optional
            The channel used for cropping. Default is None, which defaults to using the first available channel.

        Returns
        -------
        xr.Dataset
            The cropped image.
        """
        if channel is None:
            channel = self._obj.coords[Dims.CHANNELS].values.tolist()[0]
        img = self._obj.pp[channel].pp.downsample(downsample)[key].values.squeeze()
        bounds = self._obj.pl._get_bounds()
        slices = _autocrop(img, bounds=bounds, downsample=downsample, padding=padding)
        return self._obj.pp[slices[0], slices[1]]
