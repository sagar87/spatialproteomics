from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..base_logger import logger
from ..constants import Attrs, Dims, Features, Layers, Props
from .spectra import format_annotation_df, plot_expression_spectra


def _set_up_subplots(num_plots, ncols=4, width=4, height=3):
    """Set up subplots for plotting multiple factors."""

    if num_plots == 1:
        fig, ax = plt.subplots()
        return fig, ax

    nrows, reminder = divmod(num_plots, ncols)

    if num_plots < ncols:
        nrows = 1
        ncols = num_plots
    else:
        nrows, reminder = divmod(num_plots, ncols)

        if nrows == 0:
            nrows = 1
        if reminder > 0:
            nrows += 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows))
    _ = [ax.axis("off") for ax in axes.flatten()[num_plots:]]
    return fig, axes


@xr.register_dataset_accessor("pl")
class PlotAccessor:
    """Adds plotting functions to the image container."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _graph_overlay(self, ax=None):
        graph = self._obj.se.get_graph()
        x = self._obj[Layers.OBS].loc[:, Dims.X].values
        y = self._obj[Layers.OBS].loc[:, Dims.Y].values
        ax.triplot(
            x,
            y,
            graph.simplices,
            color="white",
        )

        # unique_labels = np.unique(ds[Layers.LABELS].values)

        # for label in unique_labels:
        #     label_bool = (ds[Layers.LABELS].values == label).squeeze()
        #     x = ds[Layers.OBS].loc[label_bool].loc[:, Dims.X].values
        #     y = ds[Layers.OBS].loc[label_bool].loc[:, Dims.Y].values
        #     ax.scatter(
        #         x, y, color=ds[Layers.LABELS].attrs[Attrs.LABEL_COLORS][label]
        #     )

    def _get_bounds(self):
        """Returns X/Y bounds of the dataset."""
        xmin = self._obj.coords[Dims.X].values[0]
        ymin = self._obj.coords[Dims.Y].values[0]
        xmax = self._obj.coords[Dims.X].values[-1]
        ymax = self._obj.coords[Dims.Y].values[-1]

        return [xmin, xmax, ymin, ymax]

    def _legend_background(self, **kwargs):
        """Returns legend handles for the background."""
        color_dict = self._obj[Layers.PLOT].attrs[Attrs.IMAGE_COLORS]

        elements = [Patch(facecolor=c, label=ch, **kwargs) for ch, c in color_dict.items()]
        return elements

    def _legend_labels(self):
        """Returns legend handles for the labels."""
        color_dict = self._obj.la._label_to_dict(Props.COLOR)
        names_dict = self._obj.la._label_to_dict(Props.NAME)

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
            for i in sorted(color_dict)
            if i in self._obj.coords[Dims.LABELS]
        ]

        return elements

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
                dims = table.dims
                if len(dims) != 2:
                    raise ValueError("Layer does not have the dimension.")
                if Dims.CELLS not in dims:
                    raise ValueError("Layer does not have a cell dimension.")

                dim = [d for d in dims if d != Dims.CELLS][0]
                # import pdb; pdb.set_trace()
                t = table.sel({Dims.CELLS: cell, dim: variable}).values
            else:
                t = cell.values

            # print(x,y, t)
            if cell in highlight:
                ax.text(x, y, s=f"{t:{format_string}}", **highlight_kwargs)
            else:

                ax.text(x, y, s=f"{t:{format_string}}", **text_kwargs)

        return self._obj

    def scatter(
        self,
        legend_label=False,
        size: float = 0.001,
        alpha: float = 0.9,
        zorder=10,
        ax=None,
        colorize: bool = True,
        legend_kwargs: dict = {"framealpha": 1},
        scatter_kws: dict = {},
    ) -> xr.Dataset:
        """
        Plots a scatter plot of labels.

        Parameters
        ----------
        legend_label : bool, optional
            Plots the legend of the labels. Default is False.
        size : float, optional
            Size of the dots in the scatter plot. Default is 0.001.
        alpha : float, optional
            Alpha value for transparency of the dots in the scatter plot. Default is 0.9.
        zorder : int, optional
            The z-order of the scatter plot. Default is 10.
        ax : matplotlib.axes, optional
            The matplotlib axis to plot on. If not provided, the current axis will be used.
        colorize : bool, optional
            Whether to colorize the dots based on label colors. Default is True.
        legend_kwargs : dict, optional
            Keyword arguments passed to the matplotlib legend function. Default is {"framealpha": 1}.
        scatter_kws : dict, optional
            Additional keyword arguments to be passed to the matplotlib scatter function.

        Returns
        -------
        xr.Dataset
            The updated image container.

        Notes
        -----
        - The function plots a scatter plot of labels from the data object.
        - The size, alpha, and zorder parameters control the appearance of the scatter plot.
        - You can colorize the dots based on label colors using the 'colorize' parameter.
        - The legend of the labels can be displayed using the 'legend_label' parameter.
        """
        if ax is None:
            ax = plt.gca()

        color_dict = self._obj.la._label_to_dict(Props.COLOR)
        # names_dict = self._obj.la._label_to_dict(Props.NAME)
        label_dict = self._obj.la._cells_to_label()

        for k, v in label_dict.items():
            label_subset = self._obj.la[k]
            obs_layer = label_subset[Layers.OBS]
            x = obs_layer.loc[:, Features.X]
            y = obs_layer.loc[:, Features.Y]
            if colorize:
                ax.scatter(x.values, y.values, s=size, c=color_dict[k], alpha=alpha, zorder=zorder, **scatter_kws)
            else:
                ax.scatter(x.values, y.values, s=size, alpha=alpha, zorder=zorder, **scatter_kws)

        xmin, xmax, ymin, ymax = self._obj.pl._get_bounds()
        ax.set_ylim([ymin, ymax])
        ax.set_xlim([xmin, xmax])
        # scatter_plot(sub, cell_types, axes[r, c], s=size)

        if legend_label:
            legend = self._obj.pl._legend_labels()
            ax.legend(handles=legend, **legend_kwargs)

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
            The updated image container.

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

    def bar(self, ax=None, bar_kwargs: dict = {}):
        """
        Plots a bar plot of present labels.

        Parameters
        ----------
        ax : matplotlib.axes, optional
            The matplotlib axis to plot on. If not provided, the current axis will be used.
        bar_kwargs : dict, optional
            Keyword arguments passed to the matplotlib bar function.

        Returns
        -------
        xr.Dataset
            The updated image container.

        Notes
        -----
        - The function plots a bar plot of present labels in the data object.
        - The appearance of the bar plot can be customized using 'bar_kwargs'.
        """
        if ax is None:
            ax = plt.gca()

        color_dict = self._obj.la._label_to_dict(Props.COLOR)
        names_dict = self._obj.la._label_to_dict(Props.NAME)

        obs_layer = self._obj[Layers.OBS]
        label_array = obs_layer.loc[:, Features.LABELS].values
        x, y = np.unique(label_array, return_counts=True)
        query = ~(x == 0)

        ax.bar(x[query], y[query], color=[color_dict[i] for i in x[query]], **bar_kwargs)
        ax.set_xticks(x[query])
        ax.set_xticklabels([names_dict[i] for i in x[query]], rotation=90)
        ax.set_ylabel("Label Frequency")
        ax.set_xlabel("Label")

        return self._obj

    def pie(
        self,
        wedgeprops={"linewidth": 7, "edgecolor": "white"},
        circle_radius=0.2,
        labels=True,
        ax=None,
    ):
        """
        Plots a pie chart of label frequencies.

        Parameters
        ----------
        wedgeprops : dict, optional
            Keyword arguments passed to the matplotlib pie function for wedge properties.
        circle_radius : float, optional
            The radius of the inner circle in the pie chart. Default is 0.2.
        labels : bool, optional
            Whether to display labels on the pie chart. Default is True.
        ax : matplotlib.axes, optional
            The matplotlib axis to plot on. If not provided, the current axis will be used.

        Returns
        -------
        None

        Notes
        -----
        - The function plots a pie chart of label frequencies in the data object.
        - The appearance of the pie chart can be customized using 'wedgeprops' and 'circle_radius'.
        - Labels on the pie chart can be shown or hidden using the 'labels' parameter.
        """
        if ax is None:
            ax = plt.gca()

        color_dict = self._obj.la._label_to_dict(Props.COLOR)
        names_dict = self._obj.la._label_to_dict(Props.NAME)

        obs_layer = self._obj[Layers.OBS]
        label_array = obs_layer.loc[:, Features.LABELS].values
        x, y = np.unique(label_array, return_counts=True)

        ax.pie(
            y,
            labels=[names_dict[i] for i in x] if labels else None,
            colors=[color_dict[i] for i in x],
            wedgeprops=wedgeprops,
        )
        my_circle = plt.Circle((0, 0), circle_radius, color="white")
        ax.add_artist(my_circle)

    def imshow(
        self,
        legend_background: bool = False,
        legend_label: bool = False,
        legend_kwargs: dict = {"framealpha": 1},
        downsample: int = 1,
        ax=None,
    ):
        """
        Plots the image.

        Meant to be used in conjunction with im.colorize and la.render_label.
        See examples.

        Parameters
        ----------
        legend_background : bool, optional
            Show the label of the colorized image. Default is False.
        legend_label : bool, optional
            Show the labels. Default is False.
        legend_kwargs : dict, optional
            Keyword arguments passed to the matplotlib legend function. Default is {"framealpha": 1}.
        downsample : int, optional
            Downsample factor for the image. Default is 1 (no downsampling).
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
            logger.warning("No plot defined yet.")
            channel = str(self._obj.coords[Dims.CHANNELS].values[0])
            self._obj = self._obj.im[channel].im.colorize(colors=["white"])

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

        if legend_background:
            legend += self._obj.pl._legend_background()

        if legend_label:
            legend += self._obj.pl._legend_labels()

        if legend_background or legend_label:
            ax.legend(handles=legend, **legend_kwargs)

        return self._obj

    def spectra(self, cells: Union[List[int], int], layers_key="intensity", ncols=4, width=4, height=3, ax=None):
        """
        Plots the spectra of cells.

        Parameters
        ----------
        cells : Union[List[int], int]
            The cell ID(s) whose spectra will be plotted.
        layers_key : str, optional
            The key representing the layer in the data object for plotting spectra. Default is "intensity".
        ax : matplotlib.axes, optional
            The matplotlib axis to plot on. If not provided, a new figure and axis will be created.

        Returns
        -------
        xr.Dataset
            The selected data array for the plotted cells.

        Notes
        -----
        - The function plots the spectra of the specified cell(s) using the 'layers_key' from the data object.
        """
        if type(cells) is int:
            cells = [cells]

        da = self._obj[layers_key].sel({"cells": cells})
        num_cells = len(cells)

        fig, axes = _set_up_subplots(num_cells, ncols=ncols, width=width, height=height)

        # fig, axes = plt.subplots(1, num_cells, figsize=(4 * num_cells, 3))

        if num_cells > 1:
            for i, ax in zip(range(da.values.shape[0]), axes.flatten()):
                ax.bar(np.arange(da.values.shape[1]), da.values[i])
                ax.set_xticks(np.arange(da.values.shape[1]))
                ax.set_xticklabels(da.channels.values, rotation=90)
                ax.set_title(f"Cell {da.cells.values[i]}")
        else:
            axes.bar(np.arange(da.values.squeeze().shape[0]), da.values.squeeze())
            axes.set_xticks(np.arange(da.values.squeeze().shape[0]))
            axes.set_xticklabels(da.channels.values, rotation=90)
            axes.set_title(f"Cell {da.cells.values[0]}")

        # if ax is isinstance(ax, np.ndarray):
        # assert np.prod(ax.shape) >= num_cells, "Must provide at least one axis for each cell to plot."

        return da

    def spectra_with_annotation(
        self,
        cells: Union[List[int], None] = None,
        layers_key="intensity",
        format_df=None,
        plot_kwargs: dict = {
            "width": 12,
            "height": 2,
            "hspace": 1.0,
            "wspace": 0.0001,
            "xticks": True,
        },
    ):
        """
        Plots the spectra of cells with annotation.

        Parameters
        ----------
        cells : Union[List[int], None], optional
            The cell ID(s) whose spectra will be plotted. If None, all cells will be plotted.
        layers_key : str, optional
            The key representing the layer in the data object for plotting spectra. Default is "intensity".
        format_df : pd.DataFrame or None, optional
            A DataFrame containing annotation information for the plotted cells. Default is None (no annotation).
        plot_kwargs : dict, optional
            Additional keyword arguments for setting up subplots and plot appearance.

        Returns
        -------
        xr.Dataset
            The image container.

        Notes
        -----
        - The function plots the spectra of the specified cell(s) using the 'layers_key' from the data object.
        - Annotates the spectra using the provided 'format_df' DataFrame.
        """

        if cells is None:
            cells = self._obj.coords[Dims.CELLS].values.tolist()

        # da = self._obj.se.quantify_cells(cells)
        da = self._obj[layers_key].sel({"cells": cells})
        annot = format_annotation_df(format_df, da)

        plot_expression_spectra(
            da.values,
            annot,
            titles=[f"{i}" for i in da.coords[Dims.CELLS]],
            **plot_kwargs,
        )

        return self._obj

    def draw_edges(self, color="white", linewidths=0.5, zorder=0, ax=None):
        """
        Draws edges connecting neighboring cells.

        Parameters
        ----------
        color : str, optional
            The color of the edges. Default is "white".
        linewidths : float, optional
            The linewidth of the edges. Default is 0.5.
        zorder : int, optional
            The z-order of the edges in the plot. Default is 0.
        ax : matplotlib.axes, optional
            The matplotlib axis to plot on. If not provided, the current axis will be used.

        Returns
        -------
        xr.Dataset
            The updated image container.

        Notes
        -----
        - The function draws edges connecting neighboring cells in the plot.
        - The appearance of the edges can be customized using 'color' and 'linewidths'.
        """
        coords = self._obj[Layers.OBS].loc[:, [Features.X, Features.Y]]
        neighbors = self._obj[Layers.NEIGHBORS].values.reshape(-1)
        cell_dim = self._obj.dims[Dims.CELLS]
        neighbor_dim = self._obj.dims[Dims.NEIGHBORS]

        # set up edgelist
        origin = coords.values
        target = coords.sel({Dims.CELLS: neighbors}).values.reshape(cell_dim, neighbor_dim, 2)

        # line segments
        all_lines = []
        for k in range(target.shape[1]):
            lines = [[i, j] for i, j in zip(map(tuple, origin), map(tuple, target[:, k]))]
            all_lines.extend(lines)

        # Line collection
        # REFACTOR
        lc = LineCollection(all_lines, colors=color, linewidths=linewidths, zorder=zorder)
        if ax is None:
            ax = plt.gca()

        ax.add_collection(lc)
        xmin, xmax, ymin, ymax = self._obj.pl._get_bounds()
        ax.set_ylim([ymin, ymax])
        ax.set_xlim([xmin, xmax])

        return self._obj

    def channel_histogram(
        self,
        intensity_key: str,
        bins: int = 50,
        ncols: int = 4,
        width: float = 4,
        height: float = 3,
        log_scale: bool = False,
        ax=None,
        **kwargs,
    ):
        """
        Plots histograms of intensity values for each channel.

        Parameters
        ----------
        intensity_key : str
            The key representing the intensity values in the data object.
        bins : int, optional
            The number of bins for histogram bins. Default is 50.
        ncols : int, optional
            The number of columns for subplot arrangement. Default is 4.
        width : float, optional
            The width of the figure. Default is 4.
        height : float, optional
            The height of the figure. Default is 3.
        log_scale : bool, optional
            Whether to use a logarithmic scale for the y-axis. Default is False.
        ax : matplotlib.axes, optional
            The matplotlib axis to plot on. If not provided, subplots will be created.
        **kwargs : dict, optional
            Additional keyword arguments passed to the matplotlib hist function.

        Returns
        -------
        xr.Dataset
            The image container.

        Notes
        -----
        - The function plots histograms of intensity values for each channel using the 'intensity_key' from the data object.
        - The histograms are arranged in subplots with 'ncols' columns.
        - Additional keyword arguments can be passed to customize the appearance of the histograms.
        """
        intensities = self._obj[intensity_key]
        channels = self._obj.coords[Dims.CHANNELS].values
        num_channels = len(channels)

        # if num_channels > 1 and ax is not None:
        #     logger.warning("More than one channel. Plotting on first axis.")
        #     # assert np.prod(ax.shape) >= num_channels, "Must provide at least one axis for each channel to plot."
        # else:
        #     if ax is None:
        #         ax = plt.gca()
        fig, axes = _set_up_subplots(num_channels, ncols=ncols, width=width, height=height)

        if num_channels > 1:

            for ch, ax in zip(channels, axes.flatten()):
                data = intensities.sel({Dims.CHANNELS: ch}).values
                ax.hist(data, bins=bins, **kwargs)
                ax.set_title(ch)
                if log_scale:
                    ax.set_yscale("log")
        else:
            ch = channels[0]
            data = intensities.sel({Dims.CHANNELS: ch}).values
            axes.hist(data, bins=bins, **kwargs)
            axes.set_title(ch)
            if log_scale:
                axes.set_yscale("log")

        return self._obj
