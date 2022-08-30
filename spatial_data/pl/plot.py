import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D

from ..base_logger import logger
from ..constants import Attrs, Dims, Features, Layers, Props


@xr.register_dataset_accessor("pl")
class PlotAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _get_bounds(self):
        xmin = self._obj.coords[Dims.X].values[0]
        ymin = self._obj.coords[Dims.Y].values[0]
        xmax = self._obj.coords[Dims.X].values[-1]
        ymax = self._obj.coords[Dims.Y].values[-1]

        return [xmin, xmax, ymin, ymax]

    def annotate(
        self,
        ax=None,
        highlight: list = [],
        text_kwargs: dict = {"color": "w", "fontsize": 12},
        highlight_kwargs: dict = {"color": "w", "fontsize": 16, "fontweight": "bold"},
    ):
        if ax is None:
            ax = plt.gca()
        for cell in self._obj.coords[Dims.CELLS]:
            x, y = self._obj[Layers.OBS].loc[cell, [Features.X, Features.Y]].values
            if cell in highlight:
                ax.text(x, y, s=f"{cell}", **highlight_kwargs)
            else:
                ax.text(x, y, s=f"{cell}", **text_kwargs)

        return self._obj

    def scatter(
        self,
        legend_label=False,
        size: float = 0.001,
        alpha: float = 0.9,
        ax=None,
        legend_kwargs: dict = {"framealpha": 1},
    ) -> xr.Dataset:
        """Plots a scatter plot of labels

        Parameters
        ----------
        legend_label: bool
            Plots the legend of the labels.
        size: float
            Size of the dots.
        ax: matplotlib.axes
            Matplotlib axis to plot on (default: None)


        Returns
        -------
        xr.Dataset
            The image container.
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
            ax.scatter(x.values, y.values, s=size, c=color_dict[k], alpha=alpha)

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
        xlim=[2800, 3200],
        ylim=[1500, 2000],
        color: str = "w",
        linewidth: float = 2,
        ax=None,
    ):
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

    def bar(self, ax=None):
        """Plots a bar plot present labels."""
        if ax is None:
            ax = plt.gca()

        color_dict = self._obj.la._label_to_dict(Props.COLOR)
        names_dict = self._obj.la._label_to_dict(Props.NAME)

        obs_layer = self._obj[Layers.OBS]
        label_array = obs_layer.loc[:, Features.LABELS].values
        x, y = np.unique(label_array, return_counts=True)

        ax.bar(x, y, color=[color_dict[i] for i in x])
        ax.set_xticks(x)
        ax.set_xticklabels([names_dict[i] for i in x], rotation=90)
        ax.set_ylabel("Label Frequency")
        ax.set_xlabel("Label")

        return self._obj

    def imshow(
        self,
        legend_background: bool = False,
        legend_label: bool = False,
        legend_kwargs: dict = {"framealpha": 1},
        ax=None,
    ):
        if Layers.PLOT not in self._obj:
            logger.warning("No plot defined yet.")
            channel = str(self._obj.coords[Dims.CHANNELS].values[0])
            self._obj = self._obj.im[channel].im.colorize(colors=["white"])

        if ax is None:
            ax = plt.gca()

        bounds = self._obj.pl._get_bounds()

        ax.imshow(
            self._obj[Layers.PLOT].values,
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

    def _legend_background(self):
        color_dict = self._obj[Layers.PLOT].attrs[Attrs.IMAGE_COLORS]

        elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=ch,
                markerfacecolor=c,
                markersize=15,
            )
            for ch, c in color_dict.items()
        ]
        return elements

    def _legend_labels(self):

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

    def plot(
        self,
        legend_image=False,
        legend_labels=False,
        ax=None,
        legend_fontsize=None,
        legend_alpha=1.0,
        annotate=False,
        highlight=[],
        show_graph=False,
    ):
        if Layers.PLOT not in self._obj:
            logger.warning("No plot defined yet.")
            channel = str(self._obj.coords[Dims.CHANNELS].values[0])
            ds = self._obj.im[channel].im.colorize(colors=["white"])
        else:
            ds = self._obj

        if ax is None:
            ax = plt.gca()

        ax.imshow(ds[Layers.PLOT].values, origin="lower")

        if show_graph:
            graph = ds.se.get_graph()
            x = ds[Layers.OBS].loc[:, Dims.X].values
            y = ds[Layers.OBS].loc[:, Dims.Y].values
            ax.triplot(
                x,
                y,
                graph.simplices,
                color="white",
            )

            unique_labels = np.unique(ds[Layers.LABELS].values)

            for label in unique_labels:
                label_bool = (ds[Layers.LABELS].values == label).squeeze()
                x = ds[Layers.OBS].loc[label_bool].loc[:, Dims.X].values
                y = ds[Layers.OBS].loc[label_bool].loc[:, Dims.Y].values
                ax.scatter(
                    x, y, color=ds[Layers.LABELS].attrs[Attrs.LABEL_COLORS][label]
                )
