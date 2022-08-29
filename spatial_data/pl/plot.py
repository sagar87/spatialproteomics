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

        xmin = self._obj.coords[Dims.X].values[0]
        ymin = self._obj.coords[Dims.Y].values[0]
        xmax = self._obj.coords[Dims.X].values[-1]
        ymax = self._obj.coords[Dims.Y].values[-1]

        print(xmin, xmax, ymin, ymax)

        ax.imshow(
            self._obj[Layers.PLOT].values,
            origin="lower",
            interpolation="none",
            extent=[xmin, xmax, ymin, ymax],
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
