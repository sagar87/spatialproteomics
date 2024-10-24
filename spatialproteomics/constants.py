Red = "#e6194B"
Green = "#3cb44b"
Yellow = "#ffe119"
Blue = "#4363d8"
Orange = "#f58231"
Purple = "#911eb4"
Cyan = "#42d4f4"
Magenta = "#f032e6"
Lime = "#bfef45"
Pink = "#fabed4"
Teal = "#469990"
Lavender = "#dcbeff"
Brown = "#9A6324"
Beige = "#fffac8"
Maroon = "#800000"
Mint = "#aaffc3"
Olive = "#808000"
Apricot = "#ffd8b1"
Navy = "#000075"
Grey = "#a9a9a9"
White = "#ffffff"
Black = "#000000"


class Layers(object):
    IMAGE = "_image"
    SEGMENTATION = "_segmentation"
    LA_PROPERTIES = "_la_properties"
    NH_PROPERTIES = "_nh_properties"
    OBS = "_obs"
    LA_LAYERS = "_la_layers"
    INTENSITY = "_intensity"
    PLOT = "_plot"
    MASK = "_mask"
    NEIGHBORHOODS = "_neighborhoods"
    ADJACENCY_MATRIX = "_adjacency_matrix"
    DISTANCE_MATRIX = "_distance_matrix"


class Dims(object):
    CHANNELS = "channels"
    X = "x"
    Y = "y"
    RGBA = "rgba"
    CELLS = "cells"
    # unfortunately, xarray does not support same-name coordinates
    # this means that we have to use different names for the same coordinates when storing the adjacency matrix
    CELLS_2 = "cells_2"
    COORDINATES = "coordinates"
    LABELS = "labels"
    FEATURES = "features"
    LA_FEATURES = "la_features"
    LA_PROPS = "la_props"
    NH_PROPS = "nh_props"
    NEIGHBORHOODS = "neighborhoods"
    IMAGE = ["channels", "x", "y"]
    COLORED_IMAGE = ["channels", "x", "y", "rgba"]
    SEGMENTATION = ["x", "y"]


class Attrs(object):
    IMAGE_COLORS = "image_colors"
    SEGMENTATION_COLORS = "segmentation_colors"
    LABEL_COLORS = "label_colors"
    OBS_COLORS = "obs_colors"
    LABEL_NAMES = "label_names"


class Props(object):
    COLOR = "_color"
    NAME = "_name"


class Features(object):
    X = "centroid-1"
    Y = "centroid-0"
    LABELS = "_labels"
    NEIGHBORHOODS = "_neighborhoods"


class Labels(object):
    UNLABELED = "Unlabeled"


COLORS = [
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#6A3A4C",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
]
