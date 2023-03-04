class Layers(object):
    IMAGE = "_image"
    SEGMENTATION = "_segmentation"
    COORDINATES = "_coordinates"
    LABELS = "_labels"
    DATA = "_data"
    PLOT = "_plot"
    OBS = "_obs"
    NEIGHBORS = "_neighbors"
    INTENSITY = "_intensity"


class Dims(object):
    CHANNELS = "channels"
    X = "x"
    Y = "y"
    RGBA = "rgba"
    CELLS = "cells"
    COORDINATES = "coordinates"
    LABELS = "labels"
    FEATURES = "features"
    PROPS = "props"
    NEIGHBORS = "neighbors"

    IMAGE = ["channels", "x", "y"]
    COLORED_IMAGE = ["channels", "x", "y", "rgba"]
    SEGMENTATION = ["x", "y"]
    # COORDINATES = ["cell_idx", "coords"]
    DATA = ["cell_idx", "channels"]


class Attrs(object):
    IMAGE_COLORS = "image_colors"
    LABEL_COLORS = "label_colors"
    LABEL_NAMES = "label_names"


class Props(object):
    COLOR = "_color"
    NAME = "_name"


class Features(object):
    LABELS = "_labels"
    ORIGINAL_LABELS = "_original_labels"
    X = "_x"
    Y = "_y"


COLORS = [
    "#000000",
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#ffffff",
]
