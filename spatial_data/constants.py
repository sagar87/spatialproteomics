class Layers(object):
    IMAGE = "_image"
    SEGMENTATION = "_segmentation"
    COORDINATES = "_coordinates"
    DATA = "_data"


class Dims(object):
    IMAGE = ["channels", "x", "y"]
    COLORED_IMAGE = ["channels", "x", "y", "rgba"]
    SEGMENTATION = ["x", "y"]
    COORDINATES = ["cell_idx", "coords"]
    DATA = ["cell_idx", "channels"]


class Image(object):
    CHANNEL = "channel"
    X = "x"
    Y = "y"


class ColoredImage(object):
    CHANNEL = "channel"
    X = "x"
    Y = "y"
    RGBA = "rgba"


class Segmentation(object):
    X = "x"
    Y = "y"


IMAGE = "_image"
SEGMENTATION = "_segmentation"
CHANNEL = "channel"
XDIM = "x"
YDIM = "y"


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
