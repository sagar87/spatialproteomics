class Layers(object):
    IMAGE = "_image"
    SEGMENTATION = "_segmentation"
    COORDINATES = "_coordinates"
    DATA = "_data"


class Dims(object):
    IMAGE = ["channels", "x", "y"]
    SEGMENTATION = ["x", "y"]
    COORDINATES = ["cell_idx", "coords"]
    DATA = ["cell_idx", "channels"]


class Image(object):
    CHANNEL = "channel"
    X = "x"
    Y = "y"


class Segmentation(object):
    X = "x"
    Y = "y"


IMAGE = "_image"
SEGMENTATION = "_segmentation"
CHANNEL = "channel"
XDIM = "x"
YDIM = "y"
