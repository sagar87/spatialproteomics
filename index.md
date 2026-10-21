# Spatialproteomics: an interoperable toolbox for analyzing highly multiplexed fluorescence images

Spatialproteomics is an interoperable toolbox for analyzing high-dimensional imaging datasets. This analysis involves a sequence of steps, including segmentation, image processing, marker quantification, cell type classification, and neighborhood analysis. Built on top of xarray and dask, spatialproteomics can process images that are larger than the working memory. It supports synchronization of shared coordinates across data modalities such as images, segmentation masks, and expression matrices, which facilitates easy and safe subsetting and transformation.

<img src="_static/img/figure_1.png" class="align-center" style="width:100.0%" alt="Spatialproteomics orchestrates analysis workflows for high-dimensional imaging datasets." />
