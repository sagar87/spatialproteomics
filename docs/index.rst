.. Spatialproteomics documentation master file, created by
   sphinx-quickstart on Tue Aug 16 14:14:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Spatialproteomics - an interoperable toolbox for analyzing highly multiplexed fluorescence images
============================================================================================

Spatialproteomics is an interoperable toolbox for analyzing high-dimensional imaging datasets. 
This analysis involves a sequence of steps, including segmentation, image processing, marker quantification, cell type classification, and neighborhood analysis.
Built on top of xarray and dask, spatialproteomics can process images that are larger than the working memory. 
It supports synchronization of shared coordinates across data modalities such as images, segmentation masks, and expression matrices, which facilitates easy and safe subsetting and transformation.

.. image:: _static/img/figure_1.png
   :width: 100%
   :align: center
   :alt: Spatialproteomics orchestrates analysis workflows for high-dimensional imaging datasets.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   preprocessing
   label
   neighborhood
   plot
   tool
   image_container

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:
   
   notebooks/Installation
   notebooks/ExampleWorkflow
   notebooks/ExampleWorkflowSpatialdata
   notebooks/Plotting
   notebooks/Segmentation
   notebooks/ImageProcessing
   notebooks/CellTypePrediction
   notebooks/Neighborhoods
   notebooks/Interoperability
   notebooks/DownstreamAnalysis
   notebooks/Interactivity
   notebooks/Customizability
   notebooks/Slicing
   notebooks/ArtifactRemoval
   notebooks/FAQ


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
