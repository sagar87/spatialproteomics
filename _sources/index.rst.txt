.. Spatialproteomics documentation master file, created by
   sphinx-quickstart on Tue Aug 16 14:14:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Spatialproteomics - a unified framework for analyzing highly multiplexed fluorescence images
============================================================================================

Spatialproteomics is a lightweight python package that provides a unified framework which facilitates the parallel (pre-)processing and analysis of high-dimensional imaging datasets. 
The underlying data structure ensures synchronization of shared coordinates across data modalities, which enables users to run analyses on subsets of their data. 
Built on top of xarray and dask, spatialproteomics can process high-dimensional larger-than-memory imaging datasets.

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
   notebooks/Slicing
   notebooks/Segmentation
   notebooks/ArtifactRemoval
   notebooks/Plotting
   notebooks/ImageProcessing
   notebooks/CellTypePrediction
   notebooks/Neighborhoods
   notebooks/Interoperability
   notebooks/Interactivity
   notebooks/FAQ


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
