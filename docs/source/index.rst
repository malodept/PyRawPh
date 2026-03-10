.. PyRawPh documentation master file, created by
   sphinx-quickstart on Mon Mar  9 19:30:35 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyRawPh's documentation!
===================================

PyRawPh is a Python package for loading, exploring, tiling, visualizing,
and exporting local ΦSat-2 L1 products.

It is intended for users who want a lightweight and practical interface for:

- loading local multiband or per-band ΦSat-2 GeoTIFF products,

- accessing spectral bands by index, wavelength, or alias,

- building quick RGB or normalized-difference products,

- splitting scenes into tiles,

- exporting processed outputs to GeoTIFF.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   concepts
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_core
   api_utils
