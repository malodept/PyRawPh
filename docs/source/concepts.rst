Concepts
========

L1_event
--------

``L1_event`` is the main high-level object in PyRawPh.

It represents a full local ΦSat-2 L1 scene and provides methods for:

- loading a product from disk,
- accessing spectral bands,
- computing quick spectral products,
- cropping the scene in pixel coordinates,
- splitting the scene into tiles,
- plotting the scene location,
- exporting arrays to GeoTIFF.

Typical entry point:

.. code-block:: python

   from pyrawph.l1.l1_event import L1_event

   ev = L1_event.from_path("path/to/product_folder")

L1_tile
-------

``L1_tile`` represents a local tile extracted from an event.

A tile stores:
- a local multiband array,
- its associated metadata,
- helper methods for visualization and geographic inspection.

Tiles are usually created from an event with:

.. code-block:: python

   ev.to_tiles(tile_size=512, overlap=0)
   tile = ev.get_tile(0)

meta
----

Both ``L1_event`` and ``L1_tile`` use a metadata dictionary named ``meta``.

Depending on what is available on disk, it may contain fields such as:

- ``path``
- ``scene_id``
- ``product_kind``
- ``count``
- ``width``
- ``height``
- ``dtype``
- ``crs``
- ``transform``
- ``bounds``
- ``band_wavelength_nm``
- ``gl_path``
- ``processing_config_path``
- ``session_metadata_path``

This metadata is used by:
- band resolution,
- crop and tile georeferencing,
- plotting,
- GeoTIFF export.

Band selection
--------------

Band selection is intentionally flexible.

Supported selectors include:
- integer indices, for example ``0`` or ``3``,
- wavelengths in nanometers, for example ``842.0``,
- string selectors such as ``"3"``, ``"B3"``, ``"BAND_3"``, or ``"842nm"``,
- aliases such as ``"BLUE"``, ``"GREEN"``, ``"RED"``, ``"RE1"``, ``"RE2"``,
  ``"RE3"``, and ``"NIR"``.

Examples:

.. code-block:: python

   ev.get_band(3)
   ev.get_band(842.0)
   ev.get_band("B3")
   ev.get_band("842nm")
   ev.get_band("NIR")

For wavelength-based and alias-based resolution, PyRawPh uses the band
wavelengths stored in metadata and selects the closest available band.