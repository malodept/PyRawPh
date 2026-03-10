Quickstart
==========

This short example shows the typical PyRawPh workflow:

1. load a local ΦSat-2 L1 product,
2. inspect the event,
3. access one band,
4. build an RGB image,
5. compute NDVI,
6. create tiles,
7. export a result to GeoTIFF.

Load a product
--------------

.. code-block:: python

   from pyrawph.l1.l1_event import L1_event

   ev = L1_event.from_path(
       product_folder="path/to/product_folder",
       scene_id=0,
       product_kind="BC",
   )

Inspect the event
-----------------

.. code-block:: python

   ev.show_event_info()

Access one band
---------------

Bands can be selected by:
- integer index, for example ``3``,
- wavelength in nanometers, for example ``842.0``,
- string selectors such as ``"B3"`` or ``"842nm"``,
- aliases such as ``"RED"`` or ``"NIR"``.

.. code-block:: python

   nir = ev.get_band("NIR")
   red = ev.get_band("RED")

Build an RGB image
------------------

.. code-block:: python

   rgb = ev.rgb()

Compute NDVI
------------

.. code-block:: python

   ndvi = ev.index("NDVI")

Create tiles
------------

.. code-block:: python

   tiles = ev.to_tiles(tile_size=512, overlap=0)
   print(ev.get_tiles_names())

Inspect one tile
----------------

.. code-block:: python

   t0 = ev.get_tile(0)
   t0.show_bands()

Export a result
---------------

.. code-block:: python

   ev.export_to_tif(
       out_path="outputs/ndvi.tif",
       arr=ndvi,
       meta=ev.get_meta(),
   )

Notes
-----

The local reader expects a ΦSat-2 product folder containing a ``bands/``
directory and may also use sidecar files such as:

- ``geolocation/GL_scene_<id>.json``
- ``processing_config.json``
- ``session_*_metadata.json``

When available, these files enrich the event metadata with geospatial
information and band wavelengths.