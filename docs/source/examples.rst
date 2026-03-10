Examples
========

Load and inspect a scene
------------------------

.. code-block:: python

   from pyrawph.l1.l1_event import L1_event

   ev = L1_event.from_path(
       product_folder="path/to/product_folder",
       scene_id=0,
       product_kind="BC",
   )

   ev.show_event_info()

Access spectral bands
---------------------

.. code-block:: python

   blue = ev.get_band("BLUE")
   red = ev.get_band("RED")
   nir = ev.get_band("NIR")

   print(blue.shape, red.shape, nir.shape)

Build RGB and NDVI
------------------

.. code-block:: python

   rgb = ev.rgb()
   ndvi = ev.index("NDVI")

Tile a scene
------------

.. code-block:: python

   tiles = ev.to_tiles(tile_size=512, overlap=64)

   print(ev.get_tiles_names())
   print(ev.get_tiles_info())

   tile = ev.get_tile(0)
   tile.show_bands(["RED", "NIR"])

Plot scene location
-------------------

.. code-block:: python

   ev.plot_location(mode="bounds")
   ev.plot_location(mode="footprint")

Export a derived product
------------------------

.. code-block:: python

   ndvi = ev.index("NDVI")

   ev.export_to_tif(
       out_path="outputs/ndvi.tif",
       arr=ndvi,
       meta=ev.get_meta(),
   )

Read only selected bands
------------------------

.. code-block:: python

   ev = L1_event.from_path(
       product_folder="path/to/product_folder",
       scene_id=0,
       product_kind="BC",
       bands=[0, 1, 2],
   )

   ev.show_event_info()