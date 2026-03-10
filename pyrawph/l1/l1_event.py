# pyrawph/l1/l1_event.py
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
except Exception:  
    torch = None 

from pyrawph.utils.optional_plots import plot_bounds, plot_gl_footprint
import rasterio
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform

try:
    from termcolor import colored
except Exception:  
    def colored(x, *_args, **_kwargs): 
        return x

from .l1_tile import L1_tile
from ..utils.l1_utils import read_L1_event_from_folder_phisat2
from ..utils.geo_utils import normalize_bounds
from ..utils.processing_utils import make_rgb, normalized_difference
from ..utils.export_utils import export_to_tif as _export_to_tif

BandSpec = Union[int, str, float]


def _try_parse_product_times(product_folder: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to parse two timestamps from the product folder name:
      ..._<YYYYMMDDhhmmss>_<YYYYMMDDhhmmss>_...
    Returns ISO-like strings (or None).
    """
    base = os.path.basename(product_folder.rstrip("\\/"))
    m = re.search(r"_(\d{14})_(\d{14})_", base)
    if not m:
        return None, None

    t0, t1 = m.group(1), m.group(2)

    def _fmt(s: str) -> str:
        dt = datetime.strptime(s, "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    try:
        return _fmt(t0), _fmt(t1)
    except Exception:
        return None, None


class L1_event:
    """
    Represent a local ΦSat-2 L1 scene together with its array data, metadata,
    and derived in-memory tiles.

    This class is the main high-level entry point for loading a ΦSat-2 product,
    accessing spectral bands, computing simple spectral products, cropping the
    scene in pixel coordinates, splitting it into tiles, plotting its geographic
    location, and exporting arrays to GeoTIFF.

    An event stores the full scene as a NumPy array with shape `(C, H, W)`, where
    `C` is the number of bands, and keeps a metadata dictionary containing
    geospatial information such as CRS, affine transform, bounds, wavelengths,
    and product-related paths.

    The recommended constructor is :meth:`from_path`, which reads a local product
    folder and initializes the event from disk.
    """

    # alias wavelengths (nm) -> resolve by closest wavelength
    _ALIAS_NM: Dict[str, int] = {
        "BLUE": 490, "B": 490,
        "GREEN": 560, "G": 560,
        "RED": 665, "R": 665,
        "REDEDGE1": 705, "RE1": 705,
        "REDEDGE2": 740, "RE2": 740,
        "REDEDGE3": 783, "RE3": 783,
        "NIR": 842,
    }

    def __init__(
        self,
        arr: np.ndarray,
        meta: Dict[str, Any],
        product_folder: str,
        scene_id: int,
        product_kind: str,
        device: str = "cpu",
    ):
        self._arr = arr
        self._meta = meta
        self._product_folder = product_folder
        self._scene_id = int(scene_id)
        self._product_kind = product_kind.upper()
        self._device = device

        # ensure times exist 
        st, ct = _try_parse_product_times(product_folder)
        if "sensing_time" not in self._meta:
            self._meta["sensing_time"] = st
        if "creation_time" not in self._meta:
            self._meta["creation_time"] = ct

        # default "tile" is the whole scene
        self._tiles: List[L1_tile] = [
            L1_tile(
                tile_name=f"scene_{self._scene_id}_{self._product_kind}",
                arr=self._arr,
                meta=self._meta,
                device=self._device,
            )
        ]
        self.n_tiles = 1

    # constructors
    @classmethod
    def from_path(
        cls,
        product_folder: str,
        scene_id: int = 0,
        product_kind: str = "BC",
        multiband: bool = True,
        bands: Optional[List[int]] = None,
        as_float32: bool = True,
        verbose: bool = True,
        device: str = "cpu",
    ) -> "L1_event":
        
        """
        Create an :class:`L1_event` from a local ΦSat-2 product folder.

        This class method loads one scene from disk using the ΦSat-2 L1 reader,
        builds the underlying `(C, H, W)` array, and attaches the corresponding
        metadata dictionary.

        Args:
            product_folder: Path to the local ΦSat-2 product folder.
            scene_id: Scene index to load from the product.
            product_kind: Product variant to load, for example `"BC"`.
            multiband: Whether to read the product as a multiband array.
            bands: Optional subset of band indices to read. If `None`, all available
                bands are loaded.
            as_float32: Whether to cast the loaded array to `float32`.
            verbose: If `True`, print a short loading message.
            device: Target device used when converting the event to a torch tensor.

        Returns:
            A new :class:`L1_event` instance initialized from the requested product.

        Raises:
            FileNotFoundError: If the product folder or required files are missing.
            ValueError: If the requested scene or product configuration is invalid.
        """

        if verbose:
            print("[PyRawPh] Loading ΦSat-2 L1 from:", product_folder)

        arr, meta = read_L1_event_from_folder_phisat2(
            product_folder=product_folder,
            scene_id=scene_id,
            product_kind=product_kind,
            multiband=multiband,
            bands=bands,
            as_float32=as_float32,
        )
        return cls(
            arr=arr,
            meta=meta,
            product_folder=product_folder,
            scene_id=scene_id,
            product_kind=product_kind,
            device=device,
        )

    # basic getters
    def as_numpy(self) -> np.ndarray:
        """
        Return the event data as a NumPy array.

        The returned array is the internal scene array stored by the event and is
        expected to have shape `(C, H, W)`.

        Returns:
            The event data as a NumPy array of shape `(C, H, W)`.
        """
        return self._arr

    def as_tensor(self, as_float32: bool = True):
        """
        Return the event data as a PyTorch tensor on the configured device.

        Args:
            as_float32: If `True`, cast the tensor to `torch.float32` before moving
                it to the target device.

        Returns:
            A PyTorch tensor containing the event data, typically with shape
            `(C, H, W)`.

        Raises:
            ImportError: If PyTorch is not available in the current environment.
        """

        if torch is None:
            raise ImportError("torch is not available")
        t = torch.from_numpy(self._arr)
        if as_float32 and t.dtype != torch.float32:
            t = t.float()
        return t.to(self._device)

    def get_meta(self) -> Dict[str, Any]:
        """
        Return the metadata dictionary associated with the event.

        The metadata may contain fields such as CRS, affine transform, bounds,
        wavelengths, product paths, sensing time, and creation time.

        Returns:
            The event metadata dictionary.
        """
        return self._meta

    def get_wavelengths(self) -> List[Optional[int]]:
        """
        Return the list of band center wavelengths in nanometers.

        The values are read from the `"band_wavelength_nm"` entry of the metadata.
        If no wavelength information is available, an empty list is returned.

        Returns:
            A list of wavelengths in nanometers, or an empty list if unavailable.
        """
        w = self._meta.get("band_wavelength_nm", None)
        return list(w) if isinstance(w, (list, tuple)) else []

    # band resolving (closest wavelength)
    def _resolve_band(self, band: BandSpec) -> int:
        """
        Resolve a band specification to a zero-based band index.

        The resolver accepts several selector formats:
        - integer band indices,
        - float wavelengths in nanometers,
        - strings such as `"842nm"`, `"3"`, `"B3"`, or `"BAND_3"`,
        - spectral aliases such as `"BLUE"`, `"GREEN"`, `"RED"`, `"RE1"`,
        `"RE2"`, `"RE3"`, or `"NIR"`.

        When a wavelength or alias is provided, the closest available wavelength in
        the event metadata is used.

        Args:
            band: Band selector to resolve.

        Returns:
            The resolved zero-based band index.

        Raises:
            ValueError: If the band specification is invalid, out of range, or cannot
                be resolved from the available wavelength metadata.
        """
        
        C = int(self._arr.shape[0])

        if isinstance(band, int):
            if not (0 <= band < C):
                raise ValueError(f"Band index out of range: {band} (C={C})")
            return band

        if isinstance(band, float):
            wl = int(round(band))
            wls = self.get_wavelengths()
            if not wls:
                raise ValueError("No wavelengths in metadata; cannot resolve float wavelength.")

            valid = [(i, v) for i, v in enumerate(wls) if v is not None]
            if not valid:
                raise ValueError("No valid wavelengths in metadata; cannot resolve float wavelength.")

            idx = min(valid, key=lambda iv: abs(int(iv[1]) - wl))[0]
            return int(idx)

        s = str(band).strip().upper()
        s_clean = s.replace(" ", "").replace("_", "")
        if s_clean.endswith("NM") and s_clean[:-2].isdigit():
            return self._resolve_band(float(int(s_clean[:-2])))

        if s.isdigit():
            return self._resolve_band(int(s))

        for prefix in ("BAND_", "BAND", "B"):
            if s.startswith(prefix) and s[len(prefix):].isdigit():
                return self._resolve_band(int(s[len(prefix):]))

        if s in self._ALIAS_NM:
            target_nm = int(self._ALIAS_NM[s])
            wls = self.get_wavelengths()
            if not wls:
                raise ValueError("No wavelengths in metadata; cannot resolve band name.")

            valid = [(i, v) for i, v in enumerate(wls) if v is not None]
            if not valid:
                raise ValueError("No valid wavelengths in metadata; cannot resolve band name.")

            idx = min(valid, key=lambda iv: abs(int(iv[1]) - target_nm))[0]
            return int(idx)

        raise ValueError(f"Cannot resolve band spec: {band!r}")

    def get_band(self, band: BandSpec) -> np.ndarray:
        """
        Return one band from the event as a 2D array.

        The band can be selected by:
        - integer band index,
        - float wavelength in nanometers,
        - string specification such as `"NIR"`, `"RED"`, `"B3"`, `"BAND_7"`,
            or `"842nm"`.

        String aliases are resolved through the event wavelength metadata using the
        closest matching wavelength when needed.

        Args:
            band: Band selector.

        Returns:
            A 2D NumPy array of shape `(H, W)` corresponding to the selected band.

        Raises:
            ValueError: If the band specification cannot be resolved.
        """
        i = self._resolve_band(band)
        return self._arr[i]

    # processing
    def rgb(
        self,
        bands=("RED", "GREEN", "BLUE"),
        stretch=(2, 98),
        arr: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build an RGB composite from three selected bands.

        By default, the composite uses the `"RED"`, `"GREEN"`, and `"BLUE"` aliases.
        An optional source array can be provided instead of the event internal array,
        but it must follow the same band ordering and use shape `(C, H, W)`.

        Args:
            bands: A length-3 sequence describing the red, green, and blue channels.
                Each entry can use any valid band selector supported by
                :meth:`get_band`.
            stretch: Percentile stretch applied before composing the RGB image.
            arr: Optional source array with shape `(C, H, W)`. If `None`, the event
                internal array is used.

        Returns:
            An RGB image as a NumPy array, typically with shape `(H, W, 3)`.
        """
        src = self._arr if arr is None else arr

        r = src[self._resolve_band(bands[0])].astype(np.float32)
        g = src[self._resolve_band(bands[1])].astype(np.float32)
        b = src[self._resolve_band(bands[2])].astype(np.float32)

        return make_rgb(r, g, b, stretch=stretch)

    def index(self, name: str, **kwargs) -> np.ndarray:
        """
        Compute a built-in normalized spectral index.

        Currently supported indices are:
        - `"NDVI"`: `(NIR - RED) / (NIR + RED)`
        - `"NDWI"`: `(GREEN - NIR) / (GREEN + NIR)`

        Band selectors can be overridden through keyword arguments. For example,
        `nir`, `red`, and `green` may each be given as an integer band index, a
        wavelength in nanometers, or a string alias such as `"NIR"` or `"B3"`.

        Args:
            name: Name of the spectral index to compute.
            **kwargs: Optional band selector overrides used by the selected index.

        Returns:
            A 2D NumPy array containing the computed index.

        Raises:
            ValueError: If the requested index name is not supported.
        """
        n = name.strip().upper()

        if n == "NDVI":
            nir = kwargs.get("nir", "NIR")
            red = kwargs.get("red", "RED")
            return normalized_difference(self.get_band(nir), self.get_band(red))

        if n == "NDWI":
            green = kwargs.get("green", "GREEN")
            nir = kwargs.get("nir", "NIR")
            return normalized_difference(self.get_band(green), self.get_band(nir))

        raise ValueError(f"Unknown index: {name!r}")

    # geo / crop
    def crop_px(self, y0: int, y1: int, x0: int, x1: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Crop the event in pixel coordinates and return the cropped array and metadata.

        The crop is defined using half-open intervals `[y0:y1, x0:x1]` in image
        coordinates. Bounds are clamped to the valid image extent. When geospatial
        metadata is available, the affine transform and geographic bounds are updated
        to match the cropped window.

        Args:
            y0: Start row index.
            y1: End row index (exclusive).
            x0: Start column index.
            x1: End column index (exclusive).

        Returns:
            A tuple `(arr_crop, meta_crop)` where:
            - `arr_crop` is the cropped array with shape `(C, h, w)`,
            - `meta_crop` is a copy of the metadata updated with the cropped size and,
                when possible, updated transform and bounds.

        Raises:
            ValueError: If the resulting crop is empty or invalid after clamping.
        """
        H, W = int(self._arr.shape[1]), int(self._arr.shape[2])

        y0c = max(0, min(H, int(y0)))
        y1c = max(0, min(H, int(y1)))
        x0c = max(0, min(W, int(x0)))
        x1c = max(0, min(W, int(x1)))

        if y0c >= y1c or x0c >= x1c:
            raise ValueError(f"Invalid crop after clamp: y[{y0c},{y1c}) x[{x0c},{x1c}) for H={H}, W={W}")

        arr_c = self._arr[:, y0c:y1c, x0c:x1c]

        meta_c = dict(self._meta)
        meta_c["height"] = int(y1c - y0c)
        meta_c["width"] = int(x1c - x0c)

        t0 = self._meta.get("transform", None)
        if t0 is not None:
            win = Window(col_off=x0c, row_off=y0c, width=(x1c - x0c), height=(y1c - y0c))
            meta_c["transform"] = window_transform(win, t0)
            try:
                left, bottom, right, top = window_bounds(win, t0)
                meta_c["bounds"] = normalize_bounds(
                    rasterio.coords.BoundingBox(left=left, bottom=bottom, right=right, top=top)
                )
            except Exception:
                pass

        return arr_c, meta_c

    # tiling
    def to_tiles(self, tile_size: int = 512, overlap: int = 0, drop_last: bool = False) -> List[L1_tile]:
        """
        Split the event into a regular grid of in-memory tiles.

        Tiles are extracted from the event array using a sliding window with optional
        overlap. Each produced tile is stored as an :class:`L1_tile` instance and the
        internal tile collection of the event is replaced by the generated tiles.

        If `drop_last=False`, border tiles are kept even if they are smaller than
        `tile_size`. If `drop_last=True`, only full tiles are retained.

        Args:
            tile_size: Tile size in pixels for both height and width.
            overlap: Overlap in pixels between consecutive tiles.
            drop_last: Whether to discard incomplete border tiles.

        Returns:
            A list of generated :class:`L1_tile` objects.

        Raises:
            ValueError: If `overlap` does not satisfy
                `0 <= overlap < tile_size`.
        """
        H, W = int(self._arr.shape[1]), int(self._arr.shape[2])
        if overlap < 0 or overlap >= tile_size:
            raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
        step = max(1, tile_size - overlap)

        tiles: List[L1_tile] = []
        t0 = self._meta.get("transform", None)

        for y0 in range(0, H, step):
            y1 = y0 + tile_size
            if y1 > H:
                if drop_last:
                    break
                y1 = H

            for x0 in range(0, W, step):
                x1 = x0 + tile_size
                if x1 > W:
                    if drop_last:
                        break
                    x1 = W

                arr_t = self._arr[:, y0:y1, x0:x1]
                meta_t = dict(self._meta)
                meta_t["height"] = int(y1 - y0)
                meta_t["width"] = int(x1 - x0)

                if t0 is not None:
                    win = Window(col_off=x0, row_off=y0, width=(x1 - x0), height=(y1 - y0))
                    meta_t["transform"] = window_transform(win, t0)
                    left, bottom, right, top = window_bounds(win, t0)
                    meta_t["bounds"] = normalize_bounds(
                        rasterio.coords.BoundingBox(left=left, bottom=bottom, right=right, top=top)
                    )

                name = f"tile_y{y0}_x{x0}_s{tile_size}_o{overlap}"
                tiles.append(L1_tile(tile_name=name, arr=arr_t, meta=meta_t, device=self._device))

        self._tiles = tiles
        self.n_tiles = len(tiles)
        return tiles

    def make_tiles(self, tile_size: int = 512, overlap: int = 0, drop_last: bool = False) -> List[L1_tile]:
        """
        Alias for :meth:`to_tiles`.

        This method exists for API convenience and forwards all arguments to
        :meth:`to_tiles`.

        Args:
            tile_size: Tile size in pixels for both height and width.
            overlap: Overlap in pixels between consecutive tiles.
            drop_last: Whether to discard incomplete border tiles.

        Returns:
            A list of generated :class:`L1_tile` objects.
        """
        return self.to_tiles(tile_size=tile_size, overlap=overlap, drop_last=drop_last)

    # tiles infos
    def get_tiles_names(self, tiles_idx=None) -> List[str]:
        """
        Return the names of the currently available tiles.

        If `tiles_idx` is not provided, names for all tiles in the current tile
        collection are returned. If a subset of indices is given, only the
        corresponding tile names are returned, in the requested order.

        Args:
            tiles_idx: Optional iterable of tile indices to query.

        Returns:
            A list of tile names.

        Raises:
            ValueError: If the tile collection is empty.
            IndexError: If one of the requested tile indices is out of range.
        """
        if len(self._tiles) == 0:
            raise ValueError("Empty tiles lists.")

        if tiles_idx is None:
            tiles_idx = range(len(self._tiles))

        names: List[str] = []
        for i in tiles_idx:
            names.append(self.get_tile(i).tile_name)
        return names

    def get_tiles_info(self, tiles_idx=None) -> Dict[str, Any]:
        """
        Return structured information for the currently available tiles.

        Each entry is obtained from :meth:`L1_tile.get_tile_info` and stored in a
        dictionary keyed by tile name. The corresponding value is a tuple containing
        the tile name, sensing time, creation time, tile corner coordinates, and tile
        footprint coordinates.

        Args:
            tiles_idx: Optional iterable of tile indices to query. If `None`, all
                tiles are included.

        Returns:
            A dictionary mapping each tile name to its information tuple.

        Raises:
            ValueError: If the tile collection is empty.
            IndexError: If one of the requested tile indices is out of range.
        """
        if len(self._tiles) == 0:
            raise ValueError("Empty tiles lists.")

        if tiles_idx is None:
            tiles_idx = range(len(self._tiles))

        tiles_names: List[str] = []
        tiles_info: List[Any] = []
        for i in tiles_idx:
            t = self.get_tile(i)
            info = t.get_tile_info()
            tiles_info.append(info)
            tiles_names.append(info[0])

        return dict(zip(tiles_names, tiles_info))

    def show_tiles_info(self) -> None:
        """
        Print human-readable information for all currently available tiles.

        For each tile, this method prints a PyRawS-style summary including the tile
        name, sensing time, creation time, corner coordinates, and footprint
        coordinates.

        Returns:
            None.

        Raises:
            ValueError: If the tile collection is empty.
        """
        tiles_info = self.get_tiles_info()
        tiles_names = list(tiles_info.keys())

        for i in range(len(tiles_names)):
            print(colored("------------------Tile " + str(i) + " ----------------------------", "blue"))
            print("Name: ", colored(tiles_info[tiles_names[i]][0], "red"))
            print("Sensing time: ", colored(str(tiles_info[tiles_names[i]][1]), "red"))
            print("Creation time: ", colored(str(tiles_info[tiles_names[i]][2]), "red"))

            coordinates = tiles_info[tiles_names[i]][3]
            footprint_coordinates = tiles_info[tiles_names[i]][4]

            print("Corners coordinates: \n")
            for k in range(len(coordinates)):
                print(colored("\tP_" + str(k), "blue") + " : " + colored(str(coordinates[k]) + "\n", "red"))

            print("\n")
            print("Footprint's coordinates: \n")
            for k in range(len(footprint_coordinates)):
                print(colored("\tP_" + str(k), "blue") + " : " + colored(str(footprint_coordinates[k]) + "\n", "red"))
            print("\n")
    
    def show_bands(self, bands=None, tile=0, **kwargs) -> None:
        """
        Display one or more bands for a selected tile.

        This is a convenience wrapper around :meth:`L1_tile.show_bands`. By default,
        it visualizes the first tile in the current collection.

        Args:
            bands: Optional sequence of band selectors to display. If `None`, all
                bands of the selected tile are shown.
            tile: Tile index or tile name identifying which tile to visualize.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`L1_tile.show_bands`, such as downsampling, max_size, stretch,
                or cmap.

        Returns:
            None.

        Raises:
            IndexError: If the requested tile index is out of range.
            KeyError: If the requested tile name does not exist.
            ValueError: If one of the requested band selectors cannot be resolved.
        """
        t = self.get_tile(tile)
        t.show_bands(bands=bands, **kwargs)

    def get_tile(self, idx_or_name: Union[int, str]) -> L1_tile:
        """
        Return one tile from the current tile collection.

        A tile can be retrieved either by integer index or by its `tile_name`.

        Args:
            idx_or_name: Tile index or tile name.

        Returns:
            The requested :class:`L1_tile` instance.

        Raises:
            IndexError: If an integer index is out of range.
            KeyError: If a tile name is requested but no matching tile exists.
        """
        if isinstance(idx_or_name, int):
            return self._tiles[idx_or_name]
        name = str(idx_or_name)
        for t in self._tiles:
            if t.tile_name == name:
                return t
        raise KeyError(f"Tile not found: {name}")

    def show_event_info(self) -> None:
        """
        Print a concise summary of the current event.

        The printed summary includes the scene identifier, product kind, product
        folder, source path, array shape and dtype, CRS, geographic bounds, band
        wavelengths, GL file path, processing configuration path, and the current
        number of in-memory tiles.

        Returns:
            None.
        """
        print(colored("Event:", "blue"), f"scene_id={self._scene_id} kind={self._product_kind}")
        print("  folder:", colored(self._product_folder, "red"))
        print("  path:", colored(str(self._meta.get("path", None)), "red"))
        print("  shape:", colored(str(tuple(self._arr.shape)), "red"), "  dtype:", colored(str(self._arr.dtype), "red"))
        print("  crs:", colored(str(self._meta.get("crs", None)), "red"))
        print("  bounds:", colored(str(self._meta.get("bounds", None)), "red"))
        print("  wavelengths_nm:", colored(str(self.get_wavelengths()), "red"))
        print("  gl_path:", colored(str(self._meta.get("gl_path", None)), "red"))
        print("  processing_config:", colored(str(self._meta.get("processing_config_path", None)), "red"))
        print("  n_tiles:", colored(str(self.n_tiles), "red"))
    
    def plot_location(
        self,
        mode: str = "bounds",              # "bounds" | "footprint"
        world: bool = True,
        tiles_idx=None,
        title: Optional[str] = None,
    ):
        """
        Plot the geographic location of the event and optionally overlay tile bounds.

        Two plotting modes are supported:
        - `"bounds"`: plot the axis-aligned geographic bounding box stored in the
        event metadata,
        - `"footprint"`: plot the scene footprint derived from the GL JSON file
        referenced by the metadata.

        If `tiles_idx` is provided, the geographic bounds of the selected tiles are
        overlaid as rectangles on top of the scene plot.

        Args:
            mode: Plotting mode, either `"bounds"` or `"footprint"`.
            world: If `True`, attempt to display the scene on top of a world basemap.
            tiles_idx: Optional iterable of tile indices to overlay.
            title: Optional plot title. If `None`, a default title based on the scene
                identifier and product kind is used.

        Returns:
            The Matplotlib axes used for the plot.

        Raises:
            ValueError: If the requested mode is unknown, or if the required
                geospatial metadata is missing for the selected mode.
        """
        m = str(mode).strip().lower()

        if m == "bounds":
            from ..utils.optional_plots import plot_bounds

            b0 = self._meta.get("bounds", None)
            if b0 is None:
                raise ValueError("No bounds in event meta; cannot plot location (bounds).")

            ax = plot_bounds(
                b0,
                world=world,
                title=title or f"scene_{self._scene_id}_{self._product_kind}",
            )

            if tiles_idx is not None:
                for i in tiles_idx:
                    bi = self.get_tile(i).meta.get("bounds", None)
                    if bi is not None:
                        plot_bounds(bi, ax=ax, world=False, title=None, linewidth=1.0)

            return ax

        if m == "footprint":
            from ..utils.optional_plots import plot_gl_footprint, plot_bounds

            gl = self._meta.get("gl_path", None)
            if gl is None:
                raise ValueError("No gl_path in meta; cannot plot location (footprint).")

            ax = plot_gl_footprint(
                gl,
                world=world,
                title=title or f"scene_{self._scene_id}_{self._product_kind}",
            )

            # optional overlay: tiles bounds (rectangles)
            if tiles_idx is not None:
                for i in tiles_idx:
                    bi = self.get_tile(i).meta.get("bounds", None)
                    if bi is not None:
                        plot_bounds(bi, ax=ax, world=False, title=None, linewidth=1.0)

            return ax

        raise ValueError(f"Unknown mode: {mode!r} (expected 'bounds' or 'footprint').")
    
    

    # export
    def export_to_tif(
        self,
        out_path: str,
        arr: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Export the event data, or a provided array, to a GeoTIFF file.

        If `arr` is not provided, the event internal array is exported. If `meta` is
        not provided, the event metadata is used. Additional keyword arguments are
        forwarded to the low-level GeoTIFF export utility.

        Args:
            out_path: Output path of the GeoTIFF file to create.
            arr: Optional array to export. If `None`, the event internal array is
                used.
            meta: Optional metadata dictionary to use for export. If `None`, the
                event metadata is used.
            **kwargs: Additional keyword arguments forwarded to the underlying export
                utility.

        Returns:
            The output path of the written GeoTIFF file.

        Raises:
            ValueError: If the provided metadata is incomplete or incompatible with
                GeoTIFF export.
            OSError: If the file cannot be written.
        """
        if arr is None:
            arr = self._arr
        if meta is None:
            meta = self._meta
        return _export_to_tif(out_path=out_path, arr=arr, meta=meta, **kwargs)