# pyrawph/l1/l1_tile.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
except Exception:  
    torch = None  


BandSpec = Union[int, float, str]

_ALIAS_NM: Dict[str, int] = {
    "BLUE": 490, "B": 490,
    "GREEN": 560, "G": 560,
    "RED": 665, "R": 665,
    "REDEDGE1": 705, "RE1": 705,
    "REDEDGE2": 740, "RE2": 740,
    "REDEDGE3": 783, "RE3": 783,
    "NIR": 842,
}


def _resolve_band_from_meta(
    band: BandSpec,
    wavelengths_nm: Optional[Sequence[Optional[int]]],
    C: int,
) -> int:
    if isinstance(band, int):
        if not (0 <= band < C):
            raise ValueError(f"Band index out of range: {band} (C={C})")
        return band

    # float wavelength (nm) => closest wavelength (ignore None)
    if isinstance(band, float):
        wl = int(round(band))
        wls = list(wavelengths_nm or [])
        valid = [(i, v) for i, v in enumerate(wls) if v is not None]
        if not valid:
            raise ValueError("No valid wavelengths in metadata; cannot resolve wavelength.")
        return int(min(valid, key=lambda iv: abs(int(iv[1]) - wl))[0])

    # string cases
    s = str(band).strip().upper()
    s_clean = s.replace(" ", "").replace("_", "")

    # "842nm"
    if s_clean.endswith("NM") and s_clean[:-2].isdigit():
        return _resolve_band_from_meta(float(int(s_clean[:-2])), wavelengths_nm, C)

    # "3"
    if s.isdigit():
        return _resolve_band_from_meta(int(s), wavelengths_nm, C)

    # "B3" / "BAND_3"
    for prefix in ("BAND_", "BAND", "B"):
        if s.startswith(prefix) and s[len(prefix):].isdigit():
            return _resolve_band_from_meta(int(s[len(prefix):]), wavelengths_nm, C)

    # aliases => closest wavelength
    if s in _ALIAS_NM:
        return _resolve_band_from_meta(float(_ALIAS_NM[s]), wavelengths_nm, C)

    raise ValueError(f"Cannot resolve band spec: {band!r}")


@dataclass
class L1_tile:
    tile_name: str
    arr: np.ndarray
    meta: Dict[str, Any]
    device: str = "cpu"

    # basic getters
    def as_numpy(self) -> np.ndarray:
        return self.arr

    def as_tensor(self, as_float32: bool = True):
        if torch is None:
            raise ImportError("torch is not available")
        t = torch.from_numpy(self.arr)
        if as_float32 and t.dtype != torch.float32:
            t = t.float()
        return t.to(self.device)

    # Helpers
    def get_tile_coordinates(self, latlon_format: bool = True) -> List[List[float]]:
        b = self.meta.get("bounds", None)
        crs = self.meta.get("crs", None)
        if b is None:
            return []

        left = float(getattr(b, "left", b[0]))
        bottom = float(getattr(b, "bottom", b[1]))
        right = float(getattr(b, "right", b[2]))
        top = float(getattr(b, "top", b[3]))

        # corners as (x, y)
        corners_xy = [(left, top), (left, bottom), (right, bottom), (right, top)]

        if latlon_format and crs == "EPSG:4326":
            return [[y, x] for (x, y) in corners_xy]  # [lat, lon]

        return [[y, x] for (x, y) in corners_xy]

    def get_tile_footprint_coordinates(self, latlon_format: bool = True, closed: bool = False) -> List[List[float]]:
        coords = self.get_tile_coordinates(latlon_format=latlon_format)
        if closed and coords:
            return coords + [coords[0]]
        return coords

    def get_tile_info(self):
        tile_name = self.tile_name
        sensing_time = self.meta.get("sensing_time", None)
        creation_time = self.meta.get("creation_time", None)
        return (
            tile_name,
            sensing_time,
            creation_time,
            self.get_tile_coordinates(latlon_format=True),
            self.get_tile_footprint_coordinates(latlon_format=True),
        )

    # Visualization
    def show_bands(
        self,
        bands: Optional[Sequence[BandSpec]] = None,
        downsampling: bool = True,
        max_size: int = 512,
        stretch: Tuple[float, float] = (2.0, 98.0),
        cmap: str = "viridis",
    ) -> None:
        """
        Visual inspection of bands.

        bands:
          - None -> show all bands
          - else: list of BandSpec (int, "B3", "842nm", aliases "NIR"/"RED"/...)
        downsampling:
          - if True, stride chosen so that max(H,W) ~ max_size
        stretch:
          - percentile stretch on each displayed band
        """
        arr = self.arr
        C, H, W = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
        wls = self.meta.get("band_wavelength_nm", None)

        if bands is None:
            idxs = list(range(C))
            labels = []
            for i in idxs:
                wl = None
                if isinstance(wls, (list, tuple)) and i < len(wls):
                    wl = wls[i]
                labels.append(f"B{i}" if wl is None else f"B{i} ({wl}nm)")
        else:
            idxs = [_resolve_band_from_meta(b, wls, C) for b in bands]
            labels = []
            for spec, i in zip(bands, idxs):
                wl = None
                if isinstance(wls, (list, tuple)) and i < len(wls):
                    wl = wls[i]
                base = str(spec)
                labels.append(base if wl is None else f"{base} -> B{i} ({wl}nm)")

        stride = 1
        if downsampling:
            stride = max(1, int(math.ceil(max(H, W) / float(max_size))))

        n = len(idxs)
        ncols = min(4, n)
        nrows = int(math.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)

        for k, ax in enumerate(axes):
            if k >= n:
                ax.axis("off")
                continue

            bi = idxs[k]
            x = arr[bi][::stride, ::stride].astype(np.float32)

            lo, hi = np.nanpercentile(x, stretch)
            x = (x - lo) / (hi - lo + 1e-6)
            x = np.clip(x, 0, 1)

            ax.imshow(x, cmap=cmap)
            ax.set_title(labels[k])
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def plot_location(self, world: bool = True, title: Optional[str] = None):
        """
        Optional world-map plot (requires geopandas available in optional_plots).
        Falls back to just the rectangle if basemap isn't available.
        """
        from ..utils.optional_plots import plot_bounds

        b = self.meta.get("bounds", None)
        if b is None:
            raise ValueError("No bounds in tile meta; cannot plot location.")
        return plot_bounds(b, world=world, title=title or self.tile_name)