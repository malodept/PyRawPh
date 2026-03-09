from __future__ import annotations

from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    import geopandas as gpd
except Exception:  
    gpd = None  

import json
import numpy as np
from pathlib import Path

def _load_world():
    """
    World polygons (geopandas only).
    Priority:
      1) geopandas built-in dataset (if available)
      2) local asset: pyrawph/assets/world_countries.geojson
    """
    if gpd is None:
        return None

    # (1) old geopandas built-in datasets (if present)
    try:
        path = gpd.datasets.get_path("naturalearth_lowres")  
        return gpd.read_file(path)
    except Exception:
        pass

    # (2) local asset
    try:
        here = Path(__file__).resolve()
        asset = here.parent.parent / "assets" / "world_countries.geojson"  # pyrawph/assets/...
        if asset.exists():
            return gpd.read_file(asset)
    except Exception:
        pass

    return None


def plot_bounds(
    bounds,
    ax=None,
    world: bool = True,
    title: Optional[str] = None,
    linewidth: float = 2.0,
    pad_deg: float = 1.0,
):
    """
    Plot a bounds rectangle (left,bottom,right,top) on an optional world basemap.

    Parameters
    ----------
    bounds : BoundingBox | tuple
        Either rasterio BoundingBox-like (left,bottom,right,top attrs) or a 4-tuple.
    ax : matplotlib axis | None
        If None -> creates a new figure/axis.
    world : bool
        If True -> tries to plot Natural Earth lowres (if available in geopandas datasets).
        If not available -> still plots the rectangle.
    title : str | None
    linewidth : float
    pad_deg : float
        Padding (in degrees) added around the rectangle when setting axis limits.
        (Works well for EPSG:4326; for projected CRSs, set to a larger value.)
    """
    left = float(getattr(bounds, "left", bounds[0]))
    bottom = float(getattr(bounds, "bottom", bounds[1]))
    right = float(getattr(bounds, "right", bounds[2]))
    top = float(getattr(bounds, "top", bounds[3]))

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    if world:
        world_gdf = _load_world()
        if world_gdf is not None:
            world_gdf.plot(ax=ax, linewidth=0.5, alpha=0.35)

    rect = patches.Rectangle(
        (min(left, right), min(bottom, top)),
        abs(right - left),
        abs(top - bottom),
        fill=False,
        linewidth=linewidth,
    )
    ax.add_patch(rect)

    ax.set_xlim(min(left, right) - pad_deg, max(left, right) + pad_deg)
    ax.set_ylim(min(bottom, top) - pad_deg, max(bottom, top) + pad_deg)

    if title:
        ax.set_title(title)

    return ax

def plot_gl_footprint(
    gl_json_path: str,
    ax=None,
    world: bool = True,
    title: Optional[str] = None,
    linewidth: float = 2.0,
    pad_deg: float = 0.02,
):
    """
    Plot a rotated footprint polygon derived from GL_scene_<id>.json.

    Notes
    -----
    - Works in EPSG:4326 (lon/lat) since GL provides Lon/Lat.
    - If world=True, tries to draw a basemap (geopandas built-in dataset) if available.
    - Automatically recenters/zooms around the footprint (pad_deg in degrees).
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    if world:
        world_gdf = _load_world()
        if world_gdf is not None:
            world_gdf.plot(ax=ax, linewidth=0.5, alpha=0.35)

    with open(gl_json_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    pts = gj.get("Geolocated_Points", [])
    if not pts:
        raise ValueError("No Geolocated_Points in GL json")

    xs = np.array([p["X_coordinate"] for p in pts], dtype=int)
    ys = np.array([p["Y_coordinate"] for p in pts], dtype=int)
    lats = np.array([p["Lat"] for p in pts], dtype=float)
    lons = np.array([p["Lon"] for p in pts], dtype=float)

    x_unique = np.unique(xs)
    y_unique = np.unique(ys)
    W = x_unique.size
    H = y_unique.size

    # sort by (y, x) then reshape to (H, W)
    order = np.lexsort((xs, ys))
    xs, ys, lats, lons = xs[order], ys[order], lats[order], lons[order]
    lat_grid = lats.reshape(H, W)
    lon_grid = lons.reshape(H, W)

    # perimeter points: top row + right col + bottom row reversed + left col reversed
    top = np.c_[lon_grid[0, :], lat_grid[0, :]]
    right = np.c_[lon_grid[1:, -1], lat_grid[1:, -1]]
    bottom = np.c_[lon_grid[-1, -2::-1], lat_grid[-1, -2::-1]]
    left = np.c_[lon_grid[-2:0:-1, 0], lat_grid[-2:0:-1, 0]]
    poly = np.vstack([top, right, bottom, left, top[:1]])

    # draw polygon
    ax.plot(poly[:, 0], poly[:, 1], linewidth=2.5)

    # zoom tightly around footprint
    xmin, ymin = float(poly[:, 0].min()), float(poly[:, 1].min())
    xmax, ymax = float(poly[:, 0].max()), float(poly[:, 1].max())
    ax.set_xlim(xmin - pad_deg, xmax + pad_deg)
    ax.set_ylim(ymin - pad_deg, ymax + pad_deg)

    # avoid "Ignoring fixed y limits..." warnings
    ax.set_aspect("equal", adjustable="box")

    if title:
        ax.set_title(title)

    return ax