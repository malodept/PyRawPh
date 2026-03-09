from __future__ import annotations

from typing import Any, Dict, Optional
import os
import numpy as np
import rasterio


def export_to_tif(
    out_path: str,
    arr: np.ndarray,
    meta: Dict[str, Any],
    dtype: Optional[str] = None,
    nodata: Optional[float] = None,
    compress: str = "deflate",
    bigtiff: str = "if_safer",
) -> str:
    """
    Export an array to GeoTIFF using meta["crs"] and meta["transform"].

    Accepted shapes:
      - (H, W)
      - (C, H, W)
      - (H, W, 3) / (H, W, 4)  (converted to (C, H, W))
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if arr.ndim == 2:
        data = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[-1] in (3, 4) and arr.shape[0] not in (3, 4):
            data = np.moveaxis(arr, -1, 0)
        else:
            data = arr
    else:
        raise ValueError(f"arr must be 2D or 3D, got shape={arr.shape}")

    C, H, W = map(int, data.shape)

    crs = meta.get("crs")
    transform = meta.get("transform")
    if crs is None:
        raise KeyError("meta['crs'] is missing.")
    if transform is None:
        raise KeyError("meta['transform'] is missing.")

    out_dtype = np.dtype(dtype).name if dtype is not None else data.dtype.name
    out_data = data.astype(out_dtype, copy=False) if out_dtype != data.dtype.name else data

    if nodata is not None and np.issubdtype(out_data.dtype, np.floating):
        out_data = out_data.copy()
        out_data[np.isnan(out_data)] = nodata

    profile = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": C,
        "dtype": out_dtype,
        "crs": crs,
        "transform": transform,
        "compress": compress,
        "bigtiff": bigtiff,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_data)

    return out_path