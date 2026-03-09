from __future__ import annotations

import json
import os
import glob as globlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio

from .geo_utils import normalize_bounds


def _find_one_any(patterns: List[str]) -> Optional[str]:
    """Return first matching path across a list of glob patterns."""
    for pat in patterns:
        hits = globlib.glob(pat)
        if hits:
            hits.sort()
            return hits[0]
    return None

def read_L1_event_from_folder_phisat2(
    product_folder: str,
    scene_id: int = 0,
    product_kind: str = "BC",           # BC / RR / AC / DN ... if GeoTIFFs exist
    multiband: bool = True,             # True -> scene_0_BC_multiband.tiff ; False -> per-band files
    bands: Optional[List[int]] = None,  # None -> all bands ; else e.g. [0,1,2]
    as_float32: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    ΦSat-2 L1 local reader.

    Expected layout:
      <product_folder>/
        bands/
          scene_0_BC_multiband.tiff
          scene_0_BC_band_0.tiff ... scene_0_BC_band_*.tiff
        geolocation/GL_scene_0.json
        processing_config.json
        session_4663_metadata.json   (or logs/session_4663_metadata.json)
        ...

    Returns:
      arr: np.ndarray  shape (C,H,W)
      meta: dict       minimal metadata (crs, transform, bounds, wavelengths if found, etc.)
    """
    product_kind = product_kind.upper()

    
    # 1 - Read GeoTIFF
    if multiband:
        tif = os.path.join(product_folder, "bands", f"scene_{scene_id}_{product_kind}_multiband.tiff")
        if not os.path.exists(tif):
            raise FileNotFoundError(f"Missing multiband GeoTIFF: {tif}")

        with rasterio.open(tif) as ds:
            if bands is None:
                arr = ds.read()  # (C,H,W)
                picked_bands = list(range(ds.count))
            else:
                arr = ds.read([b + 1 for b in bands]) 
                picked_bands = list(bands)

            meta: Dict[str, Any] = {
                "path": tif,
                "scene_id": scene_id,
                "product_kind": product_kind,
                "picked_bands": picked_bands,
                "count": int(arr.shape[0]),
                "width": int(ds.width),
                "height": int(ds.height),
                "dtype": str(arr.dtype),
                "crs": str(ds.crs) if ds.crs is not None else None,
                "transform": ds.transform,
                "bounds": normalize_bounds(ds.bounds),
            }

    else:
        # If bands not specified, infer from file presence (scene_{id}_{kind}_band_*.tiff),
        # else fallback to 8.
        if bands is None:
            pat = os.path.join(product_folder, "bands", f"scene_{scene_id}_{product_kind}_band_*.tiff")
            hits = globlib.glob(pat)
            if hits:
                idxs = []
                for p in hits:
                    base = os.path.basename(p)
                    try:
                        i = int(os.path.splitext(base)[0].split("_band_")[-1])
                        idxs.append(i)
                    except Exception:
                        pass
                bands = sorted(set(idxs)) if idxs else list(range(8))
            else:
                bands = list(range(8))

        stacks: List[np.ndarray] = []
        first_meta: Optional[Dict[str, Any]] = None

        for b in bands:
            tif_b = os.path.join(product_folder, "bands", f"scene_{scene_id}_{product_kind}_band_{b}.tiff")
            if not os.path.exists(tif_b):
                raise FileNotFoundError(f"Missing band GeoTIFF: {tif_b}")

            with rasterio.open(tif_b) as ds:
                stacks.append(ds.read(1))
                if first_meta is None:
                    first_meta = {
                        "path": tif_b,
                        "width": int(ds.width),
                        "height": int(ds.height),
                        "crs": str(ds.crs) if ds.crs is not None else None,
                        "transform": ds.transform,
                        "bounds": normalize_bounds(ds.bounds),
                    }

        arr = np.stack(stacks, axis=0)
        meta = {
            **(first_meta or {}),
            "scene_id": scene_id,
            "product_kind": product_kind,
            "picked_bands": list(bands),
            "count": int(arr.shape[0]),
            "dtype": str(arr.dtype),
        }

    if as_float32 and arr.dtype != np.float32:
        arr = arr.astype(np.float32)



    # 2 - Parse session metadata (NoBands + wavelengths)
    session_meta_path = _find_one_any([
        os.path.join(product_folder, "session_*_metadata.json"),
        os.path.join(product_folder, "logs", "session_*_metadata.json"),
    ])
    meta["session_metadata_path"] = session_meta_path
    meta["band_wavelength_nm"] = None
    meta["no_bands"] = None

    if session_meta_path is not None:
        try:
            with open(session_meta_path, "r", encoding="utf-8") as f:
                sm = json.load(f)

            session_key = next((k for k in sm.keys() if k.startswith("Session ")), None)
            if session_key:
                imcfg = sm.get(session_key, {}).get("ImagerConfig", {}) or {}

                no_bands = imcfg.get("NoBands", None)
                try:
                    no_bands = int(no_bands) if no_bands is not None else None
                except Exception:
                    no_bands = None
                if no_bands is None:
                    no_bands = 8
                meta["no_bands"] = no_bands

                bcw = imcfg.get("BandCentreWavelength", {}) or {}
                if bcw:
                    all_w = [bcw.get(f"Band {i}") for i in range(no_bands)]
                    all_w = [int(x) if x is not None else None for x in all_w]

                    picked = meta.get("picked_bands", list(range(meta["count"])))
                    wl = []
                    for i in picked:
                        ii = int(i)
                        wl.append(all_w[ii] if 0 <= ii < len(all_w) else None)
                    meta["band_wavelength_nm"] = wl
        except Exception:
            pass

    # 3 - Attach minimal sidecar paths

    gl = os.path.join(product_folder, "geolocation", f"GL_scene_{scene_id}.json")
    if os.path.exists(gl):
        meta["gl_path"] = gl

    pc = os.path.join(product_folder, "processing_config.json")
    if os.path.exists(pc):
        meta["processing_config_path"] = pc

    return arr, meta