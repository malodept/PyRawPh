# PyRawPh — ΦSat-2 L1 Python API 

PyRawPh is a lightweight Python API to **open and process ΦSat-2 Level-1 (L1)** products.  
It intentionally follows the **look & feel of PyRawS** (same “event/tile” mental model, similar `show_*_info()` outputs), but it is **much simpler**:

- **No database indexing**
- **No raw ↔ L1 linkage**
- Focus: **open the data** + **basic processing utilities**.

---

## What you can do

**Open**
- Load a ΦSat-2 L1 product from a local folder (`L1_event.from_path(...)`)
- Read multi-band GeoTIFFs (or per-band files)
- Extract key metadata (CRS, transform, bounds, timestamps, wavelengths, GL footprint path)

**Process**
- Compute **RGB composites** (for visualization)
- Compute **indices** (NDVI, NDWI)
- **Crop** in pixel coordinates (with correct geo transform/bounds update)
- Create an in-memory **tile grid**
- **Export** arrays to GeoTIFF (tiles, crops, indices)
- Plot **location** on a world map using either:
  - `bounds` (rectangle) or
  - `footprint` (parallelogram-like polygon derived from `GL_scene_*.json`)

---

## Installation

### 1) Create a virtual environment
Windows (PowerShell):
```bash
python -m venv .venv
.venv\Scripts\activate

