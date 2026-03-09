from __future__ import annotations

from typing import Tuple, Union

from rasterio.coords import BoundingBox


BoundsLike = Union[BoundingBox, Tuple[float, float, float, float]]


def normalize_bounds(bb: BoundsLike | None) -> BoundingBox | None:
    """
    Ensure bounds are always (left < right) and (bottom < top).

    Accepts either a rasterio BoundingBox or a (left, bottom, right, top) tuple.
    Returns a rasterio BoundingBox (or None).

    Note: no need to import rasterio here; BoundingBox is enough.
    """
    if bb is None:
        return None

    if isinstance(bb, BoundingBox):
        left, bottom, right, top = bb.left, bb.bottom, bb.right, bb.top
    else:
        left, bottom, right, top = bb

    return BoundingBox(
        left=min(left, right),
        bottom=min(bottom, top),
        right=max(left, right),
        top=max(bottom, top),
    )