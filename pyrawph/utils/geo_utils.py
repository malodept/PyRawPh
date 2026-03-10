from __future__ import annotations

from typing import Tuple, Union

from rasterio.coords import BoundingBox


BoundsLike = Union[BoundingBox, Tuple[float, float, float, float]]


def normalize_bounds(bb: BoundsLike | None) -> BoundingBox | None:
    """
    Normalize geographic bounds to a standard raster-style ordering.

    This function accepts either a `rasterio.coords.BoundingBox` or a
    `(left, bottom, right, top)` tuple and returns a `BoundingBox` such that:

    - `left < right`
    - `bottom < top`

    If the input is `None`, `None` is returned unchanged.

    Args:
        bb: Input bounds as a `BoundingBox`, a 4-tuple, or `None`.

    Returns:
        A normalized `BoundingBox`, or `None` if the input is `None`.
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