from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import numpy as np

from .types import Detection
from .nms import nms_detections


@dataclass(frozen=True)
class TilerParams:
    """Parameters controlling tiling and merge."""
    tile_w: int = 640
    tile_h: int = 640
    overlap: int = 96               # overlap (px) in both x and y
    nms_iou: float = 0.4            # final global NMS IoU
    cap_per_tile: int = 400         # safety cap to avoid huge per-tile candidate sets

    # If True, a tile "owns" only its inner core (without right/bottom half-overlaps).
    # Boxes whose centers fall outside the core are dropped (except for last row/col tiles).
    use_ownership: bool = True


def _tiles(w: int, h: int, p: TilerParams) -> Iterable[Tuple[int, int, int, int, Tuple[int, int, int, int]]]:
    """Yield tiles as (x0,y0,x1,y1, core) with ownership core for de-duplication."""
    tw, th, ov = int(p.tile_w), int(p.tile_h), int(p.overlap)
    step_x = max(1, tw - ov)
    step_y = max(1, th - ov)

    xs = list(range(0, max(1, w - tw + 1), step_x))
    ys = list(range(0, max(1, h - th + 1), step_y))
    if xs[-1] != w - tw:
        xs.append(max(0, w - tw))
    if ys[-1] != h - th:
        ys.append(max(0, h - th))

    for yi, y0 in enumerate(ys):
        for xi, x0 in enumerate(xs):
            x1 = min(w, x0 + tw)
            y1 = min(h, y0 + th)

            # Core area (ownership): shave off half-overlaps on right and bottom,
            # except for last col/row where the tile keeps full extent.
            if p.use_ownership:
                half = ov // 2
                core_x1 = x1 if xi == len(xs) - 1 else min(x1, x0 + tw - half)
                core_y1 = y1 if yi == len(ys) - 1 else min(y1, y0 + th - half)
                core = (x0, y0, core_x1, core_y1)
            else:
                core = (x0, y0, x1, y1)

            yield x0, y0, x1, y1, core


def _center_in_core(det: Detection, core: Tuple[int, int, int, int]) -> bool:
    cx = 0.5 * (det.x1 + det.x2)
    cy = 0.5 * (det.y1 + det.y2)
    x0, y0, x1, y1 = core
    return (x0 <= cx < x1) and (y0 <= cy < y1)


def run_tiled(
    image_bgr: np.ndarray,
    detector: Callable[[np.ndarray], List[Detection]],
    params: TilerParams = TilerParams(),
) -> List[Detection]:
    """Run `detector` over overlapping tiles and merge detections with global NMS.

    Parameters
    ----------
    image_bgr : np.ndarray (H,W,3) uint8/float32
        Input image.
    detector : Callable[[tile_bgr], List[Detection]]
        Function returning detections in tile-local coordinates.
    params : TilerParams
        Tiling/merging parameters.

    Returns
    -------
    List[Detection] in global coordinates (image space).
    """
    h, w = image_bgr.shape[:2]
    all_dets: List[Detection] = []

    for x0, y0, x1, y1, core in _tiles(w, h, params):
        tile = image_bgr[y0:y1, x0:x1, ...]
        local = detector(tile) or []

        # Safety cap
        if params.cap_per_tile and len(local) > params.cap_per_tile:
            local = sorted(local, key=lambda d: d.score, reverse=True)[: params.cap_per_tile]

        # Lift to global coords and (optionally) apply ownership
        for d in local:
            g = Detection(d.x1 + x0, d.y1 + y0, d.x2 + x0, d.y2 + y0, d.score, d.label)
            if not params.use_ownership or _center_in_core(g, core):
                all_dets.append(g)

    # Final global NMS
    return nms_detections(all_dets, params.nms_iou)
