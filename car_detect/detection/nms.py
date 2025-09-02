from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .types import Detection

__all__ = ["iou_xyxy", "nms", "nms_detections"]


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes in XYXY format.

    Parameters
    ----------
    a : np.ndarray, shape (N, 4)
        Boxes [x1, y1, x2, y2].
    b : np.ndarray, shape (M, 4)
        Boxes [x1, y1, x2, y2].

    Returns
    -------
    iou : np.ndarray, shape (N, M)
        IoU matrix in [0, 1].

    Notes
    -----
    Vectorized, no Python loops. Assumes coordinates are in pixels, with x2>x1 and y2>y1.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    # Areas
    a_wh = np.clip(a[:, 2:4] - a[:, 0:2], 0, None)
    b_wh = np.clip(b[:, 2:4] - b[:, 0:2], 0, None)
    a_area = a_wh[:, 0] * a_wh[:, 1]
    b_area = b_wh[:, 0] * b_wh[:, 1]

    # Intersections
    tl = np.maximum(a[:, None, 0:2], b[None, :, 0:2])  # (N, M, 2)
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])  # (N, M, 2)
    wh = np.clip(br - tl, 0, None)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # Union
    union = a_area[:, None] + b_area[None, :] - inter
    # Avoid divide by zero
    iou = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    return iou


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Non-maximum suppression (greedy) over boxes with scores.

    Parameters
    ----------
    boxes : np.ndarray, shape (N, 4)
        Boxes in [x1, y1, x2, y2].
    scores : np.ndarray, shape (N,)
        Confidence scores.
    iou_threshold : float
        Suppress boxes with IoU > threshold w.r.t. a kept higher-score box.

    Returns
    -------
    keep_idx : np.ndarray, shape (K,)
        Indices of kept boxes (sorted by descending score). Deterministic tie-break.

    Notes
    -----
    - Deterministic tie-break: higher score first; if equal, lower original index first.
    - Works on CPU; complexity is acceptable for typical post-NN or template outputs.
    """
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)

    boxes = boxes.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)
    idx = np.arange(boxes.shape[0], dtype=np.int64)

    # Sort by score desc; for equal scores, by index asc (deterministic)
    order = np.lexsort((idx, -scores))  # secondary key: idx asc; primary: -scores
    order = order.astype(np.int64, copy=False)

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        # Compute IoU of the top box against the rest
        iou = iou_xyxy(boxes[i][None, :], boxes[rest]).reshape(-1)
        mask = iou <= float(iou_threshold)
        order = rest[mask]

    # Return indices sorted by descending score
    # (Already in descending order due to greedy selection.)
    return np.array(keep, dtype=np.int64)


def nms_detections(dets: Sequence[Detection], iou_threshold: float) -> List[Detection]:
    """Convenience wrapper for NMS on a list of `Detection`."""
    boxes, scores, labels = Detection.list_to_arrays(list(dets))
    keep = nms(boxes, scores, iou_threshold)
    kept = [dets[i] for i in keep.tolist()]
    return kept