from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .types import Detection

logger = logging.getLogger(__name__)


class YoloNotAvailableError(RuntimeError):
    """Raised when Ultralytics is not installed or not importable."""


def _require_ultralytics():
    """Try to import Ultralytics YOLO and return the class.

    Raises
    ------
    YoloNotAvailableError
        If 'ultralytics' is not installed. The message contains install hint.
    """
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except Exception as exc:  # pragma: no cover
        raise YoloNotAvailableError(
            "Ultralytics is not available. Install it with:\n"
            "    pip install ultralytics\n"
            "and provide proper weights (e.g., 'yolov8n.pt' or custom)."
        ) from exc


@dataclass(frozen=True)
class YOLOParams:
    """YOLO inference and post-filtering parameters.

    Attributes
    ----------
    weights : str
        Path or name of weights. Example: 'yolov8n.pt'. If not present locally,
        Ultralytics may download it (internet required).
    conf : float
        Confidence threshold at model level (0..1). Higher → fewer, cleaner boxes.
    iou : float
        NMS IoU inside the model (0..1). Lower → more aggressive suppression.
    imgsz : int
        Inference image size (long side). Larger → better recall for small cars,
        slower inference and more VRAM.
    keep_classes : Optional[Tuple[int, ...]]
        Keep only these class IDs (per model's class map). For COCO, 'car' is 2.
        Set None to keep all classes.
    width_range : Tuple[float, float]
        Accept only boxes with width in pixels within [min, max].
        Use to match expected car size at your image scale (~120 px in width).
    height_range : Tuple[float, float]
        Same for height in pixels.
    aspect : float
        Expected width/height ratio of a car (~120/50 ≈ 2.4 for top-down).
    aspect_tol : float
        Allowed absolute deviation around 'aspect' (e.g., 0.8 → keep 1.6..3.2).
    label : str
        Label to write into Detection objects.
    """

    weights: str = "yolov8n.pt"
    conf: float = 0.25
    iou: float = 0.50
    imgsz: int = 1280

    keep_classes: Optional[Tuple[int, ...]] = (2,)  # COCO: car=2
    width_range: Tuple[float, float] = (80.0, 170.0)   # tuned for ~120×50 px cars
    height_range: Tuple[float, float] = (35.0, 90.0)
    aspect: float = 120.0 / 50.0  # 2.4
    aspect_tol: float = 0.8

    label: str = "car_yolo"


def _coerce_np(x) -> np.ndarray:
    """Convert torch/ultralytics tensor-like to np.ndarray (cpu)."""
    try:
        # Ultralytics tensors expose .cpu() for torch types
        return x.cpu().numpy()  # type: ignore[attr-defined]
    except Exception:
        return np.asarray(x)


def _post_filter_xyxy(
    xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    p: YOLOParams,
) -> List[Detection]:
    """Filter raw YOLO outputs by class, size (px), and aspect ratio."""
    dets: List[Detection] = []

    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = map(float, xyxy[i, :4])
        score = float(scores[i])
        cls_id = int(classes[i]) if classes is not None and classes.size else -1

        if p.keep_classes is not None and cls_id not in p.keep_classes:
            continue

        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if not (p.width_range[0] <= w <= p.width_range[1]):
            continue
        if not (p.height_range[0] <= h <= p.height_range[1]):
            continue

        a = w / (h if h > 1e-6 else 1e-6)
        if abs(a - p.aspect) > p.aspect_tol:
            continue

        dets.append(Detection(x1, y1, x2, y2, score=score, label=p.label))
    return dets


def detect_cars_by_yolo(image_bgr: np.ndarray, params: YOLOParams = YOLOParams()) -> List[Detection]:
    """Run Ultralytics YOLO and filter detections to car-like boxes.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR (OpenCV style), H×W×3 uint8 or float32 in [0..255].
    params : YOLOParams
        Inference and filtering parameters.

    Returns
    -------
    List[Detection]
        Filtered detections (car candidates) as our dataclass list.

    Raises
    ------
    YoloNotAvailableError
        If ultralytics is missing. The message suggests how to install it.
    """
    YOLO = _require_ultralytics()
    model = YOLO(params.weights)

    # Prefer the explicit predict() API for consistency across versions
    try:
        results = model.predict(
            source=image_bgr,
            imgsz=int(params.imgsz),
            conf=float(params.conf),
            iou=float(params.iou),
            verbose=False,
        )
    except TypeError:
        # Fallback to callable if predict signature differs
        results = model(image_bgr, imgsz=int(params.imgsz), verbose=False)

    if not results:
        return []

    r = results[0]
    # Ultralytics v8: r.boxes has .xyxy, .conf, .cls
    try:
        xyxy = _coerce_np(r.boxes.xyxy)
        scores = _coerce_np(r.boxes.conf)
        classes = _coerce_np(r.boxes.cls)
    except Exception as exc:
        # Fallback: some versions expose .data (N×6: x1 y1 x2 y2 conf cls)
        data = _coerce_np(r.boxes.data)
        if data.size == 0:
            return []
        xyxy = data[:, 0:4]
        scores = data[:, 4]
        classes = data[:, 5]

    dets = _post_filter_xyxy(xyxy, scores, classes, params)
    logger.debug("YOLO raw=%d → kept=%d (conf≥%.2f, w∈%s, h∈%s, aspect≈%.2f±%.2f)",
                 xyxy.shape[0], len(dets), params.conf, params.width_range,
                 params.height_range, params.aspect, params.aspect_tol)
    return dets