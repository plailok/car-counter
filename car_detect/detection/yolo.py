from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Literal

import cv2
import numpy as np

from .types import Detection
import sys, os
from functools import lru_cache
import threading
_model_lock = threading.Lock()

@lru_cache(maxsize=4)
def _load_yolo(weights_path: str):
    from ultralytics import YOLO
    with _model_lock:
        return YOLO(weights_path)

def resource_path(rel_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, rel_path)
    return os.path.join(os.path.abspath("."), rel_path)

WEIGHTS = {
    "n": resource_path("models/yolov8n.pt"),
    "m": resource_path("models/yolov8m.pt"),
    "x": resource_path("models/yolov8x.pt"),
}
AspectMode = Literal["axis", "rotated", "auto"]

logger = logging.getLogger(__name__)


class YoloNotAvailableError(RuntimeError):
    """Raised when Ultralytics is not installed or not importable."""


def _estimate_rotated_geometry(patch_gray: np.ndarray, edge_thr: float, min_pts: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Возвращает (angle_deg, r_width, r_height) по облаку edge-пикселей в патче.
    angle: [-90, 90], 0 = горизонтальная длинная ось.
    Ширина/высота считаются как проекции edge-точек на главные оси (robust).
    """
    if patch_gray.size == 0:
        return None, None, None

    # бинаризуем «ребристые» пиксели по градиенту
    gx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ys, xs = np.where(mag > float(edge_thr))
    if xs.size < min_pts:
        return None, None, None

    # PCA по координатам edge-пикселей
    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]  # [major, minor]
    major = evecs[:, 0]
    minor = evecs[:, 1]

    # угол главной оси (x вправо, y вниз)
    angle = np.degrees(np.arctan2(major[1], major[0]))  # -90..90

    # проекции точек на оси → ориентированные ширина/высота
    proj_major = pts @ major
    proj_minor = pts @ minor
    r_w = float(proj_major.max() - proj_major.min())
    r_h = float(proj_minor.max() - proj_minor.min())

    return float(angle), r_w, r_h

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

    aspect_mode: AspectMode = "auto"  # "axis" (старое), "rotated" (ориентированное), "auto" (лучшее из двух)
    angle_edge_thr: float = 35.0  # порог градиента для набора edge-пикселей
    angle_min_points: int = 60  # мин. число edge-пикселей для оценки угла


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
    model = _load_yolo(params.weights)

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

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    filtered: list[Detection] = []
    for d in dets:  # то, что вы собираете из модели -> Detection(x1,y1,x2,y2,score,label)
        w = d.width
        h = d.height
        a_axis = (w / h) if h > 1e-6 else 1e9

        # --- angle-aware ---
        angle = None
        r_w = None
        r_h = None
        if params.aspect_mode in ("rotated", "auto"):
            xi1 = max(0, int(np.floor(d.x1)));
            yi1 = max(0, int(np.floor(d.y1)))
            xi2 = min(gray.shape[1], int(np.ceil(d.x2)));
            yi2 = min(gray.shape[0], int(np.ceil(d.y2)))
            patch = gray[yi1:yi2, xi1:xi2]
            angle, r_w, r_h = _estimate_rotated_geometry(patch, params.angle_edge_thr, params.angle_min_points)

        # выберем аспект в зависимости от режима
        if params.aspect_mode == "axis":
            a_use = a_axis
            w_use, h_use = w, h
        elif params.aspect_mode == "rotated" and (angle is not None) and (r_w and r_h and r_h > 1e-6):
            a_use = r_w / r_h
            w_use, h_use = r_w, r_h
        else:  # auto
            if (angle is not None) and (r_w and r_h and r_h > 1e-6):
                a_rot = r_w / r_h
                # берём «лучший» аспект: ближе к ожидаемому
                if abs(a_rot - params.aspect) < abs(a_axis - params.aspect):
                    a_use = a_rot;
                    w_use, h_use = r_w, r_h
                else:
                    a_use = a_axis;
                    w_use, h_use = w, h
            else:
                a_use = a_axis;
                w_use, h_use = w, h

        # --- фильтры размеров/аспекта ---
        wmin, wmax = params.width_range
        hmin, hmax = params.height_range
        a_lo = params.aspect * (1.0 - params.aspect_tol)
        a_hi = params.aspect * (1.0 + params.aspect_tol)

        size_ok = (wmin <= w_use <= wmax) and (hmin <= h_use <= hmax)
        aspect_ok = (a_lo <= a_use <= a_hi)

        if size_ok and aspect_ok:
            d.angle = angle
            d.r_width = w_use
            d.r_height = h_use
            filtered.append(d)

    return filtered