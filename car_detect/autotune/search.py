from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from car_detect.detection.types import Detection
from car_detect.detection.yolo import detect_cars_by_yolo, YOLOParams, YoloNotAvailableError, _load_yolo
from car_detect.utils.io import safe_imread
from car_detect.utils.io import safe_imread_first_frame

import logging
log = logging.getLogger("car_counter.autotune")



@dataclass(frozen=True)
class ImageTask:
    path: Path
    bgr: Optional[np.ndarray] = None  # если заранее загружено


@dataclass(frozen=True)
class SearchSpec:
    weights_path: str
    conf_list: Tuple[float, ...]
    iou_list: Tuple[float, ...]
    imgsz_list: Tuple[int, ...]
    width_range: Tuple[float, float]
    height_range: Tuple[float, float]
    aspect: float
    aspect_tol: float


@dataclass(frozen=True)
class TrialParams:
    conf: float
    iou: float
    imgsz: int


@dataclass
class TrialResult:
    image_path: Path
    conf: float
    iou: float
    imgsz: int
    n: int                # количество детекций
    pass_ratio: float     # доля боксов, прошедших фильтры размера/аспекта
    avg_score: float
    runtime_ms: float
    dets: List[Detection]


def _measure_quality(dets: List[Detection],
                     width_range: Tuple[float, float],
                     height_range: Tuple[float, float],
                     aspect: float,
                     aspect_tol: float) -> Tuple[int, float, float]:
    """Возвращает (n_dets, pass_ratio, avg_score)."""
    if not dets:
        return 0, 0.0, 0.0
    wmin, wmax = width_range
    hmin, hmax = height_range
    a_lo = aspect * (1.0 - aspect_tol)
    a_hi = aspect * (1.0 + aspect_tol)

    passed = 0
    ssum = 0.0
    for d in dets:
        w = d.width
        h = d.height
        a = (w / h) if h > 1e-6 else 1e9
        ok = (wmin <= w <= wmax) and (hmin <= h <= hmax) and (a_lo <= a <= a_hi)
        if ok:
            passed += 1
            ssum += float(d.score)
    pass_ratio = passed / max(1, len(dets))
    avg_score = (ssum / max(1, passed)) if passed else 0.0
    return len(dets), pass_ratio, avg_score


def _run_trial(img: np.ndarray, spec: SearchSpec, tp: TrialParams) -> TrialResult:
    yp = YOLOParams(
        weights=spec.weights_path,
        conf=tp.conf,
        iou=tp.iou,
        imgsz=tp.imgsz,
        width_range=spec.width_range,
        height_range=spec.height_range,
        aspect=spec.aspect,
        aspect_tol=spec.aspect_tol,
        keep_classes=(2,),  # COCO "car"
        label="car_yolo",
    )
    t0 = time.perf_counter()
    dets = detect_cars_by_yolo(img, yp)
    dt = (time.perf_counter() - t0) * 1000.0
    n, pr, av = _measure_quality(dets, spec.width_range, spec.height_range, spec.aspect, spec.aspect_tol)
    return TrialResult(Path("<in-memory>"), tp.conf, tp.iou, tp.imgsz, n, pr, av, dt, dets)


def grid_for_mode(mode: str) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[int, ...]]:
    """Возвращает (conf_list, iou_list, imgsz_list) по режиму."""
    m = mode.lower()
    if m.startswith("quick"):
        return (0.15, 0.25, 0.35), (0.4, 0.5), (960, 1280)
    if m.startswith("accurate"):
        return (0.12, 0.18, 0.25, 0.32), (0.35, 0.45, 0.55), (960, 1280, 1536)
    # very accurate
    return (0.10, 0.15, 0.20, 0.25, 0.30), (0.35, 0.45, 0.55, 0.65), (960, 1280, 1536, 1664)


def run_autotune(
    images: List[ImageTask],
    spec: SearchSpec,
    max_workers: int = max(1, cv2.getNumberOfCPUs() // 2),
) -> List[TrialResult]:
    """Параллельный перебор по всем изображениям и сетке параметров."""
    log.info("Auto-Tune start: %d images, grid conf=%s iou=%s imgsz=%s",
             len(images), spec.conf_list, spec.iou_list, spec.imgsz_list)
    results: List[TrialResult] = []
    tasks: List[Tuple[TrialParams, ImageTask]] = []
    _ = _load_yolo(spec.weights_path)
    if max_workers is None:
        max_workers = 1

    for img_task in images:
        # загрузка
        if img_task.bgr is None:
            bgr = safe_imread_first_frame(img_task.path)
            if bgr is None:
                continue
        else:
            bgr = img_task.bgr

        for c in spec.conf_list:
            for i in spec.iou_list:
                for sz in spec.imgsz_list:
                    tp = TrialParams(c, i, sz)
                    tasks.append((tp, ImageTask(img_task.path, bgr)))

    if not tasks:
        return results

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut2meta: Dict[Any, Tuple[TrialParams, ImageTask]] = {}
            for tp, it in tasks:
                fut = ex.submit(_run_trial, it.bgr, spec, tp)
                fut2meta[fut] = (tp, it)
            for fut in as_completed(fut2meta):
                tp, it = fut2meta[fut]
                try:
                    r = fut.result()
                    r.image_path = it.path
                    results.append(r)
                    log.info("Trial: %s conf=%.2f iou=%.2f sz=%d -> n=%d pass=%.1f%% time=%.0fms",
                             it.path.name, tp.conf, tp.iou, tp.imgsz, r.n, 100 * r.pass_ratio, r.runtime_ms)
                except YoloNotAvailableError:
                    raise
                except Exception as e:
                    results.append(TrialResult(it.path, tp.conf, tp.iou, tp.imgsz, 0, 0.0, 0.0, 0.0, []))
    except YoloNotAvailableError:
        # For GUI
        raise
    return results
