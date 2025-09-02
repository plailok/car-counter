from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import cv2
import numpy as np

from .nms import nms_detections
from .types import Detection


@dataclass(frozen=True)
class TemplateParams:
    """Parameters for template-matching car detection."""
    threshold: float = 0.72
    scales: Tuple[float, ...] = (0.9, 1.0, 1.1)
    nms_iou: float = 0.4
    base_sizes: Tuple[Tuple[int, int], ...] = ((120, 50), (50, 120))  # (w, h)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    # Peak picking controls
    peak_min_distance: int = 3  # absolute floor (px)
    relative_peak_distance: float = 0.35  # fraction of min(w,h) per template
    max_peaks_per_template: int = 400  # hard cap to keep NMS fast


# ----------------------- preprocessing ------------------------------------- #

def to_gray_clahe(bgr: np.ndarray, clip: float, tile: Tuple[int, int]) -> np.ndarray:
    """Convert BGR→GRAY and apply CLAHE to improve local contrast (uint8)."""
    if bgr.ndim == 2:
        gray = bgr
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tile))
    out = clahe.apply(gray)
    return out


# ----------------------- template construction ----------------------------- #

def _make_template(w: int, h: int) -> np.ndarray:
    """Mean-zero, unit-variance rectangular template with a bright core and
    a negative ring (difference-of-rectangles), then softly blurred.

    Это даёт острый максимум на прямоугольниках нужного размера и низкий отклик на фоне.
    """
    w = int(w); h = int(h)
    tpl = np.zeros((h, w), np.float32)

    # Внутренний яркий прямоугольник (оставим поля 8% по каждой стороне)
    mx = max(1, int(round(0.08 * w)))
    my = max(1, int(round(0.08 * h)))
    core_x1, core_y1 = mx, my
    core_x2, core_y2 = max(core_x1 + 2, w - mx - 1), max(core_y1 + 2, h - my - 1)

    # Бинарные маски ядра и всего шаблона
    core = np.zeros_like(tpl)
    core[core_y1:core_y2, core_x1:core_x2] = 1.0
    full = np.ones_like(tpl)

    # «Кольцо» = full - core
    ring = full - core

    # Выберем веса так, чтобы среднее по tpl было нулевым: A*core + B*ring,  mean=0
    area_core = float(core.sum())
    area_ring = float(ring.sum())
    if area_ring < 1:  # на всякий случай
        area_ring = 1.0
    A = 1.0
    B = - area_core / area_ring  # чтобы среднее было 0
    tpl = A * core + B * ring

    # Мягкие края, чтобы избежать звонков
    tpl = cv2.GaussianBlur(tpl, (0, 0), sigmaX=max(w, h) * 0.03)

    # Нормализация (zero-mean, unit-std)
    tpl -= tpl.mean()
    std = tpl.std()
    if std < 1e-6:
        std = 1e-6
    tpl /= std
    return tpl


def _build_templates(
    base_sizes: Sequence[Tuple[int, int]],
    scales: Iterable[float],
) -> List[Tuple[np.ndarray, int, int]]:
    """Generate templates for all (base_size × scale). Returns (template, w, h)."""
    out: List[Tuple[np.ndarray, int, int]] = []
    for (bw, bh) in base_sizes:
        for s in scales:
            w = max(4, int(round(bw * float(s))))
            h = max(4, int(round(bh * float(s))))
            tpl = _make_template(w, h)
            out.append((tpl, w, h))
    return out


# ----------------------- peak picking on scoremap -------------------------- #

# --- replace this function in car_detect/detection/template.py ---

def _find_local_maxima(
    score: np.ndarray,
    thr: float,
    min_dist: int,
    top_k: Optional[int] = None,
) -> List[Tuple[int, int, float]]:
    """Find unique local maxima ≥ thr on possibly flat plateaus.

    Strategy:
      1) Morphological dilation to locate regional maxima.
      2) Build mask of points equal to dilation AND above threshold.
      3) Connected components on that mask -> one plateau = one component.
      4) For each component pick a single representative (argmax by score).
      5) Sort by score desc and keep top_k if requested.
    """
    if score.size == 0:
        return []

    # Kernel size tied to min_dist (guarantees spatial sparsity after step 4)
    k = 2 * int(max(1, min_dist)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dil = cv2.dilate(score, kernel)

    # Regional maxima (allow tiny epsilon to swallow FP32 noise)
    maxima_mask = (score >= thr) & (np.abs(score - dil) <= 1e-6)
    if not np.any(maxima_mask):
        return []

    # Connected components on 8-neighborhood
    # OpenCV expects uint8 mask {0,1}
    mask_u8 = maxima_mask.astype(np.uint8)
    num, labels = cv2.connectedComponents(mask_u8, connectivity=8)

    peaks: List[Tuple[int, int, float]] = []
    # For each component (skip background 0)
    for comp_id in range(1, num):
        ys, xs = np.where(labels == comp_id)
        vals = score[ys, xs]
        # Choose argmax; if many equal, pick smallest (y,x) for determinism
        best = np.flatnonzero(vals == vals.max())
        idx = int(best[0])  # deterministic: the first in scanning order
        x = int(xs[idx])
        y = int(ys[idx])
        v = float(score[y, x])
        peaks.append((x, y, v))

    # Sort by score desc
    peaks.sort(key=lambda t: t[2], reverse=True)
    if top_k is not None and len(peaks) > top_k:
        peaks = peaks[:top_k]
    return peaks



# ----------------------- main detection ------------------------------------ #

def detect_cars_by_template(
    image_bgr: np.ndarray,
    params: TemplateParams = TemplateParams(),
    label: str = "car_tpl",
    zscore_stats: Tuple[float, float] | None = None,  # <— НОВОЕ
) -> List[Detection]:
    """Detect cars via normalized template matching across scales and orientations.

    Steps:
      1) GRAY + CLAHE
      2) For each template (two orientations × scales): cv2.matchTemplate(TMN_CCOEFF_NORMED)
      3) Collect local maxima ≥ threshold (distance depends on template size, top-K limited)
      4) Convert to boxes, run per-template NMS
      5) Concatenate and run global NMS
    """
    gray = to_gray_clahe(image_bgr, params.clahe_clip_limit, params.clahe_tile_grid)

    # Z-score нормализация: либо глобальная (из zscore_stats), либо локальная (по текущему тайлу)
    if zscore_stats is None:
        m, s = float(gray.mean()), float(gray.std())
    else:
        m, s = zscore_stats
    g = (gray.astype(np.float32) - m) / (s + 1e-6)

    h_img, w_img = g.shape[:2]

    templates = _build_templates(params.base_sizes, params.scales)
    all_dets: List[Detection] = []

    for tpl, w, h in templates:
        res = cv2.matchTemplate(g, tpl, cv2.TM_CCOEFF_NORMED)
        # Backroll: if res is NaN/Inf
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
        # Distance tied to template size (prevents multiple peaks within one car)
        min_dist_tpl = max(
            int(round(min(w, h) * float(params.relative_peak_distance))),
            int(params.peak_min_distance),
        )

        peaks = _find_local_maxima(
            res,
            thr=params.threshold,
            min_dist=min_dist_tpl,
            top_k=params.max_peaks_per_template,
        )

        # Build local detections and clip to image just in case
        local: List[Detection] = []
        x_max = float(w_img - 1)
        y_max = float(h_img - 1)
        for x, y, v in peaks:
            x1 = float(x)
            y1 = float(y)
            x2 = min(float(x + w), x_max)
            y2 = min(float(y + h), y_max)
            local.append(Detection(x1, y1, x2, y2, score=float(v), label=label))

        # Per-template NMS to shrink duplicates even before the final merge
        if local:
            local = nms_detections(local, params.nms_iou)
            all_dets.extend(local)

    # Global NMS across all templates/scales
    all_dets = nms_detections(all_dets, params.nms_iou)
    return all_dets


# ----------------------- synthetic scene generator ------------------------- #

@dataclass(frozen=True)
class SyntheticParams:
    width: int = 1920
    height: int = 1080
    n_cars: int = 12
    base_size: Tuple[int, int] = (120, 50)  # (w, h) canonical
    jitter_scale: float = 0.08              # ±8% random scale
    orientations: Tuple[str, ...] = ("h", "v")  # horizontal (w×h) or vertical (h×w)
    bg_level: int = 80
    car_level: int = 200
    noise_sigma: float = 8.0
    shadow_prob: float = 0.35
    seed: int = 42


def _rand_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _add_shadow(img: np.ndarray, center: Tuple[int, int], radius: int) -> None:
    """Darken a circular region (soft) to emulate shadows."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.circle(mask, center, int(radius), color=1.0, thickness=-1, lineType=cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.6)
    img[:] = np.clip(img.astype(np.float32) - 40.0 * mask, 0, 255).astype(np.uint8)


def generate_synthetic_scene(p: SyntheticParams = SyntheticParams()) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """Make a synthetic grayscale scene with bright rectangular 'cars' on darker background.

    Returns
    -------
    image_bgr : np.ndarray (H, W, 3) uint8
    gt_boxes  : List of GT boxes [x1, y1, x2, y2] in pixels
    """
    rng = _rand_rng(p.seed)
    H, W = p.height, p.width

    base = np.full((H, W), p.bg_level, np.uint8)
    noise = rng.normal(0.0, p.noise_sigma, size=(H, W)).astype(np.float32)
    img = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    gt: List[Tuple[int, int, int, int]] = []

    def _place_ok(x1: int, y1: int, w: int, h: int) -> bool:
        x2, y2 = x1 + w, y1 + h
        if x2 >= W - 1 or y2 >= H - 1 or x1 < 0 or y1 < 0:
            return False
        # Avoid heavy overlaps (IoU > 0.1)
        for gx1, gy1, gx2, gy2 in gt:
            xx1 = max(x1, gx1)
            yy1 = max(y1, gy1)
            xx2 = min(x2, gx2)
            yy2 = min(y2, gy2)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            a = (x2 - x1) * (y2 - y1)
            b = (gx2 - gx1) * (gy2 - gy1)
            union = a + b - inter
            if union > 0 and inter / union > 0.1:
                return False
        return True

    for _ in range(p.n_cars):
        orient = rng.choice(p.orientations)
        bw, bh = p.base_size
        w0, h0 = (bw, bh) if orient == "h" else (bh, bw)
        s = float(rng.uniform(1.0 - p.jitter_scale, 1.0 + p.jitter_scale))
        w = max(8, int(round(w0 * s)))
        h = max(8, int(round(h0 * s)))

        ok = False
        for _try in range(200):
            x1 = int(rng.integers(0, max(1, W - w - 1)))
            y1 = int(rng.integers(0, max(1, H - h - 1)))
            if _place_ok(x1, y1, w, h):
                ok = True
                break
        if not ok:
            continue

        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(img, (x1, y1), (x2, y2), color=int(p.car_level), thickness=-1, lineType=cv2.LINE_AA)

        if rng.random() < p.shadow_prob:
            cx = int(np.clip(x1 + int(0.3 * w), 0, W - 1))
            cy = int(np.clip(y1 + int(0.7 * h), 0, H - 1))
            _add_shadow(img, (cx, cy), radius=max(6, int(0.6 * max(w, h))))

        gt.append((x1, y1, x2, y2))

    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return bgr, gt
