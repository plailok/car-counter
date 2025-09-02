from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

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
    peak_min_distance: int = 3  # pixels between local maxima


# ----------------------- preprocessing ------------------------------------- #

def to_gray_clahe(bgr: np.ndarray, clip: float, tile: Tuple[int, int]) -> np.ndarray:
    """Convert BGR→GRAY and apply CLAHE to improve local contrast.

    Returns uint8 image.
    """
    if bgr.ndim == 2:
        gray = bgr
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tile))
    out = clahe.apply(gray)
    return out


# ----------------------- template construction ----------------------------- #

def _make_template(w: int, h: int) -> np.ndarray:
    """Create a soft-edged rectangular template of size (h, w)."""
    # base = ones with soft gaussian edge to reduce ringing
    base = np.ones((h, w), np.float32)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=max(w, h) * 0.05)
    # Normalize for TM_CCOEFF_NORMED
    base = base - base.mean()
    denom = base.std() + 1e-6
    base = base / denom
    return base


def _build_templates(
    base_sizes: Sequence[Tuple[int, int]],
    scales: Iterable[float],
) -> List[Tuple[np.ndarray, int, int]]:
    """Generate templates for all (base_size × scale).

    Returns list of (template, w, h).
    """
    out: List[Tuple[np.ndarray, int, int]] = []
    for (bw, bh) in base_sizes:
        for s in scales:
            w = max(4, int(round(bw * float(s))))
            h = max(4, int(round(bh * float(s))))
            tpl = _make_template(w, h)
            out.append((tpl, w, h))
    return out


# ----------------------- peak picking on scoremap -------------------------- #

def _find_local_maxima(score: np.ndarray, thr: float, min_dist: int) -> List[Tuple[int, int, float]]:
    """Find local maxima ≥ thr using dilation-based NMS on the score map.

    Returns list of (x, y, value).
    """
    if score.size == 0:
        return []

    # Dilate to get local maxima
    k = 2 * int(max(1, min_dist)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dil = cv2.dilate(score, kernel)
    peaks = (score >= thr) & (score >= dil - 1e-6)

    ys, xs = np.where(peaks)
    vals = score[ys, xs]
    order = np.argsort(-vals)
    return [(int(xs[i]), int(ys[i]), float(vals[i])) for i in order]


# ----------------------- main detection ------------------------------------ #

def detect_cars_by_template(
    image_bgr: np.ndarray,
    params: TemplateParams = TemplateParams(),
    label: str = "car_tpl",
) -> List[Detection]:
    """Detect cars via normalized template matching across scales and orientations.

    Steps:
      1) GRAY + CLAHE
      2) For each template (two orientations × scales): cv2.matchTemplate(TMN_CCOEFF_NORMED)
      3) Collect local maxima ≥ threshold
      4) Convert to boxes and run NMS

    Returns
    -------
    List[Detection]
        Detections with scores in [0, 1].
    """
    gray = to_gray_clahe(image_bgr, params.clahe_clip_limit, params.clahe_tile_grid)
    h_img, w_img = gray.shape[:2]

    templates = _build_templates(params.base_sizes, params.scales)
    dets: List[Detection] = []

    for tpl, w, h in templates:
        # matchTemplate expects: image (float32) and template (float32)
        # We normalize image per-template subtly by converting to float32
        res = cv2.matchTemplate(gray.astype(np.float32), tpl, cv2.TM_CCOEFF_NORMED)
        # res has shape (H - h + 1, W - w + 1), value in [-1, 1]
        peaks = _find_local_maxima(res, params.threshold, params.peak_min_distance)
        for x, y, v in peaks:
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
            # Clip to image bounds just in case
            x2 = min(x2, float(w_img - 1))
            y2 = min(y2, float(h_img - 1))
            dets.append(Detection(x1, y1, x2, y2, score=float(v), label=label))

    # Greedy NMS across all templates
    dets = nms_detections(dets, params.nms_iou)
    return dets


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

    # Base background + noise
    base = np.full((H, W), p.bg_level, np.uint8)
    noise = rng.normal(0.0, p.noise_sigma, size=(H, W)).astype(np.float32)
    img = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    gt: List[Tuple[int, int, int, int]] = []

    def _place_ok(x1: int, y1: int, w: int, h: int) -> bool:
        x2, y2 = x1 + w, y1 + h
        if x2 >= W - 1 or y2 >= H - 1 or x1 < 0 or y1 < 0:
            return False
        # Light IoU constraint to avoid heavy overlaps that hurt counting
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

    # Place cars
    for _ in range(p.n_cars):
        orient = rng.choice(p.orientations)
        bw, bh = p.base_size
        if orient == "h":
            w0, h0 = bw, bh
        else:
            w0, h0 = bh, bw
        s = float(rng.uniform(1.0 - p.jitter_scale, 1.0 + p.jitter_scale))
        w = max(8, int(round(w0 * s)))
        h = max(8, int(round(h0 * s)))

        # Try several times to find a free spot
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

        # Occasionally add a soft shadow nearby
        if rng.random() < p.shadow_prob:
            cx = int(np.clip(x1 + int(0.3 * w), 0, W - 1))
            cy = int(np.clip(y1 + int(0.7 * h), 0, H - 1))
            _add_shadow(img, (cx, cy), radius=max(6, int(0.6 * max(w, h))))

        gt.append((x1, y1, x2, y2))

    # Return as BGR
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return bgr, gt
