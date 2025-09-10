from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import os

def _ext(p: str | Path) -> str:
    return os.path.splitext(str(p))[-1].lower()

def safe_imread(path: str | Path):
    """
    Надёжная загрузка 1-кадровых изображений, вне патча ultralytics:
    читаем как байты -> imdecode. Возвращает BGR или None.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def safe_imread_first_frame(path: str | Path):
    """
    Универсальная загрузка: если TIFF/TIF — берём ПЕРВУЮ страницу через imreadmulti,
    иначе используем безопасное imdecode. Возвращает BGR или None.
    """
    p = str(path)
    if _ext(p) in (".tif", ".tiff"):
        try:
            # imreadmulti не патчится ultralytics; возвращает (ok, [frames])
            ok, frames = cv2.imreadmulti(p, flags=cv2.IMREAD_UNCHANGED)
            if ok and frames:
                img = frames[0]
                # Приведём к BGR
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.ndim == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                elif img.ndim == 3 and img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return img
        except Exception:
            # упадём в запасной путь
            pass
    return safe_imread(p)