from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import cv2



from PySide6.QtWidgets import (
    QApplication, QFileDialog, QLabel, QMainWindow, QDockWidget, QWidget, QVBoxLayout, QMessageBox, QToolBar
)
from PySide6.QtCore import Qt, QRectF, QSize, QPointF
from PySide6.QtGui import QAction, QPainter, QPen, QPixmap, QImage, QWheelEvent, QMouseEvent, QKeyEvent, QActionGroup

from .ui.car_detection_panel import CarDetectionPanel
from .detection.config import PipelineConfig
from .detection.template import detect_cars_by_template, to_gray_clahe
from .detection.yolo import detect_cars_by_yolo, YoloNotAvailableError
from .detection.tiler import run_tiled
from .detection.types import Detection
from .detection.export import save_detections_csv, save_detections_coco


def _cv_to_qimage(bgr: np.ndarray) -> QImage:
    if bgr.ndim == 2:
        h, w = bgr.shape
        return QImage(bgr.data, w, h, w, QImage.Format_Grayscale8).copy()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class ImageView(QWidget):
    """Image viewer with zoom/pan and detection overlays."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pix: Optional[QPixmap] = None
        self._dets: List[Detection] = []
        self._show_scores: bool = True
        self._line_w: int = 2

        # Navigation
        self._zoom: float = 1.0
        self._pan: QPointF = QPointF(0.0, 0.0)
        self._panning: bool = False
        self._last_mouse: QPointF = QPointF()
        self.setMinimumSize(QSize(400, 300))
        self.setFocusPolicy(Qt.StrongFocus)

        # ROI
        self._roi: Optional[QRectF] = None
        self._roi_dragging: bool = False
        self._roi_anchor: Optional[QPointF] = None

        # Tools
        self._tool: Optional[str] = None
        self._tool_active = False
        self._tool_anchor = None
        self._tool_point = None

    # ---- API ---------------------------------------------------------------

    def set_tool(self, mode: str) -> None:
        self._tool = mode
        # при смене инструмента — не мешаем ROI/детекции
        self._tool_active = False
        self._tool_anchor = None
        self._tool_point = None
        self.update()

    def set_image(self, path: Path | None = None, bgr: Optional[np.ndarray] = None) -> None:
        if bgr is None:
            if path is None:
                return
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                self._pix = None
                self.update()
                return
        qi = _cv_to_qimage(bgr)
        self._pix = QPixmap.fromImage(qi)
        self._dets = []
        self._roi = None
        self.reset_view()
        self.update()

    def set_detections(self, dets: List[Detection]) -> None:
        self._dets = dets or []
        self.update()

    def set_overlay_options(self, show_scores: bool, line_width: int) -> None:
        self._show_scores = bool(show_scores)
        self._line_w = max(1, int(line_width))
        self.update()

    def clear_roi(self) -> None:
        self._roi = None
        self.update()

    def get_roi_xyxy(self) -> Optional[Tuple[int, int, int, int]]:
        if self._pix is None or self._roi is None:
            return None
        w, h = self._pix.width(), self._pix.height()
        x1 = int(max(0, min(self._roi.left(), self._roi.right())))
        y1 = int(max(0, min(self._roi.top(), self._roi.bottom())))
        x2 = int(min(w, max(self._roi.left(), self._roi.right())))
        y2 = int(min(h, max(self._roi.top(), self._roi.bottom())))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return x1, y1, x2, y2

    # ---- Navigation ----
    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def _fit_scale_origin(self) -> Tuple[float, float, float]:
        """Вернёт (scale, x0, y0): image→view: view = origin + scale * image."""
        assert self._pix is not None
        w, h = self._pix.width(), self._pix.height()
        fit = min(self.width() / w, self.height() / h)
        scale = float(fit * self._zoom)
        x0 = (self.width() - w * scale) * 0.5 + self._pan.x()
        y0 = (self.height() - h * scale) * 0.5 + self._pan.y()
        return scale, x0, y0

    def _view_to_image(self, pt: QPointF) -> QPointF:
        if self._pix is None:
            return QPointF(0, 0)
        scale, x0, y0 = self._fit_scale_origin()
        x = (pt.x() - x0) / (scale if scale != 0 else 1.0)
        y = (pt.y() - y0) / (scale if scale != 0 else 1.0)
        # без жёсткого clamp — клипнем при чтении ROI
        return QPointF(x, y)

    def _image_to_view_rect(self, r: QRectF) -> QRectF:
        scale, x0, y0 = self._fit_scale_origin()
        return QRectF(x0 + r.x() * scale, y0 + r.y() * scale, r.width() * scale, r.height() * scale)

    # ---- Events ----
    def wheelEvent(self, ev: QWheelEvent) -> None:  # type: ignore[override]
        if not self._pix:
            return
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        factor = 1.1 if delta > 0 else 1 / 1.1
        old_zoom = self._zoom
        self._zoom = float(np.clip(self._zoom * factor, 0.2, 10.0))
        pos = ev.position()
        self._pan = pos - (self._zoom / old_zoom) * (pos - self._pan)
        self.update()

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # type: ignore[override]
        if ev.button() == Qt.MiddleButton:
            self._panning = True
            self._last_mouse = ev.position()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if ev.button() == Qt.LeftButton and self._pix is not None:
            img_pt = self._view_to_image(ev.position())

            if self._tool == "cursor":
                # поведение как раньше: ЛКМ — ROI
                self._roi_dragging = True
                self._roi_anchor = img_pt
                self._roi = QRectF(self._roi_anchor, self._roi_anchor)
                self.update()
                return
            else:
                # режим инструментов (line/rect/circle)
                self._tool_active = True
                self._tool_anchor = img_pt
                self._tool_point = img_pt
                self.update()
                return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:  # type: ignore[override]
        if self._panning:
            d = ev.position() - self._last_mouse
            self._pan += d
            self._last_mouse = ev.position()
            self.update()
            return

        if self._pix is not None:
            img_pt = self._view_to_image(ev.position())

            if self._tool == "cursor":
                if self._roi_dragging and self._roi is not None and self._roi_anchor is not None:
                    x1 = min(self._roi_anchor.x(), img_pt.x())
                    y1 = min(self._roi_anchor.y(), img_pt.y())
                    x2 = max(self._roi_anchor.x(), img_pt.x())
                    y2 = max(self._roi_anchor.y(), img_pt.y())
                    self._roi = QRectF(x1, y1, x2 - x1, y2 - y1)
                    self.update()
                    return
            else:
                if self._tool_active and self._tool_anchor is not None:
                    # snap для линии при Shift
                    if self._tool == "line" and (ev.modifiers() & Qt.ShiftModifier):
                        dx = img_pt.x() - self._tool_anchor.x()
                        dy = img_pt.y() - self._tool_anchor.y()
                        if abs(dx) >= abs(dy):
                            img_pt.setY(self._tool_anchor.y())
                        else:
                            img_pt.setX(self._tool_anchor.x())
                    self._tool_point = img_pt
                    self.update()
                    return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:  # type: ignore[override]
        if ev.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return

        if ev.button() == Qt.LeftButton:
            if self._tool == "cursor":
                self._roi_dragging = False
                self._roi_anchor = None
                r = self.get_roi_xyxy()
                if r is None:
                    self._roi = None
                self.update()
                return
            else:
                self._tool_active = False
                # оставим последнюю фигуру «на экране» (как замер); можно отменить Esc
                self.update()
                return

        super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev: QKeyEvent) -> None:  # type: ignore[override]
        if ev.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self._zoom = float(np.clip(self._zoom * 1.1, 0.2, 10.0))
            self.update()
        elif ev.key() == Qt.Key_Minus:
            self._zoom = float(np.clip(self._zoom / 1.1, 0.2, 10.0))
            self.update()
        elif ev.key() in (Qt.Key_0, Qt.Key_Insert):
            self.reset_view()
        elif ev.key() == Qt.Key_Escape:
            # очистить текущий измерительный чертёж
            self._tool_active = False
            self._tool_anchor = None
            self._tool_point = None
            self.update()
        else:
            super().keyPressEvent(ev)

    def paintEvent(self, ev) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if not self._pix:
            p.setPen(Qt.white)
            p.drawText(self.rect(), Qt.AlignCenter, "Open an image (File → Open)")
            p.end()
            return

        # базовая геометрия
        w, h = self._pix.width(), self._pix.height()
        scale, x0, y0 = self._fit_scale_origin()
        target = QRectF(x0, y0, w * scale, h * scale)
        p.drawPixmap(target, self._pix, QRectF(0, 0, w, h))

        # детекции
        if self._dets:
            sx = target.width() / w
            sy = target.height() / h
            pen = QPen(Qt.green)
            pen.setWidth(self._line_w)
            p.setPen(pen)
            for d in self._dets:
                x1 = x0 + d.x1 * sx; y1 = y0 + d.y1 * sy
                x2 = x0 + d.x2 * sx; y2 = y0 + d.y2 * sy
                p.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
                if self._show_scores:
                    p.drawText(QRectF(x1, y1 - 16, 120, 14), Qt.AlignLeft | Qt.AlignVCenter, f"{d.score:.2f}")

        # ROI (жёлтым пунктиром)
        if self._pix is not None and self._roi is not None and self._roi.width() > 1 and self._roi.height() > 1:
            vr = self._image_to_view_rect(self._roi)
            pen = QPen(Qt.yellow)
            pen.setWidth(max(1, self._line_w))
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawRect(vr)

        # инструменты (оверлей измерений — белый)
        if self._tool_anchor is not None and self._tool_point is not None:
            ax, ay = self._tool_anchor.x(), self._tool_anchor.y()
            bx, by = self._tool_point.x(), self._tool_point.y()
            pen = QPen(Qt.white); pen.setWidth(max(1, self._line_w)); p.setPen(pen)

            def to_view(x, y):
                return QPointF(x0 + x * scale, y0 + y * scale)

            if self._tool == "line":
                A = to_view(ax, ay); B = to_view(bx, by)
                p.drawLine(A, B)
                dx = bx - ax; dy = by - ay
                length = (dx**2 + dy**2) ** 0.5
                angle = np.degrees(np.arctan2(dy, dx)) if length > 1e-6 else 0.0
                label = f"{length:.1f}px  @{angle:.1f}°"
                p.drawText(QRectF((A.x()+B.x())/2 + 6, (A.y()+B.y())/2 - 18, 160, 16),
                           Qt.AlignLeft | Qt.AlignVCenter, label)

            elif self._tool == "rect":
                x1, y1 = min(ax, bx), min(ay, by)
                x2, y2 = max(ax, bx), max(ay, by)
                vr = self._image_to_view_rect(QRectF(x1, y1, x2 - x1, y2 - y1))
                p.drawRect(vr)
                ww = x2 - x1; hh = y2 - y1
                if hh < 1e-6: aspect = float("inf")
                else: aspect = ww / hh
                label = f"{ww:.1f}×{hh:.1f}px  a={aspect:.2f}"
                p.drawText(QRectF(vr.x()+6, vr.y()-18, 220, 16), Qt.AlignLeft | Qt.AlignVCenter, label)

            elif self._tool == "circle":
                # трактуем якорь как центр, текущую точку как радиус-вектор
                cx, cy = ax, ay
                r = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
                # нарисуем окружность через bounding-box
                vr = self._image_to_view_rect(QRectF(cx - r, cy - r, 2 * r, 2 * r))
                p.drawEllipse(vr)
                label = f"R={r:.1f}px  D={2*r:.1f}px"
                p.drawText(QRectF(vr.x()+6, vr.y()-18, 220, 16), Qt.AlignLeft | Qt.AlignVCenter, label)

        p.end()


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Car Counter")
        self.resize(1280, 840)

        # Central image view
        self.image_view = ImageView(self)
        self.setCentralWidget(self.image_view)
        self._last_bgr: Optional[np.ndarray] = None

        # Right dock
        self._create_right_dock()

        # Menu
        self._create_menu()

        # Toolbar
        self._create_toolbar()

        self.statusBar().showMessage("Ready")

        self._limit_to_roi: bool = False

    def _create_right_dock(self) -> None:
        dock = QDockWidget("Car Detection", self)
        dock.setObjectName("CarDetectionDock")
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)

        panel = CarDetectionPanel(self)
        panel.detectRequested.connect(self._on_detect)
        # НОВОЕ: связываем оверлей/экспорт
        panel.overlayParamsChanged.connect(self._on_overlay_changed)
        panel.exportCsvClicked.connect(self._on_export_csv)
        panel.exportCocoClicked.connect(self._on_export_coco)
        # ROI signals
        panel.limitToRoiChanged.connect(self._on_limit_roi_changed)
        panel.clearRoiClicked.connect(self._on_clear_roi)
        panel.exportRoiClicked.connect(self._on_export_roi)

        dock.setWidget(panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.panel = panel

    def _on_overlay_changed(self, params: dict) -> None:
        self.image_view.set_overlay_options(
            show_scores=params.get("show_scores", True),
            line_width=params.get("line_width", 2),
        )

    def _on_export_csv(self) -> None:
        if not self._last_bgr or not getattr(self.image_view, "_dets", []):
            QMessageBox.information(self, "Export CSV", "No detections to export.");
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
        if not path:
            return
        save_detections_csv(path, self.image_view._dets)  # простота > инкапсуляция
        self.statusBar().showMessage(f"CSV saved: {path}")

    def _on_export_coco(self) -> None:
        if not self._last_bgr or not getattr(self.image_view, "_dets", []):
            QMessageBox.information(self, "Export COCO", "No detections to export.");
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save COCO", "", "JSON (*.json)")
        if not path:
            return
        h, w = self._last_bgr.shape[:2]
        save_detections_coco(path, self.image_view._dets, (w, h), categories={1: "car"})
        self.statusBar().showMessage(f"COCO saved: {path}")

    def _on_limit_roi_changed(self, checked: bool) -> None:
        self._limit_to_roi = bool(checked)

    def _on_clear_roi(self) -> None:
        self.image_view.clear_roi()

    def _on_export_roi(self) -> None:
        if self._last_bgr is None:
            QMessageBox.information(self, "Export ROI", "Open an image first.");
            return
        roi = self.image_view.get_roi_xyxy()
        if roi is None:
            QMessageBox.information(self, "Export ROI", "No ROI drawn.");
            return
        x1, y1, x2, y2 = roi
        crop = self._last_bgr[y1:y2, x1:x2, :]
        path, _ = QFileDialog.getSaveFileName(self, "Save ROI image", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path:
            return
        ok = cv2.imwrite(path, crop)
        if ok:
            self.statusBar().showMessage(f"ROI saved: {path}")
        else:
            QMessageBox.warning(self, "Export ROI", "Failed to save ROI image.")

    def _create_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("Open image…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_image_dialog)
        file_menu.addAction(open_action)

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _create_toolbar(self) -> None:
        tb = QToolBar("Tools", self)
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)

        group = QActionGroup(tb)
        group.setExclusive(True)

        act_cursor = QAction("Cursor", self)
        act_cursor.setCheckable(True)
        act_cursor.setChecked(True)
        act_cursor.setData("cursor")
        act_line = QAction("Line", self)
        act_line.setCheckable(True)
        act_line.setData("line")
        act_rect = QAction("Rect", self)
        act_rect.setCheckable(True)
        act_rect.setData("rect")
        act_circle = QAction("Circle", self)
        act_circle.setCheckable(True)
        act_circle.setData("circle")

        for a in (act_cursor, act_line, act_rect, act_circle):
            group.addAction(a);
            tb.addAction(a)

        def on_tool_changed(a: QAction):
            mode = str(a.data())
            self.image_view.set_tool(mode)
            self.statusBar().showMessage(f"Tool: {mode}")

        group.triggered.connect(on_tool_changed)

    def _open_image_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if not path_str:
            return
        bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if bgr is None:
            QMessageBox.warning(self, "Open", "Failed to load image.")
            return
        self._last_bgr = bgr
        self.image_view.set_image(bgr=bgr)

    # ---- detection runner ----
    def _on_detect(self, pc: PipelineConfig) -> None:
        if self._last_bgr is None:
            QMessageBox.information(self, "Detect", "Open an image first.");
            return

        base_bgr = self._last_bgr
        offset_x = 0
        offset_y = 0

        # Если нужно — ограничиваемся ROI
        if self._limit_to_roi:
            roi = self.image_view.get_roi_xyxy()
            if roi is None:
                QMessageBox.information(self, "Detect", "Draw ROI or uncheck 'Limit detect to ROI'.");
                return
            x1, y1, x2, y2 = roi
            base_bgr = base_bgr[y1:y2, x1:x2, :].copy()
            offset_x, offset_y = x1, y1

        dets: List[Detection] = []
        try:
            if pc.backend == "template":
                tpl = pc.template.to_params()
                if pc.tiler.enabled:
                    # ВАЖНО: считать статистики для z-score по текущей области (всему изображению или ROI)
                    gray_full = to_gray_clahe(base_bgr, tpl.clahe_clip_limit, tpl.clahe_tile_grid)
                    stats = (float(gray_full.mean()), float(gray_full.std()))
                    dets = run_tiled(
                        base_bgr,
                        detector=lambda t: detect_cars_by_template(t, tpl, zscore_stats=stats),
                        params=pc.tiler.to_params(),
                    )
                else:
                    dets = detect_cars_by_template(base_bgr, tpl)
            else:
                from .detection.yolo import detect_cars_by_yolo
                if pc.tiler.enabled:
                    dets = run_tiled(base_bgr, detector=lambda t: detect_cars_by_yolo(t, pc.yolo.to_params()),
                                     params=pc.tiler.to_params())
                else:
                    dets = detect_cars_by_yolo(base_bgr, pc.yolo.to_params())
        except YoloNotAvailableError as e:
            QMessageBox.warning(self, "YOLO", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "Detect error", f"{type(e).__name__}: {e}")
            return

        # Смещение координат, если работали по ROI
        if offset_x or offset_y:
            for d in dets:
                d.x1 += offset_x
                d.x2 += offset_x
                d.y1 += offset_y
                d.y2 += offset_y

        self.image_view.set_detections(dets)
        mode = "ROI" if self._limit_to_roi else "full"
        self.statusBar().showMessage(
            f"Detections: {len(dets)} (backend={pc.backend}, mode={mode}, tiler={pc.tiler.enabled})")

def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
