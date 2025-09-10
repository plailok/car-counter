from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QListWidget,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QComboBox, QTableWidget,
    QTableWidgetItem, QWidget, QMessageBox
)

from car_detect.autotune.search import run_autotune, SearchSpec, ImageTask, grid_for_mode, TrialResult
from car_detect.detection.types import Detection
from car_detect.detection.yolo import YoloNotAvailableError


def _cv_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


@dataclass
class YoloWeightOption:
    name: str
    path: str


class AutoTuneDialog(QDialog):
    """Окно автоподбора параметров YOLO. Сравнение сетки конфигураций по нескольким изображениям."""

    applyParams = Signal(float, float, int)  # conf, iou, imgsz (для переноса в основную панель)

    def __init__(self, yolo_weights: List[YoloWeightOption], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto-Tuner (YOLO)")
        self.resize(1200, 800)

        self._weights = yolo_weights
        self._images: List[ImageTask] = []
        self._results: List[TrialResult] = []
        self._preview_img: Optional[np.ndarray] = None
        self._preview_dets: List[Detection] = []

        root = QVBoxLayout(self)

        # Top: images list + add/remove
        gb_imgs = QGroupBox("Images")
        limgs = QHBoxLayout(gb_imgs)
        self.list_imgs = QListWidget()
        btn_add = QPushButton("Add images…")
        btn_clr = QPushButton("Clear")
        limgs.addWidget(self.list_imgs, 1)
        lbtns = QVBoxLayout()
        lbtns.addWidget(btn_add)
        lbtns.addWidget(btn_clr)
        lbtns.addStretch(1)
        limgs.addLayout(lbtns)
        root.addWidget(gb_imgs)

        # Middle: params
        gb_params = QGroupBox("Expected car size / Search mode / Weights")
        f = QFormLayout(gb_params)

        self.sp_wmin = QDoubleSpinBox(); self.sp_wmin.setRange(1, 4096); self.sp_wmin.setValue(70)
        self.sp_wmax = QDoubleSpinBox(); self.sp_wmax.setRange(1, 4096); self.sp_wmax.setValue(150)
        self.sp_hmin = QDoubleSpinBox(); self.sp_hmin.setRange(1, 4096); self.sp_hmin.setValue(30)
        self.sp_hmax = QDoubleSpinBox(); self.sp_hmax.setRange(1, 4096); self.sp_hmax.setValue(70)
        self.sp_aspect = QDoubleSpinBox(); self.sp_aspect.setRange(0.2, 10); self.sp_aspect.setValue(2.4)
        self.sp_atol = QDoubleSpinBox(); self.sp_atol.setRange(0.0, 5.0); self.sp_atol.setSingleStep(0.1); self.sp_atol.setValue(0.8)

        self.cb_mode = QComboBox(); self.cb_mode.addItems(["Quick", "Accurate", "Very accurate"])
        self.cb_weights = QComboBox(); [self.cb_weights.addItem(w.name) for w in yolo_weights]

        f.addRow("w[min,max] (px)", self._row(self.sp_wmin, self.sp_wmax))
        f.addRow("h[min,max] (px)", self._row(self.sp_hmin, self.sp_hmax))
        f.addRow("aspect ± tol", self._row(self.sp_aspect, self.sp_atol))
        f.addRow("Mode", self.cb_mode)
        f.addRow("Weights", self.cb_weights)
        root.addWidget(gb_params)

        # Run / table
        row_run = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_apply = QPushButton("Apply to main panel")
        row_run.addWidget(self.btn_run)
        row_run.addStretch(1)
        row_run.addWidget(self.btn_apply)
        root.addLayout(row_run)

        # Table
        self.tbl = QTableWidget(0, 9)
        self.tbl.setHorizontalHeaderLabels(["Image", "conf", "iou", "imgsz", "n", "pass%", "avg score", "time, ms", "preview"])
        self.tbl.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        root.addWidget(self.tbl, 2)

        # Preview
        self.lbl_preview = QLabel("Preview"); self.lbl_preview.setMinimumSize(QSize(400, 260))
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        root.addWidget(self.lbl_preview, 1)

        # handlers
        btn_add.clicked.connect(self._on_add)
        btn_clr.clicked.connect(self._on_clear)
        self.btn_run.clicked.connect(self._on_run)
        self.tbl.itemSelectionChanged.connect(self._on_select_row)
        self.btn_apply.clicked.connect(self._on_apply)

    def _row(self, a: QWidget, b: QWidget) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(a); lay.addWidget(b); lay.addStretch(1)
        return w

    def _on_add(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add images", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        for fp in files:
            p = Path(fp)
            self._images.append(ImageTask(path=p))
            self.list_imgs.addItem(str(p.name))

    def _on_clear(self) -> None:
        self._images.clear()
        self._results.clear()
        self.list_imgs.clear()
        self.tbl.setRowCount(0)
        self.lbl_preview.setText("Preview")
        self._preview_img = None; self._preview_dets = []

    def _spec(self) -> SearchSpec:
        mode = self.cb_mode.currentText()
        confs, ious, sizes = grid_for_mode(mode)
        w = self._weights[self.cb_weights.currentIndex()].path
        return SearchSpec(
            weights_path=w,
            conf_list=confs,
            iou_list=ious,
            imgsz_list=sizes,
            width_range=(self.sp_wmin.value(), self.sp_wmax.value()),
            height_range=(self.sp_hmin.value(), self.sp_hmax.value()),
            aspect=self.sp_aspect.value(),
            aspect_tol=self.sp_atol.value(),
        )

    def _on_run(self) -> None:
        if len(self._images) < 2:
            QMessageBox.information(self, "Auto-Tuner", "Please add at least two images.");
            return

        self.btn_run.setEnabled(False)
        self.tbl.setRowCount(0)
        self.lbl_preview.setText("Running…")
        spec = self._spec()
        try:
            self._results = run_autotune(self._images, spec)  # теперь надёжнее
        except KeyboardInterrupt:
            QMessageBox.warning(self, "Auto-Tuner", "Interrupted by user.")
            self._results = []
        except Exception as e:
            QMessageBox.critical(self, "Auto-Tuner", f"Failed: {type(e).__name__}: {e}")
            self._results = []
        finally:
            self.btn_run.setEnabled(True)

        # fill table
        self.tbl.setRowCount(0)
        for r in self._results:
            row = self.tbl.rowCount()
            self.tbl.insertRow(row)
            self.tbl.setItem(row, 0, QTableWidgetItem(r.image_path.name))
            self.tbl.setItem(row, 1, QTableWidgetItem(f"{r.conf:.2f}"))
            self.tbl.setItem(row, 2, QTableWidgetItem(f"{r.iou:.2f}"))
            self.tbl.setItem(row, 3, QTableWidgetItem(f"{r.imgsz}"))
            self.tbl.setItem(row, 4, QTableWidgetItem(str(r.n)))
            self.tbl.setItem(row, 5, QTableWidgetItem(f"{100.0*r.pass_ratio:.1f}"))
            self.tbl.setItem(row, 6, QTableWidgetItem(f"{r.avg_score:.2f}"))
            self.tbl.setItem(row, 7, QTableWidgetItem(f"{r.runtime_ms:.0f}"))
            self.tbl.setItem(row, 8, QTableWidgetItem("View"))
        if self._results:
            self.tbl.selectRow(0)
            self._show_preview(self._results[0])

    def _on_select_row(self) -> None:
        rows = self.tbl.selectionModel().selectedRows()
        if not rows or not self._results:
            return
        idx = rows[0].row()
        if idx < 0 or idx >= len(self._results):
            return
        self._show_preview(self._results[idx])

    def _show_preview(self, r: TrialResult) -> None:
        # reload image if necessary
        from car_detect.utils.io import safe_imread_first_frame
        bgr = safe_imread_first_frame(r.image_path)
        if bgr is None:
            self.lbl_preview.setText("Failed to load image (TIFF/unsupported).")
            return

        qi = _cv_to_qimage(bgr)
        pm = QPixmap.fromImage(qi)
        # draw boxes
        from PySide6.QtGui import QPainter, QPen
        from PySide6.QtCore import QRectF
        p = QPainter(pm)
        pen = QPen(Qt.green); pen.setWidth(2); p.setPen(pen)
        for d in r.dets:
            p.drawRect(QRectF(d.x1, d.y1, d.width, d.height))
        p.end()
        self.lbl_preview.setPixmap(pm.scaled(self.lbl_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # save for Apply (not needed strictly, but might be useful)
        self._preview_img = bgr
        self._preview_dets = r.dets

    def _on_apply(self) -> None:
        rows = self.tbl.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(self, "Auto-Tuner", "Select a row first."); return
        idx = rows[0].row()
        r = self._results[idx]
        self.applyParams.emit(r.conf, r.iou, r.imgsz)
        QMessageBox.information(self, "Auto-Tuner", f"Applied: conf={r.conf:.2f}, iou={r.iou:.2f}, imgsz={r.imgsz}")