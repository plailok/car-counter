from __future__ import annotations

from typing import Optional, Tuple
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox, QLabel, QDoubleSpinBox,
    QSpinBox, QCheckBox, QLineEdit, QPushButton, QFileDialog
)

from car_detect.detection.config import PipelineConfig, TemplateConfig, YOLOConfig, TilerConfig


class CarDetectionPanel(QWidget):
    """Right-side control panel: backend params, tiling, presets, and Detect button."""

    detectRequested = Signal(object)  # emits PipelineConfig
    overlayParamsChanged = Signal(object)
    exportCsvClicked = Signal()
    exportCocoClicked = Signal()

    limitToRoiChanged = Signal(bool)
    clearRoiClicked = Signal()
    exportRoiClicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Backend selector
        gb_backend = QGroupBox("Backend")
        l_backend = QHBoxLayout(gb_backend)
        self.cb_backend = QComboBox()
        self.cb_backend.addItems(["template", "yolo"])
        l_backend.addWidget(QLabel("Type:"))
        l_backend.addWidget(self.cb_backend, 1)
        root.addWidget(gb_backend)

        # ---- Template group ----
        gb_tpl = QGroupBox("Template Matching")
        l_tpl = QVBoxLayout(gb_tpl)

        row1 = QHBoxLayout()
        self.sp_tpl_thr = QDoubleSpinBox()
        self.sp_tpl_thr.setRange(0.0, 1.0)
        self.sp_tpl_thr.setSingleStep(0.01)
        self.sp_tpl_thr.setValue(0.70)
        self.sp_tpl_thr.setToolTip("Score threshold on TM_CCOEFF_NORMED map. Higher → fewer candidates.")
        row1.addWidget(QLabel("Threshold"))
        row1.addWidget(self.sp_tpl_thr)

        self.sp_tpl_nms = QDoubleSpinBox()
        self.sp_tpl_nms.setRange(0.0, 1.0)
        self.sp_tpl_nms.setSingleStep(0.05)
        self.sp_tpl_nms.setValue(0.30)
        self.sp_tpl_nms.setToolTip("IoU threshold for greedy NMS (template candidates).")
        row1.addWidget(QLabel("NMS IoU"))
        row1.addWidget(self.sp_tpl_nms)
        l_tpl.addLayout(row1)

        row2 = QHBoxLayout()
        self.ed_tpl_scales = QLineEdit("0.85,0.9,0.95,1.0,1.05,1.1,1.15")
        self.ed_tpl_scales.setToolTip("Comma-separated scales for template size around 120×50 px.")
        row2.addWidget(QLabel("Scales"))
        row2.addWidget(self.ed_tpl_scales)
        l_tpl.addLayout(row2)

        row3 = QHBoxLayout()
        self.sp_tpl_rpd = QDoubleSpinBox()
        self.sp_tpl_rpd.setRange(0.1, 1.0)
        self.sp_tpl_rpd.setSingleStep(0.05)
        self.sp_tpl_rpd.setValue(0.40)
        self.sp_tpl_rpd.setToolTip("Relative peak distance: fraction of min(w,h) to suppress nearby peaks.")
        row3.addWidget(QLabel("Rel. peak dist"))
        row3.addWidget(self.sp_tpl_rpd)

        self.sp_tpl_pmd = QSpinBox()
        self.sp_tpl_pmd.setRange(1, 50)
        self.sp_tpl_pmd.setValue(3)
        self.sp_tpl_pmd.setToolTip("Absolute floor for peak distance in pixels.")
        row3.addWidget(QLabel("Min dist (px)"))
        row3.addWidget(self.sp_tpl_pmd)
        l_tpl.addLayout(row3)

        row4 = QHBoxLayout()
        self.sp_tpl_clip = QDoubleSpinBox()
        self.sp_tpl_clip.setRange(0.1, 10.0)
        self.sp_tpl_clip.setSingleStep(0.1)
        self.sp_tpl_clip.setValue(2.0)
        self.sp_tpl_grid_x = QSpinBox()
        self.sp_tpl_grid_x.setRange(2, 64)
        self.sp_tpl_grid_x.setValue(8)
        self.sp_tpl_grid_y = QSpinBox()
        self.sp_tpl_grid_y.setRange(2, 64)
        self.sp_tpl_grid_y.setValue(8)
        self.sp_tpl_maxk = QSpinBox()
        self.sp_tpl_maxk.setRange(10, 5000)
        self.sp_tpl_maxk.setValue(400)
        row4.addWidget(QLabel("CLAHE clip"))
        row4.addWidget(self.sp_tpl_clip)
        row4.addWidget(QLabel("Grid X"))
        row4.addWidget(self.sp_tpl_grid_x)
        row4.addWidget(QLabel("Grid Y"))
        row4.addWidget(self.sp_tpl_grid_y)
        row4.addWidget(QLabel("Max peaks/tpl"))
        row4.addWidget(self.sp_tpl_maxk)
        l_tpl.addLayout(row4)

        row5 = QHBoxLayout()
        self.sp_tpl_minstd = QDoubleSpinBox()
        self.sp_tpl_minstd.setRange(0.0, 100.0)
        self.sp_tpl_minstd.setValue(12.0)
        self.sp_tpl_edge_thr = QDoubleSpinBox()
        self.sp_tpl_edge_thr.setRange(0.0, 255.0)
        self.sp_tpl_edge_thr.setValue(35.0)
        self.sp_tpl_edge_den = QDoubleSpinBox()
        self.sp_tpl_edge_den.setRange(0.0, 1.0)
        self.sp_tpl_edge_den.setSingleStep(0.005)
        self.sp_tpl_edge_den.setValue(0.02)
        self.sp_tpl_minstd.setToolTip(
            "Reject boxes with too uniform intensity (σ < min_std). Empty slots are very flat.")
        self.sp_tpl_edge_thr.setToolTip("Sobel gradient magnitude threshold for counting 'edge' pixels.")
        self.sp_tpl_edge_den.setToolTip(
            "Minimum fraction of pixels above edge threshold. Cars have richer edges than empty slabs.")

        row5.addWidget(QLabel("Min σ")); row5.addWidget(self.sp_tpl_minstd)
        row5.addWidget(QLabel("Edge thr")); row5.addWidget(self.sp_tpl_edge_thr)
        row5.addWidget(QLabel("Min edge dens")); row5.addWidget(self.sp_tpl_edge_den)
        l_tpl.addLayout(row5)

        root.addWidget(gb_tpl)

        # ---- YOLO group ----
        gb_yolo = QGroupBox("YOLO")
        l_yolo = QVBoxLayout(gb_yolo)

        ry1 = QHBoxLayout()
        self.ed_yolo_weights = QLineEdit("yolov8n.pt")
        self.sp_yolo_conf = QDoubleSpinBox()
        self.sp_yolo_conf.setRange(0.01, 0.99)
        self.sp_yolo_conf.setValue(0.25)
        self.sp_yolo_iou = QDoubleSpinBox()
        self.sp_yolo_iou.setRange(0.01, 0.99)
        self.sp_yolo_iou.setValue(0.50)
        self.sp_yolo_imgsz = QSpinBox()
        self.sp_yolo_imgsz.setRange(320, 2048)
        self.sp_yolo_imgsz.setSingleStep(64)
        self.sp_yolo_imgsz.setValue(1280)
        ry1.addWidget(QLabel("Weights"))
        ry1.addWidget(self.ed_yolo_weights)
        ry1.addWidget(QLabel("conf"))
        ry1.addWidget(self.sp_yolo_conf)
        ry1.addWidget(QLabel("iou"))
        ry1.addWidget(self.sp_yolo_iou)
        ry1.addWidget(QLabel("imgsz"))
        ry1.addWidget(self.sp_yolo_imgsz)
        l_yolo.addLayout(ry1)

        ry2 = QHBoxLayout()
        self.sp_yolo_wmin = QDoubleSpinBox()
        self.sp_yolo_wmin.setRange(1, 4096)
        self.sp_yolo_wmin.setValue(80)
        self.sp_yolo_wmax = QDoubleSpinBox()
        self.sp_yolo_wmax.setRange(1, 4096)
        self.sp_yolo_wmax.setValue(170)
        self.sp_yolo_hmin = QDoubleSpinBox()
        self.sp_yolo_hmin.setRange(1, 4096)
        self.sp_yolo_hmin.setValue(35)
        self.sp_yolo_hmax = QDoubleSpinBox()
        self.sp_yolo_hmax.setRange(1, 4096)
        self.sp_yolo_hmax.setValue(90)
        self.sp_yolo_aspect = QDoubleSpinBox()
        self.sp_yolo_aspect.setRange(0.2, 10)
        self.sp_yolo_aspect.setValue(2.4)
        self.sp_yolo_tol = QDoubleSpinBox()
        self.sp_yolo_tol.setRange(0.0, 5.0)
        self.sp_yolo_tol.setSingleStep(0.1)
        self.sp_yolo_tol.setValue(0.8)
        ry2.addWidget(QLabel("w[min,max]"))
        ry2.addWidget(self.sp_yolo_wmin)
        ry2.addWidget(self.sp_yolo_wmax)
        ry2.addWidget(QLabel("h[min,max]"))
        ry2.addWidget(self.sp_yolo_hmin)
        ry2.addWidget(self.sp_yolo_hmax)
        ry2.addWidget(QLabel("aspect±tol"))
        ry2.addWidget(self.sp_yolo_aspect)
        ry2.addWidget(self.sp_yolo_tol)
        l_yolo.addLayout(ry2)

        root.addWidget(gb_yolo)

        # ---- Tiler group ----
        gb_t = QGroupBox("Tiling / Merge")
        lt = QHBoxLayout(gb_t)
        self.cb_tiler_enabled = QCheckBox("Enable tiling")
        self.cb_tiler_enabled.setChecked(False)
        self.sp_tile_w = QSpinBox()
        self.sp_tile_w.setRange(128, 4096)
        self.sp_tile_w.setValue(640)
        self.sp_tile_h = QSpinBox()
        self.sp_tile_h.setRange(128, 4096)
        self.sp_tile_h.setValue(640)
        self.sp_overlap = QSpinBox()
        self.sp_overlap.setRange(0, 1024)
        self.sp_overlap.setValue(96)
        self.sp_tnms = QDoubleSpinBox()
        self.sp_tnms.setRange(0.0, 1.0)
        self.sp_tnms.setValue(0.40)
        lt.addWidget(self.cb_tiler_enabled)
        lt.addWidget(QLabel("tile WxH"))
        lt.addWidget(self.sp_tile_w)
        lt.addWidget(self.sp_tile_h)
        lt.addWidget(QLabel("overlap"))
        lt.addWidget(self.sp_overlap)
        lt.addWidget(QLabel("global NMS IoU"))
        lt.addWidget(self.sp_tnms)
        root.addWidget(gb_t)

        # ---- ROI (NEW) ----
        gb_roi = QGroupBox("ROI (Region of Interest)")
        lroi = QHBoxLayout(gb_roi)
        self.cb_use_roi = QCheckBox("Limit detect to ROI")
        self.cb_use_roi.setToolTip("If checked, detection runs only inside the drawn ROI.")
        self.btn_clear_roi = QPushButton("Clear ROI")
        self.btn_export_roi = QPushButton("Export ROI image")
        lroi.addWidget(self.cb_use_roi)
        lroi.addStretch(1)
        lroi.addWidget(self.btn_clear_roi)
        lroi.addWidget(self.btn_export_roi)
        root.addWidget(gb_roi)

        # ---- Overlay/Export ----
        gb_ov = QGroupBox("Overlay / Export")
        lov = QHBoxLayout(gb_ov)
        self.cb_show_scores = QCheckBox("Show scores")
        self.cb_show_scores.setChecked(True)
        self.sp_line_w = QSpinBox()
        self.sp_line_w.setRange(1, 10)
        self.sp_line_w.setValue(2)
        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_coco = QPushButton("Export COCO")
        lov.addWidget(self.cb_show_scores)
        lov.addWidget(QLabel("Line width"))
        lov.addWidget(self.sp_line_w)
        lov.addStretch(1)
        lov.addWidget(self.btn_export_csv)
        lov.addWidget(self.btn_export_coco)
        root.addWidget(gb_ov)

        # Buttons
        row_btn = QHBoxLayout()
        self.btn_detect = QPushButton("Detect")
        self.btn_save = QPushButton("Save preset")
        self.btn_load = QPushButton("Load preset")
        row_btn.addWidget(self.btn_detect, 1)
        row_btn.addWidget(self.btn_save)
        row_btn.addWidget(self.btn_load)
        root.addLayout(row_btn)

        root.addStretch()

        # Handlers
        # Export/Import
        self.btn_detect.clicked.connect(self._emit_detect)
        self.btn_save.clicked.connect(self._save_preset)
        self.btn_load.clicked.connect(self._load_preset)
        # Overlay
        self.cb_show_scores.stateChanged.connect(self._emit_overlay)
        self.sp_line_w.valueChanged.connect(self._emit_overlay)
        self.btn_export_csv.clicked.connect(self.exportCsvClicked.emit)
        self.btn_export_coco.clicked.connect(self.exportCocoClicked.emit)
        # ROI

    # --- helpers ---

    def _parse_scales(self) -> Tuple[float, ...]:
        txt = self.ed_tpl_scales.text().strip()
        vals = []
        for t in txt.split(","):
            t = t.strip()
            if not t:
                continue
            try:
                vals.append(float(t))
            except ValueError:
                pass
        return tuple(vals) if vals else (1.0,)

    def overlay_params(self) -> dict:
        return {
            "show_scores": bool(self.cb_show_scores.isChecked()),
            "line_width": int(self.sp_line_w.value()),
        }

    def _emit_overlay(self) -> None:
        self.overlayParamsChanged.emit(self.overlay_params())

    def _build_config(self) -> PipelineConfig:
        pc = PipelineConfig()
        pc.backend = self.cb_backend.currentText()

        # template
        pc.template = TemplateConfig(
            threshold=self.sp_tpl_thr.value(),
            scales=self._parse_scales(),
            nms_iou=self.sp_tpl_nms.value(),
            clahe_clip_limit=self.sp_tpl_clip.value(),
            clahe_tile_grid=(self.sp_tpl_grid_x.value(), self.sp_tpl_grid_y.value()),
            peak_min_distance=self.sp_tpl_pmd.value(),
            relative_peak_distance=self.sp_tpl_rpd.value(),
            max_peaks_per_template=self.sp_tpl_maxk.value(),
            min_std=self.sp_tpl_minstd.value(),
            edge_grad_thresh=self.sp_tpl_edge_thr.value(),
            min_edge_density=self.sp_tpl_edge_den.value(),
        )

        # yolo
        pc.yolo = YOLOConfig(
            weights=self.ed_yolo_weights.text().strip(),
            conf=self.sp_yolo_conf.value(),
            iou=self.sp_yolo_iou.value(),
            imgsz=self.sp_yolo_imgsz.value(),
            width_range=(self.sp_yolo_wmin.value(), self.sp_yolo_wmax.value()),
            height_range=(self.sp_yolo_hmin.value(), self.sp_yolo_hmax.value()),
            aspect=self.sp_yolo_aspect.value(),
            aspect_tol=self.sp_yolo_tol.value(),
        )

        # tiler
        pc.tiler = TilerConfig(
            enabled=self.cb_tiler_enabled.isChecked(),
            tile_w=self.sp_tile_w.value(),
            tile_h=self.sp_tile_h.value(),
            overlap=self.sp_overlap.value(),
            nms_iou=self.sp_tnms.value(),
        )
        return pc

    def _emit_detect(self) -> None:
        self.detectRequested.emit(self._build_config())

    def _save_preset(self) -> None:
        pc = self._build_config()
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", "", "JSON (*.json)")
        if path:
            pc.save_json(path)

    def _load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load preset", "", "JSON (*.json)")
        if not path:
            return
        from car_detect.detection.config import PipelineConfig
        pc = PipelineConfig.load_json(path)
        # Apply loaded config to widgets (minimal impl)
        self.cb_backend.setCurrentText(pc.backend)
        self.sp_tpl_thr.setValue(pc.template.threshold)
        self.ed_tpl_scales.setText(",".join(str(s) for s in pc.template.scales))
        self.sp_tpl_nms.setValue(pc.template.nms_iou)
        self.sp_tpl_clip.setValue(pc.template.clahe_clip_limit)
        self.sp_tpl_grid_x.setValue(pc.template.clahe_tile_grid[0])
        self.sp_tpl_grid_y.setValue(pc.template.clahe_tile_grid[1])
        self.sp_tpl_pmd.setValue(pc.template.peak_min_distance)
        self.sp_tpl_rpd.setValue(pc.template.relative_peak_distance)
        self.sp_tpl_maxk.setValue(pc.template.max_peaks_per_template)
        self.sp_tpl_minstd.setValue(pc.template.min_std)
        self.sp_tpl_edge_thr.setValue(pc.template.edge_grad_thresh)
        self.sp_tpl_edge_den.setValue(pc.template.min_edge_density)

        self.ed_yolo_weights.setText(pc.yolo.weights)
        self.sp_yolo_conf.setValue(pc.yolo.conf)
        self.sp_yolo_iou.setValue(pc.yolo.iou)
        self.sp_yolo_imgsz.setValue(pc.yolo.imgsz)
        self.sp_yolo_wmin.setValue(pc.yolo.width_range[0])
        self.sp_yolo_wmax.setValue(pc.yolo.width_range[1])
        self.sp_yolo_hmin.setValue(pc.yolo.height_range[0])
        self.sp_yolo_hmax.setValue(pc.yolo.height_range[1])
        self.sp_yolo_aspect.setValue(pc.yolo.aspect)
        self.sp_yolo_tol.setValue(pc.yolo.aspect_tol)

        self.cb_tiler_enabled.setChecked(pc.tiler.enabled)
        self.sp_tile_w.setValue(pc.tiler.tile_w)
        self.sp_tile_h.setValue(pc.tiler.tile_h)
        self.sp_overlap.setValue(pc.tiler.overlap)
        self.sp_tnms.setValue(pc.tiler.nms_iou)
