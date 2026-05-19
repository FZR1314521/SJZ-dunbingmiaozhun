"""
main_window.py - Main application window.

Layout
------
┌─────────────────────────────────────────────────────┐
│  Toolbar: [Select Window] [Unfocus] [▶/⏸] interval │
├───────────────────────────┬─────────────────────────┤
│                           │  Detection Log          │
│   Live Preview            │  ─────────────────────  │
│   (annotated frame)       │  [timestamp] label conf │
│                           │  ...                    │
├───────────────────────────┴─────────────────────────┤
│  Status bar: window name | device | FPS             │
└─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSize, Qt, pyqtSlot
from PyQt6.QtGui import QFont, QImage, QPainter
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from panopticon.capture.classifier import AimingClassifier
from panopticon.capture.manager import CaptureManager
from panopticon.capture.screenshot_capture import ScreenshotCaptureManager
from panopticon.ui.window_selector import WindowSelectorDialog
from panopticon.utils.platform import WindowInfo


class PreviewWidget(QWidget):
    """
    Live-frame preview widget.

    Accepts a QImage via set_frame() and draws it scaled to fit via
    paintEvent/QPainter.drawImage().  This avoids creating a QPixmap
    (and therefore a Win32 HBITMAP GDI object) on every frame, which
    was the cause of GDI handle exhaustion after extended runtime.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        self.setMinimumSize(320, 240)
        self._image: QImage | None = None

    def set_frame(self, image: QImage):
        self._image = image
        self.update()  # schedule a repaint; does not block

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        if self._image is not None:
            # drawImage scales the source to fit the destination rect while
            # letting Qt handle aspect-ratio alignment internally via the
            # painter transform.  No QPixmap / HBITMAP is created.
            scaled = self._image.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawImage(x, y, scaled)
        painter.end()


class DetectionLogWidget(QPlainTextEdit):
    """Read-only auto-scrolling log of detection events."""

    MAX_LINES = 500

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        # Qt trims oldest blocks automatically when the document exceeds this
        # limit, with no manual cursor loop required.
        self.setMaximumBlockCount(self.MAX_LINES)
        font = QFont("Monospace", 9)
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        self.setFont(font)
        self.setStyleSheet("background: #0d0d0d; color: #e0e0e0; border: 1px solid #333;")

    def append_detections(self, detections: list[DetectionResult]):
        if not detections:
            return
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        # Build all lines for this frame and append in one call to avoid
        # triggering a layout/repaint for every individual detection.
        lines = "\n".join(
            f"[{ts}] {d.label:<15} {d.confidence:5.1%}  box=({d.x1},{d.y1},{d.x2},{d.y2})"
            for d in detections
        )
        self.appendPlainText(lines)
        # appendPlainText already scrolls to the bottom; the scrollbar update
        # below keeps the view pinned when the user has not manually scrolled.
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def clear_log(self):
        self.clear()


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FZR数据收集工具")
        self.setMinimumSize(960, 600)

        self._manager = CaptureManager(interval_ms=100)
        self._current_window: WindowInfo | None = None
        
        # 截图管理器
        self._screenshot_mgr = ScreenshotCaptureManager(
            aimed_dir="准心数据集/瞄准状态",
            not_aimed_dir="准心数据集/未瞄准状态",
            auto_interval=1
        )
        
        # 准心状态分类器
        self._classifier = AimingClassifier()
        
        # 瞄准状态截图保存目录
        self._aimed_capture_dir = Path("准心数据集/检测到瞄准状态的截图存档")
        self._aimed_capture_dir.mkdir(parents=True, exist_ok=True)
        
        # 点击脚本路径
        self._click_script = Path("click_at_position.py")
        
        # 点击冷却机制（秒）- 避免连续快速点击
        self._click_cooldown = 2.0
        self._last_click_time = 0.0

        # FPS tracking — keep at most 1 s worth of timestamps.
        # deque with maxlen avoids rebuilding the list on every frame.
        self._frame_times: deque[float] = deque()

        self._build_ui()
        self._build_toolbar()
        self._build_statusbar()
        
        # 设置截图热键（F9单次截图，F10定时截图）
        self._screenshot_mgr.set_hotkey(aimed_key="f9", auto_key="f10")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left: live preview
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._preview = PreviewWidget()
        left_layout.addWidget(self._preview)
        self._no_target_label = QLabel('未选择窗口。\n请使用"选择窗口"开始。')
        self._no_target_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_target_label.setStyleSheet("color: #888; font-size: 14px;")
        self._no_target_label.setVisible(True)
        self._preview.setVisible(False)
        left_layout.addWidget(self._no_target_label)
        splitter.addWidget(left)

        # Right: detection log
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        log_header = QLabel("检测日志")
        log_header.setStyleSheet("font-weight: bold; padding: 2px 4px;")
        right_layout.addWidget(log_header)

        self._log = DetectionLogWidget()
        right_layout.addWidget(self._log)

        clear_btn = QPushButton("清空日志")
        clear_btn.setFixedHeight(24)
        clear_btn.clicked.connect(self._log.clear_log)
        right_layout.addWidget(clear_btn)

        splitter.addWidget(right)
        splitter.setSizes([640, 320])
        layout.addWidget(splitter)

    def _build_toolbar(self):
        tb = QToolBar("Controls")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        self._select_btn = QPushButton("选择窗口")
        self._select_btn.setFixedHeight(28)
        self._select_btn.clicked.connect(self._on_select_window)
        tb.addWidget(self._select_btn)

        self._unfocus_btn = QPushButton("取消聚焦")
        self._unfocus_btn.setFixedHeight(28)
        self._unfocus_btn.setEnabled(False)
        self._unfocus_btn.clicked.connect(self._on_unfocus)
        tb.addWidget(self._unfocus_btn)

        tb.addSeparator()

        self._toggle_btn = QPushButton("开始")
        self._toggle_btn.setFixedHeight(28)
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.clicked.connect(self._on_toggle_capture)
        tb.addWidget(self._toggle_btn)

        tb.addSeparator()
        
        # F9单次截图按钮
        self._screenshot_btn = QPushButton("截图 (F9)")
        self._screenshot_btn.setFixedHeight(28)
        self._screenshot_btn.setEnabled(False)
        self._screenshot_btn.clicked.connect(self._on_take_screenshot)
        tb.addWidget(self._screenshot_btn)
        
        # F9截图计数标签
        self._aimed_count_label = QLabel("瞄准: 0")
        self._aimed_count_label.setStyleSheet("color: #4CAF50; padding: 0 4px;")  # 绿色
        tb.addWidget(self._aimed_count_label)
        
        tb.addSeparator()
        
        # F10定时截图按钮
        self._auto_screenshot_btn = QPushButton("自动截图 (F10)")
        self._auto_screenshot_btn.setFixedHeight(28)
        self._auto_screenshot_btn.setEnabled(False)
        self._auto_screenshot_btn.clicked.connect(self._on_toggle_auto_screenshot)
        tb.addWidget(self._auto_screenshot_btn)
        
        # F10截图计数标签
        self._not_aimed_count_label = QLabel("未瞄准: 0")
        self._not_aimed_count_label.setStyleSheet("color: #FF9800; padding: 0 4px;")  # 橙色
        tb.addWidget(self._not_aimed_count_label)

        tb.addSeparator()

        interval_label = QLabel(" 间隔 (毫秒): ")
        tb.addWidget(interval_label)

        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(10, 5000)
        self._interval_spin.setValue(100)
        self._interval_spin.setSingleStep(50)
        self._interval_spin.setFixedWidth(80)
        self._interval_spin.valueChanged.connect(self._on_interval_changed)
        tb.addWidget(self._interval_spin)

        tb.addSeparator()
        
        # 裁剪大小设置
        crop_label = QLabel(" 裁剪大小: ")
        tb.addWidget(crop_label)
        
        self._crop_size_spin = QSpinBox()
        self._crop_size_spin.setRange(32, 2048)
        self._crop_size_spin.setValue(640)
        self._crop_size_spin.setSingleStep(32)
        self._crop_size_spin.setFixedWidth(80)
        self._crop_size_spin.valueChanged.connect(self._on_crop_size_changed)
        tb.addWidget(self._crop_size_spin)
        
        # 裁剪大小单位标签
        self._crop_unit_label = QLabel("px")
        self._crop_unit_label.setStyleSheet("color: #888; padding: 0 4px;")
        tb.addWidget(self._crop_unit_label)

    def _build_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

        self._status_window = QLabel("无目标")
        self._status_fps = QLabel("-- FPS")
        
        # 准心状态标签
        self._status_aiming = QLabel("状态: --")
        self._status_aiming.setStyleSheet("padding: 0 8px; color: #888;")

        for lbl in (self._status_window, self._status_fps, self._status_aiming):
            lbl.setStyleSheet("padding: 0 8px;")
            self._statusbar.addPermanentWidget(lbl)

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------

    def _on_select_window(self):
        dlg = WindowSelectorDialog(self)
        if dlg.exec() == WindowSelectorDialog.DialogCode.Accepted and dlg.selected_window:
            win = dlg.selected_window
            self._current_window = win
            self._status_window.setText(str(win))
            self._toggle_btn.setEnabled(True)
            self._unfocus_btn.setEnabled(True)
            self._screenshot_btn.setEnabled(True)  # 启用F9截图按钮
            self._auto_screenshot_btn.setEnabled(True)  # 启用F10自动截图按钮
            self._no_target_label.setVisible(False)
            self._preview.setVisible(True)

            # If already running, swap the target
            if self._manager.is_running:
                self._manager.set_window(win)
            else:
                self._toggle_btn.setText("开始")

    def _on_unfocus(self):
        self._manager.stop()
        # 停止自动截图
        if self._screenshot_mgr.is_auto_capturing:
            self._screenshot_mgr.stop_auto_capture()
        self._current_window = None
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.setText("开始")
        self._screenshot_btn.setEnabled(False)
        self._auto_screenshot_btn.setEnabled(False)
        self._auto_screenshot_btn.setText("自动截图 (F10)")
        self._aimed_count_label.setText("瞄准: 0")
        self._not_aimed_count_label.setText("未瞄准: 0")
        self._unfocus_btn.setEnabled(False)
        self._preview.setVisible(False)
        self._no_target_label.setVisible(True)
        self._status_window.setText("无目标")
        self._frame_times.clear()
        self._status_fps.setText("-- FPS")

    def _on_toggle_capture(self):
        if self._manager.is_running:
            self._manager.stop()
            self._toggle_btn.setText("开始")
        else:
            if self._current_window is None:
                return
            self._start_capture(self._current_window)
            self._toggle_btn.setText("停止")

    def _on_interval_changed(self, value: int):
        self._manager.set_interval(value)
    
    def _on_crop_size_changed(self, value: int):
        """裁剪大小变化响应"""
        # 更新检测用的裁剪大小（用于YOLO检测，提升性能）
        self._manager.set_crop_size(value)
        # 更新截图用的裁剪大小
        self._screenshot_mgr.crop_size = value
        self._statusbar.showMessage(f"裁剪大小已设置为: {value}x{value}", 2000)
    
    def _on_take_screenshot(self):
        """F9手动截图按钮响应"""
        if self._current_window is None:
            self._statusbar.showMessage("请先选择窗口", 2000)
            return
        
        # 请求截图
        self._screenshot_mgr.request_aimed_screenshot()
        self._statusbar.showMessage("瞄准状态截图已保存", 1000)
    
    def _on_toggle_auto_screenshot(self):
        """F10自动截图按钮响应"""
        if self._current_window is None:
            self._statusbar.showMessage("请先选择窗口", 2000)
            return
        
        # 切换自动截图状态
        self._screenshot_mgr.toggle_auto_capture()
        
        if self._screenshot_mgr.is_auto_capturing:
            self._auto_screenshot_btn.setText("停止 (F10)")
            self._statusbar.showMessage("自动截图已开始（每5秒一张）- 保存到: 准心数据集/未瞄准状态", 5000)
        else:
            self._auto_screenshot_btn.setText("自动截图 (F10)")
            self._statusbar.showMessage(f"自动截图已停止（未瞄准状态: {self._screenshot_mgr.not_aimed_count}张）", 3000)

    # ------------------------------------------------------------------
    # Capture helpers
    # ------------------------------------------------------------------

    def _start_capture(self, window: WindowInfo):
        self._manager.start(window)
        worker = self._manager.worker
        worker.frame_ready.connect(self._on_frame_ready)
        worker.error.connect(self._on_capture_error)
        worker.status.connect(self._on_capture_status)

    @pyqtSlot(object)
    def _on_frame_ready(self, frame: np.ndarray):
        # Track FPS — drop timestamps older than 1 second from the left.
        now = time.monotonic()
        self._frame_times.append(now)
        cutoff = now - 1.0
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.popleft()
        fps = len(self._frame_times)
        self._status_fps.setText(f"{fps} FPS")

        # 更新截图管理器中的当前帧（用于按F9/F10时保存）
        self._screenshot_mgr.update_frame(frame)

        # 更新截图计数显示
        self._aimed_count_label.setText(f"瞄准: {self._screenshot_mgr.aimed_count}")
        self._not_aimed_count_label.setText(f"未瞄准: {self._screenshot_mgr.not_aimed_count}")
        
        # 使用CNN分类器判断准心状态
        if self._classifier.is_ready:
            result = self._classifier.predict(frame)
            status = result['status']
            confidence = result['confidence']
            
            if status == 'aimed':
                self._status_aiming.setText(f"状态: 瞄准中 ({confidence:.2%})")
                self._status_aiming.setStyleSheet("padding: 0 8px; color: #4CAF50; font-weight: bold;")
                
                # 检测到瞄准状态：保存截图并执行点击
                print(f"[检测到瞄准] 置信度: {confidence:.2%}")
                self._on_aimed_detected(frame, confidence)
                
            elif status == 'not_aimed':
                self._status_aiming.setText(f"状态: 未瞄准 ({confidence:.2%})")
                self._status_aiming.setStyleSheet("padding: 0 8px; color: #FF9800;")
            else:
                self._status_aiming.setText("状态: 未知")
                self._status_aiming.setStyleSheet("padding: 0 8px; color: #888;")
        else:
            self._status_aiming.setText("状态: 模型未加载")
            self._status_aiming.setStyleSheet("padding: 0 8px; color: #f44336;")

        # Convert frame to QImage for preview
        h, w, ch = frame.shape
        rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB, ensure C-contiguous
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._preview.set_frame(qimg)

    @pyqtSlot(str)
    def _on_capture_error(self, msg: str):
        self._statusbar.showMessage(f"Error: {msg}", 4000)

    @pyqtSlot(str)
    def _on_capture_status(self, msg: str):
        self._statusbar.showMessage(msg, 2000)
    
    def _on_aimed_detected(self, frame: np.ndarray, confidence: float):
        """
        当检测到瞄准状态时调用：保存截图并执行点击
        
        参数：
            frame: 当前帧（已裁剪）
            confidence: 置信度
        """
        try:
            import cv2
            
            # 检查冷却时间
            current_time = time.time()
            if current_time - self._last_click_time < self._click_cooldown:
                # 还在冷却期内，只保存截图
                self._save_aimed_screenshot(frame, confidence)
                return
            
            # 保存瞄准状态截图
            self._save_aimed_screenshot(frame, confidence)
            
            # 执行点击（使用默认参数）
            if self._click_script.exists():
                try:
                    # 使用默认参数运行点击脚本
                    result = subprocess.run(
                        [sys.executable, str(self._click_script)],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                    
                    if result.returncode == 0:
                        self._last_click_time = current_time
                        print(f"[瞄准触发] 点击已执行，置信度: {confidence:.2%}")
                    else:
                        print(f"[瞄准触发] 点击执行失败: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print("[瞄准触发] 点击脚本超时")
                except Exception as e:
                    print(f"[瞄准触发] 点击脚本错误: {e}")
            else:
                print(f"[瞄准触发] 点击脚本不存在: {self._click_script}")
                
        except Exception as e:
            print(f"[瞄准触发] 处理错误: {e}")
    
    def _save_aimed_screenshot(self, frame: np.ndarray, confidence: float):
        """
        保存瞄准状态截图
        
        参数：
            frame: 当前帧
            confidence: 置信度
        """
        try:
            import cv2
            
            # 生成文件名：时间戳_置信度.jpg
            timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
            confidence_int = int(confidence * 100)  # 转换为整数，如 96.36% -> 9636
            filename = f"{timestamp}_c{confidence_int}.jpg"
            filepath = self._aimed_capture_dir / filename
            
            print(f"[保存截图] 目录: {self._aimed_capture_dir}")
            print(f"[保存截图] 文件: {filename}")
            print(f"[保存截图] 完整路径: {filepath}")
            
            # 使用imencode处理中文路径保存
            result, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if result:
                with open(str(filepath), 'wb') as f:
                    encoded.tofile(f)
                print(f"[保存截图] 成功!")
            else:
                print(f"[保存截图] imencode失败")
            
        except Exception as e:
            print(f"[保存截图] 失败: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._manager.stop()
        self._screenshot_mgr.stop_hotkey_listener()
        super().closeEvent(event)
