"""
manager.py - QThread-based capture loop.

CaptureManager runs in a background QThread.  Each tick it:
  1. Grabs a screenshot of the target window via ScreenshotCapture.
  2. Crops the center region (for CNN input).
  3. Emits `frame_ready` with the cropped frame.

Signals are the only communication path back to the main thread -
no direct UI calls are made here.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from PyQt6.QtCore import QMutex, QMutexLocker, QThread, pyqtSignal

from panopticon.capture.screenshot import ScreenshotCapture
from panopticon.utils.platform import WindowInfo

log = logging.getLogger(__name__)

MAX_FAILURES = 30


class CaptureWorker(QThread):
    """
    Background thread that continuously captures frames.

    Signals:
        frame_ready(np.ndarray)
            Emitted after each successful capture.
            The ndarray is the cropped BGR frame; safe to convert to QImage
            on the receiving side.
        error(str)
            Emitted when a non-fatal error occurs (e.g. window disappeared).
        status(str)
            Emitted for informational status messages.
    """

    # Signals must be class-level attributes
    frame_ready = pyqtSignal(object)  # (cropped frame)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(
        self,
        window: WindowInfo,
        interval_ms: int = 100,
        crop_size: int = 640,
        parent=None,
    ):
        super().__init__(parent)
        self._window = window
        self._interval_ms = interval_ms
        self._crop_size = crop_size

        self._mutex = QMutex()
        self._running = False
        self._paused = False

        self._capture: ScreenshotCapture | None = None
    
    def set_crop_size(self, size: int):
        """设置裁剪大小（实时生效）"""
        with QMutexLocker(self._mutex):
            if size > 0:
                self._crop_size = size
                log.info(f"裁剪大小已更新为: {self._crop_size}x{self._crop_size}")
    
    def _crop_center(self, frame) -> np.ndarray | None:
        """
        裁剪帧的中心区域（正方形）。
        
        参数：
            frame: BGR格式的numpy数组
            
        返回：
            裁剪后的中心区域（正方形）
        """
        if frame is None:
            return None
        
        height, width = frame.shape[:2]
        size = min(height, width, self._crop_size)
        
        start_x = (width - size) // 2
        start_y = (height - size) // 2
        
        return frame[start_y:start_y + size, start_x:start_x + size]

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def set_interval(self, ms: int):
        with QMutexLocker(self._mutex):
            self._interval_ms = max(10, ms)

    def set_window(self, window: WindowInfo):
        with QMutexLocker(self._mutex):
            self._window = window
            if self._capture is not None:
                self._capture.update_window(window)

    def pause(self):
        with QMutexLocker(self._mutex):
            self._paused = True

    def resume(self):
        with QMutexLocker(self._mutex):
            self._paused = False

    def stop(self):
        with QMutexLocker(self._mutex):
            self._running = False

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self):
        with QMutexLocker(self._mutex):
            self._running = True
            window = self._window
            interval = self._interval_ms

        # Initialize capture
        self._capture = ScreenshotCapture(window)
        log.info("Capture started for window: %s", window)
        self.status.emit("Capture started.")

        consecutive_failures = 0

        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break
                if self._paused:
                    time.sleep(0.05)
                    continue
                interval = self._interval_ms

            t_start = time.monotonic()

            frame = self._capture.grab()

            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    log.warning(
                        "Window lost — no frame received after %d attempts", consecutive_failures
                    )
                    self.error.emit("Window lost - no frame received.")
                    consecutive_failures = 0
                time.sleep(interval / 1000.0)
                continue

            consecutive_failures = 0

            # 裁剪中心区域（用于CNN输入）
            cropped_frame = self._crop_center(frame)
            
            if cropped_frame is None:
                log.warning("帧裁剪失败")
                time.sleep(interval / 1000.0)
                continue

            # 传递裁剪后的帧
            self.frame_ready.emit(cropped_frame)

            # Throttle to maintain the requested interval
            elapsed_ms = (time.monotonic() - t_start) * 1000
            sleep_ms = interval - elapsed_ms
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        if self._capture:
            self._capture.close()
        log.info("Capture stopped for window: %s", window)
        self.status.emit("Capture stopped.")


class CaptureManager:
    """
    High-level manager that owns the CaptureWorker thread.

    Handles starting, stopping, and swapping the target window.
    The caller connects to the worker's signals directly:

        manager.worker.frame_ready.connect(my_slot)
        manager.worker.error.connect(my_error_slot)
    """

    def __init__(self, interval_ms: int = 100, crop_size: int = 640):
        self._interval_ms = interval_ms
        self._crop_size = crop_size
        self.worker: CaptureWorker | None = None

    @property
    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()
    
    def set_crop_size(self, size: int):
        """设置裁剪大小（实时生效）"""
        self._crop_size = size
        if self.worker:
            self.worker.set_crop_size(size)

    def start(self, window: WindowInfo):
        """Start (or restart) capture on the given window."""
        self.stop()
        self.worker = CaptureWorker(window, self._interval_ms, self._crop_size)
        self.worker.start()

    def stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.worker = None

    def set_interval(self, ms: int):
        """Update the capture interval."""
        self._interval_ms = ms
        if self.worker:
            self.worker.set_interval(ms)
