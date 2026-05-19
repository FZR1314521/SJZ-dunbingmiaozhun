"""
screenshot.py - Cross-platform window screenshot capture.

Uses mss for the pixel grab (fast, cross-platform) and platform-specific
APIs to locate the window rect each frame so it follows the window if it moves.

On Windows, falls back to PrintWindow (win32gui) for capturing windows that
are minimized or occluded by other windows.

On KDE Wayland, uses spectacle (ships with KDE Plasma) for a full-screen
grab then crops to the window region, since XGetImage is unavailable.
"""

from __future__ import annotations

import logging
import sys

import numpy as np

from panopticon.utils.platform import WindowInfo, get_window_geometry

log = logging.getLogger(__name__)


def _is_kde_wayland() -> bool:
    import os

    return (
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        and os.environ.get("XDG_CURRENT_DESKTOP", "").upper() == "KDE"
    )


def _win32_printwindow_available() -> bool:
    """
    Check once at startup whether win32gui.PrintWindow exists.
    Some pywin32 builds omit it; calling it per-frame and swallowing
    AttributeError would spam the log on every capture tick.
    """
    try:
        import win32gui
        import ctypes
        from ctypes import wintypes

        # 方法1: 检查 pywin32 是否有 PrintWindow
        if hasattr(win32gui, "PrintWindow"):
            return True
        
        # 方法2: 尝试通过 ctypes 直接调用 user32.dll 的 PrintWindow
        # 即使 pywin32 没有封装，我们也可以自己实现
        user32 = ctypes.windll.user32
        if hasattr(user32, "PrintWindow"):
            return True
            
        return False
    except ImportError:
        return False


_WIN32_PRINTWINDOW_AVAILABLE = _win32_printwindow_available()
if sys.platform == "win32":
    if _WIN32_PRINTWINDOW_AVAILABLE:
        log.info("win32gui.PrintWindow 可用 - 支持捕获最小化和被遮挡窗口")
    else:
        log.warning("win32gui.PrintWindow 不可用 - 回退到 mss 捕获（仅可见区域）")


class ScreenshotCapture:
    """
    Captures screenshots of a target window.

    Usage:
        cap = ScreenshotCapture(window)
        frame = cap.grab()   # returns np.ndarray (H, W, 3) BGR or None
    """

    def __init__(self, window: WindowInfo):
        self.window = window
        self._last_geometry: dict | None = window.geometry
        self._kde_wayland = _is_kde_wayland()
        if not self._kde_wayland:
            self._init_mss()
        # 记录窗口是否曾经最小化（避免反复最小化/恢复导致闪烁）
        self._was_minimized_at_start = False
        self._restored_from_minimize = False  # 标记窗口是否已从最小化恢复

    def _init_mss(self):
        import mss

        self._mss = mss.mss()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grab(self) -> np.ndarray | None:
        """
        Capture the current frame of the tracked window.
        Returns a BGR numpy array (H, W, 3), or None on failure.
        """
        geom = self._refresh_geometry()
        if geom is None:
            return None

        if (
            sys.platform == "win32"
            and _WIN32_PRINTWINDOW_AVAILABLE
            and self.window.handle is not None
        ):
            frame = self._grab_win32_printwindow(geom)
            if frame is not None:
                return frame

        if self._kde_wayland:
            return self._grab_spectacle(geom)

        return self._grab_mss(geom)

    def update_window(self, window: WindowInfo):
        """Swap the target window (e.g. user re-selects)."""
        self.window = window
        self._last_geometry = window.geometry
        # 重置最小化状态标志
        self._was_minimized_at_start = False
        self._restored_from_minimize = False

    def close(self):
        import contextlib

        if not self._kde_wayland:
            with contextlib.suppress(Exception):
                self._mss.close()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _refresh_geometry(self) -> dict | None:
        """Get the latest window geometry, falling back to the last known."""
        geom = get_window_geometry(self.window)
        if geom and geom["width"] > 0 and geom["height"] > 0:
            self._last_geometry = geom
        return self._last_geometry

    # ------------------------------------------------------------------
    # KDE Wayland: spectacle fullscreen grab + crop
    # ------------------------------------------------------------------

    def _grab_spectacle(self, geom: dict) -> np.ndarray | None:
        """
        Capture the full screen via spectacle then crop to the window region.

        spectacle ships with KDE Plasma and is authorised by KWin to capture
        the compositor surface, which bypasses the XGetImage restriction.
        """
        import subprocess
        import tempfile

        import cv2

        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
                outfile = fh.name

            result = subprocess.run(
                ["spectacle", "-b", "-f", "-n", "-o", outfile],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                log.warning("spectacle exited with code %d", result.returncode)
                return None

            full = cv2.imread(outfile)
            if full is None:
                return None

            x = max(0, geom["left"])
            y = max(0, geom["top"])
            w = geom["width"]
            h = geom["height"]
            fh_img, fw_img = full.shape[:2]
            x2 = min(x + w, fw_img)
            y2 = min(y + h, fh_img)

            if x2 <= x or y2 <= y:
                return None

            return full[y:y2, x:x2]
        except Exception:
            log.debug("spectacle grab failed", exc_info=True)
            return None
        finally:
            import contextlib
            import os

            with contextlib.suppress(Exception):
                os.unlink(outfile)

    # ------------------------------------------------------------------
    # mss capture (Linux X11, macOS, Windows fallback)
    # ------------------------------------------------------------------

    def _grab_mss(self, geom: dict) -> np.ndarray | None:
        try:
            monitor = {
                "left": geom["left"],
                "top": geom["top"],
                "width": geom["width"],
                "height": geom["height"],
            }
            sct_img = self._mss.grab(monitor)
            # mss returns BGRA; drop alpha channel -> BGR
            frame = np.frombuffer(sct_img.raw, dtype=np.uint8)
            frame = frame.reshape((sct_img.height, sct_img.width, 4))
            return frame[:, :, :3]  # BGR
        except Exception:
            log.debug("mss grab failed", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Windows: PrintWindow (captures minimized / occluded windows)
    # ------------------------------------------------------------------

    def _grab_win32_printwindow(self, geom: dict) -> np.ndarray | None:
        """
        Use Win32 PrintWindow to render the window into a DC.
        Works even when the window is minimized or behind other windows.
        """
        try:
            import win32gui
            import win32ui
            import win32con

            hwnd = self.window.handle
            
            # 检查窗口是否最小化
            is_minimized = win32gui.IsIconic(hwnd)
            
            # 第一次检测到最小化时，记录并恢复窗口
            if is_minimized:
                if not self._restored_from_minimize:
                    # 第一次最小化：临时恢复窗口（不激活到前台）
                    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)
                    # 等待窗口恢复
                    import time
                    time.sleep(0.05)
                    self._restored_from_minimize = True
                    self._was_minimized_at_start = True
            
            # 获取窗口实际尺寸
            rect = win32gui.GetWindowRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]

            if width <= 0 or height <= 0:
                return None

            # 检查是否有 pywin32 的 PrintWindow
            if hasattr(win32gui, "PrintWindow"):
                hwnd_dc = win32gui.GetWindowDC(hwnd)
                mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
                save_dc = mfc_dc.CreateCompatibleDC()
                bitmap = win32ui.CreateBitmap()
                bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
                save_dc.SelectObject(bitmap)

                # 尝试多种标志组合以获得最佳效果
                flags_list = [
                    0x00000002,  # PW_RENDERFULLCONTENT
                    0x00000000,  # PW_DEFAULT
                    0x00000001,  # PW_CLIENTONLY
                ]
                
                frame = None
                for flags in flags_list:
                    try:
                        result = win32gui.PrintWindow(hwnd, save_dc.GetSafeHdc(), flags)
                        if result:
                            bmp_info = bitmap.GetInfo()
                            bmp_str = bitmap.GetBitmapBits(True)
                            img = np.frombuffer(bmp_str, dtype=np.uint8)
                            img = img.reshape((bmp_info["bmHeight"], bmp_info["bmWidth"], 4))
                            frame = img[:, :, :3]  # drop alpha, keep BGR
                            # 检查是否是有效的帧（不是全黑）
                            if frame.mean() > 1:
                                break
                    except:
                        continue

                # Cleanup
                save_dc.DeleteDC()
                mfc_dc.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwnd_dc)
                win32ui.DeleteObject(bitmap.GetHandle())
            else:
                # 回退到 ctypes 实现
                frame = self._grab_win32_printwindow_ctypes(hwnd, width, height)

            return frame
        except Exception:
            log.debug("win32 PrintWindow grab failed", exc_info=True)
            return None
    
    def _grab_win32_printwindow_ctypes(self, hwnd, width, height) -> np.ndarray | None:
        """
        备用实现：使用 ctypes 直接调用 Windows API
        """
        try:
            import ctypes
            from ctypes import wintypes
            
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32
            
            # 定义类型
            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ("biSize", wintypes.DWORD),
                    ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG),
                    ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD),
                    ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD),
                    ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG),
                    ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD),
                ]
            
            class BITMAPINFO(ctypes.Structure):
                _fields_ = [
                    ("bmiHeader", BITMAPINFOHEADER),
                    ("bmiColors", wintypes.DWORD * 3),
                ]
            
            # 获取 DC
            hdcWindow = user32.GetWindowDC(hwnd)
            hdcMemDC = gdi32.CreateCompatibleDC(hdcWindow)
            
            # 创建位图
            hbmScreen = gdi32.CreateCompatibleBitmap(hdcWindow, width, height)
            hbmOld = gdi32.SelectObject(hdcMemDC, hbmScreen)
            
            # 尝试多种标志
            flags_list = [0x00000002, 0x00000000, 0x00000001]
            success = False
            
            for flags in flags_list:
                try:
                    result = user32.PrintWindow(hwnd, hdcMemDC, flags)
                    if result != 0:
                        success = True
                        break
                except:
                    continue
            
            frame = None
            if success:
                # 准备位图信息
                bmi = BITMAPINFO()
                bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
                bmi.bmiHeader.biWidth = width
                bmi.bmiHeader.biHeight = -height  # 负高度表示从上到下
                bmi.bmiHeader.biPlanes = 1
                bmi.bmiHeader.biBitCount = 32
                bmi.bmiHeader.biCompression = 0  # BI_RGB
                
                # 分配缓冲区
                buffer_len = width * height * 4
                buffer = ctypes.create_string_buffer(buffer_len)
                
                # 获取位图数据
                gdi32.GetDIBits(hdcMemDC, hbmScreen, 0, height, buffer, ctypes.byref(bmi), 0)
                
                # 转换为 numpy 数组
                img = np.frombuffer(buffer, dtype=np.uint8)
                img = img.reshape((height, width, 4))
                frame = img[:, :, :3]  # 丢弃 alpha 通道，保留 BGR
            
            # 清理
            gdi32.SelectObject(hdcMemDC, hbmOld)
            gdi32.DeleteObject(hbmScreen)
            gdi32.DeleteDC(hdcMemDC)
            user32.ReleaseDC(hwnd, hdcWindow)
            
            return frame
        except Exception:
            log.debug("ctypes PrintWindow grab failed", exc_info=True)
            return None


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------


def capture_window(window: WindowInfo) -> np.ndarray | None:
    """One-shot capture of a window. Creates and destroys a ScreenshotCapture."""
    cap = ScreenshotCapture(window)
    try:
        return cap.grab()
    finally:
        cap.close()
