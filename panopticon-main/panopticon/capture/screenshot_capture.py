"""
screenshot_capture.py - YOLO训练数据集截图管理模块。

功能：
- F9单次截图：每次按F9截取一帧，保存到"瞄准状态"目录
- F10定时截图：按F10开始自动截图（每1秒一张），保存到"未瞄准状态"目录
- 保存原始帧（无标注）
- 支持热键绑定
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Thread, Event

import numpy as np

log = logging.getLogger(__name__)


class ScreenshotCaptureManager:
    """
    截图管理器，负责保存YOLO训练数据集。

    功能：
    - F9单次截图：每次按F9截取一帧，保存到"瞄准状态"目录
    - F10定时截图：按F10开始自动截图（每1秒一张），保存到"未瞄准状态"目录
    - 保存原始帧（无标注）
    - 支持热键绑定
    """

    def __init__(
        self,
        aimed_dir: str = "准心数据集/瞄准状态",
        not_aimed_dir: str = "准心数据集/未瞄准状态",
        auto_interval: int = 1,  # 自动截图间隔（秒）
        crop_size: int = 640,  # 中心裁剪区域大小（正方形，用于YOLO训练）
    ):
        # F9单次截图保存目录
        self.aimed_dir = Path(aimed_dir)
        self.aimed_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"F9截图保存目录: {self.aimed_dir.absolute()}")
        
        # F10定时截图保存目录
        self.not_aimed_dir = Path(not_aimed_dir)
        self.not_aimed_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"F10截图保存目录: {self.not_aimed_dir.absolute()}")
        
        self.auto_interval = auto_interval  # 自动截图间隔（秒）
        self._crop_size = crop_size  # 中心裁剪区域大小（正方形）
        
        # F9截图计数
        self._aimed_count = 0
        # F10定时截图计数
        self._not_aimed_count = 0
        
        self._listener_thread: Thread | None = None
        self._current_frame: np.ndarray | None = None  # 当前帧缓存
        
        # F9单次截图请求
        self._aimed_requested = False
        
        # F10定时截图控制
        self._auto_capturing = False
        self._auto_capture_thread: Thread | None = None
        self._auto_capture_stop_event = Event()

    @property
    def aimed_count(self) -> int:
        """F9截图数量"""
        return self._aimed_count
    
    @property
    def not_aimed_count(self) -> int:
        """F10截图数量"""
        return self._not_aimed_count
    
    @property
    def is_auto_capturing(self) -> bool:
        """是否正在自动截图"""
        return self._auto_capturing
    
    @property
    def total_count(self) -> int:
        """总截图数量"""
        return self._aimed_count + self._not_aimed_count
    
    @property
    def crop_size(self) -> int:
        """获取当前裁剪大小"""
        return self._crop_size
    
    @crop_size.setter
    def crop_size(self, value: int):
        """设置裁剪大小（实时生效）"""
        if value > 0:
            self._crop_size = value
            log.info(f"裁剪大小已更新为: {self._crop_size}x{self._crop_size}")
        else:
            log.warning("裁剪大小必须大于0")

    def request_aimed_screenshot(self):
        """请求F9单次截图"""
        self._aimed_requested = True
        log.debug("F9单次截图请求已记录")
    
    def toggle_auto_capture(self):
        """切换F10自动截图状态"""
        if self._auto_capturing:
            self.stop_auto_capture()
        else:
            self.start_auto_capture()
    
    def start_auto_capture(self):
        """开始F10定时自动截图"""
        if self._auto_capturing:
            log.warning("自动截图已经在进行中")
            return
        
        self._auto_capturing = True
        self._auto_capture_stop_event.clear()
        
        # 启动自动截图线程
        self._auto_capture_thread = Thread(target=self._auto_capture_loop, daemon=True)
        self._auto_capture_thread.start()
        
        log.info(f"F10自动截图开始（每{self.auto_interval}秒一张）")
    
    def stop_auto_capture(self):
        """停止F10定时自动截图"""
        if not self._auto_capturing:
            log.warning("自动截图未在进行中")
            return
        
        self._auto_capturing = False
        self._auto_capture_stop_event.set()
        
        if self._auto_capture_thread and self._auto_capture_thread.is_alive():
            self._auto_capture_thread.join(timeout=1)
        
        log.info(f"F10自动截图停止，共保存 {self._not_aimed_count} 张图片")
    
    def _auto_capture_loop(self):
        """自动截图循环"""
        while not self._auto_capture_stop_event.is_set():
            # 等待指定间隔
            if self._auto_capture_stop_event.wait(timeout=self.auto_interval):
                break
            
            # 检查是否停止
            if not self._auto_capturing:
                break
            
            # 保存当前帧
            self._do_save_screenshot(self.not_aimed_dir, "not_aimed")
    
    def update_frame(self, frame: np.ndarray):
        """
        更新当前帧缓存。每帧调用此方法更新缓存。
        
        参数：
            frame: BGR格式的numpy数组（原始帧，无标注）
        """
        self._current_frame = frame.copy() if frame is not None else None
        
        # F9单次截图请求处理
        if self._aimed_requested and self._current_frame is not None:
            self._do_save_screenshot(self.aimed_dir, "aimed")
            self._aimed_requested = False
    
    def _crop_center(self, frame: np.ndarray) -> np.ndarray:
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
        
        # 计算裁剪区域
        start_x = (width - size) // 2
        start_y = (height - size) // 2
        
        # 裁剪中心区域
        cropped = frame[start_y:start_y + size, start_x:start_x + size]
        
        return cropped
    
    def _do_save_screenshot(self, save_dir: Path, mode: str):
        """执行截图保存"""
        if self._current_frame is None:
            log.warning("没有可用的帧进行截图")
            return
        
        try:
            import cv2
            
            # 裁剪中心区域
            cropped_frame = self._crop_center(self._current_frame)
            if cropped_frame is None:
                log.warning("裁剪失败")
                return
            
            # 生成文件名：时间戳_序号.jpg
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            if mode == "aimed":
                filename = f"aimed_{timestamp}_{self._aimed_count:04d}.jpg"
                filepath = save_dir / filename
                # 使用imencode处理中文路径
                result, encoded = cv2.imencode('.jpg', cropped_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if result:
                    with open(str(filepath), 'wb') as f:
                        encoded.tofile(f)
                    self._aimed_count += 1
                    log.info(f"F9截图已保存: {filepath} (瞄准状态: {self._aimed_count})")
                else:
                    log.error(f"F9截图保存失败: {filepath}")
                    
            elif mode == "not_aimed":
                filename = f"not_aimed_{timestamp}_{self._not_aimed_count:04d}.jpg"
                filepath = save_dir / filename
                # 使用imencode处理中文路径
                result, encoded = cv2.imencode('.jpg', cropped_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if result:
                    with open(str(filepath), 'wb') as f:
                        encoded.tofile(f)
                    self._not_aimed_count += 1
                    log.info(f"F10截图已保存: {filepath} (未瞄准状态: {self._not_aimed_count})")
                else:
                    log.error(f"F10截图保存失败: {filepath}")
                
        except Exception as e:
            log.error(f"保存截图时出错: {e}")
    
    def save_aimed_screenshot(self, frame: np.ndarray) -> str | None:
        """
        立即保存F9截图（手动调用）。
        
        参数：
            frame: BGR格式的numpy数组

        返回：
            保存的文件路径，或None（如果保存失败）
        """
        try:
            import cv2
            
            # 裁剪中心区域
            cropped_frame = self._crop_center(frame)
            if cropped_frame is None:
                log.warning("裁剪失败")
                return None
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"aimed_{timestamp}_{self._aimed_count:04d}.jpg"
            filepath = self.aimed_dir / filename
            
            # 使用imencode处理中文路径
            result, encoded = cv2.imencode('.jpg', cropped_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if result:
                with open(str(filepath), 'wb') as f:
                    encoded.tofile(f)
                self._aimed_count += 1
                log.info(f"F9截图已保存: {filepath} (瞄准状态: {self._aimed_count})")
                return str(filepath)
            else:
                log.error(f"F9截图保存失败: {filepath}")
                return None
                
        except Exception as e:
            log.error(f"保存截图时出错: {e}")
            return None

    def set_hotkey(self, aimed_key: str = "f9", auto_key: str = "f10"):
        """
        设置热键（使用pynput库）。

        参数：
            aimed_key: F9单次截图的热键
            auto_key: F10定时截图的热键
        """
        try:
            from pynput import keyboard
            
            def on_press(key):
                try:
                    key_str = self._key_to_string(key)
                    
                    # F9单次截图
                    if key_str == aimed_key.lower() or key_str == aimed_key.upper():
                        self.request_aimed_screenshot()
                        log.info(f"热键 '{aimed_key}' - F9单次截图")
                    
                    # F10定时截图
                    elif key_str == auto_key.lower() or key_str == auto_key.upper():
                        self.toggle_auto_capture()
                        if self._auto_capturing:
                            log.info(f"热键 '{auto_key}' - F10自动截图开始")
                        else:
                            log.info(f"热键 '{auto_key}' - F10自动截图停止")
                
                except Exception as e:
                    log.error(f"热键处理错误: {e}")
            
            # 在新线程中运行监听器
            self._listener_thread = keyboard.Listener(on_press=on_press)
            self._listener_thread.daemon = True
            self._listener_thread.start()
            
            log.info(f"热键已设置: {aimed_key} 单次截图, {auto_key} 定时截图")
            
        except ImportError:
            log.warning("pynput未安装，热键功能不可用。请运行: pip install pynput")
        except Exception as e:
            log.error(f"设置热键失败: {e}")

    def _key_to_string(self, key) -> str:
        """将pynput的Key对象转换为字符串"""
        try:
            if hasattr(key, 'char') and key.char:
                return key.char.lower()
            elif hasattr(key, 'name'):
                return key.name
            else:
                return str(key).replace('Key.', '').replace("'", "")
        except:
            return str(key)

    def stop_hotkey_listener(self):
        """停止热键监听"""
        # 先停止自动截图
        if self._auto_capturing:
            self.stop_auto_capture()
        
        # 停止热键监听
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.stop()
            log.info("热键监听已停止")
