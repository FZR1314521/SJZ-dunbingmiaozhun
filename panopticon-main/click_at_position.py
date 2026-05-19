"""
后台鼠标点击脚本 - 使用 PostMessage

功能：
- 向指定窗口句柄发送鼠标点击消息
- 支持客户端坐标（相对于窗口内部）
- 窗口可以在后台，不会被激活

使用方法：
python click_at_position.py [窗口句柄/标题] [X坐标] [Y坐标]

示例：
python click_at_position.py 135094 100 200    # 使用窗口句柄
python click_at_position.py "手机投屏" 100 200  # 使用窗口标题
"""

import ctypes
import sys
import time
from ctypes import wintypes

# Windows API 常量
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_MOUSEMOVE = 0x0200
MK_LBUTTON = 0x0001

# 加载 user32.dll
user32 = ctypes.windll.user32


def find_window_by_title(title: str):
    """根据标题查找窗口"""
    # 精确匹配
    hwnd = user32.FindWindowW(None, title)
    if hwnd and user32.IsWindowVisible(hwnd):
        return hwnd
    
    # 部分匹配
    windows = []
    
    def enum_callback(hwnd, lParam):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buff, length + 1)
                if title.lower() in buff.value.lower():
                    windows.append(hwnd)
        return True
    
    EnumWindowsProc = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        wintypes.HWND,
        ctypes.c_void_p
    )
    user32.EnumWindows(EnumWindowsProc(enum_callback), 0)
    
    return windows[0] if windows else None


def click_at_window(hwnd: int, x: int, y: int):
    """
    向窗口句柄发送鼠标点击（客户端坐标）
    
    参数：
        hwnd: 窗口句柄
        x: 客户端坐标X（相对于窗口左上角）
        y: 客户端坐标Y（相对于窗口左上角）
    
    返回：
        成功返回True
    """
    if not hwnd or not user32.IsWindow(hwnd):
        print(f"错误：无效的窗口句柄 {hwnd}")
        return False
    
    # 组合坐标到lParam：x在低16位，y在高16位
    lParam = (x & 0xFFFF) | ((y & 0xFFFF) << 16)
    
    print("Sending PostMessage to HWND " + str(hwnd))
    print("Coordinates: (" + str(x) + ", " + str(y) + ") -> lParam: 0x" + format(lParam, '08X'))
    
    # Method 1: Send WM_LBUTTONDOWN and WM_LBUTTONUP (client coordinates)
    # PostMessage accepts client coordinates directly
    result_down = user32.PostMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lParam)
    print("WM_LBUTTONDOWN result: " + str(result_down))
    
    time.sleep(0.05)  # Short delay to simulate real click
    
    result_up = user32.PostMessageW(hwnd, WM_LBUTTONUP, 0, lParam)
    print("WM_LBUTTONUP result: " + str(result_up))
    
    return result_down and result_up


def click_at_window_screen_coords(hwnd: int, screen_x: int, screen_y: int):
    """
    向窗口句柄发送鼠标点击（屏幕坐标）
    
    参数：
        hwnd: 窗口句柄
        screen_x: 屏幕坐标X
        screen_y: 屏幕坐标Y
    """
    if not hwnd or not user32.IsWindow(hwnd):
        print(f"错误：无效的窗口句柄 {hwnd}")
        return False
    
    # 将屏幕坐标转换为客户端坐标
    point = wintypes.POINT(screen_x, screen_y)
    if not user32.ScreenToClient(hwnd, ctypes.byref(point)):
        print("错误：ScreenToClient 失败")
        return False
    
    client_x, client_y = point.x, point.y
    print(f"屏幕坐标 ({screen_x}, {screen_y}) -> 客户端坐标 ({client_x}, {client_y})")
    
    return click_at_window(hwnd, client_x, client_y)


def main():
    # 默认值：手机投屏窗口，坐标 (1450, 400)
    DEFAULT_WINDOW = "手机投屏"
    DEFAULT_X = 1450
    DEFAULT_Y = 400
    
    if len(sys.argv) < 2:
        # 无参数：使用默认设置
        window_arg = DEFAULT_WINDOW
        x = DEFAULT_X
        y = DEFAULT_Y
        print("Using default settings: window='" + DEFAULT_WINDOW + "', coords=(" + str(DEFAULT_X) + ", " + str(DEFAULT_Y) + ")")
    elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
        print("Usage:")
        print("  python click_at_position.py [window_title/handle] [X] [Y]")
        print()
        print("Default settings:")
        print("  Window: " + DEFAULT_WINDOW)
        print("  Coords: (" + str(DEFAULT_X) + ", " + str(DEFAULT_Y) + ")")
        print()
        print("Examples:")
        print("  python click_at_position.py")
        print("  python click_at_position.py 135094 100 200")
        print("  python click_at_position.py PhoneScreen 100 200")
        return 0
    elif len(sys.argv) < 4:
        print("Usage:")
        print("  python click_at_position.py [window_title/handle] [X] [Y]")
        print()
        print("Examples:")
        print("  python click_at_position.py 135094 100 200")
        print("  python click_at_position.py PhoneScreen 100 200")
        return 1
    else:
        # 解析参数
        window_arg = sys.argv[1]
        x = int(float(sys.argv[2]))
        y = int(float(sys.argv[3]))
    
    # 获取窗口句柄
    if window_arg.isdigit():
        # 数字，被当作窗口句柄
        hwnd = int(window_arg)
        if not user32.IsWindow(hwnd):
            print("ERROR: Invalid window handle " + str(hwnd))
            return 1
        
        # 获取窗口标题
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        window_title = buff.value
    else:
        # 字符串，被当作窗口标题
        window_title = window_arg
        hwnd = find_window_by_title(window_title)
        if not hwnd:
            print("ERROR: Window not found: " + window_title)
            return 1
    
    # 获取窗口信息
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    
    print("=" * 50)
    print("Window handle: " + str(hwnd))
    print("Window title: " + window_title)
    print("Window rect: (" + str(rect.left) + ", " + str(rect.top) + ") - (" + str(rect.right) + ", " + str(rect.bottom) + ")")
    print("Window size: " + str(rect.right - rect.left) + " x " + str(rect.bottom - rect.top))
    print("=" * 50)
    print("Click coordinates: (" + str(x) + ", " + str(y) + ") - client coords")
    print("Absolute position: (" + str(rect.left + x) + ", " + str(rect.top + y) + ")")
    print("=" * 50)
    print()
    
    # 执行点击
    print("Executing click...")
    
    success = click_at_window(hwnd, x, y)
    
    if success:
        print()
        print("OK: Click sent successfully!")
        return 0
    else:
        print()
        print("ERROR: Click failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
