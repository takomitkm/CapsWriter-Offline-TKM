# coding: utf-8
"""Windows 键盘映射：VK 码、win32 消息常量、pynput KeyTranslator。

从 key_mapper.py 平移而来（行为逐行不变），仅 Windows 平台加载；
vk_to_name 的语义与拆分前完全一致。
"""

from pynput import keyboard
from pynput._util.win32 import KeyTranslator

# 创建键盘翻译器实例（用于 VK 到字符的转换）
_key_translator = KeyTranslator()

# 特殊键 VK 映射（从 pynput 复制）
_SPECIAL_KEYS = {
    key.value.vk: key
    for key in keyboard.Key
}

# 小键盘按键映射（VK -> 名称）
NUMPAD_KEYS = {
    0x60: 'numpad0',  0x61: 'numpad1',  0x62: 'numpad2',  0x63: 'numpad3',
    0x64: 'numpad4',  0x65: 'numpad5',  0x66: 'numpad6',  0x67: 'numpad7',
    0x68: 'numpad8',  0x69: 'numpad9',
    0x6A: 'numpad_multiply',  # *
    0x6B: 'numpad_add',       # +
    0x6C: 'numpad_separator', # （通常未使用）
    0x6D: 'numpad_subtract',  # -
    0x6E: 'numpad_decimal',   # 小数点
    0x6F: 'numpad_divide',    # /
}

# Windows 键盘消息常量
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105

# Windows 鼠标消息常量
WM_XBUTTONDOWN = 0x020B
WM_XBUTTONUP = 0x020C
XBUTTON1 = 0x0001
XBUTTON2 = 0x0002

# 按键消息集合
KEYBOARD_MESSAGES = (WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP)
KEY_UP_MESSAGES = (WM_KEYUP, WM_SYSKEYUP)
KEY_DOWN_MESSAGES = (WM_KEYDOWN, WM_SYSKEYDOWN)
MOUSE_MESSAGES = (WM_XBUTTONDOWN, WM_XBUTTONUP)


def vk_to_name(vk: int) -> str:
    """
    将虚拟键码转换为按键名称

    Args:
        vk: 虚拟键码

    Returns:
        str: 按键名称（与 Shortcut.key 格式一致）
    """
    # 首先检查是否是特殊键（pynput Key 枚举）
    if vk in _SPECIAL_KEYS:
        return _SPECIAL_KEYS[vk].name

    # 检查是否是小键盘按键
    if vk in NUMPAD_KEYS:
        return NUMPAD_KEYS[vk]

    # 使用 pynput 的 KeyTranslator 获取字符（字母、数字、符号键）
    try:
        params = _key_translator(vk, is_press=True)
        if 'char' in params and params['char'] is not None:
            return params['char']
    except Exception:
        pass

    # 未知键码，返回 vk_ 格式
    return f'vk_{vk}'
