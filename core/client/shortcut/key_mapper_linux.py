# coding: utf-8
"""Linux 键盘映射：pynput Key/KeyCode 名称规范化。

从 shortcut_manager._pynput_key_to_name 平移而来（行为不变），
供 Linux 的 on_press/on_release 回调路径使用。
"""

from pynput import keyboard


def pynput_key_to_name(key) -> 'str | None':
    """将 pynput Key/KeyCode 转换为字符串名称 (Linux 用)"""
    if isinstance(key, keyboard.Key):
        name = key.name
        # 规范化常见名称
        if name == 'caps_lock':
            return 'caps_lock'
        if name == 'num_lock':
            return 'num_lock'
        if name == 'scroll_lock':
            return 'scroll_lock'
        if name == 'ctrl_l':
            return 'ctrl_l'
        if name == 'ctrl_r':
            return 'ctrl_r'
        if name == 'shift_l':
            return 'shift'
        if name == 'shift_r':
            return 'shift_r'
        if name == 'alt_l':
            return 'alt_l'
        if name == 'alt_gr':
            return 'alt_gr'
        return name
    elif isinstance(key, keyboard.KeyCode):
        # 普通字符键
        if key.char:
            return key.char
        return f'vk_{key.vk}'
    return None


def pynput_mouse_to_name(button) -> 'str | None':
    """将 pynput mouse Button 转换为字符串名称。仅处理侧键，其余返回 None。"""
    name = getattr(button, 'name', '')
    if name in ('x1', 'x2'):
        return name
    return None
