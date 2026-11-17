# coding: utf-8
"""macOS 键盘映射（孵化中，行为未在真机验证）。

素材来源：EdgarZhong/CapsWriter-for-macOS（#399）的键位实践。
当前只做占位与已知差异记录；真机验证通过后在此实现。

已知 macOS 差异（与 Windows/Linux 相比）：
- pynput 可用（keyboard/mouse.Listener 支持 darwin），但需要「辅助功能」权限，
  否则监听不到任何事件；
- Listener 无 win32_event_filter，也不支持 Linux 式全部按键回调语义的
  suppress_event() —— 按键阻塞（suppress）在 macOS 上不可用，
  快捷键需换成不与输入冲突的键（如右 Command，chaser-mimo 的实践）；
- keyboard.Key 枚举含 cmd/cmd_r 等专用键名，规范化表需补充：
  cmd_l -> cmd、cmd_r -> cmd_r、alt_l -> option 等；
- KeyCode.from_char 与 KeyTranslator 均为 Windows 专有，不可用。
"""

from pynput import keyboard

from . import logger


def pynput_key_to_name(key) -> 'str | None':
    """将 pynput Key/KeyCode 转换为字符串名称 (macOS 用，占位实现)。

    与 Linux 版同构，补充 macOS 专有键名规范化。
    未在真机验证：cmd 系列键是否能作为录音快捷键待测。
    """
    if isinstance(key, keyboard.Key):
        name = key.name
        # macOS 专有键名规范化（占位，未验证）
        if name in ('cmd_l', 'cmd_r', 'cmd'):
            return name
        if name == 'option_l':
            return 'alt_l'
        if name == 'option_r':
            return 'alt_gr'
        # 其余与 Linux 命名一致（caps_lock/shift_l/ctrl_l/f1-f12/...）
        return name
    elif isinstance(key, keyboard.KeyCode):
        if key.char:
            return key.char
        return f'vk_{key.vk}'
    return None


def pynput_mouse_to_name(button) -> 'str | None':
    """将 pynput mouse Button 转换为字符串名称（macOS 占位）。"""
    name = getattr(button, 'name', '')
    if name in ('x1', 'x2'):
        return name
    return None


def check_permissions() -> bool:
    """检查辅助功能权限（占位：实现为尝试监听一个键并超时）。

    macOS 上 pynput 没有权限时静默收不到事件，必须显式提示用户去
    系统设置 > 隐私与安全性 > 辅助功能 授权。
    """
    logger.warning(
        "macOS：请确认已在 系统设置 > 隐私与安全性 > 辅助功能 中授权本程序"
    )
    return True
