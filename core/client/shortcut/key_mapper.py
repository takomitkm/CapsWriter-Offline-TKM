# coding: utf-8
"""
按键映射 —— 公共接口与平台分发

平台实现拆分（同目录）：
- key_mapper_win.py    Windows：VK 码 + pynput KeyTranslator + win32 消息常量
- key_mapper_linux.py  Linux：pynput Key/KeyCode 名称规范化（on_press/on_release 用）
- key_mapper_macos.py  macOS：键位规范化占位（孵化中，见 feat/macos-adapter 分支）

本文件只保留：平台检测、跨平台常量、公共 KeyMapper 门面。
对外导入方式不变（from ... import KeyMapper, IS_WINDOWS, ...），调用方无需改动。
"""

import sys

from pynput import keyboard

from . import logger

# ------------------------------------------------------------------
# 平台检测
# ------------------------------------------------------------------
IS_WINDOWS = sys.platform == 'win32'
IS_LINUX = sys.platform.startswith('linux')
IS_MACOS = sys.platform == 'darwin'


class KeyMapper:
    """按键映射器（平台无关门面：特殊键枚举 + 名称/对象互转）"""

    # pynput 特殊键对象缓存
    _SPECIAL_KEY_OBJECTS = None

    @classmethod
    def _get_special_key_objects(cls):
        """获取 pynput 特殊键对象（延迟初始化）"""
        if cls._SPECIAL_KEY_OBJECTS is None:
            # 直接从 pynput keyboard.Key 枚举构建 name -> key 映射，
            # 覆盖全部特殊键（含 insert/home/end/方向键/page_up 等），避免手工枚举遗漏
            cls._SPECIAL_KEY_OBJECTS = {
                key.name: key
                for key in keyboard.Key
            }
        return cls._SPECIAL_KEY_OBJECTS

    @staticmethod
    def name_to_key(key_name: str):
        """
        将按键名称转换为 pynput 按键对象

        Args:
            key_name: 按键名称

        Returns:
            pynput 按键对象或 None
        """
        # 特殊按键
        special_keys = KeyMapper._get_special_key_objects()
        if key_name in special_keys:
            return special_keys[key_name]

        # 单个字符按键
        if len(key_name) == 1:
            return keyboard.KeyCode.from_char(key_name)

        logger.warning(f"未知按键名称: {key_name}")
        return None


# ------------------------------------------------------------------
# Windows 专有实现（VK 码、win32 消息常量、KeyTranslator）
# ------------------------------------------------------------------
if IS_WINDOWS:
    from .key_mapper_win import (          # noqa: F401
        vk_to_name,
        NUMPAD_KEYS,
        WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP,
        WM_XBUTTONDOWN, WM_XBUTTONUP,
        XBUTTON1, XBUTTON2,
        KEYBOARD_MESSAGES, KEY_UP_MESSAGES, KEY_DOWN_MESSAGES, MOUSE_MESSAGES,
    )
    # 平台版补齐 VK 查询能力（公共门面只管名称/对象互转）
    KeyMapper.vk_to_name = staticmethod(vk_to_name)

# ------------------------------------------------------------------
# Linux / macOS：走 on_press/on_release 回调路径，没有 VK 查询，
# 名称规范化在 shortcut_manager 的 _pynput_key_to_name 中完成。
# macOS 键位表占位见 key_mapper_macos.py（feat/macos-adapter 分支孵化）。
# ------------------------------------------------------------------
