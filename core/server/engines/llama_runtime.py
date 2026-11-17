"""共享 llama.cpp 原生运行库位置（所有 GGUF 引擎共用）。"""

from __future__ import annotations

from pathlib import Path


REQUIRED_RUNTIME_FILES = (
    'ggml.dll',
    'ggml-base.dll',
    'ggml-vulkan.dll',
    'llama.dll',
)


def llama_runtime_dir() -> Path:
    """返回所有 GGUF 引擎共用的原生运行库目录。"""
    return Path(__file__).resolve().parent / 'llama' / 'bin'


def require_llama_runtime() -> Path:
    """打包运行库不完整时给出精确报错。"""
    runtime_dir = llama_runtime_dir()
    missing = [name for name in REQUIRED_RUNTIME_FILES if not (runtime_dir / name).is_file()]
    if missing:
        missing_text = ', '.join(missing)
        raise FileNotFoundError(
            f'llama.cpp 运行库不完整：{runtime_dir}；缺少：{missing_text}'
        )
    return runtime_dir
