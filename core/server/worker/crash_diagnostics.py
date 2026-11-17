# coding: utf-8
"""识别 Worker 崩溃诊断报告生成。

子进程在模型加载阶段崩溃时，收集运行时库 / ONNX / GPU 状态，
连同 Python 异常栈一起写入 logs/worker_crash_latest.log，供主进程展示。
移植自 VoxCaps b50c3318（去品牌化）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import platform
import traceback

from config_server import BASE_DIR


REPORT_PATH = Path(BASE_DIR) / 'logs' / 'worker_crash_latest.log'


def _runtime_diagnostics() -> list[str]:
    """检查共享 llama.cpp 运行库是否齐全。"""
    lines: list[str] = []
    try:
        from core.server.engines.llama_runtime import REQUIRED_RUNTIME_FILES, llama_runtime_dir

        runtime_dir = llama_runtime_dir()
        lines.append(f"runtime_dir: {runtime_dir}")
        for name in REQUIRED_RUNTIME_FILES:
            lines.append(f"{name}: exists={(runtime_dir / name).is_file()}")
    except Exception as error:
        lines.append(f"runtime check failed: {error!r}")
    return lines


def _onnx_diagnostics() -> list[str]:
    try:
        import onnxruntime

        return [f"onnxruntime version={onnxruntime.__version__}",
                f"available_providers={onnxruntime.get_available_providers()}"]
    except Exception as error:
        return [f"onnxruntime unavailable: {error!r}"]


def _gpu_diagnostics() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"Name : {name}"
        return "CUDA not available"
    except Exception as error:
        return f"torch unavailable: {error!r}"


def write_worker_crash_report(error: Exception, python_traceback: str) -> Path:
    """写入崩溃报告并返回报告路径。"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "Worker crash report",
        f"time: {datetime.now().isoformat(timespec='seconds')}",
        f"python: {platform.python_version()} / {platform.platform()}",
        "",
        "== exception ==",
        f"{type(error).__name__}: {error}",
        python_traceback or traceback.format_exc(),
        "",
        "== runtime libs ==",
        *_runtime_diagnostics(),
        "",
        "== onnxruntime ==",
        *_onnx_diagnostics(),
        "",
        "== gpu ==",
        _gpu_diagnostics(),
    ]
    REPORT_PATH.write_text('\n'.join(sections), encoding='utf-8')
    return REPORT_PATH
