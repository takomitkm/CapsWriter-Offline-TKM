# coding: utf-8
"""
识别子进程工作包 (Worker Package)

包含模型加载、任务处理和 Worker 门面类。
"""

from multiprocessing import Queue
from multiprocessing.managers import ListProxy
import faulthandler
from pathlib import Path
from config_server import BASE_DIR
from .. import logger
from .worker import RecognizerWorker

def start_worker(queue_in: Queue, queue_out: Queue, sockets_id: ListProxy, stdin_fn: int):
    """识别子进程启动入口"""
    native_log_path = Path(BASE_DIR) / 'logs' / 'worker_native_crash_latest.log'
    native_log_path.parent.mkdir(parents=True, exist_ok=True)
    with native_log_path.open('w', encoding='utf-8') as native_log:
        native_log.write('Worker native crash trace\n')
        native_log.flush()
        faulthandler.enable(file=native_log, all_threads=True)
        try:
            worker = RecognizerWorker(queue_in, queue_out, sockets_id, stdin_fn)
            worker.run()
        except Exception as e:
            # 主进程在等 True/错误状态；把真实异常打包回传，避免只看到裸退出码
            from .diagnostics import write_worker_crash_report
            report = write_worker_crash_report(e, '')
            queue_out.put({
                'kind': 'worker_startup_error',
                'message': f'{type(e).__name__}: {e}',
                'traceback': '',
                'diagnostics_file': str(report),
            })
            return
        finally:
            faulthandler.disable()

__all__ = ['RecognizerWorker', 'start_worker']
