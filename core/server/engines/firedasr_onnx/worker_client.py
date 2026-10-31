# coding: utf-8
"""常驻 GPU 工作进程的客户端：拉起、发请求、超时与重启。

为什么是独立进程而不是在本进程里 import onnxruntime：
CapsWriter 自带的 ORT 只有 DirectML/CPU，DirectML 可能挑中 AMD 核显；CUDA 版 ORT
又与自带那份同名同模块，装进同一个环境会互相顶掉。所以另建一个带 CUDA EP 的
Python 环境，把 aed_ort / FireRedPunc 放进独立进程，模型只加载一次，主进程只走
JSON 行协议。
"""
import base64
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from core.server.engines import logger

IS_WIN = os.name == "nt"
ROOT = Path(__file__).resolve().parents[4]
WORKER_PY = Path(__file__).resolve().parent / "fired_worker.py"


class WorkerError(RuntimeError):
    pass


class FireRedWorkerClient:
    def __init__(self, config):
        self.cfg = config
        self._proc = None
        self._q = None
        self._lines = None
        self._lock = threading.Lock()
        self._next_id = 0
        self._log_path = ROOT / "logs" / "firedasr_worker.log"

    # ---------- 生命周期 ----------
    def _cmd(self):
        c = self.cfg
        cmd = [c.python_exe, "-u", str(WORKER_PY),
               "--chain-dir", c.chain_dir,
               "--asr-dir", c.model_dir,
               "--punc-dir", c.punc_dir,
               "--provider", c.onnx_provider,
               "--threads", str(c.threads),
               "--punc-threads", str(c.punc_threads),
               "--graph", c.graph,
               "--mode", c.mode,
               "--piece-max", str(c.piece_max)]
        if c.use_vad and c.vad_dir:
            cmd += ["--vad-dir", c.vad_dir]
        else:
            cmd += ["--no-vad"]
        return cmd

    def _env(self):
        env = os.environ.copy()
        # 父进程是 PyInstaller 打出来的，这些变量会把外部解释器的 sys.path 带歪
        for k in ("PYTHONHOME", "PYTHONPATH", "_PYI_APPLICATION_HOME_DIR",
                  "_PYI_ARCHIVE_FILE", "_PYI_PARENT_PROCESS_LEVEL", "_MEIPASS2"):
            env.pop(k, None)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def start(self, timeout=None):
        c = self.cfg
        for p in (c.python_exe, c.model_dir):
            if not os.path.exists(p):
                raise WorkerError(f"小红书 ONNX 链路缺件：{p}")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self._log_path, "a", encoding="utf-8", errors="replace")
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start =====\n")
        log.flush()
        flags = subprocess.CREATE_NO_WINDOW if IS_WIN else 0
        logger.info(f"拉起 FireRedASR2-AED 工作进程：{c.python_exe} ({c.onnx_provider}/{c.graph})")
        self._proc = subprocess.Popen(
            self._cmd(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log, cwd=str(WORKER_PY.parent), env=self._env(),
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=flags)
        self._log = log
        self._q = queue.Queue()
        self._lines = threading.Thread(target=self._pump, daemon=True,
                                       name="fired-worker-reader")
        self._lines.start()
        buf = []
        deadline = time.time() + (timeout or c.start_timeout)
        while True:
            left = deadline - time.time()
            if left <= 0:
                self.close(force=True)
                raise WorkerError(f"工作进程 {c.start_timeout}s 内没就绪（日志：{self._log_path}）")
            try:
                item = self._q.get(timeout=min(left, 1.0))
            except queue.Empty:
                if self._proc.poll() is not None:
                    raise WorkerError(f"工作进程启动即退出 rc={self._proc.returncode}"
                                      f"（日志：{self._log_path}）")
                continue
            if item is None:
                raise WorkerError(f"工作进程在就绪前退出 rc={self._proc.poll()}"
                                  f"（日志：{self._log_path}）")
            try:
                msg = json.loads(item)
            except Exception:
                buf.append(item)
                continue
            if msg.get("fatal"):
                self.close(force=True)
                raise WorkerError(f"工作进程拒绝启动：{msg['fatal']}（日志：{self._log_path}）")
            if msg.get("ready"):
                logger.info(f"FireRedASR2-AED 就绪 enc={msg.get('enc_ep')} "
                            f"dec={msg.get('dec_ep')} vad={msg.get('vad')} "
                            f"load={msg.get('load_s')}s")
                return msg

    def _pump(self):
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if line:
                    self._q.put(line)
        except Exception:
            pass
        finally:
            self._q.put(None)

    def _alive(self):
        return self._proc is not None and self._proc.poll() is None

    def ensure(self):
        if not self._alive():
            logger.warning("FireRedASR2-AED 工作进程不在了，重新拉起")
            self.close(force=True)
            self.start()

    def request(self, pcm, sr, timeout=None):
        with self._lock:
            self.ensure()
            self._next_id += 1
            rid = self._next_id
            payload = {
                "id": rid, "sr": int(sr),
                "pcm": base64.b64encode(pcm.astype("<f4").tobytes()).decode("ascii"),
            }
            t0 = time.time()
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except Exception as ex:
                self.close(force=True)
                raise WorkerError(f"写工作进程失败：{ex}")
            dur = len(pcm) / max(1, sr)
            limit = timeout or (30.0 + 6.0 * dur)
            while True:
                left = limit - (time.time() - t0)
                if left <= 0:
                    self.close(force=True)
                    raise WorkerError(f"识别超时（{dur:.1f}s 音频等了 {limit:.0f}s）")
                try:
                    line = self._q.get(timeout=left)
                except queue.Empty:
                    continue
                if line is None:
                    rc = self._proc.poll()
                    self.close(force=True)
                    raise WorkerError(f"工作进程中途退出 rc={rc}（日志：{self._log_path}）")
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                if msg.get("id") != rid:
                    continue
                if not msg.get("ok"):
                    raise WorkerError(msg.get("error", "未知错误"))
                return msg

    def close(self, force=False):
        proc, log = self._proc, getattr(self, "_log", None)
        self._proc = None
        if proc is not None:
            try:
                if proc.poll() is None:
                    if not force:
                        try:
                            proc.stdin.write('{"cmd": "quit"}\n')
                            proc.stdin.flush()
                        except Exception:
                            pass
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=3)
                    except Exception:
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except Exception:
                            proc.kill()
            except Exception:
                pass
        if log is not None:
            try:
                log.close()
            except Exception:
                pass
        self._log = None
        self._q = None
