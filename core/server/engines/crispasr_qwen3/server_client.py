# coding: utf-8
"""CrispASR 常驻 HTTP 服务的客户端：拉起、健康检查、按请求送 WAV。

为什么是常驻服务而不是每次调 crispasr.exe：
CLI 每跑一次都要重新加载 2.51 GB（q8_0，2,506,723,200 B）的 qwen3 GGUF
（本机实测冷 30.6 s / 热 8.6 s），
语音输入等不起。`crispasr.exe --server` 把模型只加载一次，之后每次 POST 只做推理
（16.45 s 音频实测 3.1 s）。这里用标准库 urllib 说话，不额外依赖任何 HTTP 包。
"""
import io
import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

import numpy as np

from .. import logger

IS_WIN = os.name == "nt"
ROOT = Path(__file__).resolve().parents[4]
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class CrispAsrError(RuntimeError):
    pass


def _wav_bytes(pcm, sr: int) -> bytes:
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    pcm = np.clip(pcm, -1.0, 1.0)
    data = (pcm * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(data)
    return buf.getvalue()


class CrispAsrServerClient:
    def __init__(self, config):
        self.cfg = config
        self._proc = None
        self._log = None
        self._port = None
        self._lock = threading.Lock()
        self._log_path = ROOT / "logs" / "crispasr_qwen3_server.log"
        self._pid_path = ROOT / "logs" / "crispasr_qwen3_server.pid"

    # ---------- 生命周期 ----------
    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def _cmd(self):
        c = self.cfg
        cmd = [c.exe_path, "--server",
               "--host", "127.0.0.1", "--port", str(self._port),
               "--backend", c.backend,
               "-m", c.model_path,
               "--gpu-backend", c.gpu_backend]
        if c.language:
            cmd += ["-l", c.language]
        return cmd

    def _kill_stale(self):
        """上一次服务端被强杀时 crispasr 会变成孤儿并占着显存。
        按上次记下的 PID 收尾；/FI 限定镜像名，PID 被复用时也伤不到别的进程。"""
        try:
            pid = int(self._pid_path.read_text().strip())
        except Exception:
            return
        try:
            r = subprocess.run(
                ["taskkill", "/PID", str(pid), "/FI", "IMAGENAME eq crispasr.exe", "/F"],
                capture_output=True, text=True, timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW if IS_WIN else 0)
            if r.returncode == 0:
                logger.info(f"已清理上次残留的 CrispASR 服务 PID={pid}")
        except Exception:
            pass
        try:
            self._pid_path.unlink()
        except Exception:
            pass

    def start(self, timeout=None):
        c = self.cfg
        for p in (c.exe_path, c.model_path):
            if not os.path.exists(p):
                raise CrispAsrError(f"Qwen3-ASR 缺件：{p}")
        self._kill_stale()
        self._port = self._free_port()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self._log_path, "a", encoding="utf-8", errors="replace")
        self._log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start port={self._port} =====\n")
        self._log.flush()
        flags = subprocess.CREATE_NO_WINDOW if IS_WIN else 0
        logger.info(f"拉起 CrispASR Qwen3 服务：{c.exe_path} :{self._port} "
                    f"({c.gpu_backend}/{c.backend})")
        self._proc = subprocess.Popen(
            self._cmd(), cwd=str(Path(c.exe_path).parent),
            stdout=self._log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=flags)
        try:
            self._pid_path.write_text(str(self._proc.pid))
        except Exception:
            pass

        deadline = time.time() + (timeout or c.start_timeout)
        while True:
            if self._proc.poll() is not None:
                rc = self._proc.returncode
                self.close(force=True)
                raise CrispAsrError(f"CrispASR 服务启动即退出 rc={rc}"
                                    f"（日志：{self._log_path}）")
            if time.time() > deadline:
                self.close(force=True)
                raise CrispAsrError(f"CrispASR 服务 {timeout or c.start_timeout}s 内没就绪"
                                    f"（日志：{self._log_path}）")
            try:
                body = self._get("/health", timeout=1.0)
                info = json.loads(body.decode("utf-8"))
            except Exception:
                time.sleep(0.25)
                continue
            logger.info(f"Qwen3-ASR 服务就绪：{info.get('backend')} "
                        f"model={Path(c.model_path).name} port={self._port}")
            if c.warmup:
                self._warmup()
            return info

    def _warmup(self):
        t0 = time.time()
        try:
            noise = (np.random.RandomState(0).randn(int(0.5 * self.cfg.sample_rate))
                     * 1e-3).astype(np.float32)
            # 直接走 _post_wav：request() 里 ensure() 可能正握着锁重入到这里
            self._post_wav(_wav_bytes(noise, self.cfg.sample_rate), 120)
        except Exception:
            logger.warning("Qwen3-ASR 预热失败（不影响后续请求）", exc_info=True)
        logger.info(f"Qwen3-ASR 预热完成 {time.time() - t0:.2f}s")

    def _alive(self):
        return self._proc is not None and self._proc.poll() is None

    def ensure(self):
        if not self._alive():
            logger.warning("CrispASR Qwen3 服务不在了，重新拉起")
            self.close(force=True)
            self.start()

    # ---------- HTTP ----------
    def _get(self, path: str, timeout: float = 2.0) -> bytes:
        url = f"http://127.0.0.1:{self._port}{path}"
        with _OPENER.open(url, timeout=timeout) as r:
            return r.read()

    def _post_wav(self, wav: bytes, timeout: float) -> dict:
        boundary = "----crispasr" + uuid.uuid4().hex
        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n')
        body.write(b"Content-Type: audio/wav\r\n\r\n")
        body.write(wav)
        body.write(f"\r\n--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            f"http://127.0.0.1:{self._port}/inference", data=body.getvalue(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST")
        try:
            with _OPENER.open(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as ex:
            detail = ex.read().decode("utf-8", "replace")[:500]
            raise CrispAsrError(f"HTTP {ex.code}: {detail}")
        except urllib.error.URLError as ex:
            raise CrispAsrError(f"连不上 CrispASR 服务：{ex.reason}")

    def request(self, pcm, sr, timeout=None):
        with self._lock:
            self.ensure()
            dur = len(pcm) / max(1, int(sr))
            limit = timeout or (30.0 + 4.0 * dur)
            t0 = time.time()
            data = self._post_wav(_wav_bytes(pcm, int(sr)), limit)
            total = time.time() - t0
            text = (data.get("text") or "").strip()
            return {
                "text": text,
                "audio_s": float(data.get("duration") or dur),
                "segments": data.get("segments") or [],
                "ms": {"total": round(total, 3)},
            }

    def close(self, force=False):
        proc, log = self._proc, self._log
        self._proc = None
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except Exception:
                        proc.kill()
                        try:
                            proc.wait(timeout=3)
                        except Exception:
                            pass
            except Exception:
                pass
        if log is not None:
            try:
                log.close()
            except Exception:
                pass
        self._log = None
        try:
            self._pid_path.unlink()
        except Exception:
            pass
