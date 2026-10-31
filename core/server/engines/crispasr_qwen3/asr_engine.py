# coding: utf-8
"""Qwen3-ASR-1.7B（CrispASR q8_0 GGUF）的 CapsWriter 引擎适配器。

模型由 CrispASR 的常驻 HTTP 服务持有（见 server_client.py）：本适配器只把
float32 PCM 递过去、把文本拿回来，再把字符均匀铺成时间戳供下游拼接/字幕用。
qwen3 自己出标点，所以不挂 CT-Transformer。
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..base import BaseASREngine, RecognitionStream, EngineCapabilities
from .. import logger
from .server_client import CrispAsrServerClient, CrispAsrError


@dataclass
class CrispAsrQwen3Config:
    """与 config_server.CrispAsrQwen3Args 的字段一一对应（工厂按字段名展开）。"""
    exe_path: str
    model_path: str
    backend: str = 'qwen3'
    language: str = 'zh'
    gpu_backend: str = 'cuda'
    sample_rate: int = 16000
    warmup: bool = True
    start_timeout: int = 240


class CrispAsrQwen3Stream(RecognitionStream):
    def __init__(self, engine, sample_rate: int = 16000):
        super().__init__(sample_rate)
        self.engine = engine
        self._audio = None
        self._sr = sample_rate

    def accept_waveform(self, sample_rate: int, audio):
        self._sr = int(sample_rate or 16000)
        self._audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    def finish(self):
        if self._audio is None or self._audio.size == 0:
            return
        try:
            r = self.engine.client.request(self._audio, self._sr)
        except CrispAsrError as ex:
            logger.error(f"Qwen3-ASR 识别失败：{ex}")
            return
        text = r.get("text", "")
        self.result.text = text
        self.result.duration = float(r.get("audio_s") or 0.0)
        self.result.performance = r.get("ms") or {}
        chars = [c for c in text if not c.isspace()]
        if chars and self.result.duration > 0:
            step = self.result.duration / len(chars)
            self.result.tokens = chars
            self.result.timestamps = [i * step for i in range(len(chars))]
        ms = self.result.performance
        total = ms.get("total") or 0.0
        logger.info(f"Qwen3-ASR：{r.get('audio_s')}s 音频 / {total}s "
                    f"RTF={(total / max(0.01, r.get('audio_s') or 0)):.3f}")


class CrispAsrQwen3Engine(BaseASREngine):
    """Qwen3-ASR-1.7B：自带标点；时间戳是字符均匀铺开的粗粒度估计。"""

    def __init__(self, config):
        super().__init__(config)
        self.client = CrispAsrServerClient(config)
        self.client.start()

    @property
    def capabilities(self) -> List[EngineCapabilities]:
        return [
            EngineCapabilities.ASR,
            EngineCapabilities.PUNC,
            EngineCapabilities.TIMESTAMPS,
        ]

    def create_stream(self, hotwords: Optional[str] = None) -> CrispAsrQwen3Stream:
        return CrispAsrQwen3Stream(self, sample_rate=self.config.sample_rate)

    def decode_stream(self, stream: CrispAsrQwen3Stream,
                      context: Optional[str] = None,
                      language: Optional[str] = None, **kwargs):
        stream.finish()

    def cleanup(self):
        self.client.close()
