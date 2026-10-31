# coding: utf-8
"""FireRedASR2-AED 纯 ONNX 链路的 CapsWriter 引擎适配器。

推理全在独立进程里跑（见 worker_client.py / fired_worker.py）：
FireRedVAD 切语音 -> FireRedASR2-AED 贪心解码（f32 编码器上 CUDA、int8 解码器留 CPU）
-> FireRedPunc 补标点。本适配器只负责把 float32 PCM 递过去、把文本与 token 拿回来。
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..base import BaseASREngine, RecognitionStream, EngineCapabilities
from .. import logger
from .worker_client import FireRedWorkerClient, WorkerError


@dataclass
class FireRedOnnxConfig:
    """与 config_server.FireRedOnnxArgs 的字段一一对应（工厂按字段名展开）。"""
    chain_dir: str
    model_dir: str
    punc_dir: str
    vad_dir: str
    python_exe: str
    sample_rate: int = 16000
    onnx_provider: str = 'cuda'
    graph: str = 'mixed'
    mode: str = 'greedy'
    threads: int = 4
    punc_threads: int = 4
    use_vad: bool = True
    piece_max: float = 10.0
    start_timeout: int = 240


class FireRedOnnxStream(RecognitionStream):
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
        except WorkerError as ex:
            logger.error(f"FireRedASR2-AED 识别失败：{ex}")
            return
        self.result.text = r.get("text", "")
        self.result.tokens = list(r.get("tokens") or [])
        self.result.timestamps = list(r.get("timestamps") or [])
        self.result.duration = float(r.get("audio_s") or 0.0)
        self.result.performance = r.get("ms") or {}
        ms = self.result.performance
        logger.info(f"FireRed AED：{r.get('audio_s')}s 音频 / {ms.get('total')}s "
                    f"(vad {ms.get('vad')} enc {ms.get('enc')} dec {ms.get('dec')} "
                    f"punc {ms.get('punc')}) RTF="
                    f"{(ms.get('total') or 0) / max(0.01, r.get('audio_s') or 0):.3f}")


class FireRedOnnxEngine(BaseASREngine):
    """FireRedASR2-AED；PUNC 与 TIMESTAMPS 都由本引擎给出（时间戳是语音段级粗粒度）。"""

    def __init__(self, config):
        super().__init__(config)
        self.client = FireRedWorkerClient(config)
        self.client.start()

    @property
    def capabilities(self) -> List[EngineCapabilities]:
        return [
            EngineCapabilities.ASR,
            EngineCapabilities.PUNC,
            EngineCapabilities.TIMESTAMPS,
        ]

    def create_stream(self, hotwords: Optional[str] = None) -> FireRedOnnxStream:
        return FireRedOnnxStream(self, sample_rate=self.config.sample_rate)

    def decode_stream(self, stream: FireRedOnnxStream,
                      context: Optional[str] = None,
                      language: Optional[str] = None, **kwargs):
        stream.finish()

    def cleanup(self):
        self.client.close()
