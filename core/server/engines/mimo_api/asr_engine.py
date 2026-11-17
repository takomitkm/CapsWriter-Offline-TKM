# coding: utf-8
"""MiMo 云端 ASR 的 CapsWriter 引擎适配器。

识别走小米 MiMo 的 OpenAI 兼容接口（chat.completions + input_audio）：
本适配器把 float32 PCM 转成 WAV 递过去、把文本拿回来，再把字符均匀铺成
时间戳供下游拼接/字幕用。云端模型自带标点，不挂 CT-Transformer。

与本地引擎的区别：没有模型文件、没有独立进程，密钥与地址走配置；
断网/欠费/超时会直接在日志里报错并给空结果，不会拖垮服务端。
"""
import base64
import io
import struct
import time
import wave
from dataclasses import dataclass
from typing import List, Optional

import httpx

from ..base import BaseASREngine, RecognitionStream, EngineCapabilities
from .. import logger


@dataclass
class MimoApiConfig:
    """与 config_server.MimoApiArgs 的字段一一对应（工厂按字段名展开）。"""
    api_key: str = ''
    base_url: str = 'https://token-plan-cn.xiaomimimo.com/v1'
    model: str = 'mimo-v2.5-asr'
    language: str = 'auto'
    sample_rate: int = 16000
    timeout: int = 60
    max_upload_bytes: int = 10 * 1024 * 1024  # 官方限制：base64 后不超过 10MB


def _pcm_to_wav_bytes(pcm: 'array.array', sample_rate: int) -> bytes:
    """float32 单声道 PCM -> 16bit WAV 字节流。"""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        # clip 到 [-1, 1] 再量化，避免爆音
        frames = bytearray()
        for s in pcm:
            v = max(-1.0, min(1.0, float(s)))
            frames += struct.pack('<h', int(v * 32767))
        w.writeframes(bytes(frames))
    return buf.getvalue()


class MimoApiStream(RecognitionStream):
    def __init__(self, engine, sample_rate: int = 16000):
        super().__init__(sample_rate)
        self.engine = engine
        self._audio = None
        self._sr = sample_rate

    def accept_waveform(self, sample_rate: int, audio):
        self._sr = int(sample_rate or 16000)
        self._audio = list(audio)

    def finish(self):
        if not self._audio:
            return
        started = time.time()
        try:
            text = self.engine.transcribe(self._audio, self._sr)
        except Exception as ex:
            logger.error(f"MiMo 云端识别失败：{type(ex).__name__}: {ex}")
            return
        elapsed = time.time() - started
        self.result.text = text
        audio_s = len(self._audio) / max(1, self._sr)
        self.result.duration = audio_s
        # 云端接口不回时间戳：与本地引擎同策略，字符均匀铺开
        chars = [c for c in text if not c.isspace()]
        if chars and audio_s > 0:
            step = audio_s / len(chars)
            self.result.tokens = chars
            self.result.timestamps = [i * step for i in range(len(chars))]
        rtf = elapsed / max(0.01, audio_s)
        logger.info(f"MiMo 云端：{audio_s:.2f}s 音频 / {elapsed:.2f}s RTF={rtf:.3f}")


class MimoApiEngine(BaseASREngine):
    """MiMo 云端 ASR：自带标点；需要网络与有效密钥。"""

    def __init__(self, config: MimoApiConfig):
        super().__init__(config)
        self.client = httpx.Client(timeout=config.timeout)

    @property
    def capabilities(self) -> List[EngineCapabilities]:
        return [
            EngineCapabilities.ASR,
            EngineCapabilities.PUNC,
            EngineCapabilities.TIMESTAMPS,
        ]

    def transcribe(self, pcm: 'array.array', sample_rate: int) -> str:
        cfg = self.config
        wav = _pcm_to_wav_bytes(pcm, sample_rate)
        b64 = base64.b64encode(wav).decode('ascii')
        if len(b64) > cfg.max_upload_bytes:
            raise ValueError(f"音频过大（base64 后 {len(b64)} 字节 > 上限），请缩短单段时长")

        resp = self.client.post(
            f"{cfg.base_url.rstrip('/')}/chat/completions",
            headers={'Authorization': f'Bearer {cfg.api_key}'},
            json={
                'model': cfg.model,
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'input_audio',
                             'input_audio': {'data': f'data:audio/wav;base64,{b64}'}},
                        ],
                    }
                ],
                'extra_body': {'asr_options': {'language': cfg.language}},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return (data['choices'][0]['message'].get('content') or '').strip()

    def create_stream(self, hotwords: Optional[str] = None) -> MimoApiStream:
        return MimoApiStream(self, sample_rate=self.config.sample_rate)

    def decode_stream(self, stream: MimoApiStream,
                      context: Optional[str] = None,
                      language: Optional[str] = None, **kwargs):
        # 云端引擎整段识别，language 在请求里带；这里只负责收尾
        stream.finish()

    def cleanup(self):
        self.client.close()
