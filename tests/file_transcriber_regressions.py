# coding: utf-8
"""文件转录未绑定变量修复的运行时测试（chore/unbound-fix-and-check 分支）。"""
import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

client_pkg = types.ModuleType("core.client")
client_pkg.__path__ = [str(REPO / "core" / "client")]
client_pkg.logger = __import__("logging").getLogger("fake.client")
sys.modules["core.client"] = client_pkg

conn_pkg = types.ModuleType("core.client.connection")
conn_pkg.__path__ = [str(REPO / "core" / "client" / "connection")]
conn_pkg.WebSocketManager = object
sys.modules["core.client.connection"] = conn_pkg

trans_pkg = types.ModuleType("core.client.transcribe")
trans_pkg.__path__ = [str(REPO / "core" / "client" / "transcribe")]
trans_pkg.logger = __import__("logging").getLogger("fake.transcribe")
sys.modules["core.client.transcribe"] = trans_pkg


class MediaTool:
    @staticmethod
    async def get_audio_duration(file):
        return 0.0

    @staticmethod
    def build_ffmpeg_cmd(file):
        return ["ffmpeg"]


class ResultHandler:
    calls = []

    @staticmethod
    def save_results(file, message):
        ResultHandler.calls.append((file, message))
        return "saved"


media = types.ModuleType("core.client.transcribe.media_tool")
media.MediaTool = MediaTool
sys.modules["core.client.transcribe.media_tool"] = media

handler = types.ModuleType("core.client.transcribe.result_handler")
handler.ResultHandler = ResultHandler
sys.modules["core.client.transcribe.result_handler"] = handler

token_sync = types.ModuleType("core.tools.token_sync")
token_sync.sync_tokens_from_text = lambda *a, **k: None
sys.modules["core.tools.token_sync"] = token_sync

cfg = types.ModuleType("config_client")
cfg.ClientConfig = SimpleNamespace(
    file_seg_duration=15.0,
    file_seg_overlap=2.0,
    context="",
    language="auto",
)
sys.modules["config_client"] = cfg

spec = importlib.util.spec_from_file_location(
    "core.client.transcribe.file_transcriber",
    REPO / "core" / "client" / "transcribe" / "file_transcriber.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class FakeConsole:
    def __init__(self):
        self.lines = []

    def print(self, *a, **k):
        self.lines.append(" ".join(str(x) for x in a))


class FakeWS:
    def __init__(self, msgs=()):
        self.msgs = list(msgs)
        self.sent = []

    async def receive(self):
        return self.msgs.pop(0) if self.msgs else None

    async def send(self, message):
        self.sent.append(message)
        return True


class FakeStdout:
    def __init__(self, data=b""):
        self.data = data

    async def read(self, n):
        out, self.data = self.data, b""
        return out


class FakeProc:
    def __init__(self, stdout=None):
        self.stdout = FakeStdout() if stdout is None else stdout
        self.returncode = None
        self.terminated = False

    async def wait(self):
        self.returncode = 0
        return 0

    def terminate(self):
        self.terminated = True


def msg(final=False):
    return SimpleNamespace(
        error="", is_final=final, duration=1.0,
        time_complete=10.0, time_start=5.0, text="ok",
    )


def make(msgs=()):
    t = mod.FileTranscriber.__new__(mod.FileTranscriber)
    t.app = SimpleNamespace(ws=FakeWS(msgs))
    t.file = Path("demo.mp3")
    t._apply_hotwords = lambda m: None
    return t


async def main():
    fake = FakeConsole()
    mod.console = fake

    # 1) 连接断开：receive() 返回 None，不能抛 NameError，也不能保存
    ResultHandler.calls.clear()
    t = make([None])
    await t.receive()
    assert ResultHandler.calls == [], ResultHandler.calls

    # 2) 正常终帧仍然保存一次
    ResultHandler.calls.clear()
    t = make([msg(final=True)])
    await t.receive()
    assert len(ResultHandler.calls) == 1, ResultHandler.calls

    # 3) 空音频：while 不执行，progress 未赋值也不能崩
    t = make()
    proc = FakeProc()
    original_create = mod.asyncio.create_subprocess_exec

    async def fake_create(*a, **k):
        return proc

    mod.asyncio.create_subprocess_exec = fake_create
    try:
        await t.send()
    finally:
        mod.asyncio.create_subprocess_exec = original_create
    assert t._audio_duration == 0.0, t._audio_duration
    assert len(t.app.ws.sent) == 1, t.app.ws.sent

    # 4) 子进程创建失败：process 为 None，terminate 分支不能崩
    t = make()

    async def fail_create(*a, **k):
        raise OSError("no ffmpeg")

    mod.asyncio.create_subprocess_exec = fail_create
    try:
        await t.send()
    finally:
        mod.asyncio.create_subprocess_exec = original_create

    # 5) stdout 为 None：显式判断后走 ConnectionError 分支并 terminate
    t = make()
    proc = FakeProc(stdout=None)
    proc.stdout = None
    mod.asyncio.create_subprocess_exec = fake_create
    try:
        await t.send()
    finally:
        mod.asyncio.create_subprocess_exec = original_create
    assert proc.terminated, "stdout 为 None 时应终止子进程"

    print("FILETRANS_UNBOUND_OK")


asyncio.run(main())