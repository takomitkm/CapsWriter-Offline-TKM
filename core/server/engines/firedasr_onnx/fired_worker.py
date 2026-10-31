# coding: utf-8
"""CapsWriter 的 FireRedASR2-AED 常驻工作进程。

父进程是 CapsWriter 服务端的识别子进程。本文件必须用带 CUDA EP 的独立 Python 环境跑
（见 chain/README.md）：CapsWriter 自带的那份 onnxruntime 只有 DirectML/CPU 两个 EP，
DirectML 可能挑中 AMD 核显。

协议（stdin/stdout 逐行 JSON，stderr 归父进程写的日志）：
  父 -> 子  {"id": 1, "sr": 16000, "pcm": "<base64 float32 LE mono>"}
  子 -> 父  {"id": 1, "ok": true, "text": "...", "tokens": [...], "timestamps": [...], ...}
            {"id": 1, "ok": false, "error": "..."}
  启动成功   {"ready": true, "enc_ep": [...], "dec_ep": [...], ...}
  退出       {"cmd": "quit"} 或 stdin EOF
"""
import argparse
import base64
import json
import os
import sys
import tempfile
import time
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(msg):
    sys.stderr.write(str(msg) + "\n")
    sys.stderr.flush()


def need_space(a, b):
    """中文直接拼；两侧都是 ASCII 字母数字时补一个空格，免得英文单词粘在一起。"""
    return bool(a) and bool(b) and a[-1].isascii() and a[-1].isalnum() \
        and b[0].isascii() and b[0].isalnum()


def plan_pieces(segs, dur, piece_max):
    """把 VAD 段按停顿归组成不超过 piece_max 秒的片段。

    整段一次编码上下文最全，但解码每步代价随片段变长暴涨：16.45 s 音频整段
    RTF 0.38，而逐 VAD 段 0.22。逐段又太碎，会把“寄递网络”解成“Gd 网络”。
    实测按停顿聚到 ≤10 s 既保住整段的输出、耗时也接近逐段，所以默认 10 s。
    单个 VAD 段本身超长时在 piece_max 处硬切。
    """
    if not segs:
        return []
    out = []
    cur_s = cur_e = None
    for s, e in segs:
        s = max(0.0, s)
        e = min(dur, e)
        while e - s > piece_max:
            if cur_s is not None:
                out.append((cur_s, cur_e))
                cur_s = cur_e = None
            out.append((s, s + piece_max))
            s += piece_max
        if e - s <= 0.1:
            continue
        if cur_s is None:
            cur_s, cur_e = s, e
        elif e - cur_s <= piece_max:
            cur_e = e
        else:
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_s is not None:
        out.append((cur_s, cur_e))
    return out


def recognize(wav, sr, aed, punc, vad, tmp_wav, np, sf, piece_max=10.0):
    t_all = time.perf_counter()
    dur = len(wav) / sr
    ms = {"vad": 0.0, "enc": 0.0, "dec": 0.0, "punc": 0.0}
    res = {"text": "", "tokens": [], "timestamps": [], "pieces": [],
           "audio_s": round(dur, 3), "speech_s": 0.0, "ms": ms}
    if dur < 0.1:
        return res

    if vad is not None:
        t0 = time.perf_counter()
        sf.write(tmp_wav, wav, sr, subtype="PCM_16")
        out, _ = vad.detect(tmp_wav)
        segs = [(float(a), float(b)) for a, b in out["timestamps"]]
        ms["vad"] = round(time.perf_counter() - t0, 3)
    else:
        segs = [(0.0, dur)]
    res["speech_s"] = round(sum(e - s for s, e in segs), 3)

    pieces = plan_pieces(segs, dur, piece_max)
    tokens, stamps = [], []
    for s0, e0 in pieces:
        seg = wav[int(s0 * sr): int(e0 * sr)]
        if seg.size < int(0.25 * sr):
            continue
        aed.times = {"feat": 0.0, "enc": 0.0, "dec": 0.0}
        aed.steps = 0
        r = aed.transcribe_wav(seg, sr)
        ms["enc"] += aed.times["enc"]
        ms["dec"] += aed.times["dec"]
        raw = r["text"].strip()
        if not raw:
            continue
        tp = time.perf_counter()
        try:
            txt = punc.add(raw) if punc is not None else raw
        except Exception as ex:
            log(f"[punc] {type(ex).__name__}: {ex}")
            txt = raw
        ms["punc"] += time.perf_counter() - tp
        res["pieces"].append({"start": round(s0, 2), "end": round(e0, 2),
                              "chars": len(raw), "steps": r["decode_steps"]})
        if tokens and need_space(tokens[-1], txt[0]):
            tokens.append(" ")
            stamps.append(round(s0, 3))
        chars = list(txt)
        step = (e0 - s0) / max(1, len(chars))
        tokens.extend(chars)
        stamps.extend(round(s0 + i * step, 3) for i in range(len(chars)))

    for k in ms:
        ms[k] = round(ms[k], 3)
    ms["total"] = round(time.perf_counter() - t_all, 3)
    res["text"] = "".join(tokens)
    res["tokens"] = tokens
    res["timestamps"] = stamps
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-dir", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--punc-dir", required=True)
    ap.add_argument("--vad-dir", default="")
    ap.add_argument("--provider", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--punc-threads", type=int, default=4)
    ap.add_argument("--graph", default="mixed")
    ap.add_argument("--mode", default="greedy")
    ap.add_argument("--piece-max", type=float, default=10.0)
    ap.add_argument("--no-vad", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, a.chain_dir)
    import numpy as np
    import soundfile as sf
    import aed_ort
    import asr_chain

    t0 = time.perf_counter()
    enc_ep = asr_chain.ep_list(a.provider)
    dec_ep = enc_ep if a.graph == "f32" else ["CPUExecutionProvider"]
    aed = aed_ort.AedOnnx(a.asr_dir, threads=a.threads, mode=a.mode, graph=a.graph,
                          cache_len="auto", enc_providers=enc_ep, dec_providers=dec_ep)
    if a.provider == "cuda" and "CUDAExecutionProvider" not in aed.enc_ep:
        emit({"fatal": f"请求了 CUDA，但编码器会话落在 {aed.enc_ep}（拒绝静默退回 CPU）"})
        return 2
    punc = asr_chain.Punc(threads=a.punc_threads, model_dir=a.punc_dir, provider="cpu")
    vad = None
    if not a.no_vad and a.vad_dir:
        sys.path.insert(0, a.vad_dir)
        from infer_onnx import FireRedVadOnnx
        vad = FireRedVadOnnx(model_dir=a.vad_dir)
    tmpdir = tempfile.mkdtemp(prefix="cw_fired_")
    tmp_wav = os.path.join(tmpdir, "in.wav")

    # 首次推理要付 CUDA 选型/升频的一次性开销（同一段 16 s 音频首请求 6.2 s、热态 3.7 s）。
    # 用 1.5 s 低噪先跑一遍，别让开机后的第一句听写吃这 2 秒。
    t_warm = time.perf_counter()
    try:
        aed.transcribe_wav((np.random.RandomState(0).randn(int(1.5 * 16000))
                            * 1e-3).astype(np.float32), 16000)
    except Exception:
        log(traceback.format_exc())
    log(f"[warmup] {time.perf_counter() - t_warm:.2f}s")

    emit({"ready": True, "enc_ep": aed.enc_ep, "dec_ep": aed.dec_ep,
          "punc_ep": punc.ep, "vad": vad is not None,
          "weight_mb": aed.weight_mb, "load_s": round(time.perf_counter() - t0, 2)})
    log(f"[ready] enc={aed.enc_ep} dec={aed.dec_ep} punc={punc.ep} "
        f"vad={vad is not None} load={time.perf_counter() - t0:.1f}s")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            log(f"[bad-line] {line[:120]}")
            continue
        if req.get("cmd") == "quit":
            break
        rid = req.get("id")
        try:
            pcm = np.frombuffer(base64.b64decode(req["pcm"]), dtype=np.float32)
            sr = int(req.get("sr", 16000))
            if sr != 16000:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(sr, 16000)
                pcm = resample_poly(pcm, 16000 // g, sr // g).astype(np.float32)
                sr = 16000
            r = recognize(pcm, sr, aed, punc, vad, tmp_wav, np, sf, piece_max=a.piece_max)
            r.update({"id": rid, "ok": True})
            emit(r)
        except Exception as ex:
            log(traceback.format_exc())
            emit({"id": rid, "ok": False, "error": f"{type(ex).__name__}: {ex}"})
    log("[bye] stdin closed")


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception as ex:
        log(traceback.format_exc())
        emit({"fatal": f"{type(ex).__name__}: {ex}"})
        sys.exit(1)
