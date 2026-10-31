# FireRedASR2-AED 纯 ONNX 链路

这是 CapsWriter-Offline fork 中 `firedasr_onnx` 引擎的推理实现：FireRedVAD 切语音 →
FireRedASR2-AED 贪心解码（f32 编码器 + int8 解码器）→ FireRedPunc 补标点。
全链路只依赖 onnxruntime，不需要 PyTorch、sherpa-onnx 或 funasr。

## 来源与改动

代码取自 [takomitkm/High-Accuracy-Offline-ASR](https://github.com/takomitkm/High-Accuracy-Offline-ASR)
的 `gpu` 分支 `CrispASR-FireRed/onnx/`（commit `1bac7e6`，2026-10-17），
`models/fireredvad-onnx/infer_onnx.py` 也一并放在本目录，方便 worker 直接 import。
该仓库的链代码以 Unlicense 捐献到公有领域，见 `licenses/UNLICENSE.txt`。

相对来源仓库，本 fork 只做了这些改动：

- 删掉适配器与注释里的本机绝对路径；
- `fetch_assets.py` 的默认落点改为本仓库的 `models/FireRedASR2-ONNX-AED/`；
- 增加 `requirements-cuda.txt`（实测的 CUDA 运行时）和 `licenses/`；
- 增加本 README 与 `THIRD_PARTY_NOTICES.md`。

推理逻辑、`GRAPH_BUILDS` 与逐件 sha256 清单未改，`fetch_assets.py --check` 仍按原哈希验收。

## 链路形状

1. `FireRedVAD` 把整段音频切成语音段，去掉长静音；
2. 语音段按停顿聚成不超过 `piece_max`（默认 10 s）的片段，单个超长段在 10 s 处硬切；
3. `FireRedASR2-AED` 对每个片段贪心解码，输出无标点文本；
4. `FireRedPunc` 逐片段补标点；
5. 片段文本按顺序拼接，字符在片段内按时间均匀铺开，供 CapsWriter 的字幕/时间戳使用。

时间戳是片段级粗粒度估计，不是强制对齐的结果。

## 文件

| 文件 | 作用 |
|---|---|
| `aed_ort.py` | AED 的 onnxruntime 推理（特征、编码器、单步解码器、贪心/beam） |
| `asr_chain.py` | 特征、VAD 包装、FireRedPunc、EP 选择等公共件 |
| `infer_onnx.py` | FireRedVAD 的 ONNX 推理（来自 FireRedVAD 上游，Apache-2.0） |
| `dequant_aed.py` | 把 AED 的 int8 动态量化图重写成 f32（本地建造，逐字节可复现） |
| `dequant_punc.py` | 把 Punc 的 weight-only 8-bit MatMulNBits 逆量化成 f32 |
| `fetch_assets.py` | 下载 + 建造 + 逐件 sha256 验收，资源清单与公开 URL 都在这里 |
| `requirements.txt` | CPU 版运行时依赖 |
| `requirements-cuda.txt` | CUDA 版运行时依赖（本 fork 实测） |
| `requirements-build.txt` | 建造 f32 图时额外需要的 `onnx` |

## 复原

运行时环境（Windows x64，CPython 3.12/3.13 都验证过）：

```bat
python -m venv models\FireRedASR2-ONNX-AED\env
models\FireRedASR2-ONNX-AED\env\Scripts\python -m pip install -r core\server\engines\firedasr_onnx\chain\requirements-cuda.txt
```

没有 CUDA 的机器把最后一行换成 `-r ...\requirements.txt`，并把
`FIREDASR_ONNX_PROVIDER` 设为 `cpu`；出厂默认是 `cuda`，请求了 CUDA 而 ORT 没有 CUDA EP 时
worker 会直接报错退出，不会静默退回 CPU。

下载并建造资源（`--graph mixed` 对应运行时的默认 `graph='mixed'`）：

```bat
python core\server\engines\firedasr_onnx\chain\fetch_assets.py --graph mixed
```

需要代理时加 `--proxy http://127.0.0.1:<你的代理端口>`（本机走的是代理，直连情况未测）。
建造 f32 需要 `onnx`：

```bat
python -m pip install -r core\server\engines\firedasr_onnx\chain\requirements-build.txt
```

如果 `onnx` 装在另一个解释器里，用 `--builder-python <那个python.exe>` 指过去。
`fetch_assets.py --check` 只核对现状，`--list` 打印所有 URL、字节数与 sha256。

下载件合计 1,004,110,068 B（AED tar 838,589,068 + Punc q8w 162,771,205 + VAD/分词器等），
建造件 `punc.f32.onnx` 406,954,097 B、`encoder.f32.onnx` 图 956,747 B + 权重
`encoder.f32.onnx.data` 3,103,167,616 B。磁盘上还要留 AED tar 解包后的
`encoder.int8.onnx` 817,286,833 B 与 `decoder.int8.onnx` 417,291,928 B。

完成后的目录：

```text
models/FireRedASR2-ONNX-AED/
├─ env/                                  # requirements-cuda.txt 装出来的运行时
├─ sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/
│  ├─ encoder.int8.onnx                  # 下载原料
│  ├─ decoder.int8.onnx                  # 运行时用这份
│  ├─ encoder.f32.onnx                   # 建造产物
│  ├─ encoder.f32.onnx.data              # 建造产物
│  └─ tokens.txt
├─ fireredpunc-onnx/
│  ├─ punc.q8w.onnx                      # 下载原料
│  ├─ punc.f32.onnx                      # 建造产物，运行时用这份
│  ├─ tokenizer.json
│  ├─ out_dict
│  └─ LICENSE
└─ fireredvad-onnx/
   ├─ model.onnx
   ├─ cmvn.bin / cmvn.json / config.json
   └─ README.md
```

`infer_onnx.py` 不必复制到 `fireredvad-onnx/`：worker 先把这个 chain 目录放进
`sys.path`，再从 VAD 目录 import；两处都找不到才会报错。

## 接入 CapsWriter

`config_server.py` 里把 `ServerConfig.model_type` 改成 `'firedasr_onnx'`，
默认路径就是上面的目录。模型放在别处时用环境变量覆盖：

| 环境变量 | 对应参数 |
|---|---|
| `FIREDASR_ONNX_MODEL_DIR` | AED 模型目录 |
| `FIREDASR_ONNX_PUNC_DIR` | FireRedPunc 目录 |
| `FIREDASR_ONNX_VAD_DIR` | FireRedVAD 目录 |
| `FIREDASR_ONNX_PYTHON` | 带 CUDA EP 的 `python.exe` |
| `FIREDASR_ONNX_CHAIN_DIR` | 本目录 |
| `FIREDASR_ONNX_PROVIDER` | `cuda`（默认）或 `cpu` |

这两路引擎没有打进 PyInstaller 生成的 `start_server.exe`，必须从源码运行：

```bat
python start_server.py
```

启动后日志里会有一行 `FireRedASR2-AED 就绪 enc=[...] dec=[...]`；worker 拒绝启动时，
详细原因在 `logs/firedasr_worker.log`。

## 实测

RTX 4060 Laptop，16.45 s 中文音频，热态：

| 项目 | 耗时 |
|---|---:|
| VAD | 0.085 s |
| AED 编码器 | 1.005 s |
| AED 解码器 | 2.610 s |
| FireRedPunc | 0.097 s |
| 引擎总计 | 3.858 s（RTF 0.235） |
| CapsWriter 整条 pipeline | 3.903 s（RTF 0.237） |

冷态首次请求约 6.2 s（RTF 0.377），相差的主要是 CUDA 选型与升频的一次性开销；
服务端启动时已经用 1.5 s 低噪预热过一遍。

## 边界与排查

- **只支持贪心档做默认**：`beam` 是官方 beam search 的 numpy 复刻，慢且没有稳定收益。
- **片段上限 10 s**：AED 每片解码预算固定，片段过长会撞顶丢句；不要随意改大 `piece_max`。
- **必须用独立 Python 环境**：CapsWriter 自带的 ORT 只有 DirectML/CPU，DirectML 可能挑中 AMD
  核显；CUDA 版 ORT 与自带那份同名同模块，不能装进同一个环境。
- **`encoder.f32.onnx` 缺失**：`fetch_assets.py` 只跑了 `--graph int8`；重跑 `--graph mixed`。
- **Punc 加载失败**：确认 `punc.f32.onnx` 在 `fireredpunc-onnx/`，且没有混用 `punc.q8w.onnx`
  （ORT 1.20 的 contrib 只支持 4-bit MatMulNBits，加载 q8w 会报 `Only 4b quantization is supported`）。
- 模型权重与二进制不入库；`fetch_assets.py` 的哈希是唯一验收口径。

## 许可

链代码来自上述仓库，Unlicense（公有领域）；`infer_onnx.py` 来自 FireRedVAD 的
Apache-2.0 发布件；FireRedPunc 的 LICENSE 随资源下载到 `fireredpunc-onnx/LICENSE`。
完整出处见 `THIRD_PARTY_NOTICES.md`。
