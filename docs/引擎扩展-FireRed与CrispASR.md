# 引擎扩展：FireRedASR2-AED 纯 ONNX 与 CrispASR Qwen3-ASR

这篇文档只讲本 fork 新增的两路引擎。上游的 `paraformer` / `sensevoice` /
`fun_asr_nano` / `qwen_asr` 四路照旧，配置与下载方式见上游文档
（`docs/模型下载的若干问题.md` 等）。

TL;DR：在 `config_server.py` 里把 `ServerConfig.model_type` 改成
`'firedasr_onnx'` 或 `'crispasr_qwen3'`，按下面对应小节把模型补齐，然后用
`python start_server.py` 启动。打包好的 `start_server.exe` 里没有这两路代码，
从 exe 启动只会得到"不支持的模型类型"。

| `model_type` | 链路 | 依赖形态 | 实测 RTF（16.45 s 中文音频，热态） |
|---|---|---|---:|
| `firedasr_onnx` | FireRedVAD → FireRedASR2-AED（f32 编码器 + int8 解码器，贪心）→ FireRedPunc | 纯 onnxruntime，独立 Python 进程 | 0.235 |
| `crispasr_qwen3` | Qwen3-ASR-1.7B q8_0 GGUF，CrispASR 常驻 HTTP 服务 | 单个 `crispasr.exe` + CUDA DLL | 0.178~0.189 |

两路的准确率、显存占用和适用场景没有在本仓库里做过横向评测，表里只给速度。
测试机是 RTX 4060 Laptop（8 GB 显存）、Windows x64、CPython 3.13。

---

## 0. 五件先要弄清的事

1. **必须从源码运行。** `start_server.exe` 是 PyInstaller 打的，`core/` 被编译进去时
   还没有这两路引擎，运行时找不到 `engines/firedasr_onnx/` 与 `chain/` 里的 `.py`。
   同理，`chain/` 目录必须真实存在于磁盘上，不能只留 exe。
2. **模型与二进制不入库。** 两路运行时真正要读的外部件合计 7,390,806,133 B
   （≈7.39 GB）：`firedasr_onnx` 3,931,201,719 B（AED 3,521,495,463 + Punc 目录
   407,223,327 + VAD 目录 2,482,929），`crispasr_qwen3` 3,459,604,414 B
   （程序包 146,153,022 + 三个 CUDA DLL 806,728,192 + q8_0 权重 2,506,723,200）。
   这是"装完之后"的必需件口径，不含 FireRed 下载原料与另建的 int8 编码器，
   也不含 FireRed 那个独立 CUDA Python 环境。全部在 `.gitignore` 里，
   `git clone` 之后必须按下面两节各跑一次下载脚本或手工摆放。
3. **只用 NVIDIA CUDA。** 两路的出厂默认都是 `cuda`，不是因为别的后端不能用，
   而是本 fork 只在 CUDA 上实测过。FireRed 这一路如果请求了 CUDA 而 onnxruntime
   没有 CUDA EP，worker 会直接报错退出，不会静默退回 CPU——静默退回会让速度掉到
   没测过的量级却看不出原因，宁可让它显式失败。想跑 CPU 就把
   `FIREDASR_ONNX_PROVIDER` 显式设成 `cpu`。
4. **`hot-server.txt` 对这两路不起作用。** 服务端热词只在引擎声明
   `EngineCapabilities.HOTWORDS` 时才注入解码，这两路都没有（FireRed 是贪心 ONNX、
   Qwen3 走 CrispASR 的 HTTP 接口，都没有热词入口）。客户端的 `hot.txt` 音素模糊替换
   和 `hot-rule.txt` 正则替换照常生效，因为它们发生在识别之后、与引擎无关。
5. **时间戳是估计值，不是对齐结果。** 两路都声明了 `EngineCapabilities.TIMESTAMPS`
   （FireRed 在片段内按字符均匀铺开，CrispASR 同理），所以 `ModelLoader` 不会再挂
   ForceAligner。做字幕够用，要逐字精确对齐请换上游的 `qwen_asr`——它只声明
   ASR + PUNC，加载时会把对齐器挂上。

---

## 1. `firedasr_onnx`：FireRedASR2-AED 纯 ONNX 链路

代码随仓库，位置 `core/server/engines/firedasr_onnx/`：
`asr_engine.py` 是给 CapsWriter 的引擎适配器，`worker_client.py` 负责拉起并维持
一个独立 Python 子进程，`fired_worker.py` 是子进程本体，`chain/` 是推理实现。

### 为什么要独立子进程

CapsWriter 自带的 onnxruntime 只有 DirectML 与 CPU 两个 Execution Provider（打包目录里
有 `DirectML.dll`、没有 CUDA provider）。CUDA 版 onnxruntime 与它同名同模块，装进同一个
环境会互相顶掉，所以推理只能另起一个带 CUDA EP 的 Python 环境，主进程通过 stdin/stdout
的逐行 JSON 协议说话。代价是启动多 9~15 s（模型加载在那个子进程里），换来的是编码器能吃到 CUDA。

那个子进程环境里不装 sherpa-onnx、funasr 或 PyTorch，`requirements-cuda.txt` 只有 8 个包。
CapsWriter 主进程自带的 sherpa-onnx 与这条链无关（上游只有 Paraformer 和 CT-Transformer
标点调它，SenseVoice 跟本链一样直接用 onnxruntime）。AED 权重是从 k2-fsa/sherpa-onnx 的
Release 下载的，所以目录名叫 `sherpa-onnx-fire-red-asr2-...`——那是模型来源，不是依赖。

### 复原步骤

细节日志在 `core/server/engines/firedasr_onnx/chain/README.md`，摘要如下。

```bat
:: 1) 建 CUDA 运行时环境（注意不是 CapsWriter 自己那个 env）
python -m venv models\FireRedASR2-ONNX-AED\env
models\FireRedASR2-ONNX-AED\env\Scripts\python -m pip install ^
  -r core\server\engines\firedasr_onnx\chain\requirements-cuda.txt

:: 2) 下载 + 本地建造 f32 图 + 逐件 sha256 验收
python core\server\engines\firedasr_onnx\chain\fetch_assets.py --graph mixed
```

`fetch_assets.py` 的默认落点就是 `models/FireRedASR2-ONNX-AED/`，需要代理时加
`--proxy http://127.0.0.1:<你的代理端口>`（测试机走了代理，直连是否可用未测）。建造 `encoder.f32.onnx` 还要
`onnx` 包：`pip install -r chain\requirements-build.txt`，如果装在另一个解释器里
就用 `--builder-python <那个python.exe>` 指过去。`--check` 只核对现状，`--list`
打印全部 URL、字节数与 sha256。

### 路径与参数

`config_server.py` 里 `FireRedOnnxArgs` 的五个路径都支持环境变量覆盖，
不覆盖就用仓库内的默认位置：

| 环境变量 | 指向 | 默认（相对仓库根） |
|---|---|---|
| `FIREDASR_ONNX_MODEL_DIR` | AED 模型目录 | `models/FireRedASR2-ONNX-AED/sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26/` |
| `FIREDASR_ONNX_PUNC_DIR` | FireRedPunc 目录 | `models/FireRedASR2-ONNX-AED/fireredpunc-onnx/` |
| `FIREDASR_ONNX_VAD_DIR` | FireRedVAD 目录 | `models/FireRedASR2-ONNX-AED/fireredvad-onnx/` |
| `FIREDASR_ONNX_PYTHON` | 带 CUDA EP 的 `python.exe` | `models/FireRedASR2-ONNX-AED/env/Scripts/python.exe` |
| `FIREDASR_ONNX_CHAIN_DIR` | 链代码目录 | `core/server/engines/firedasr_onnx/chain/` |
| `FIREDASR_ONNX_PROVIDER` | `cuda`（默认）或 `cpu` | `cuda` |

其余参数（`graph='mixed'`、`mode='greedy'`、`piece_max=10.0`、`threads=4`）在
`FireRedOnnxArgs` 里改，含义见 chain/README.md 的"链路形状"。

这一路没有 `language` 参数：模型是 `sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26`，
只覆盖中/英，其余语种没有可用的解码图。`FireRedPunc` 的标签集是
`<space> ， 。 ？ ！` 五类（`out_dict`），不会补顿号、分号之类。

---

## 2. `crispasr_qwen3`：Qwen3-ASR-1.7B（CrispASR GGUF）

代码位置 `core/server/engines/crispasr_qwen3/`：`asr_engine.py` 是适配器，
`server_client.py` 负责拉起 `crispasr.exe --server` 并维持这个常驻 HTTP 服务。

### 为什么是常驻服务

`crispasr.exe` 每跑一次都要重新加载 2,506,723,200 B 的 q8_0 GGUF，本机实测冷
30.6 s、热 8.6 s，语音输入等不起。`--server` 把模型只加载一次，之后每个请求只做
推理，16.45 s 音频实测 3.1 s。适配器用随机空闲端口在 `127.0.0.1` 上起服务、
轮询 `/health` 等就绪、把 PCM 包成 WAV 以 multipart POST 给 `/inference`，
只用标准库，不额外依赖 HTTP 包。

服务被强杀时 CrispASR 会变成孤儿进程继续占着显存，所以适配器会把 PID 写进
`logs/crispasr_qwen3_server.pid`，下次启动按 PID 加镜像名条件收尾；服务意外消失时
`ensure()` 会重新拉起。停止请发 SIGTERM/正常退出，别直接关窗口。

### 下载清单

这一路是"外部二进制 + 权重"，脚本不替你下载，按表摆放即可。URL 与 sha256
来自 CrispASR 官方 Release 与 HuggingFace，本机文件已逐件核对。

程序包（CrispASR v0.8.37，git sha `d08ec2dd`，2026-09-25）：

| 文件 | 字节 | sha256 |
|---|---:|---|
| [crispasr-windows-x86_64-cuda-non-cuda.zip](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.37/crispasr-windows-x86_64-cuda-non-cuda.zip) | 146,153,022 | `74d04c30034f33c6270df294bb4b66f98c05fd152f4846b81ddeef0a7f62d712` |

解出来的 `crispasr.exe` / `crispasr.dll` / `ggml*.dll` 之外，CUDA 后端还需要
这三个运行库，放在同一个目录：

| 文件 | 字节 | sha256 |
|---|---:|---|
| [cublas64_12.dll](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.37/cublas64_12.dll) | 113,712,640 | `0a60a7ecbf906f7f3842826fecaf412f1587ff6269729339245a1ec224364161` |
| [cublasLt64_12.dll](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.37/cublasLt64_12.dll) | 692,441,600 | `a21ddfc1c9cd090ab07d2c6aad235aa4f15fe60fe896d5db28adf7a279c09ef3` |
| [cudart64_12.dll](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.37/cudart64_12.dll) | 573,952 | `9d9b868955149875ac4ef43442aaaa8913d4a7b3e4d6dd60ee871626aa045768` |

另外要 Microsoft Visual C++ 运行库（`https://aka.ms/vc14/vc_redist.x64.exe`，
系统已装则跳过）。

模型（HuggingFace `cstr/qwen3-asr-1.7b-GGUF`）：

| 文件 | 字节 | sha256 | 备注 |
|---|---:|---|---|
| [qwen3-asr-1.7b-q8_0.gguf](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF/resolve/main/qwen3-asr-1.7b-q8_0.gguf) | 2,506,723,200 | `9851ab996591a2d0cb0efb216002764b509c86bd40c95e613d7b65b8e69c8a6e` | 实测用的这份 |
| [qwen3-asr-1.7b-q4_k.gguf](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF/resolve/main/qwen3-asr-1.7b-q4_k.gguf) | 1,490,915,200 | `ec197cef7ccc589fdcae1becc3f4a3de119d0a41e790b898b519b1a048dad8d4` | 显存不够时的档，本 fork 未实测 |

q8_0 的 sha256 是按本机文件算出来的，与 HF 上的字节数一致；q4_k 那份来自
[takomitkm/High-Accuracy-Offline-ASR](https://github.com/takomitkm/High-Accuracy-Offline-ASR)
`gpu` 分支的 `CrispASR-Qwen/assets.json`。

摆放（文件名可换，用环境变量指过去）：

```text
models/CrispASR/
├─ crispasr.exe            以及 zip 里的其余 dll
├─ cublas64_12.dll
├─ cublasLt64_12.dll
├─ cudart64_12.dll
└─ qwen3-asr-1.7b-q8_0.gguf
```

### 路径与参数

| 环境变量 | 对应参数 | 默认（相对仓库根） |
|---|---|---|
| `CRISPASR_EXE` | `exe_path` | `models/CrispASR/crispasr.exe` |
| `CRISPASR_MODEL` | `model_path` | `models/CrispASR/qwen3-asr-1.7b-q8_0.gguf` |

`CrispAsrQwen3Args` 里还有 `backend='qwen3'`、`gpu_backend='cuda'`、
`language='zh'`、`warmup=True`、`start_timeout=240`。

- `backend='qwen3'` 是加载这份 1.7B q8_0 实测可用的后端名；上游 README 里另有一行
  `qwen3-1.7b`，本 fork 没有改用它，因为换后端是否等价没有验证过。
- `language='zh'`：与测试记录一致，中文听写固定 `zh`。`auto`（自动语种判断）没有测过，
  所以出厂不留 `auto`。
- `warmup=True`：服务就绪后用 0.5 s 噪声打一次推理，吃掉 CUDA 首次选型/升频的开销。

---

## 3. 实测数据

同一份 16.45 s 中文音频，RTX 4060 Laptop，热态。RTF = 处理耗时 / 音频时长。

| 指标 | `firedasr_onnx` | `crispasr_qwen3` |
|---|---:|---:|
| 引擎推理 | 3.858 s（RTF 0.235） | 2.921~3.115 s（RTF 0.178~0.189） |
| CapsWriter 整条 pipeline | 3.903 s（RTF 0.237） | 3.030~3.148 s（RTF 0.184~0.191） |
| 冷态首次请求 | ~6.2 s（RTF 0.377） | 请求正好压在服务重启上：pipeline 7.733 s（RTF 0.470），其中引擎仍只有 2.921 s |
| 长音频 | 未测 | 82.40 s 音频 / 15.548 s（RTF 0.189） |

RTF 由日志原样给出，两个引擎各有多次采样，表里取最小~最大。
`firedasr_onnx` 的分项拆解：VAD 0.085 s + AED 编码器 1.005 s + AED 解码器 2.610 s +
FireRedPunc 0.097 s = 3.797 s，比引擎总计 3.858 s 少 0.061 s（分片之间的调度开销）。
解码器占引擎总计的 67.7%（2.610/3.858）——那是带自回归 step 的 int8 图在 CPU EP 上跑；
编码器已经是 f32 CUDA。

短句另有一组采样（引擎行日志原样摘录，13 条 `RTF` 与 `耗时/音频` 逐条重算过，最大偏差 0.0005，是日志四舍五入）：

| 音频 | `firedasr_onnx` 引擎 | `crispasr_qwen3` 引擎 |
|---:|---|---|
| 2.25 s | 1.792 s（RTF 0.796） | 未测 |
| 3.0 s | 0.614 s（0.205） | 0.926 s（0.309） |
| 3.6 s | 0.754 s（0.209） | 未测 |
| 4.7 s | 1.286 s（0.274） | 未测 |
| 6.45 s | 1.122 s（0.174） | 未测 |
| 13.85 s | 未测 | 2.703 s（0.195） |
| 16.45 s | 3.858 s（0.235） | 2.921~3.115 s（0.178~0.189） |
| 22.40 s | 未测 | 3.659 s（0.163） |

`firedasr_onnx` 那条 0.796 不是短音频本身的代价：那次 VAD 0.728 s + Punc 0.465 s
= 1.193 s，占 1.792 s 的 66.6%，是该次启动后 VAD/Punc 会话第一次推理；同批次后来的
3.0~6.45 s 片段里 VAD 只有 0.015~0.027 s、Punc 0.026~0.037 s，RTF 落回 0.174~0.274。
编码器在 3~6.45 s 片段上有 0.377~0.462 s 的下限，跟片段长度几乎无关，所以音频越短
RTF 越难看——RTF 这个指标本身对短句不友好，看绝对耗时更准。

启动耗时（服务端 `全系统初始化完成，耗时` 那行）：

| | 采样 | 对照 |
|---|---|---|
| `firedasr_onnx` | 9.68 / 15.18 / 13.55 s | worker 的 `load=`（含 0.99~1.18 s 预热）9.4 / 14.9 / 13.3 s，适配器另加 0.25~0.28 s |
| `crispasr_qwen3` | 5.07 / 5.93 s | `/health` 就绪 3.8~4.4 s（加载 2.51 GB GGUF）+ 预热 0.92~1.19 s |

`crispasr_qwen3` 首次请求前如果服务还没就绪，`start_timeout=240` 会兜住，但那一次
请求要等模型加载完（冷 30.6 s 是每次单独调 `crispasr.exe` 的数据，`--server` 常驻后不再出现）。

---

## 4. 排查

- **"不支持的模型类型：firedasr_onnx"**：在用打包 exe 启动。改 `python start_server.py`。
- **`check_model` 报缺件**：按它提示的这份文档补，或跑 `chain/fetch_assets.py`（FireRed）。
  上游的模型发布页对这两路无效，那里没有这些文件。
- **FireRed worker 起不来但服务端只显示超时**：看 `logs/firedasr_worker.log`，真实原因
  （缺 CUDA EP、缺 `encoder.f32.onnx`、Punc 混用了 `punc.q8w.onnx`）都在那里。
- **`Only 4b quantization is supported`**：Punc 用错了文件。运行时必须是建造出来的
  `punc.f32.onnx`，不是下载的 `punc.q8w.onnx`。
- **`encoder.f32.onnx` 缺失**：`fetch_assets.py` 当时只跑了 `--graph int8`，重跑
  `--graph mixed`。
- **CrispASR 服务启动即退出**：看 `logs/crispasr_qwen3_server.log`。三个 CUDA DLL
  缺一个、或者没有 VC++ 运行库，都表现为启动即退。
- **显存被吃满 / 有两个 crispasr 进程**：上一次的服务被强杀成了孤儿。适配器已经会按
  PID 清理，手工处理就是结束那个还活着但 CapsWriter 已经不认的 `crispasr.exe`。
- **NVIDIA 显卡不在候选里**：`FIREDASR_ONNX_PROVIDER` 确认是 `cuda`，且
  `pip list` 里装的是 `onnxruntime-gpu` 而不是 `onnxruntime`。

---

## 5. 出处与本 fork 的改动边界

- 上游项目：[HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)
  （MIT）。本仓库是它的 fork，上游代码与许可完整保留，两路新引擎只做增量、不改上游逻辑，
  也没有提 PR。跟上游用 merge，不用 squash，避免把上游提交改写。
- `firedasr_onnx` 的链代码：
  [takomitkm/High-Accuracy-Offline-ASR](https://github.com/takomitkm/High-Accuracy-Offline-ASR)
  的 `gpu` 分支 `CrispASR-FireRed/onnx/`（commit `1bac7e6`，2026-10-17），Unlicense。
  相对来源仓库只做了四件事：删掉注释里的本机绝对路径、把 `fetch_assets.py` 默认落点
  改成本仓库的 `models/`、补 `requirements-cuda.txt` 与 `licenses/`、补文档。
  推理逻辑、`GRAPH_BUILDS` 和逐件 sha256 清单未改。
- `crispasr_qwen3`：推理走 [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)
  （MIT）v0.8.37；权重是 [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
  的 GGUF 转换件 `cstr/qwen3-asr-1.7b-GGUF`（Apache-2.0）。
- 完整的第三方组件与许可见
  `core/server/engines/firedasr_onnx/chain/THIRD_PARTY_NOTICES.md`。
