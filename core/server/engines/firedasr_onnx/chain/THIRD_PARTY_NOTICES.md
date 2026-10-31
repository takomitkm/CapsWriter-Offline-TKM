# 第三方声明

本仓库是 [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)
的 fork，上游代码按根目录 `LICENSE`（MIT）使用。本文件只列本 fork 新增的第三方件。

本仓库不包含任何模型权重、GGUF、可执行文件或 CUDA 运行库；以下资源均由使用者在本地下载。

## FireRed ONNX 链代码

`core/server/engines/firedasr_onnx/chain/` 下的 `aed_ort.py`、`asr_chain.py`、
`dequant_aed.py`、`dequant_punc.py`、`fetch_assets.py` 取自：

- 仓库：[takomitkm/High-Accuracy-Offline-ASR](https://github.com/takomitkm/High-Accuracy-Offline-ASR)
- 分支/路径：`gpu` 分支 `CrispASR-FireRed/onnx/`
- 版本：commit `1bac7e6`（2026-10-17）
- 许可：Unlicense（公有领域），全文见 `chain/licenses/UNLICENSE.txt`

## FireRedVAD `infer_onnx.py`

- 文件：`core/server/engines/firedasr_onnx/chain/infer_onnx.py`
- 来源：[tardigrade-doc/FireRedVAD_onnx](https://huggingface.co/tardigrade-doc/FireRedVAD_onnx)
- 基础模型：[FireRedTeam/FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD)
- 许可：Apache-2.0，全文见 `chain/licenses/Apache-2.0.txt`
- 模型卡：`https://huggingface.co/tardigrade-doc/FireRedVAD_onnx`

## FireRedPunc

- 模型与分词器：[jiangzhuo9357/fireredpunc-onnx](https://huggingface.co/jiangzhuo9357/fireredpunc-onnx)
- 许可：Apache-2.0；`fetch_assets.py` 会把上游的 `LICENSE` 下载到
  `models/FireRedASR2-ONNX-AED/fireredpunc-onnx/LICENSE`
- 本仓库只包含逆量化脚本，不包含权重

## FireRedASR2-AED 模型

- 转换与发布：[k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 的模型 release
  `sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26.tar.bz2`
- 原始模型：[FireRedTeam/FireRedASR2-AED](https://huggingface.co/FireRedTeam/FireRedASR2-AED)
- 本仓库不包含权重；下载地址、字节数与 sha256 见 `chain/fetch_assets.py`

## CrispASR

- 项目：[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)
- 许可：MIT（见其 release 包内的 `LICENSE`）
- 本仓库不包含 `crispasr.exe`、DLL 或 CUDA 运行库

## Qwen3-ASR-1.7B

- 原始模型：[Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- GGUF 转换：[cstr/qwen3-asr-1.7b-GGUF](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF)
- 许可：Apache-2.0
- 本仓库不包含 GGUF；由使用者从上述地址下载，或用 CrispASR 的 `-m auto` 获取

## 其它

`core/server/engines/firedasr_onnx/chain/requirements*.txt` 中的 Python 包各自遵循其上游许可。
