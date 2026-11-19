# 第三方声明（仓库级）

本仓库是 [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)
的 fork，自 2026-11-20 起整体以 GNU AGPL-3.0 分发（全文见根目录 [LICENSE](LICENSE)）。
在此之前的历史提交以 MIT 许可分发（当时的根目录 `LICENSE` 即为下文全文）。

上游代码原以 MIT 许可发布。MIT 允许再许可，但要求保留其版权与许可声明；
该声明对来自上游的代码继续有效，全文照录如下：

```
MIT License

Copyright (c) 2026 Haujet Zhao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

`firedasr_onnx`、`crispasr_qwen3` 两路引擎引入的第三方件（FireRed ONNX 链代码
0BSD、FireRedVAD `infer_onnx.py` Apache-2.0、其余为运行时下载的模型与工具）
另见 [core/server/engines/firedasr_onnx/chain/THIRD_PARTY_NOTICES.md](core/server/engines/firedasr_onnx/chain/THIRD_PARTY_NOTICES.md)。

本仓库不包含任何模型权重、GGUF、可执行文件或 CUDA 运行库。
