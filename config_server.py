import os
from pathlib import Path

# 版本信息
__version__ = '2.6'

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, '').strip()
    return Path(value) if value else default


# 服务端配置
class ServerConfig:
    addr = '0.0.0.0'
    port = '6016'

    # 语音模型选择：'qwen_asr', 'fun_asr_nano', 'sensevoice', 'paraformer',
    #              'firedasr_onnx', 'crispasr_qwen3'（后两路由本 fork 增加）
    model_type = 'qwen_asr'

    format_num = True       # 输出时是否将中文数字转为阿拉伯数字
    format_spell = True     # 输出时是否调整中英之间的空格

    enable_tray = True        # 是否启用托盘图标功能
    hotwords_path = Path() / 'hot-server.txt' # 全局热词配置文件路径

    # 日志配置
    log_level = 'DEBUG'        # 日志级别：'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    aligner_idle_timeout = 10  # 对齐引擎空闲多少秒后自动释放显存 (0 表示不释放)

    # GPU 预加速配置（有识别任务时，提前调高显存频率，降低延迟，需管理员权限运行）
    gpu_boost_enabled = False                   # 总开关，默认关闭
    gpu_boost_cmd = 'nvidia-smi -lmc 9000'      # GPU 预加速命令，锁定显存频率到9000MHz（根据实际 GPU 调整）
    gpu_unboost_cmd = 'nvidia-smi -rmc'         # GPU 取消预加速命令，恢复显存到默认频率
    gpu_unboost_timeout = 1                     # 空闲多少秒后取消加速

    # 集成显卡兼容性补丁
    # os.environ["GGML_VK_DISABLE_COOPMAT"] = "1"   # AMD集显无法加载 GGUF 模型时尝试
    # os.environ["GGML_VK_DISABLE_F16"] = "1"       # 集成显卡解码有误，强制熔断时尝试




class ModelDownloadLinks:
    """模型下载链接配置"""
    # 统一导向 GitHub Release 模型页面
    models_page = "https://github.com/HaujetZhao/CapsWriter-Offline/releases/tag/models"


class ModelPaths:
    """模型文件路径配置"""

    # 基础目录
    model_dir = Path() / 'models'

    # Paraformer 模型路径
    paraformer_dir = model_dir / 'Paraformer' / "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx"
    paraformer_model = paraformer_dir / 'model.onnx'
    paraformer_tokens = paraformer_dir / 'tokens.txt'

    # 标点模型路径
    punc_model_dir = model_dir / 'Punct-CT-Transformer' / 'sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12' / 'model.onnx'

    # SenseVoice 模型路径，自带标点
    sensevoice_dir = model_dir / 'SenseVoice-Small' / 'Sensevoice-Small-ONNX'
    sensevoice_encoder = sensevoice_dir / 'SenseVoice-Encoder.fp16.onnx'
    sensevoice_decoder = sensevoice_dir / 'SenseVoice-CTC.fp16.onnx'
    sensevoice_tokenizer = sensevoice_dir / 'tokenizer.bpe.model'


    # Fun-ASR-Nano 模型路径，自带标点
    fun_asr_nano_gguf_dir = model_dir / 'Fun-ASR-Nano' / 'Fun-ASR-Nano-GGUF'
    fun_asr_nano_gguf_encoder_adaptor = fun_asr_nano_gguf_dir / 'Fun-ASR-Nano-Encoder-Adaptor.fp16.onnx'
    fun_asr_nano_gguf_ctc = fun_asr_nano_gguf_dir / 'Fun-ASR-Nano-CTC.fp16.onnx'
    fun_asr_nano_gguf_llm_decode = fun_asr_nano_gguf_dir / 'Fun-ASR-Nano-Decoder.q5_k.gguf'
    fun_asr_nano_gguf_token = fun_asr_nano_gguf_dir / 'tokens.txt'
    fun_asr_nano_gguf_hotwords = Path() / 'hot-server.txt'

    # Qwen3-ASR 模型路径，自带标点
    qwen3_asr_gguf_dir = model_dir / 'Qwen3-ASR' / 'Qwen3-ASR-1.7B'
    qwen3_asr_gguf_encoder_frontend = qwen3_asr_gguf_dir / 'qwen3_asr_encoder_frontend.onnx'
    qwen3_asr_gguf_encoder_backend = qwen3_asr_gguf_dir / 'qwen3_asr_encoder_backend.onnx'
    qwen3_asr_gguf_llm_decode = qwen3_asr_gguf_dir / 'qwen3_asr_llm.gguf'

    # Force-Aligner 模型路径
    force_aligner_gguf_dir = model_dir / 'Qwen3-ForcedAligner' / 'Qwen3-ForcedAligner-0.6B'
    force_aligner_gguf_encoder_frontend = force_aligner_gguf_dir / 'qwen3_aligner_encoder_frontend.int4.onnx'
    force_aligner_gguf_encoder_backend = force_aligner_gguf_dir / 'qwen3_aligner_encoder_backend.int4.onnx'
    force_aligner_gguf_llm_decode = force_aligner_gguf_dir / 'qwen3_aligner_llm.q5_k.gguf'

    # FireRedASR2-AED 纯 ONNX 链路：代码随仓库，模型用 chain/fetch_assets.py 补全
    firedasr_onnx_dir = model_dir / 'FireRedASR2-ONNX-AED'
    firedasr_onnx_model_dir = firedasr_onnx_dir / 'sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26'
    firedasr_onnx_encoder = firedasr_onnx_model_dir / 'encoder.f32.onnx'
    firedasr_onnx_encoder_data = firedasr_onnx_model_dir / 'encoder.f32.onnx.data'
    firedasr_onnx_decoder = firedasr_onnx_model_dir / 'decoder.int8.onnx'
    firedasr_onnx_tokens = firedasr_onnx_model_dir / 'tokens.txt'
    firedasr_onnx_punc_dir = firedasr_onnx_dir / 'fireredpunc-onnx'
    firedasr_onnx_punc_model = firedasr_onnx_punc_dir / 'punc.f32.onnx'
    firedasr_onnx_vad_dir = firedasr_onnx_dir / 'fireredvad-onnx'
    firedasr_onnx_vad_model = firedasr_onnx_vad_dir / 'model.onnx'
    firedasr_onnx_chain_dir = Path(BASE_DIR) / 'core' / 'server' / 'engines' / 'firedasr_onnx' / 'chain'
    firedasr_onnx_python = firedasr_onnx_dir / (
        'env/Scripts/python.exe' if os.name == 'nt' else 'env/bin/python')

    # CrispASR：独立发布的 CLI + GGUF
    crispasr_dir = model_dir / 'CrispASR'
    crispasr_exe = crispasr_dir / ('crispasr.exe' if os.name == 'nt' else 'crispasr')
    crispasr_qwen3_model = crispasr_dir / 'qwen3-asr-1.7b-q8_0.gguf'



class ParaformerArgs:
    """Paraformer 模型参数配置"""

    paraformer = ModelPaths.paraformer_model.as_posix()
    tokens = ModelPaths.paraformer_tokens.as_posix()
    num_threads = 4
    sample_rate = 16000
    feature_dim = 80
    decoding_method = 'greedy_search'
    provider = 'cpu'
    debug = False


class SenseVoiceArgs:
    """SenseVoice 模型参数配置"""

    encoder_path = ModelPaths.sensevoice_encoder.as_posix()
    decoder_path = ModelPaths.sensevoice_decoder.as_posix()
    tokenizer_path = ModelPaths.sensevoice_tokenizer.as_posix()
    itn = True                  # 原生输出阿拉伯数字
    onnx_provider = 'CPU'       # ONNX 推理后端 (CPU, DML)
    top_k = 8                   # 热词检索的 CTC 空间大小
    dml_pad_to = 30             # 开启 DirectML 加速时，短音频统一填充到指定长度，有加速效果


class FunASRNanoGGUFArgs:
    """Fun-ASR-Nano-GGUF 模型参数配置"""

    # 模型路径
    encoder_onnx_path = ModelPaths.fun_asr_nano_gguf_encoder_adaptor.as_posix()
    ctc_onnx_path = ModelPaths.fun_asr_nano_gguf_ctc.as_posix()
    decoder_gguf_path = ModelPaths.fun_asr_nano_gguf_llm_decode.as_posix()
    tokens_path = ModelPaths.fun_asr_nano_gguf_token.as_posix()

    # 显卡加速
    onnx_provider = 'CPU'       # ONNX 推理后端 (CPU, DML)
    llm_use_gpu = True          # 是否启用 GPU 加速 GGUF 模型
    vulkan_force_fp32 = False   # 是否强制 FP32 计算（如果 GPU 是 Intel 集显且出现精度溢出，可设为 True）
    
    # 模型细节
    enable_ctc = True           # 是否启用 CTC 热词检索
    n_predict = 512             # LLM 最大生成 token 数
    n_threads = None            # 线程数，None 表示自动
    similar_threshold = 0.6     # 热词相似度阈值，超过阈值的热词会被传入 llm decoder 的上下文
    max_hotwords = 20           # 传入上下文的热词数量上限
    dml_pad_to = 30             # 开启 DirectML 加速时，短音频统一填充到指定长度，有加速效果
    verbose = False

class Qwen3ASRGGUFArgs:
    """Qwen3-ASR-GGUF 模型参数配置"""

    # 模型路径
    model_dir = ModelPaths.qwen3_asr_gguf_dir.as_posix()
    encoder_frontend_fn = ModelPaths.qwen3_asr_gguf_encoder_frontend.name
    encoder_backend_fn = ModelPaths.qwen3_asr_gguf_encoder_backend.name
    llm_fn = ModelPaths.qwen3_asr_gguf_llm_decode.name

    # 显卡加速
    onnx_provider = 'CPU'       # ONNX 推理后端 (CPU, DML)
    llm_use_gpu = True          # 是否启用 GPU 加速 GGUF 模型
    
    # 模型细节
    n_ctx = 2048                # 上下文窗口大小
    chunk_size = 80.0           # 分段长度（秒）
    memory_num = 1              # 记忆段数
    dml_pad_to = 30             # 开启 DirectML 加速时，短音频统一填充到指定长度，有加速效果
    verbose = False


class ForceAlignerGGUFArgs:
    """Force-Aligner-GGUF 模型参数配置"""

    # 模型路径
    model_dir = ModelPaths.force_aligner_gguf_dir.as_posix()
    encoder_frontend_fn = ModelPaths.force_aligner_gguf_encoder_frontend.name
    encoder_backend_fn = ModelPaths.force_aligner_gguf_encoder_backend.name
    llm_fn = ModelPaths.force_aligner_gguf_llm_decode.name

    # 显卡加速
    onnx_provider = 'CPU'       # ONNX 推理后端 (CPU, DML)(推荐保持 DML)
    llm_use_gpu = False          # 是否启用 GPU 加速 GGUF 模型
    
    # 对齐细节
    n_ctx = 4096                # 上下文窗口大小
    dml_pad_to = 30             # 开启 DirectML 加速时，短音频统一填充到指定长度，有加速效果


class FireRedOnnxArgs:
    """FireRedASR2-AED 纯 ONNX 链路参数配置。

    推理在独立进程里跑；onnx_provider 出厂 cuda，没有 CUDA EP 会直接报错退出，
    不会静默退回 CPU。路径均可用环境变量覆盖，见 docs/引擎扩展-FireRed与CrispASR.md。
    """
    chain_dir = _env_path('FIREDASR_ONNX_CHAIN_DIR', ModelPaths.firedasr_onnx_chain_dir).resolve().as_posix()
    model_dir = _env_path('FIREDASR_ONNX_MODEL_DIR', ModelPaths.firedasr_onnx_model_dir).resolve().as_posix()
    punc_dir = _env_path('FIREDASR_ONNX_PUNC_DIR', ModelPaths.firedasr_onnx_punc_dir).resolve().as_posix()
    vad_dir = _env_path('FIREDASR_ONNX_VAD_DIR', ModelPaths.firedasr_onnx_vad_dir).resolve().as_posix()
    python_exe = _env_path('FIREDASR_ONNX_PYTHON', ModelPaths.firedasr_onnx_python).resolve().as_posix()

    sample_rate = 16000
    onnx_provider = os.environ.get('FIREDASR_ONNX_PROVIDER', 'cuda').strip().lower()
    graph = 'mixed'             # f32 编码器 + int8 解码器
    mode = 'greedy'             # 贪心解码；beam 更慢且没有稳定收益
    threads = 4                 # ONNX 会话线程数
    punc_threads = 4            # FireRedPunc 的 BERT 在 CPU 上跑
    use_vad = True              # FireRedVAD 先切语音、去掉长静音
    piece_max = 10.0            # VAD 段按停顿聚到 <=10 s 再送 AED
    start_timeout = 240         # 等模型就绪的上限


class CrispAsrQwen3Args:
    """Qwen3-ASR-1.7B（CrispASR）参数配置。

    推理跑在 crispasr.exe 的常驻 HTTP 服务里，本进程只发 WAV。
    实测（RTX 4060 Laptop，16.45 s 音频热态）：2.92~3.12 s，RTF 0.178~0.189。
    """
    exe_path = _env_path('CRISPASR_EXE', ModelPaths.crispasr_exe).resolve().as_posix()
    model_path = _env_path('CRISPASR_MODEL', ModelPaths.crispasr_qwen3_model).resolve().as_posix()

    backend = 'qwen3'           # 实测用这个后端加载 1.7B q8_0
    language = 'zh'             # 中文听写固定 zh
    gpu_backend = 'cuda'        # 只用 NVIDIA CUDA
    sample_rate = 16000
    warmup = True               # 起服务后先用 0.5 s 噪声吃掉 CUDA 首推理开销
    start_timeout = 240         # 等模型加载 + 就绪的上限

