from typing import Optional
from fastapi import HTTPException
from conf.settings import SETTINGS
import os
import subprocess
import uuid
import logging
import shutil
import tempfile
from contextlib import contextmanager

logger = logging.getLogger("asr")

# ===================== 通用工具函数 =====================
@contextmanager
def temporary_file_cleanup(file_path: str):
    """临时文件自动清理上下文管理器"""
    try:
        yield file_path
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug("临时文件已清理: %s", file_path)
            except Exception as e:
                logger.warning("清理临时文件失败: %s, 错误: %s", file_path, e)

def _validate_audio_file(audio_path: str):
    """校验音频文件是否存在且非空"""
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail=f"音频文件不存在: {audio_path}")
    if os.path.getsize(audio_path) == 0:
        raise HTTPException(status_code=400, detail=f"音频文件为空: {audio_path}")

# ===================== Whisper 相关 =====================
def _transcribe_with_whisper(audio_path: str, model_name: str = "medium") -> str:
    """
    使用 faster-whisper 进行语音转文字（支持在线/离线）。
    模型名映射：
        small → Systran/faster-whisper-small
        medium → Systran/faster-whisper-medium
        large → Systran/faster-whisper-large
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise HTTPException(status_code=501, detail="未安装 faster-whisper，请先安装依赖：pip install faster-whisper")
    
    # 1. 基础校验
    _validate_audio_file(audio_path)
    logger.info("ASR whisper start model=%s file=%s", model_name, audio_path)
    
    # 2. 模型名映射（简化输入）
    model_alias = {
        "small": "Systran/faster-whisper-small",
        "medium": "Systran/faster-whisper-medium",
        "large": "Systran/faster-whisper-large",
        "small.en": "Systran/faster-whisper-small.en",
        "medium.en": "Systran/faster-whisper-medium.en",
        "large-v2": "Systran/faster-whisper-large-v2",
    }
    real_model_name = model_alias.get(model_name.lower(), model_name)
    
    # 3. 环境变量配置（优先级：系统环境变量 > 代码默认）
    env_config = {
        "HF_ENDPOINT": os.getenv("HF_ENDPOINT", "https://hf-mirror.com"),
        "HF_HUB_OFFLINE": os.getenv("HF_HUB_OFFLINE", "0"),
        "HTTP_PROXY": os.getenv("HTTP_PROXY", ""),
        "HTTPS_PROXY": os.getenv("HTTPS_PROXY", ""),
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
    }
    # 应用环境变量
    for key, value in env_config.items():
        if value:
            os.environ[key] = value
    
    # 4. 判断是否为本地模型路径，控制是否仅加载本地文件
    is_local_model = os.path.exists(real_model_name)
    local_files_only = is_local_model or env_config["HF_HUB_OFFLINE"] == "1"
    
    try:
        # 5. 加载模型（核心修复：添加 local_files_only 参数）
        model = WhisperModel(
            real_model_name,
            device="cpu",
            compute_type="int8",
            local_files_only=local_files_only  # 关键：离线环境设为True，在线设为False
        )
    except Exception as e:
        error_msg = (
            f"Whisper 模型加载失败：{str(e)}. "
            f"模型名: {model_name} → {real_model_name}, 本地路径: {is_local_model}, 仅本地加载: {local_files_only}. "
            "如为离线环境，请确保 WHISPER_MODEL 配置为本地模型路径；如为在线环境，请检查代理/网络和 HF_ENDPOINT 配置。"
        )
        logger.error("Whisper 模型加载失败: %s", error_msg)
        raise HTTPException(status_code=503, detail=error_msg)
    
    # 6. 执行转写
    try:
        segments, _ = model.transcribe(audio_path, vad_filter=True)
        text = "".join([seg.text.strip() for seg in segments])
        logger.info("ASR whisper done file=%s chars=%s text=%s", audio_path, len(text or ""), text[:50] + "..." if len(text) > 50 else text)
        return text.strip()
    except Exception as e:
        logger.error("Whisper 转写失败 file=%s error=%s", audio_path, e)
        raise HTTPException(status_code=500, detail=f"Whisper 转写失败：{str(e)}")

# ===================== Paraformer 相关 =====================
def _transcribe_with_paraformer(audio_path: str, model_name: str = "paraformer-zh") -> str:
    """
    使用 FunASR Paraformer 离线模型进行语音转文字（中文）。
    支持本地模型路径和 Hugging Face 模型名。
    """
    try:
        from funasr import AutoModel
    except ImportError:
        raise HTTPException(status_code=501, detail="未安装 funasr，请先安装依赖：pip install funasr")
    
    # 1. 模型名别名映射
    def _resolve_model(name: str) -> str:
        m = (name or "").strip().lower()
        alias = {
            "paraformer-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "paraformer-large-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "paraformer-zh-cn-16k": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        }
        resolved = alias.get(m, name)
        # 如果是本地路径，直接返回；否则返回Hugging Face模型名
        return resolved if os.path.exists(resolved) else resolved
    
    real_model = _resolve_model(model_name)
    logger.info("ASR paraformer start model=%s → %s file=%s", model_name, real_model, audio_path)
    
    # 2. 加载模型（支持本地路径，禁用更新）
    try:
        asr_model = AutoModel(
            model=real_model,
            model_type="paraformer",
            model_revision="v2.0.0",
            disable_update=True,  # 禁用自动更新
            device="cpu"  # 显式指定CPU，避免自动选GPU导致问题
        )
    except AssertionError as e:
        error_msg = f"Paraformer 模型不可用: {model_name} -> {real_model}, 请检查模型名/本地路径或网络: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Paraformer 模型加载失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=503, detail=error_msg)
    
    # 3. 执行转写
    try:
        res = asr_model.generate(input=audio_path)
        # 健壮处理结果：兼容列表/字典格式，处理空值
        if not res:
            return ""
        if isinstance(res, list):
            text = "".join([r.get("text", "").strip() for r in res if isinstance(r, dict)])
        else:
            text = res.get("text", "").strip() if isinstance(res, dict) else ""
        logger.info("ASR paraformer done file=%s chars=%s text=%s", audio_path, len(text or ""), text[:50] + "..." if len(text) > 50 else text)
        return text
    except Exception as e:
        logger.error("Paraformer 转写失败 file=%s error=%s", audio_path, e)
        raise HTTPException(status_code=500, detail=f"Paraformer 转写失败：{str(e)}")

def _ensure_wav_16k(audio_path: str) -> str:
    """
    将输入音频统一转换为 16kHz 单声道 WAV，返回转换后的路径。
    依赖系统 ffmpeg，可通过 brew install ffmpeg / apt install ffmpeg 安装。
    """
    _validate_audio_file(audio_path)
    # 使用系统临时目录，避免权限问题
    temp_dir = tempfile.gettempdir()
    out_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.wav")
    
    try:
        logger.info("FFMPEG convert start src=%s dst=%s", audio_path, out_path)
        # 执行ffmpeg转换，添加超时（10秒）
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            timeout=10
        )
        # 校验转换后的文件
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise HTTPException(status_code=500, detail="音频转换后文件为空")
        logger.info("FFMPEG convert ok dst=%s size_bytes=%s", out_path, os.path.getsize(out_path))
        return out_path
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="音频转换超时（10秒），请检查文件大小或格式")
    except FileNotFoundError:
        raise HTTPException(status_code=501, detail="未检测到 ffmpeg，请先安装：brew install ffmpeg / apt install ffmpeg")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="ignore")[:200]
        logger.error("FFMPEG 转换失败: %s, 错误信息: %s", audio_path, stderr)
        raise HTTPException(status_code=500, detail=f"音频转换失败：{stderr}")

# ===================== 主函数 =====================
def transcribe_audio(audio_path: str, backend: Optional[str] = None, model_name: Optional[str] = None) -> str:
    """
    根据配置或入参选择后端进行语音转写。
    :param audio_path: 音频文件路径
    :param backend: ASR后端，支持 whisper/paraformer/funasr
    :param model_name: 模型名，whisper支持 small/medium/large；paraformer支持 paraformer-zh 等
    :return: 转写后的文本
    """
    # 1. 标准化后端名称
    def _normalize_backend(b: Optional[str]) -> str:
        b = (b or "").strip().lower()
        if b in {"faster-whisper", "whisper", "openai-whisper"}:
            return "whisper"
        if b in {"funasr", "paraformer"}:
            return "paraformer"
        return b or "paraformer"  # 默认使用paraformer
    
    backend = _normalize_backend(backend or SETTINGS.ASR_PROVIDER or "funasr")
    logger.info("ASR select backend=%s model=%s file=%s", backend, model_name, audio_path)
    
    # 2. 分支执行转写
    try:
        if backend == "whisper":
            return _transcribe_with_whisper(audio_path, model_name or SETTINGS.WHISPER_MODEL or "medium")
        elif backend == "paraformer":
            # 转换音频格式，并自动清理临时文件
            wav_path = _ensure_wav_16k(audio_path)
            with temporary_file_cleanup(wav_path):
                return _transcribe_with_paraformer(wav_path, model_name or SETTINGS.ASR_MODEL or "paraformer-zh")
        else:
            raise HTTPException(status_code=400, detail=f"不支持的 ASR 后端: {backend}，仅支持 whisper/paraformer")
    except HTTPException:
        # 透传HTTPException
        raise
    except Exception as e:
        logger.error("ASR 整体执行失败 file=%s backend=%s error=%s", audio_path, backend, e)
        raise HTTPException(status_code=500, detail=f"语音转写失败：{str(e)}")