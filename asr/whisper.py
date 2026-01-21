import os
import logging
from typing import Optional
from fastapi import HTTPException
from asr.utils import ensure_wav_16k, temporary_file_cleanup, validate_audio_file

logger = logging.getLogger("asr")

def transcribe_whisper(audio_path: str, model_name: Optional[str] = "medium") -> str:
    """
    使用 faster-whisper 进行语音转文字，移植 whisper_demo 的稳定策略：
    - 统一将输入音频转换为16kHz单声道WAV以避免解码崩溃
    - 使用 compute_type=float32，强制单线程 cpu_threads=1，提升稳定性
    - 支持在线/离线模式，自动根据模型名是否为本地路径选择 local_files_only
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise HTTPException(status_code=501, detail="未安装 faster-whisper，请先安装依赖：pip install faster-whisper")

    validate_audio_file(audio_path)
    logger.info("📥 Whisper 加载准备 model=%s file=%s", model_name, audio_path)

    model_alias = {
        "tiny": "Systran/faster-whisper-tiny",
        "base": "Systran/faster-whisper-base",
        "small": "Systran/faster-whisper-small",
        "medium": "Systran/faster-whisper-medium",
        "large": "Systran/faster-whisper-large",
        "small.en": "Systran/faster-whisper-small.en",
        "medium.en": "Systran/faster-whisper-medium.en",
        "large-v2": "Systran/faster-whisper-large-v2",
    }
    real_model_name = model_alias.get((model_name or "medium").lower(), model_name or "medium")

    os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "0")
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    if os.getenv("HTTP_PROXY"):
        os.environ["HTTP_PROXY"] = os.getenv("HTTP_PROXY")
    if os.getenv("HTTPS_PROXY"):
        os.environ["HTTPS_PROXY"] = os.getenv("HTTPS_PROXY")

    is_local_model = os.path.exists(real_model_name)
    local_files_only = True if is_local_model or os.getenv("HF_HUB_OFFLINE", "0") == "1" else False

    try:
        logger.info("📥 在线/离线加载 Whisper compute=float32 threads=1 local_only=%s", local_files_only)
        model = WhisperModel(
            real_model_name,
            device="cpu",
            compute_type="float32",
            local_files_only=local_files_only,
            cpu_threads=1,
        )
        logger.info("✅ Whisper 模型加载完成: %s", real_model_name)
    except Exception as e:
        msg = f"Whisper 模型加载失败: {e}"
        logger.error("❌ %s", msg)
        raise HTTPException(status_code=503, detail=msg)

    wav_path = ensure_wav_16k(audio_path)
    with temporary_file_cleanup(wav_path):
        try:
            logger.info("📼 Whisper 转录开始: %s", audio_path)
            segments, _ = model.transcribe(
                wav_path,
                language="zh",
                beam_size=3,
                vad_filter=True,
                word_timestamps=False
            )
            text = "".join([seg.text.strip() for seg in segments])
            logger.info("✅ Whisper 转录完成 chars=%s preview=%s", len(text or ""), text[:50] + "..." if len(text) > 50 else text)
            return text.strip()
        except Exception as e:
            logger.error("❌ Whisper 转录失败 file=%s error=%s", audio_path, e)
            raise HTTPException(status_code=500, detail=f"Whisper 转写失败：{str(e)}")
