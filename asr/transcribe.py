from typing import Optional
import logging
from fastapi import HTTPException
from conf.settings import SETTINGS
from asr.whisper import transcribe_whisper
from asr.paraformer import transcribe_paraformer

logger = logging.getLogger("asr")

def transcribe_audio(audio_path: str, backend: Optional[str] = None, model_name: Optional[str] = None) -> str:
    """
    统一ASR入口：根据配置或入参选择 Whisper/Paraformer
    - 优先使用显式入参 backend
    - 其次使用 SETTINGS.ASR_PROVIDER
    - 若都未指定，则依据 SETTINGS.RAG_PROVIDER 映射选择（默认 whisper）
    """
    def _normalize_backend(b: Optional[str]) -> str:
        b = (b or "").strip().lower()
        if b in {"faster-whisper", "whisper", "openai-whisper"}:
            return "whisper"
        if b in {"funasr", "paraformer"}:
            return "paraformer"
        return ""

    eff_backend = _normalize_backend(backend)
    if not eff_backend:
        eff_backend = _normalize_backend(getattr(SETTINGS, "ASR_PROVIDER", None))
    if not eff_backend:
        # RAG_PROVIDER → ASR 映射：chroma→paraformer，其余→whisper
        rag = (getattr(SETTINGS, "RAG_PROVIDER", "faiss") or "faiss").lower()
        eff_backend = "paraformer" if rag == "chroma" else "whisper"
    logger.info("🎛️ ASR 选择 backend=%s model=%s file=%s", eff_backend, model_name, audio_path)

    try:
        if eff_backend == "whisper":
            return transcribe_whisper(audio_path, model_name or getattr(SETTINGS, "WHISPER_MODEL", "medium") or "medium")
        if eff_backend == "paraformer":
            return transcribe_paraformer(audio_path, model_name or getattr(SETTINGS, "ASR_MODEL", "paraformer-zh") or "paraformer-zh")
        raise HTTPException(status_code=400, detail=f"不支持的 ASR 后端: {eff_backend}，仅支持 whisper/paraformer")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ ASR 执行失败 file=%s backend=%s error=%s", audio_path, eff_backend, e)
        raise HTTPException(status_code=500, detail=f"语音转写失败：{str(e)}")
