from typing import Optional
from fastapi import HTTPException
from conf.settings import SETTINGS


def _transcribe_with_whisper(audio_path: str, model_name: str = "medium") -> str:
    """
    使用 faster-whisper 离线模型进行语音转文字。
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise HTTPException(status_code=501, detail="未安装 faster-whisper，请先安装依赖")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, vad_filter=True)
    text = "".join([seg.text for seg in segments])
    return text.strip()


def _transcribe_with_paraformer(audio_path: str, model_name: str = "paraformer-zh") -> str:
    """
    使用 FunASR Paraformer 离线模型进行语音转文字（中文）。
    """
    try:
        from funasr import AutoModel
    except ImportError:
        raise HTTPException(status_code=501, detail="未安装 funasr，请先安装依赖")
    asr_model = AutoModel(model=model_name, model_revision="v1.0.0")
    res = asr_model.generate(input=audio_path)
    text = "".join([r.get("text", "") for r in res]) if isinstance(res, list) else res.get("text", "")
    return text.strip()


def transcribe_audio(audio_path: str, backend: Optional[str] = None, model_name: Optional[str] = None) -> str:
    """
    根据配置或入参选择后端进行语音转写。
    """
    backend = (backend or SETTINGS.ASR_PROVIDER or "funasr").lower()
    if backend == "whisper":
        return _transcribe_with_whisper(audio_path, model_name or SETTINGS.WHISPER_MODEL or "medium")
    elif backend == "paraformer" or backend == "funasr":
        return _transcribe_with_paraformer(audio_path, model_name or SETTINGS.ASR_MODEL or "paraformer-zh")
    else:
        raise HTTPException(status_code=400, detail="不支持的 ASR 后端")
