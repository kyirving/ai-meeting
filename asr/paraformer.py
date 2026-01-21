import os
import logging
from typing import Optional
from fastapi import HTTPException
from asr.utils import temporary_file_cleanup, ensure_wav_16k

logger = logging.getLogger("asr")

def transcribe_paraformer(audio_path: str, model_name: Optional[str] = "paraformer-zh") -> str:
    """
    使用 FunASR Paraformer 离线模型进行语音转文字（中文）
    - 统一输入为16kHz单声道WAV
    - 支持模型名别名映射与本地路径
    """
    try:
        from funasr import AutoModel
    except ImportError:
        raise HTTPException(status_code=501, detail="未安装 funasr，请先安装依赖：pip install funasr")

    alias = {
        "paraformer-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "paraformer-large-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "paraformer-zh-cn-16k": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    }
    resolved = alias.get((model_name or "paraformer-zh").lower(), model_name or "paraformer-zh")
    real_model = resolved if os.path.exists(resolved) else resolved
    logger.info("📥 Paraformer 加载准备 model=%s → %s file=%s", model_name, real_model, audio_path)

    try:
        asr_model = AutoModel(
            model=real_model,
            model_type="paraformer",
            model_revision="v2.0.0",
            disable_update=True,
            device="cpu"
        )
        logger.info("✅ Paraformer 模型加载完成: %s", real_model)
    except AssertionError as e:
        msg = f"Paraformer 模型不可用: {model_name} -> {real_model}, 请检查模型名/本地路径或网络: {str(e)}"
        logger.error("❌ %s", msg)
        raise HTTPException(status_code=500, detail=msg)
    except Exception as e:
        msg = f"Paraformer 模型加载失败: {str(e)}"
        logger.error("❌ %s", msg)
        raise HTTPException(status_code=503, detail=msg)

    wav_path = ensure_wav_16k(audio_path)
    with temporary_file_cleanup(wav_path):
        try:
            logger.info("📼 Paraformer 转录开始: %s", audio_path)
            res = asr_model.generate(input=wav_path)
            if not res:
                return ""
            if isinstance(res, list):
                text = "".join([r.get("text", "").strip() for r in res if isinstance(r, dict)])
            else:
                text = res.get("text", "").strip() if isinstance(res, dict) else ""
            logger.info("✅ Paraformer 转录完成 chars=%s preview=%s", len(text or ""), text[:50] + "..." if len(text) > 50 else text)
            return text
        except Exception as e:
            logger.error("❌ Paraformer 转写失败 file=%s error=%s", audio_path, e)
            raise HTTPException(status_code=500, detail=f"Paraformer 转写失败：{str(e)}")
