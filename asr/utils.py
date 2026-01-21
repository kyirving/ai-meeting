import os
import uuid
import logging
import tempfile
import subprocess
from typing import Generator
from contextlib import contextmanager
from fastapi import HTTPException

logger = logging.getLogger("asr")

@contextmanager
def temporary_file_cleanup(file_path: str) -> Generator[str, None, None]:
    """
    临时文件自动清理上下文管理器
    """
    try:
        yield file_path
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug("临时文件已清理: %s", file_path)
            except Exception as e:
                logger.warning("清理临时文件失败: %s, 错误: %s", file_path, e)

def validate_audio_file(audio_path: str) -> None:
    """
    校验音频文件是否存在且非空
    """
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail=f"音频文件不存在: {audio_path}")
    if os.path.getsize(audio_path) == 0:
        raise HTTPException(status_code=400, detail=f"音频文件为空: {audio_path}")

def ensure_wav_16k(audio_path: str) -> str:
    """
    将输入音频统一转换为16kHz单声道WAV，返回转换后的路径
    """
    validate_audio_file(audio_path)
    temp_dir = tempfile.gettempdir()
    out_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.wav")
    try:
        logger.info("📼 正在转换音频 src=%s → dst=%s", audio_path, out_path)
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            timeout=10
        )
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise HTTPException(status_code=500, detail="音频转换后文件为空")
        logger.info("✅ 音频转换完成 dst=%s size_bytes=%s", out_path, os.path.getsize(out_path))
        return out_path
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="音频转换超时（10秒），请检查文件大小或格式")
    except FileNotFoundError:
        raise HTTPException(status_code=501, detail="未检测到 ffmpeg，请先安装：brew install ffmpeg / apt install ffmpeg")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="ignore")[:200]
        logger.error("❌ 音频转换失败: %s, 错误信息: %s", audio_path, stderr)
        raise HTTPException(status_code=500, detail=f"音频转换失败：{stderr}")
