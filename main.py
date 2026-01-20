import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware

from app.models import MeetingCreateResponse, MeetingDetailResponse, MeetingSummaryItem, MeetingListResponse, RagSearchResponse
from app.storage import init_db, create_meeting, list_meetings, get_meeting, update_meeting
from conf.settings import SETTINGS
from asr.transcribe import transcribe_audio
from llm.generate import generate_minutes
from rag.index import RagIndex
from common.utils import chunk_text

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("app")

# 加载配置与初始化
SETTINGS.load()
SETTINGS.ensure_dirs()
init_db(SETTINGS.DB_PATH)
rag_index = RagIndex(
    persist_dir=SETTINGS.FAISS_DIR,
    meta_path=SETTINGS.FAISS_META_PATH,
    embed_model=SETTINGS.EMBED_MODEL,
)
logger.info("Service starting with settings DB=%s AUDIO_DIR=%s FAISS_DIR=%s EMBED_MODEL=%s LLM_PROVIDER=%s",
            SETTINGS.DB_PATH, SETTINGS.AUDIO_DIR, SETTINGS.FAISS_DIR, SETTINGS.EMBED_MODEL, SETTINGS.LLM_PROVIDER)

app = FastAPI(title="AI 会议纪要服务", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _sanitize_filename(name: str) -> str:
    """
    规范化文件名，保留中英文、数字、下划线和连字符，空白转为下划线。
    """
    import re
    s = (name or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-\u4e00-\u9fff]", "", s)
    return s or uuid.uuid4().hex


def _save_upload(file: UploadFile, target_dir: str, base_name: Optional[str] = None) -> str:
    """
    保存上传的音频文件到指定目录，并返回保存后的绝对路径。
    """
    ext = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    if base_name:
        safe = _sanitize_filename(base_name)
        fname = f"{safe}{ext}"
        fpath_try = os.path.join(target_dir, fname)
        if os.path.exists(fpath_try):
            fname = f"{safe}-{uuid.uuid4().hex[:8]}{ext}"
    else:
        fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(target_dir, fname)
    with open(fpath, "wb") as f:
        f.write(file.file.read())
    logger.info("Saved upload filename_req=%s saved_as=%s path=%s size_bytes=%s", file.filename, fname, fpath, os.path.getsize(fpath))
    return fpath


def _derive_title(file: UploadFile, fallback: Optional[str]) -> str:
    """
    依据上传文件名派生会议标题，若不存在则使用入参回退值。
    """
    raw = (file.filename or "").strip()
    if raw:
        base = os.path.splitext(os.path.basename(raw))[0]
        logger.info("Derived title from filename=%s -> %s", raw, base or (fallback or "未命名会议"))
        return base or (fallback or "未命名会议")
    logger.info("No filename, use fallback title=%s", fallback or "未命名会议")
    return fallback or "未命名会议"


def _process_meeting_bg(
    meeting_id: int,
    audio_path: str,
    title: str,
    asr_backend: str,
    asr_model: Optional[str],
    llm_model: Optional[str],
) -> None:
    """
    后台任务：转写音频、生成纪要并写入RAG与数据库。
    """
    try:
        logger.info("BG start meeting_id=%s audio=%s backend=%s model=%s llm=%s", meeting_id, audio_path, asr_backend, asr_model, llm_model)
        transcript = transcribe_audio(audio_path, backend=asr_backend, model_name=asr_model)
        logger.info("BG transcript done chars=%s", len(transcript or ""))
        logger.info("BG transcript content:\n%s", transcript or "")
        minutes = generate_minutes(transcript=transcript, meeting_name=title, model=llm_model)
        logger.info("BG minutes done chars=%s", len(minutes or ""))
        logger.info("BG minutes content:\n%s", minutes or "")
        m = update_meeting(meeting_id, transcript=transcript, minutes=minutes)
        if m:
            text_to_index = minutes if (getattr(SETTINGS, "INDEX_SOURCE", "minutes") == "minutes" and minutes) else (transcript or "")
            logger.info("BG indexing source=%s len=%s", getattr(SETTINGS, "INDEX_SOURCE", "minutes"), len(text_to_index))
            for i, chunk in enumerate(chunk_text(text_to_index, chunk_size=800, overlap=100)):
                rag_index.add_texts(
                    texts=[chunk],
                    metadatas=[{"meeting_id": m.id, "meeting_name": m.name, "chunk_id": i, "recorded_at": str(m.recorded_at)}],
                )
        logger.info("BG done meeting_id=%s", meeting_id)
    except Exception:
        logger.exception("BG processing failed meeting_id=%s", meeting_id)
        pass


@app.post("/api/meetings/realtime", response_model=MeetingCreateResponse)
def create_realtime_meeting(
    audio: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    asr_backend: Optional[str] = Form(default=None),
    asr_model: Optional[str] = Form(default=None),
    llm_model: Optional[str] = Form(default="qwen2.5:7b"),  # Ollama 模型名
    async_process: Optional[bool] = Form(default=True),
    bg: BackgroundTasks = None,
):
    """
    实时会议处理接口：将上传的录音转写为文本，交给本地大模型生成会议纪要，并写入RAG。
    """
    try:
        eff_backend = (asr_backend or SETTINGS.ASR_PROVIDER)
        eff_model = asr_model or (SETTINGS.WHISPER_MODEL if str(eff_backend).lower() in ["whisper", "faster-whisper", "openai-whisper"] else SETTINGS.ASR_MODEL)
        logger.info("Realtime upload filename=%s async=%s backend=%s model=%s llm=%s", audio.filename, async_process, eff_backend, eff_model, llm_model)
        title_final = title or _derive_title(audio, title)
        audio_path = _save_upload(audio, SETTINGS.AUDIO_DIR, base_name=title_final)
        logger.info("Create meeting title=%s", title_final)
        meeting = create_meeting(
            name=title_final,
            recorded_at=datetime.now(timezone.utc),
            transcript=None,
            minutes=None,
            audio_path=audio_path,
        )
        if async_process:
            if bg is not None:
                bg.add_task(_process_meeting_bg, meeting.id, audio_path, title_final, eff_backend, eff_model, llm_model)
            logger.info("Queued BG processing meeting_id=%s", meeting.id)
            return MeetingCreateResponse(
                id=meeting.id,
                name=meeting.name,
                recorded_at=meeting.recorded_at.isoformat(),
                transcript_preview="",
                minutes_preview="",
            )
        logger.info("Sync processing meeting_id=%s", meeting.id)
        transcript = transcribe_audio(audio_path, backend=eff_backend, model_name=eff_model)
        logger.info("Transcript done chars=%s", len(transcript or ""))
        logger.info("Transcript content:\n%s", transcript or "")
        minutes = generate_minutes(transcript=transcript, meeting_name=title_final, model=llm_model)
        logger.info("Minutes done chars=%s", len(minutes or ""))
        logger.info("Minutes content:\n%s", minutes or "")
        update_meeting(meeting.id, transcript=transcript, minutes=minutes)
        text_to_index = minutes if (getattr(SETTINGS, "INDEX_SOURCE", "minutes") == "minutes" and minutes) else (transcript or "")
        logger.info("Sync indexing source=%s len=%s", getattr(SETTINGS, "INDEX_SOURCE", "minutes"), len(text_to_index))
        for i, chunk in enumerate(chunk_text(text_to_index, chunk_size=800, overlap=100)):
            rag_index.add_texts(
                texts=[chunk],
                metadatas=[{"meeting_id": meeting.id, "meeting_name": meeting.name, "chunk_id": i, "recorded_at": str(meeting.recorded_at)}],
            )
        logger.info("Sync processing complete meeting_id=%s", meeting.id)
        return MeetingCreateResponse(
            id=meeting.id,
            name=meeting.name,
            recorded_at=meeting.recorded_at.isoformat(),
            transcript_preview=transcript[:400],
            minutes_preview=minutes[:400],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Realtime meeting failed")
        raise HTTPException(status_code=500, detail=f"会议处理失败: {e}")


@app.get("/api/meetings/history", response_model=MeetingListResponse)
def get_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
):
    """
    历史会议列表接口：分页返回历史会议摘要。
    """
    meetings = list_meetings(page=page, page_size=page_size)
    items: List[MeetingSummaryItem] = []
    for m in meetings:
        items.append(
            MeetingSummaryItem(
                id=m.id,
                name=m.name,
                recorded_at=m.recorded_at.isoformat(),
                minutes_preview=(m.minutes or "")[:400],
            )
        )
    return MeetingListResponse(
        page=page,
        page_size=page_size,
        items=items,
    )


@app.get("/api/meetings/{meeting_id}", response_model=MeetingDetailResponse)
def get_meeting_detail(meeting_id: int):
    """
    会议详情接口：返回指定会议的完整转写与纪要。
    """
    m = get_meeting(meeting_id)
    if not m:
        raise HTTPException(status_code=404, detail="会议不存在")
    return MeetingDetailResponse(
        id=m.id,
        name=m.name,
        recorded_at=m.recorded_at.isoformat(),
        transcript=m.transcript or "",
        minutes=m.minutes or "",
        audio_path=m.audio_path or "",
    )


@app.post("/api/meetings/upload", response_model=MeetingCreateResponse)
def upload_meeting_audio(
    audio: UploadFile = File(...),
    name: Optional[str] = Form(default=None),
    asr_backend: Optional[str] = Form(default=None),
    asr_model: Optional[str] = Form(default=None),
    llm_model: Optional[str] = Form(default="qwen2.5:7b"),
    async_process: Optional[bool] = Form(default=True),
    bg: BackgroundTasks = None,
):
    """
    上传会议录音接口：仅保存文件与基本记录，不做转写与纪要生成。
    """
    try:
        title_final = name or _derive_title(audio, name)
        audio_path = _save_upload(audio, SETTINGS.AUDIO_DIR, base_name=title_final)
        eff_backend = (asr_backend or SETTINGS.ASR_PROVIDER)
        eff_model = asr_model or (SETTINGS.WHISPER_MODEL if str(eff_backend).lower() in ["whisper", "faster-whisper", "openai-whisper"] else SETTINGS.ASR_MODEL)
        logger.info("Upload-only title=%s async=%s backend=%s model=%s", title_final, async_process, eff_backend, eff_model)
        meeting = create_meeting(
            name=title_final,
            recorded_at=datetime.now(timezone.utc),
            transcript=None,
            minutes=None,
            audio_path=audio_path,
        )
        if async_process:
            if bg is not None:
                bg.add_task(_process_meeting_bg, meeting.id, audio_path, title_final, eff_backend, eff_model, llm_model)
            logger.info("Queued BG processing meeting_id=%s", meeting.id)
            return MeetingCreateResponse(
                id=meeting.id,
                name=meeting.name,
                recorded_at=meeting.recorded_at.isoformat(),
                transcript_preview="",
                minutes_preview="",
            )
        return MeetingCreateResponse(
            id=meeting.id,
            name=meeting.name,
            recorded_at=meeting.recorded_at.isoformat(),
            transcript_preview="",
            minutes_preview="",
        )
    except Exception as e:
        logger.exception("Upload meeting failed")
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")


@app.post("/api/rag/search", response_model=RagSearchResponse)
def rag_search(
    message: str = Body(..., description="检索问题"),
    k: int = Body(default=5, ge=1, le=20, description="检索条数"),
    temperature: float = Body(default=0.2, ge=0.0, le=1.0, description="LLM 温度"),
    llm_model: Optional[str] = Body(default="qwen2.5:7b", description="Ollama 模型名"),
    retrieval_only: bool = Body(default=False, description="仅检索，不生成答案"),
):
    """
    RAG 检索接口：从本地向量库检索相关纪要片段，并使用本地大模型生成回答。
    """
    try:
        logger.info("RAG search message=%s k=%s temp=%s llm=%s", message, k, temperature, llm_model)
        results = rag_index.search(message, top_k=k)
        logger.info("RAG hits count=%s", len(results))
        if retrieval_only:
            return RagSearchResponse(
                message=message,
                answer="",
                citations=[{"meeting_id": r["metadata"]["meeting_id"], "meeting_name": r["metadata"]["meeting_name"], "chunk_id": r["metadata"]["chunk_id"]} for r in results],
            )
        context = "\n\n".join([r["text"] for r in results])
        prompt = f"基于以下会议纪要片段回答问题：\n\n{context}\n\n问题：{message}\n请给出简洁、准确的回答，并列出关键引用点。"
        answer = generate_minutes(
            prompt_text=prompt,
            meeting_name="RAG 查询",
            model=llm_model,
            temperature=temperature,
        )
        logger.info("RAG answer chars=%s", len(answer or ""))
        return RagSearchResponse(
            message=message,
            answer=answer,
            citations=[{"meeting_id": r["metadata"]["meeting_id"], "meeting_name": r["metadata"]["meeting_name"], "chunk_id": r["metadata"]["chunk_id"]} for r in results],
        )
    except Exception as e:
        logger.exception("RAG search failed")
        raise HTTPException(status_code=500, detail=f"RAG 检索失败: {e}")
