import os
import uuid
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.models import MeetingCreateResponse, MeetingDetailResponse, MeetingSummaryItem, MeetingListResponse, RagSearchResponse
from app.storage import init_db, create_meeting, list_meetings, get_meeting
from conf.settings import SETTINGS
from asr.transcribe import transcribe_audio
from llm.generate import generate_minutes
from rag.index import RagIndex
from common.utils import chunk_text

# 加载配置与初始化
SETTINGS.load()
SETTINGS.ensure_dirs()
init_db(SETTINGS.DB_PATH)
rag_index = RagIndex(persist_dir=SETTINGS.FAISS_DIR, meta_path=SETTINGS.FAISS_META_PATH, embed_model=SETTINGS.EMBED_MODEL)

app = FastAPI(title="AI 会议纪要服务", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload(file: UploadFile, target_dir: str) -> str:
    """
    保存上传的音频文件到指定目录，并返回保存后的绝对路径。
    """
    ext = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(target_dir, fname)
    with open(fpath, "wb") as f:
        f.write(file.file.read())
    return fpath


@app.post("/api/meetings/realtime", response_model=MeetingCreateResponse)
def create_realtime_meeting(
    audio: UploadFile = File(...),
    name: str = Form(...),
    asr_backend: str = Form(default="paraformer"),  # whisper | paraformer
    asr_model: Optional[str] = Form(default="paraformer-zh"),  # whisper 模型名或 funasr 预设
    llm_model: Optional[str] = Form(default="qwen2.5:7b"),  # Ollama 模型名
):
    """
    实时会议处理接口：将上传的录音转写为文本，交给本地大模型生成会议纪要，并写入RAG。
    """
    try:
        audio_path = _save_upload(audio, SETTINGS.AUDIO_DIR)
        transcript = transcribe_audio(audio_path, backend=asr_backend, model_name=asr_model)
        minutes = generate_minutes(transcript=transcript, meeting_name=name, model=llm_model)

        meeting = create_meeting(
            name=name,
            recorded_at=datetime.now(timezone.utc),
            transcript=transcript,
            minutes=minutes,
            audio_path=audio_path,
        )

        for i, chunk in enumerate(chunk_text(minutes, chunk_size=800, overlap=100)):
            rag_index.add_texts(
                texts=[chunk],
                metadatas=[{"meeting_id": meeting.id, "meeting_name": meeting.name, "chunk_id": i, "recorded_at": str(meeting.recorded_at)}],
            )

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
    name: str = Form(...),
):
    """
    上传会议录音接口：仅保存文件与基本记录，不做转写与纪要生成。
    """
    try:
        audio_path = _save_upload(audio, SETTINGS.AUDIO_DIR)
        meeting = create_meeting(
            name=name,
            recorded_at=datetime.now(timezone.utc),
            transcript=None,
            minutes=None,
            audio_path=audio_path,
        )
        return MeetingCreateResponse(
            id=meeting.id,
            name=meeting.name,
            recorded_at=meeting.recorded_at.isoformat(),
            transcript_preview="",
            minutes_preview="",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")


@app.get("/api/rag/search", response_model=RagSearchResponse)
def rag_search(
    q: str = Query(..., description="检索问题"),
    k: int = Query(default=5, ge=1, le=20, description="检索条数"),
    temperature: float = Query(default=0.2, ge=0.0, le=1.0, description="LLM 温度"),
    llm_model: Optional[str] = Query(default="qwen2.5:7b", description="Ollama 模型名"),
):
    """
    RAG 检索接口：从本地向量库检索相关纪要片段，并使用本地大模型生成回答。
    """
    try:
        results = rag_index.search(q, top_k=k)
        context = "\n\n".join([r["text"] for r in results])
        answer = generate_minutes(
            prompt_text=f"基于以下会议纪要片段回答问题：\n\n{context}\n\n问题：{q}\n请给出简洁、准确的回答，并列出关键引用点。",
            meeting_name="RAG 查询",
            model=llm_model,
            temperature=temperature,
        )
        return RagSearchResponse(
            question=q,
            answer=answer,
            citations=[{"meeting_id": r["metadata"]["meeting_id"], "meeting_name": r["metadata"]["meeting_name"], "chunk_id": r["metadata"]["chunk_id"]} for r in results],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 检索失败: {e}")
