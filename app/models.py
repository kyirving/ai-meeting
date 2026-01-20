from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field


class Meeting(SQLModel, table=True):
    """
    会议实体表：存储会议名称、时间、录音路径、转写文本与会议纪要。
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    recorded_at: datetime
    audio_path: Optional[str] = None
    transcript: Optional[str] = None
    minutes: Optional[str] = None


class MeetingCreateResponse(SQLModel):
    """
    创建会议返回：包含会议ID、名称、时间与文本预览。
    """
    id: int
    name: str
    recorded_at: str
    transcript_preview: str
    minutes_preview: str


class MeetingSummaryItem(SQLModel):
    """
    历史会议列表项：基础信息与纪要预览。
    """
    id: int
    name: str
    recorded_at: str
    minutes_preview: str


class MeetingListResponse(SQLModel):
    """
    历史会议分页返回：包含列表与分页信息。
    """
    page: int
    page_size: int
    items: List[MeetingSummaryItem]


class MeetingDetailResponse(SQLModel):
    """
    会议详情返回：完整转写与纪要。
    """
    id: int
    name: str
    recorded_at: str
    transcript: str
    minutes: str
    audio_path: str


class RagSearchResponse(SQLModel):
    """
    RAG 检索返回：查询消息、答案与引用来源。
    """
    message: str
    answer: str
    citations: List[dict]
