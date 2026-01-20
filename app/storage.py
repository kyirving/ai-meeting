import os
from typing import Optional, List
from datetime import datetime

from sqlmodel import SQLModel, Session, create_engine, select

from .models import Meeting
import logging

_engine = None
logger = logging.getLogger("storage")


def init_db(db_path: str) -> None:
    """
    初始化数据库连接与表结构。
    """
    global _engine
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _engine = create_engine(f"sqlite:///{db_path}", echo=False)
    SQLModel.metadata.create_all(_engine)
    logger.info("DB initialized path=%s", db_path)


def create_meeting(
    name: str,
    recorded_at: datetime,
    transcript: Optional[str],
    minutes: Optional[str],
    audio_path: Optional[str],
) -> Meeting:
    """
    创建会议记录并返回实体。
    """
    with Session(_engine) as session:
        m = Meeting(
            name=name,
            recorded_at=recorded_at,
            transcript=transcript,
            minutes=minutes,
            audio_path=audio_path,
        )
        session.add(m)
        session.commit()
        session.refresh(m)
        logger.info("DB create meeting id=%s name=%s", m.id, m.name)
        return m


def list_meetings(page: int = 1, page_size: int = 10) -> List[Meeting]:
    """
    分页查询会议列表。
    """
    with Session(_engine) as session:
        stmt = select(Meeting).order_by(Meeting.recorded_at.desc()).offset((page - 1) * page_size).limit(page_size)
        return list(session.exec(stmt))


def get_meeting(meeting_id: int) -> Optional[Meeting]:
    """
    获取会议详情。
    """
    with Session(_engine) as session:
        m = session.get(Meeting, meeting_id)
        logger.info("DB get meeting id=%s found=%s", meeting_id, bool(m))
        return m


def update_meeting(
    meeting_id: int,
    transcript: Optional[str] = None,
    minutes: Optional[str] = None,
    name: Optional[str] = None,
) -> Optional[Meeting]:
    """
    更新会议记录的转写、纪要或标题并返回实体。
    """
    with Session(_engine) as session:
        m = session.get(Meeting, meeting_id)
        if not m:
            logger.info("DB update meeting id=%s not found", meeting_id)
            return None
        if name is not None:
            m.name = name
        if transcript is not None:
            m.transcript = transcript
        if minutes is not None:
            m.minutes = minutes
        session.add(m)
        session.commit()
        session.refresh(m)
        logger.info("DB update meeting id=%s name=%s transcript=%s minutes=%s",
                    m.id, m.name, bool(transcript), bool(minutes))
        return m
