import os
from typing import Optional, List
from datetime import datetime

from sqlmodel import SQLModel, Session, create_engine, select

from .models import Meeting

_engine = None


def init_db(db_path: str) -> None:
    """
    初始化数据库连接与表结构。
    """
    global _engine
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _engine = create_engine(f"sqlite:///{db_path}", echo=False)
    SQLModel.metadata.create_all(_engine)


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
        return session.get(Meeting, meeting_id)

