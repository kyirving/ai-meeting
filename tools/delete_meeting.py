import os
import sys
import glob
import shutil
import argparse
import logging
from typing import Optional, List, Dict, Any

from sqlmodel import Session, select

from conf.settings import SETTINGS
from app.storage import init_db
from app.models import Meeting
from rag.index import RagIndex

logger = logging.getLogger("cleanup")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

def _safe_unlink(path: Optional[str]) -> bool:
    """
    删除单个文件（若存在），返回是否成功
    """
    if not path:
        return False
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info("删除文件: %s", path)
            return True
    except Exception as e:
        logger.warning("删除文件失败: %s error=%s", path, e)
    return False

def _delete_minutes_md(meeting_id: int, title: str) -> int:
    """
    删除该会议的 Markdown 纪要文件，按 <规范化标题>-<id>.md 匹配
    """
    from main import _sanitize_filename  # 复用项目内规范化函数
    safe = _sanitize_filename(title)
    pattern = os.path.join(SETTINGS.MEETINGS_DIR, f"{safe}-{meeting_id}.md")
    cnt = 0
    for p in glob.glob(pattern):
        if _safe_unlink(p):
            cnt += 1
    # 兜底：按 id 后缀匹配所有文件
    pattern2 = os.path.join(SETTINGS.MEETINGS_DIR, f"*-{meeting_id}.md")
    for p in glob.glob(pattern2):
        if _safe_unlink(p):
            cnt += 1
    return cnt

def _rag_delete_by_meeting(meeting_id: int) -> None:
    """
    从向量库删除该会议的所有片段，FAISS 重建索引，Chroma 直接删除持久化数据
    """
    provider = (getattr(SETTINGS, "RAG_PROVIDER", "faiss") or "faiss").lower()
    if provider == "chroma":
        try:
            # 直接清理持久化目录，保证删除干净
            if os.path.isdir(SETTINGS.CHROMA_DIR):
                shutil.rmtree(SETTINGS.CHROMA_DIR, ignore_errors=True)
                os.makedirs(SETTINGS.CHROMA_DIR, exist_ok=True)
                logger.info("Chroma 已清空目录: %s", SETTINGS.CHROMA_DIR)
        except Exception as e:
            logger.warning("Chroma 目录清理失败: %s error=%s", SETTINGS.CHROMA_DIR, e)
        return

    # FAISS 路径：移除 metas 中 meeting_id 关联项并重建索引
    meta_path = SETTINGS.FAISS_META_PATH
    try:
        metas: List[Dict[str, Any]] = []
        if os.path.exists(meta_path):
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                metas = json.load(f) or []
        before = len(metas)
        remain = [m for m in metas if (m.get("metadata") or {}).get("meeting_id") != meeting_id]
        removed = before - len(remain)
        if removed <= 0:
            logger.info("FAISS 无需删除：未找到 meeting_id=%s 的条目", meeting_id)
            return
        # 重建索引
        rag = RagIndex(SETTINGS.FAISS_DIR, SETTINGS.FAISS_META_PATH, SETTINGS.EMBED_MODEL)
        texts = [m.get("text", "") for m in remain if m.get("text")]
        import numpy as np
        if texts:
            embs = rag._embed(texts)
            dim = embs.shape[1]
            import faiss
            index = faiss.IndexFlatIP(dim)
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / norms
            index.add(embs)
            rag.index = index
        else:
            rag.index = None
        # 覆盖 metas 并写盘
        rag._metas = remain
        if rag.index is not None:
            faiss = __import__("faiss")
            faiss.write_index(rag.index, os.path.join(SETTINGS.FAISS_DIR, "faiss.index"))
        with open(SETTINGS.FAISS_META_PATH, "w", encoding="utf-8") as f:
            import json
            json.dump(remain, f, ensure_ascii=False, indent=2)
        logger.info("FAISS 删除完成 meeting_id=%s 移除条数=%s 剩余=%s", meeting_id, removed, len(remain))
    except Exception as e:
        logger.warning("FAISS 删除失败 meeting_id=%s error=%s", meeting_id, e)

def delete_meeting(meeting_id: int) -> None:
    """
    删除指定会议记录：数据库、音频文件、纪要 MD、向量库条目
    """
    SETTINGS.load()
    SETTINGS.ensure_dirs()
    init_db(SETTINGS.DB_PATH)
    # 读取实体
    from app.storage import _engine
    with Session(_engine) as session:
        m = session.get(Meeting, meeting_id)
        if not m:
            logger.info("会议不存在 id=%s", meeting_id)
            return
        # 删除文件
        _safe_unlink(m.audio_path)
        _delete_minutes_md(m.id, m.name or "未命名会议")
        # 删除向量库
        _rag_delete_by_meeting(m.id)
        # 删除数据库记录
        session.delete(m)
        session.commit()
        logger.info("数据库记录已删除 id=%s", meeting_id)

def delete_all() -> None:
    """
    删除所有会议数据：数据库全部会议、音频、纪要、向量库
    """
    SETTINGS.load()
    SETTINGS.ensure_dirs()
    init_db(SETTINGS.DB_PATH)
    from app.storage import _engine
    with Session(_engine) as session:
        meetings = list(session.exec(select(Meeting)))
        for m in meetings:
            _safe_unlink(m.audio_path)
            _delete_minutes_md(m.id, m.name or "未命名会议")
            session.delete(m)
        session.commit()
        logger.info("数据库会议记录已全部删除 count=%s", len(meetings))
    # 清理向量库
    try:
        if getattr(SETTINGS, "RAG_PROVIDER", "faiss").lower() == "chroma":
            if os.path.isdir(SETTINGS.CHROMA_DIR):
                shutil.rmtree(SETTINGS.CHROMA_DIR, ignore_errors=True)
                os.makedirs(SETTINGS.CHROMA_DIR, exist_ok=True)
                logger.info("Chroma 已清空目录: %s", SETTINGS.CHROMA_DIR)
        else:
            if os.path.exists(SETTINGS.FAISS_META_PATH):
                os.remove(SETTINGS.FAISS_META_PATH)
            idx_path = os.path.join(SETTINGS.FAISS_DIR, "faiss.index")
            if os.path.exists(idx_path):
                os.remove(idx_path)
            logger.info("FAISS 索引与元数据已清理")
    except Exception as e:
        logger.warning("向量库清理失败 error=%s", e)
    # 可选清理音频与纪要目录残留
    for p in glob.glob(os.path.join(SETTINGS.MEETINGS_DIR, "*.md")):
        _safe_unlink(p)
    logger.info("纪要目录清理完成: %s", SETTINGS.MEETINGS_DIR)

def main():
    """
    命令行入口：支持按会议ID删除或清空全部
    """
    parser = argparse.ArgumentParser(description="会议数据删除工具")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", type=int, help="要删除的会议ID")
    g.add_argument("--all", action="store_true", help="删除全部会议数据")
    args = parser.parse_args()
    if args.all:
        delete_all()
    else:
        delete_meeting(args.id)

if __name__ == "__main__":
    main()
