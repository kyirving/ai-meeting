import os
import json
from typing import List, Dict, Any

import numpy as np
import faiss
import requests

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from conf.settings import SETTINGS
import logging

logger = logging.getLogger("rag")


class RagIndex:
    """
    简易RAG索引：使用 FAISS 存储向量，JSON 持久化元数据。
    """

    def __init__(self, persist_dir: str, meta_path: str, embed_model: str):
        self.persist_dir = persist_dir
        self.meta_path = meta_path
        self.embed_model = embed_model
        self.provider = getattr(SETTINGS, "RAG_PROVIDER", "faiss").lower()
        if self.provider == "chroma":
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings
            except ImportError:
                raise RuntimeError("未安装 chromadb，请安装后使用 RAG_PROVIDER=chromadb")
            os.makedirs(SETTINGS.CHROMA_DIR, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=SETTINGS.CHROMA_DIR, settings=ChromaSettings(allow_reset=False))
            self._collection = self._chroma_client.get_or_create_collection(name="meetings")
            logger.info("RAG(chroma) init dir=%s", SETTINGS.CHROMA_DIR)
            self.index = None
            self._metas = []
        else:
            os.makedirs(persist_dir, exist_ok=True)
            self.index_path = os.path.join(persist_dir, "faiss.index")
            self.index: faiss.IndexFlatIP = None
            self._metas: List[Dict[str, Any]] = []
            self._load()

    def _load(self) -> None:
        """
        加载索引与元数据，如果不存在则初始化。
        """
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = None
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self._metas = json.load(f)
        else:
            self._metas = []
        logger.info("RAG load index ntotal=%s metas=%s", getattr(self.index, "ntotal", 0) if self.index else 0, len(self._metas))

    def _persist(self) -> None:
        """
        持久化索引与元数据到磁盘。
        """
        if self.provider == "chroma":
            return
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metas, f, ensure_ascii=False, indent=2)
        logger.info("RAG persist index ntotal=%s metas=%s", getattr(self.index, "ntotal", 0) if self.index else 0, len(self._metas))

    def _embed_ollama(self, texts: List[str]) -> np.ndarray:
        """
        使用 Ollama embedding 生成向量。
        """
        logger.info("Embed(ollama) model=%s count=%s", self.embed_model, len(texts))
        embs = []
        for t in texts:
            r = requests.post(
                f"{SETTINGS.OLLAMA_BASE_URL.rstrip('/')}/api/embeddings",
                json={"model": self.embed_model, "prompt": t},
                timeout=120,
            )
            r.raise_for_status()
            embs.append(np.array(r.json()["embedding"], dtype="float32"))
        return np.vstack(embs)

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """
        使用 OpenAI 兼容接口生成向量。
        """
        headers = {}
        if SETTINGS.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {SETTINGS.LLM_API_KEY}"
        logger.info("Embed(openai) base=%s model=%s count=%s", SETTINGS.LLM_BASE_URL, self.embed_model, len(texts))
        embs = []
        for t in texts:
            r = requests.post(
                f"{SETTINGS.LLM_BASE_URL.rstrip('/')}/v1/embeddings",
                headers=headers,
                json={"model": self.embed_model, "input": t},
                timeout=120,
            )
            r.raise_for_status()
            embs.append(np.array(r.json()["data"][0]["embedding"], dtype="float32"))
        return np.vstack(embs)

    def _embed_dashscope(self, texts: List[str]) -> np.ndarray:
        """
        使用通义千问 DashScope 生成向量。
        """
        api_key = SETTINGS.DASHSCOPE_API_KEY or ""
        if not api_key:
            raise RuntimeError("未配置 DASHSCOPE_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding"
        model = self.embed_model
        if model in {"nomic-embed-text", "mxbai-embed-large", "text-embedding-3-large"}:
            model = "text-embedding-v1"
        logger.info("Embed(dashscope) model=%s count=%s", model, len(texts))
        payload = {"model": model, "input": texts}
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        outputs = data.get("output", {}).get("embeddings", [])
        if not outputs and "data" in data:
            outputs = data.get("data", [])
        vecs = []
        for item in outputs:
            vec = item.get("embedding")
            if vec is not None:
                vecs.append(np.array(vec, dtype="float32"))
        if not vecs:
            raise RuntimeError("DashScope embeddings 返回为空")
        return np.vstack(vecs)
    def _embed_langchain(self, texts: List[str]) -> np.ndarray:
        
        logger.info("Embed(langchain) model=%s count=%s", self.embed_model, len(texts))
        """
        使用 LangChain 生成向量。
        """
        provider = (getattr(SETTINGS, "EMBED_PROVIDER", "ollama") or "ollama").lower()
        if provider == "tongyi":
            provider = "dashscope"
        model = self.embed_model
        if provider == "openai" and model in {"nomic-embed-text", "mxbai-embed-large"}:
            model = "text-embedding-3-large"
        if provider == "dashscope" and model in {"nomic-embed-text", "mxbai-embed-large", "text-embedding-3-large"}:
            model = "text-embedding-v1"
        if provider == "ollama":
            embedder = OllamaEmbeddings(
                model=model,
                base_url=SETTINGS.OLLAMA_BASE_URL,
            )
        elif provider == "openai":
            embedder = OpenAIEmbeddings(
                model=model,
                base_url=SETTINGS.LLM_BASE_URL,
                api_key=SETTINGS.LLM_API_KEY,
            )
        elif provider == "dashscope":
            embedder = OpenAIEmbeddings(
                model=model,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=SETTINGS.DASHSCOPE_API_KEY,
            )
        else:
            embedder = OllamaEmbeddings(
                model=model,
                base_url=SETTINGS.OLLAMA_BASE_URL,
            )
        logger.info("Embed(langchain) provider=%s model=%s count=%s", provider, model, len(texts))
        # LangChain embed_documents 返回 List[List[float]]
        vecs = embedder.embed_documents(texts)
        return np.array(vecs, dtype="float32")

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        根据配置选择 embedding 提供方。
        """
        if getattr(SETTINGS, "USE_LANGCHAIN", False):
            p = (getattr(SETTINGS, "EMBED_PROVIDER", "ollama") or "ollama").lower()
            if p in ("dashscope", "tongyi"):
                return self._embed_dashscope(texts)
            return self._embed_langchain(texts)

        provider = (getattr(SETTINGS, "EMBED_PROVIDER", "ollama") or "ollama").lower()
        if provider == "tongyi":
            provider = "dashscope"
        if provider == "tongyi":
            provider = "dashscope"
        if provider == "openai":
            return self._embed_openai(texts)
        if provider == "dashscope":
            return self._embed_dashscope(texts)
        return self._embed_ollama(texts)

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """
        添加文本到向量索引。
        """
        if len(texts) != len(metadatas):
            raise ValueError("texts 与 metadatas 数量不一致")
        logger.info("RAG add_texts count=%s", len(texts))
        embs = self._embed(texts)
        if self.provider == "chroma":
            ids = [f"{m.get('meeting_id','m')}-{m.get('chunk_id',i)}-{i}" for i, m in enumerate(metadatas)]
            self._collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=[e.tolist() for e in embs])
            return
        dim = embs.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = embs / norms
        self.index.add(embs)
        for i, t in enumerate(texts):
            self._metas.append({"text": t, "metadata": metadatas[i]})
        self._persist()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        在向量索引中检索最相关的文本片段。
        """
        logger.info("RAG search query='%s' top_k=%s", query, top_k)
        if self.provider == "chroma":
            q = self._embed([query])
            res = self._collection.query(query_embeddings=[q[0].tolist()], n_results=top_k)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0] if "distances" in res else [0.0] * len(docs)
            return [{"text": d, "metadata": m, "score": float(s)} for d, m, s in zip(docs, metas, dists)]
        q = self._embed([query])
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        if self.index is None or self.index.ntotal == 0:
            logger.info("RAG search no index or empty")
            return []
        D, I = self.index.search(q, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self._metas):
                continue
            item = self._metas[idx]
            results.append({"text": item["text"], "metadata": item["metadata"], "score": float(score)})
        logger.info("RAG search results=%s", len(results))
        return results
