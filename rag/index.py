import os
import json
from typing import List, Dict, Any

import numpy as np
import faiss
import requests

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from conf.settings import SETTINGS


class RagIndex:
    """
    简易RAG索引：使用 FAISS 存储向量，JSON 持久化元数据。
    """

    def __init__(self, persist_dir: str, meta_path: str, embed_model: str):
        self.persist_dir = persist_dir
        self.meta_path = meta_path
        self.embed_model = embed_model
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

    def _persist(self) -> None:
        """
        持久化索引与元数据到磁盘。
        """
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metas, f, ensure_ascii=False, indent=2)

    def _embed_ollama(self, texts: List[str]) -> np.ndarray:
        """
        使用 Ollama embedding 生成向量。
        """
        embs = []
        for t in texts:
            r = requests.post(
                f"{SETTINGS.OLLAMA_API}/api/embeddings",
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
        embs = []
        for t in texts:
            payload = {"model": self.embed_model, "input": t}
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            vec = data.get("output", {}).get("embeddings", [{}])[0].get("embedding")
            if vec is None:
                # 兼容 openai 风格返回
                vec = data.get("data", [{}])[0].get("embedding")
            embs.append(np.array(vec, dtype="float32"))
        return np.vstack(embs)
    def _embed_langchain(self, texts: List[str]) -> np.ndarray:
        """
        使用 LangChain 生成向量。
        """
        provider = (getattr(SETTINGS, "EMBED_PROVIDER", "ollama") or "ollama").lower()
        if provider == "ollama":
            embedder = OllamaEmbeddings(
                model=self.embed_model,
                base_url=SETTINGS.OLLAMA_API,
            )
        elif provider == "openai":
            embedder = OpenAIEmbeddings(
                model=self.embed_model,
                base_url=SETTINGS.LLM_BASE_URL,
                api_key=SETTINGS.LLM_API_KEY,
            )
        elif provider == "dashscope":
            embedder = OpenAIEmbeddings(
                model=self.embed_model,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=SETTINGS.DASHSCOPE_API_KEY,
            )
        else:
            embedder = OllamaEmbeddings(
                model=self.embed_model,
                base_url=SETTINGS.OLLAMA_API,
            )
        
        # LangChain embed_documents 返回 List[List[float]]
        vecs = embedder.embed_documents(texts)
        return np.array(vecs, dtype="float32")

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        根据配置选择 embedding 提供方。
        """
        if getattr(SETTINGS, "USE_LANGCHAIN", False):
            return self._embed_langchain(texts)

        provider = (getattr(SETTINGS, "EMBED_PROVIDER", "ollama") or "ollama").lower()
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
        embs = self._embed(texts)
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
        q = self._embed([query])
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        if self.index is None or self.index.ntotal == 0:
            return []
        D, I = self.index.search(q, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self._metas):
                continue
            item = self._metas[idx]
            results.append({"text": item["text"], "metadata": item["metadata"], "score": float(score)})
        return results
