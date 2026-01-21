import os
from dataclasses import dataclass
from typing import Optional, Any, Dict
from pathlib import Path
import yaml
from dotenv import load_dotenv


@dataclass
class _Settings:
    """
    服务配置：路径与本地模型配置。
    """
    ROOT: str = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
    DATA_DIR: str = os.path.join(ROOT, "data")
    AUDIO_DIR: str = os.path.join(DATA_DIR, "audio")
    DB_PATH: str = os.path.join(DATA_DIR, "meetings.db")
    FAISS_DIR: str = os.path.join(DATA_DIR, "faiss")
    FAISS_META_PATH: str = os.path.join(DATA_DIR, "faiss_meta.json")
    MEETINGS_DIR: str = os.path.join(DATA_DIR, "mettings")

    DEFAULT_LLM_MODEL: str = "qwen2.5:7b"
    EMBED_MODEL: str = "nomic-embed-text"
    ASR_PROVIDER: str = "faster-whisper"
    ASR_MODEL: str = "paraformer-zh"
    WHISPER_MODEL: str = "tiny"
    RAG_ENABLED: bool = True
    LLM_PROVIDER: str = "tongyi"
    LLM_BASE_URL: str = "http://127.0.0.1:11434"
    LLM_API_KEY: str = ""
    EMBED_PROVIDER: str = "tongyi"
    DASHSCOPE_API_KEY: str = ""
    USE_LANGCHAIN: bool = False
    ZHIPUAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    OPENAI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    RAG_PROVIDER: str = "faiss"
    CHROMA_DIR: str = os.path.join(ROOT, "data/chroma")
    def _normalize_provider(self, p: Optional[str]) -> str:
        """
        归一化 Provider 名称。
        """
        p = (p or "").strip().lower()
        if p in {"tongyi", "dashscope", "ali", "aliyun"}:
            return "dashscope"
        if p in {"openai", "oai"}:
            return "openai"
        if p in {"ollama", "local"}:
            return "ollama"
        return p or "ollama"

    def ensure_dirs(self) -> None:
        """
        确保数据目录存在。
        """
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.AUDIO_DIR, exist_ok=True)
        os.makedirs(self.FAISS_DIR, exist_ok=True)
        os.makedirs(self.CHROMA_DIR, exist_ok=True)
        os.makedirs(self.MEETINGS_DIR, exist_ok=True)

    def _abs_from_root(self, p: str) -> str:
        """
        将相对路径转为项目根的绝对路径。
        """
        path = Path(p)
        if not path.is_absolute():
            path = Path(self.ROOT) / path
        return str(path.resolve())

    def load(self, config_path: Optional[str] = None) -> None:
        """
        读取 .env 与 YAML 配置并刷新设置。
        """
        app_env = os.getenv("APP_ENV", "prod").lower()
        env_file = ".env.example" if app_env == "dev" else ".env"
        env_path = os.path.join(self.ROOT, env_file)
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path, override=True)
        else:
            load_dotenv()
        cfg_path = config_path or os.getenv("APP_CONFIG") or os.path.join(self.ROOT, "conf", "config.yaml")
        cfg: Dict[str, Any] = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        paths = cfg.get("paths", {})
        models = cfg.get("models", {})
        asr = cfg.get("asr", {})
        rag = cfg.get("rag", {})

        data_dir = os.getenv("DATA_DIR", paths.get("data_dir", self.DATA_DIR))
        audio_subdir = paths.get("audio_subdir", "audio")
        db_name = paths.get("db_name", "meetings.db")
        faiss_subdir = paths.get("faiss_subdir", "faiss")
        faiss_meta_name = paths.get("faiss_meta_name", "faiss_meta.json")
        meetings_subdir = paths.get("meetings_subdir", "mettings")

        self.DATA_DIR = self._abs_from_root(data_dir)
        self.AUDIO_DIR = self._abs_from_root(os.path.join(self.DATA_DIR, audio_subdir))
        self.DB_PATH = self._abs_from_root(os.path.join(self.DATA_DIR, db_name))
        self.FAISS_DIR = self._abs_from_root(os.path.join(self.DATA_DIR, faiss_subdir))
        self.FAISS_META_PATH = self._abs_from_root(os.path.join(self.DATA_DIR, faiss_meta_name))
        self.MEETINGS_DIR = self._abs_from_root(os.path.join(self.DATA_DIR, meetings_subdir))

        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", models.get("OLLAMA_BASE_URL", self.OLLAMA_BASE_URL))
        self.DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", models.get("default_llm_model", self.DEFAULT_LLM_MODEL))
        self.EMBED_MODEL = os.getenv("EMBED_MODEL", rag.get("embed_model", self.EMBED_MODEL))

        self.ASR_PROVIDER = os.getenv("ASR_PROVIDER", asr.get("provider", self.ASR_PROVIDER)).lower()
        self.ASR_MODEL = os.getenv("ASR_MODEL", asr.get("model", self.ASR_MODEL))
        self.WHISPER_MODEL = os.getenv("WHISPER_MODEL", asr.get("whisper_model", self.WHISPER_MODEL))
        self.RAG_ENABLED = bool(str(os.getenv("RAG_ENABLED", rag.get("enabled", self.RAG_ENABLED))).lower() in ["1", "true", "yes"])
        self.LLM_PROVIDER = self._normalize_provider(os.getenv("LLM_PROVIDER", models.get("llm_provider", self.LLM_PROVIDER)))
        # 兼容 OPENAI_BASE_URL / OPENAI_API_KEY
        self.LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", models.get("llm_base_url", self.LLM_BASE_URL)))
        self.LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", models.get("llm_api_key", self.LLM_API_KEY)))
        self.EMBED_PROVIDER = self._normalize_provider(os.getenv("EMBED_PROVIDER", rag.get("embed_provider", self.EMBED_PROVIDER)))
        # DashScope API Key
        self.DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", models.get("dashscope_api_key", self.DASHSCOPE_API_KEY))
        self.USE_LANGCHAIN = bool(str(os.getenv("USE_LANGCHAIN", models.get("use_langchain", self.USE_LANGCHAIN))).lower() in ["1", "true", "yes"])
        self.RAG_PROVIDER = os.getenv("RAG_PROVIDER", rag.get("provider", self.RAG_PROVIDER)).lower()
        self.CHROMA_DIR = os.getenv("CHROMA_DIR", rag.get("chroma_dir", self.CHROMA_DIR))
        self.MEETINGS_DIR = os.getenv("MEETINGS_DIR", paths.get("meetings_dir", self.MEETINGS_DIR))

        # 美化输出 使用什么LLM、ASR、RAG等
        print("=" * 50)
        print("配置信息")

        print(f"LLM_PROVIDER: {self.LLM_PROVIDER}")
        print(f"LLM_BASE_URL: {self.LLM_BASE_URL}")
        print(f"LLM_API_KEY: {self.LLM_API_KEY}")

        print(f"OLLAMA_BASE_URL: {self.OLLAMA_BASE_URL}")
        print(f"EMBED_PROVIDER: {self.EMBED_PROVIDER} EMBED_MODEL: {self.EMBED_MODEL}")

        print(f"ASR_PROVIDER: {self.ASR_PROVIDER}")
        print(f"ASR_MODEL: {self.ASR_MODEL}")
        print(f"WHISPER_MODEL: {self.WHISPER_MODEL}")
        print(f"RAG_ENABLED: {self.RAG_ENABLED}")
        print(f"USE_LANGCHAIN: {self.USE_LANGCHAIN}")
        print(f"RAG_PROVIDER: {self.RAG_PROVIDER} CHROMA_DIR: {self.CHROMA_DIR}")
        print(f"MEETINGS_DIR: {self.MEETINGS_DIR}")

SETTINGS = _Settings()
