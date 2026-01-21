import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from conf.settings import SETTINGS

logger = logging.getLogger("llm")

def _normalize_model(model: str) -> str:
    """
    规范化同义千问（DashScope）模型名，兼容 Ollama 风格。
    """
    m = (model or "").strip().lower()
    alias = {
        "qwen2.5:7b": "qwen-turbo",
        "qwen2.5-7b": "qwen-turbo",
        "qwen:7b": "qwen-turbo",
        "qwen2:7b": "qwen-turbo",
        "qwen2.5:32b": "qwen-plus",
        "qwen2.5-32b": "qwen-plus",
    }
    return alias.get(m, model)

def generate_tongyi(prompt: str, model: Optional[str] = None, temperature: float = 0.2) -> str:
    """
    使用 LangChain 的 ChatOpenAI 通过 DashScope 兼容模式生成文本。
    """
    model_ds = _normalize_model(model or SETTINGS.DEFAULT_LLM_MODEL)
    llm = ChatOpenAI(
        model=model_ds,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=SETTINGS.DASHSCOPE_API_KEY,
        temperature=temperature,
    )
    messages = [
        SystemMessage(content="你是一名资深会议纪要助理，输出清晰的结构化中文纪要或答案。"),
        HumanMessage(content=prompt),
    ]
    logger.info("LLM(tongyi) model=%s temp=%s prompt_chars=%s", getattr(llm, "model_name", model_ds), temperature, len(prompt or ""))
    resp = llm.invoke(messages)
    logger.info("LLM(tongyi) response_chars=%s", len(str(resp.content or "")))
    return str(resp.content).strip()
