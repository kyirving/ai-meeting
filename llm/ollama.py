import logging
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from conf.settings import SETTINGS

logger = logging.getLogger("llm")

def generate_ollama(prompt: str, model: Optional[str] = None, temperature: float = 0.2) -> str:
    """
    使用 LangChain 的 ChatOllama 生成文本（本地 Ollama）。
    """
    llm = ChatOllama(
        model=(model or SETTINGS.DEFAULT_LLM_MODEL),
        base_url=SETTINGS.LLM_BASE_URL,
        temperature=temperature,
    )
    messages = [
        SystemMessage(content="你是一名资深会议纪要助理，输出清晰的结构化中文纪要或答案。"),
        HumanMessage(content=prompt),
    ]
    logger.info("LLM(ollama) model=%s temp=%s prompt_chars=%s", getattr(llm, "model_name", model or SETTINGS.DEFAULT_LLM_MODEL), temperature, len(prompt or ""))
    resp = llm.invoke(messages)
    logger.info("LLM(ollama) response_chars=%s", len(str(resp.content or "")))
    return str(resp.content).strip()
