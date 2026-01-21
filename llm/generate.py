from typing import Optional
import logging

from conf.settings import SETTINGS
from llm.ollama import generate_ollama
from llm.openai import generate_openai
from llm.tongyi import generate_tongyi

logger = logging.getLogger("llm")

def _dispatch_generate(provider: str, prompt: str, model: str, temperature: float) -> str:
    """
    按提供方分发到对应模块生成文本。
    """
    if provider in {"ollama", "local"}:
        return generate_ollama(prompt=prompt, model=model, temperature=temperature)
    if provider in {"openai", "oai"}:
        return generate_openai(prompt=prompt, model=model, temperature=temperature)
    if provider in {"dashscope", "tongyi", "ali", "aliyun"}:
        return generate_tongyi(prompt=prompt, model=model, temperature=temperature)
    return generate_ollama(prompt=prompt, model=model, temperature=temperature)


    # LangChain 已统一调用路径，无需回退到 HTTP

def _minutes_prompt(transcript: str, meeting_name: str) -> str:
    """
    构造会议纪要生成提示词。
    """
    return (
        f"你是一名资深会议纪要助理。以下是会议《{meeting_name}》的转写文本，请生成结构化会议纪要，包含：\n"
        f"1. 会议摘要（不超过200字）；\n"
        f"2. 关键讨论点（条目化）；\n"
        f"3. 决策与结论（条目化）；\n"
        f"4. 待办事项（负责人与截止日期）；\n"
        f"5. 风险与后续跟进；\n\n"
        f"会议转写：\n{transcript}\n\n"
        f"请使用清晰的中文、保持简洁，可直接输出Markdown结构。"
    )


def generate_minutes(
    prompt_text: Optional[str] = None,
    meeting_name: Optional[str] = None,
    transcript: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """
    统一调度至不同 LLM 提供方模块（LangChain）。
    """
    provider = (getattr(SETTINGS, "LLM_PROVIDER", "ollama") or "ollama").lower()
    model = model or SETTINGS.DEFAULT_LLM_MODEL
    if prompt_text is None:
        if not transcript or not meeting_name:
            raise ValueError("transcript 与 meeting_name 不能为空")
        prompt = _minutes_prompt(transcript, meeting_name)
    else:
        prompt = prompt_text
    return _dispatch_generate(provider=provider, prompt=prompt, model=model, temperature=temperature)
