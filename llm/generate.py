import json
from typing import Optional
import requests
import logging

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from conf.settings import SETTINGS

logger = logging.getLogger("llm")


def _langchain_generate(model: str, prompt: str, temperature: float, provider: str) -> str:
    """
    使用 LangChain 调用 LLM。
    """
    if provider == "ollama":
        llm = ChatOllama(
            model=model,
            base_url=SETTINGS.LLM_BASE_URL,
            temperature=temperature,
        )
    elif provider == "openai":
        llm = ChatOpenAI(
            model=model,
            base_url=SETTINGS.LLM_BASE_URL,
            api_key=SETTINGS.LLM_API_KEY,
            temperature=temperature,
        )
    elif provider == "dashscope":
        # DashScope 兼容 OpenAI 接口
        llm = ChatOpenAI(
            model=model,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=SETTINGS.DASHSCOPE_API_KEY,
            temperature=temperature,
        )
    else:
        # 默认回退到 Ollama
        llm = ChatOllama(
            model=model,
            base_url=SETTINGS.LLM_BASE_URL,
            temperature=temperature,
        )

    messages = [
        SystemMessage(content="你是一名资深会议纪要助理，输出清晰的结构化中文纪要或答案。"),
        HumanMessage(content=prompt),
    ]
    logger.info("LLM(langchain) provider=%s model=%s temp=%s prompt_chars=%s", provider, model, temperature, len(prompt or ""))
    resp = llm.invoke(messages)
    logger.info("LLM(langchain) response_chars=%s", len(str(resp.content or "")))
    return str(resp.content).strip()


def _post_ollama_generate(model: str, prompt: str, temperature: float = 0.2) -> str:
    """
    调用本地 Ollama 生成文本。
    """
    logger.info("LLM(ollama) model=%s temp=%s prompt_chars=%s", model, temperature, len(prompt or ""))
    resp = requests.post(
        f"{SETTINGS.LLM_BASE_URL.rstrip('/')}/api/generate",    
        json={"model": model, "prompt": prompt, "options": {"temperature": temperature}},
        timeout=120,
    )
    resp.raise_for_status()
    text = ""
    for line in resp.text.splitlines():
        try:
            obj = json.loads(line)
            text += obj.get("response", "")
        except Exception:
            continue
    logger.info("LLM(ollama) response_chars=%s", len(text or ""))
    return text.strip()


def _post_openai_chat(model: str, prompt: str, temperature: float = 0.2) -> str:
    """
    调用 OpenAI 兼容接口生成文本（支持自定义 BASE_URL）。
    """
    headers = {}
    if SETTINGS.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {SETTINGS.LLM_API_KEY}"
    logger.info("LLM(openai) base=%s model=%s temp=%s prompt_chars=%s", SETTINGS.LLM_BASE_URL, model, temperature, len(prompt or ""))
    resp = requests.post(
        f"{SETTINGS.LLM_BASE_URL.rstrip('/')}/v1/chat/completions",
        headers=headers,
        json={
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": "你是一名资深会议纪要助理，输出清晰的结构化中文纪要或答案。"},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        logger.info("LLM(openai) response_tokens=%s", data.get("usage", {}).get("total_tokens"))
    except Exception:
        pass
    return data["choices"][0]["message"]["content"].strip()

def _post_dashscope_chat(model: str, prompt: str, temperature: float = 0.2) -> str:
    """
    调用通义千问 DashScope 文生文接口生成文本。
    """
    api_key = SETTINGS.DASHSCOPE_API_KEY or ""
    if not api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # DashScope 文生文接口
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    payload = {
        "model": model,
        "input": {
            "messages": [
                {"role": "system", "content": "你是一名资深会议纪要助理，输出清晰的结构化中文纪要或答案。"},
                {"role": "user", "content": prompt},
            ]
        },
        "parameters": {
            "temperature": temperature
        }
    }
    logger.info("LLM(dashscope) model=%s temp=%s prompt_chars=%s", model, temperature, len(prompt or ""))
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # 兼容不同返回格式
    # 标准格式：{"output":{"text":"..."}} 或 chat 格式 {"output":{"choices":[{"message":{"content":"..."}}]}}
    if "output" in data:
        out = data["output"]
        if isinstance(out, dict):
            if "text" in out:
                logger.info("LLM(dashscope) response_text_chars=%s", len(str(out["text"] or "")))
                return str(out["text"]).strip()
            if "choices" in out and out["choices"]:
                msg = out["choices"][0].get("message", {})
                logger.info("LLM(dashscope) response_msg_chars=%s", len(str(msg.get("content", "") or "")))
                return str(msg.get("content", "")).strip()
    return json.dumps(data, ensure_ascii=False)

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
    根据配置选择 LLM 提供方生成会议纪要或回答，默认使用 Ollama。
    """
    provider = (getattr(SETTINGS, "LLM_PROVIDER", "ollama") or "ollama").lower()
    model = model or SETTINGS.DEFAULT_LLM_MODEL
    if prompt_text is None:
        if not transcript or not meeting_name:
            raise ValueError("transcript 与 meeting_name 不能为空")
        prompt = _minutes_prompt(transcript, meeting_name)
    else:
        prompt = prompt_text
    if getattr(SETTINGS, "USE_LANGCHAIN", False):
        return _langchain_generate(model=model, prompt=prompt, temperature=temperature, provider=provider)

    if provider == "openai":
        # SETTINGS.LLM_BASE_URL 来自 OPENAI_BASE_URL 或配置
        return _post_openai_chat(model=model, prompt=prompt, temperature=temperature)
    if provider == "dashscope":
        return _post_dashscope_chat(model=model, prompt=prompt, temperature=temperature)
    # 默认 Ollama
    return _post_ollama_generate(model=model, prompt=prompt, temperature=temperature)
