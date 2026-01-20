# AI 会议纪要服务（离线链路）

本项目提供从语音识别（Whisper / Paraformer）到本地大模型（Ollama）以及向量库（FAISS）的全链路离线服务接口。

## 功能
- 实时会议：上传录音与会议名称，转写为文本并生成结构化会议纪要，自动写入向量库
- RAG 检索：基于已入库的会议纪要进行语义检索与问答，支持温度参数
- 上传与历史：支持纯上传会议音频、查看历史会议列表与详情

## 目录结构（优化后）
```
app/
  main.py        # FastAPI 入口与路由
  storage.py     # SQLite 存储
  models.py      # 数据模型与返回体
ASR/
  transcribe.py  # 语音识别（faster-whisper / paraformer）
llm/
  generate.py    # 本地 LLM（Ollama）调用
RAG/
  index.py       # FAISS 索引 + Ollama Embeddings
conf/
  settings.py    # 全局配置（路径与模型）
common/
  utils.py       # 文本分块等工具
data/
  audio/         # 上传音频
  meetings.db    # SQLite 数据库
  faiss/         # 索引文件
  faiss_meta.json
requirements.txt
```

## 环境准备
1. 安装 Python 3.11+
2. 安装依赖：
   ```bash
   # conda 安装
   conda create --name ai-meeting python=3.11

   # 从环境文件创建环境
   conda env create -f environment.yml
   # 激活环境
   conda activate ai-meeting
   # 停用环境
   conda deactivate
  # 启动服务
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```
3. 安装并启动 Ollama：
   - https://ollama.com
   - 拉取模型（示例）：
     ```bash
     ollama pull qwen2.5:7b
     ollama pull nomic-embed-text
     ```
4. 语音识别（任选其一）：
   - faster-whisper（需 ffmpeg）：`brew install ffmpeg`
   - FunASR Paraformer（中文更强）：首次调用会下载模型

## 启动
```bash
 APP_ENV=dev python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 在线 Whisper 配置（faster-whisper）
- 在 .env.example 设置：
  - ASR_PROVIDER=faster-whisper
  - WHISPER_MODEL=medium
  - HF_HUB_OFFLINE=0
  - 可选代理/镜像：HTTP_PROXY、HTTPS_PROXY、HF_ENDPOINT=https://hf-mirror.com、HF_TOKEN
- 启动时使用 APP_ENV=dev 以加载 .env.example
- 首次会在线下载模型并缓存，后续离线也可用

## 离线 Whisper 配置（本地模型路径）
- 在 .env.example 设置：
  - ASR_PROVIDER=faster-whisper
  - WHISPER_MODEL=/绝对路径/到/faster-whisper-模型目录
  - HF_HUB_OFFLINE=1
- 不需要网络，直接加载本地模型目录
## 关于 LangChain 与通义千问
- 本项目未使用 LangChain，目标是轻量与可控的离线链路；如需工作流编排、工具调用、复杂检索链，可按需引入。
- LLM Provider 支持：
  - Ollama：本地模型推理
  - OpenAI 兼容：将 OPENAI_BASE_URL 指向兼容端点即可（可用于通义千问的兼容接口）
  - DashScope：原生通义千问 API（配置 DASHSCOPE_API_KEY，并选择 LLM_PROVIDER=dashscope；RAG 可选 EMBED_PROVIDER=dashscope）

## 配置
- 文件：conf/config.yaml（支持 .env 覆盖）
- 环境变量：
  - APP_CONFIG 指定配置文件路径
  - DATA_DIR、OLLAMA_API、DEFAULT_LLM_MODEL、EMBED_MODEL 可覆盖
  - APP_ENV=dev 时优先加载 .env.example，否则加载 .env
  - LLM_PROVIDER: ollama | openai | dashscope
  - LLM_BASE_URL / LLM_API_KEY（当 provider=openai）
  - DASHSCOPE_API_KEY（当 provider=dashscope）
  - EMBED_PROVIDER: ollama | openai | dashscope


## API
- POST /api/meetings/realtime
  - multipart-form：audio(file), name(string), asr_backend(whisper|paraformer, 默认 whisper), asr_model, llm_model
  - 返回：会议ID、时间与文本预览

- POST /api/meetings/upload
  - multipart-form：audio(file), name(string)
  - 返回：会议ID与基本信息（不转写、不生成纪要）

- GET /api/meetings/history?page=1&page_size=10
  - 返回：历史会议列表（包含纪要预览）

- GET /api/meetings/{id}
  - 返回：会议详情（完整转写与纪要）

- GET /api/rag/search?q=...&k=5&temperature=0.2&llm_model=qwen2.5:7b
  - 返回：答案与引用来源（meeting_id, chunk_id）

## 注意事项
- 本项目不依赖任何云服务，所有推理与存储均在本地进行
- 初次转写或生成可能需要下载模型（请确保网络或预先准备模型）
- 建议在 macOS 上安装 ffmpeg 以支持更多音频格式
