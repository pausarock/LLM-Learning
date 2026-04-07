# 第 7 天：把 CLI 升级成服务 — FastAPI + SSE 流式接口（可部署基础版）

> **学习时间：约 2～3 小时（120～180 分钟）**（概念 50～70 min + 动手 70～110 min）  
> **目标**：把 day6 的 CLI 聊天器升级成 HTTP 服务，支持 `/health`、`/chat`、`/chat/stream`，并建立最小服务化意识（超时、错误码、日志、接口契约）。

---

## 1. 今日学习目标

完成今天学习后，你应能：

- [ ] 解释为什么要把脚本改为 API 服务（解耦前后端、便于部署和复用）  
- [ ] 用 FastAPI 写出非流式与流式（SSE）两个端点  
- [ ] 使用 Pydantic 定义请求/响应结构并做基本校验  
- [ ] 在服务端保留 day6 的核心能力：重试、统计、日志  
- [ ] 用 `curl` 或 Python 客户端验证服务可用

---

## 2. 核心概念讲解

### 2.0 为什么 day7 做服务化？

day5/day6 你已经能在本地脚本里完成调用、流式、重试、上下文管理。  
但脚本有三个局限：

1. **调用入口单一**：只能在终端里用，不便于网页或其他程序复用；  
2. **部署困难**：脚本不是标准服务形态，不容易放到云端；  
3. **接口不稳定**：输入输出格式没有“契约”，协作成本高。

所以 day7 的核心转变是：  
从“人用脚本”升级为“程序调程序”的服务接口。

**对应代码（2.0）**

```python
from fastapi import FastAPI

app = FastAPI(title="Day7 LLM Service", version="0.1.0")

@app.get("/health")
def health():
    return {"ok": True}
```

---

### 2.1 接口契约：为什么先写 Schema？

服务化后，最容易出问题的不是模型本身，而是“前后端对字段理解不一致”。  
因此先定义请求/响应模型非常重要。

建议最小请求字段：

- `user_text`: 用户输入  
- `session_id`: 会话标识（可选）  
- `stream`: 是否流式（某些设计可放到路径里而不是 body）

建议最小响应字段：

- `answer`  
- `elapsed_ms`  
- `ttft_ms`（非流式可为 `null`）

**对应代码（2.1）**

```python
from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    user_text: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    elapsed_ms: float
    ttft_ms: Optional[float] = None
```

---

### 2.2 非流式与流式：HTTP 层怎么实现？

#### 非流式
- 一次请求一次返回，适合后端调用和批处理；  
- 结构简单，便于调试。

#### 流式（SSE）
- 服务端分块返回，前端能“边收边显示”；  
- 用户体感显著更好；  
- 需要正确设置 `media_type="text/event-stream"`。

**对应代码（2.2）**

```python
from fastapi.responses import StreamingResponse
import json

def to_sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

@app.get("/chat/stream")
def chat_stream_demo():
    def event_gen():
        yield to_sse({"delta": "你好，"})
        yield to_sse({"delta": "这是流式"})
        yield to_sse({"delta": "返回示例。"})
        yield "event: done\ndata: [DONE]\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")
```

---

### 2.3 服务中的会话管理：先做内存版

day6 的 `history` 在本地变量里就够用；  
服务化后至少要考虑 `session_id -> history` 的映射。

入门阶段可用内存字典（重启会丢，适合学习）：

- `SESSIONS[session_id] = {"history": [...], "summary_json": "..."}`

以后可迁移到 Redis/数据库。

**对应代码（2.3）**

```python
SESSIONS: dict[str, dict] = {}

def get_session_state(session_id: str) -> dict:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"history": [], "summary_json": ""}
    return SESSIONS[session_id]
```

---

### 2.4 错误处理与状态码：把失败讲人话

服务层建议：

- 参数不合法 -> 422（FastAPI 自动）  
- 配置缺失/服务不可用 -> 500  
- 供应商限流/超时 -> 502 或 503（对上游错误做封装）

重点：不要把原始堆栈直接暴露给调用方。

**对应代码（2.4）**

```python
from fastapi import HTTPException

def ensure_model_ready(model: str):
    if not model:
        raise HTTPException(status_code=500, detail="服务端缺少 OPENAI_MODEL 配置")
```

---

### 2.5 日志与观测：服务版最小指标

延续 day6 的指标，建议每次请求都记录：

- `path`、`session_id`、`elapsed_ms`、`ttft_ms`、`ok`、`error_type`

这会直接帮助你做 W8 的评测与稳定性复盘。

**对应代码（2.5）**

```python
import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("logs/day7_service.jsonl")

def write_service_log(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row["ts"] = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

---

### 2.6 与 `learning-guide.md` 的衔接

- 对 W1：你把“会调用 API”推进到了“会封装 API”；  
- 对 W7：这就是 FastAPI + SSE 服务化的最小雏形；  
- 对后续 Agent/RAG：统一服务层后，工具调用和检索拼接更容易接入。

---

## 3. 总整合参考代码（`day7_service.py`）

```python
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

app = FastAPI(title="Day7 LLM Service", version="0.1.0")
LOG_PATH = Path("logs/day7_service.jsonl")
SYSTEM_PROMPT = "你是一个简洁、可靠的中文助教。"
MAX_TURNS = 6
SESSIONS: dict[str, dict] = {}


class ChatRequest(BaseModel):
    user_text: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    answer: str
    elapsed_ms: float
    ttft_ms: Optional[float] = None


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise HTTPException(status_code=500, detail="缺少 OPENAI_API_KEY 或 OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model() -> str:
    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise HTTPException(status_code=500, detail="缺少 OPENAI_MODEL")
    return model


def get_session_state(session_id: str) -> dict:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"history": []}
    return SESSIONS[session_id]


def trim_history_by_turns(history: list[dict], max_turns: int) -> list[dict]:
    max_msgs = max_turns * 2
    return history[-max_msgs:] if len(history) > max_msgs else history


def build_messages(history: list[dict], user_text: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_text})
    return msgs


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((RateLimitError, APIError, APITimeoutError)),
    reraise=True,
)
def chat_once(client: OpenAI, model: str, messages: list[dict]) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=256,
    )
    return resp.choices[0].message.content or ""


def write_service_log(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row["ts"] = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    t0 = time.perf_counter()
    client = build_client()
    model = get_model()
    state = get_session_state(req.session_id or "default")

    history = trim_history_by_turns(state["history"], MAX_TURNS)
    messages = build_messages(history, req.user_text)
    try:
        answer = chat_once(client, model, messages)
    except Exception as e:
        write_service_log({"path": "/chat", "ok": False, "error_type": type(e).__name__})
        raise HTTPException(status_code=502, detail=f"上游模型调用失败: {type(e).__name__}") from e

    history.extend([{"role": "user", "content": req.user_text}, {"role": "assistant", "content": answer}])
    state["history"] = history

    elapsed_ms = (time.perf_counter() - t0) * 1000
    write_service_log(
        {
            "path": "/chat",
            "session_id": req.session_id,
            "elapsed_ms": round(elapsed_ms, 2),
            "ttft_ms": None,
            "ok": True,
        }
    )
    return ChatResponse(answer=answer, elapsed_ms=elapsed_ms, ttft_ms=None)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    client = build_client()
    model = get_model()
    state = get_session_state(req.session_id or "default")
    history = trim_history_by_turns(state["history"], MAX_TURNS)
    messages = build_messages(history, req.user_text)

    def event_gen() -> Generator[str, None, None]:
        t0 = time.perf_counter()
        t_first = None
        answer = ""
        ok = True
        error_type = None
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=256,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    if t_first is None:
                        t_first = time.perf_counter()
                    answer += delta
                    yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
            history.extend([{"role": "user", "content": req.user_text}, {"role": "assistant", "content": answer}])
            state["history"] = history
            yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            ok = False
            error_type = type(e).__name__
            yield f"event: error\ndata: {json.dumps({'error': error_type}, ensure_ascii=False)}\n\n"
        finally:
            t1 = time.perf_counter()
            ttft_ms = None if t_first is None else (t_first - t0) * 1000
            elapsed_ms = (t1 - t0) * 1000
            write_service_log(
                {
                    "path": "/chat/stream",
                    "session_id": req.session_id,
                    "ttft_ms": None if ttft_ms is None else round(ttft_ms, 2),
                    "elapsed_ms": round(elapsed_ms, 2),
                    "ok": ok,
                    "error_type": error_type,
                }
            )

    return StreamingResponse(event_gen(), media_type="text/event-stream")
```

---

## 4. 推荐阅读（可选）

1. FastAPI 文档（Request Body、StreamingResponse）  
2. SSE 基础格式（`data: ...\n\n`）  
3. `learning-guide.md` 中 W7（FastAPI + SSE + Docker）

---

## 5. 120～180 分钟时间安排

| 时段 | 时长 | 内容 |
|------|------|------|
| A | 25 min | 通读 §2，先跑通 `/health` |
| B | 35 min | 实现 `/chat`（非流式）+ Pydantic |
| C | 40 min | 实现 `/chat/stream`（SSE）+ 计时 |
| D | 30 min | 加日志、错误处理、会话内存 |
| E | 10～20 min | 用 `curl` 做联调 + §7 自测 |

---

## 6. 练习题

### 必做 1：启动最小服务

要求：

1. 新建 `day7_service.py`；  
2. 至少提供 `/health`；  
3. 使用 `uvicorn day7_service:app --reload` 启动成功。

<details>
<summary>参考答案代码</summary>

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}
```

</details>

---

### 必做 2：实现 `/chat`（非流式）

要求：

1. 请求体使用 Pydantic；  
2. 返回 `answer` + `elapsed_ms`；  
3. 失败返回可读错误。

<details>
<summary>参考答案代码</summary>

```python
from pydantic import BaseModel
import time

class ChatRequest(BaseModel):
    user_text: str

@app.post("/chat")
def chat(req: ChatRequest):
    t0 = time.perf_counter()
    # answer = 调用模型
    answer = "示例回答"
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {"answer": answer, "elapsed_ms": elapsed_ms}
```

</details>

---

### 必做 3：实现 `/chat/stream`（SSE）

要求：

1. 返回 `StreamingResponse`；  
2. 至少返回 3 个 `delta` 事件 + 1 个 `done` 事件。

<details>
<summary>参考答案代码</summary>

```python
from fastapi.responses import StreamingResponse
import json

@app.get("/chat/stream")
def chat_stream():
    def event_gen():
        for d in ["你", "好", "！"]:
            yield f"data: {json.dumps({'delta': d}, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: [DONE]\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")
```

</details>

---

### 必做 4：记录服务日志

要求：

1. 每次 `/chat` 或 `/chat/stream` 结束都写一行 jsonl；  
2. 至少包含 `path`, `elapsed_ms`, `ok`。

<details>
<summary>参考答案代码</summary>

```python
import json
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("logs/day7_service.jsonl")

def write_log(path: str, elapsed_ms: float, ok: bool):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "path": path,
        "elapsed_ms": round(elapsed_ms, 2),
        "ok": ok,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

</details>

---

### 选做 5：`session_id` 多会话隔离

把会话状态从单个 `history` 升级到 `SESSIONS[session_id]`，验证两个 `session_id` 不互相污染。

---

### 选做 6：把 day6 的摘要记忆接入服务

当某个会话历史超过阈值时，触发结构化 summary，再保留最近 4 轮原文。

---

## 7. 自测检查点

- [ ] 我能解释为什么 CLI 到服务化是必要步骤。  
- [ ] 我能区分 `/chat` 与 `/chat/stream` 在体验和实现上的差异。  
- [ ] 我能写出至少一个 Pydantic 请求模型。  
- [ ] 我能让服务在失败时返回可读错误而不是直接崩溃。  
- [ ] 我能通过日志回答“这次请求为什么慢”。

<details>
<summary>自测要点</summary>

- 服务化核心：接口契约、复用、部署  
- SSE 核心：分块推送、更好交互体验  
- 观测核心：`ttft_ms` / `elapsed_ms` / `ok` / `error_type`

</details>

---

## 8. 今日学习总结与记录

**七天串联**

| 天 | 主题 |
|----|------|
| Day 1 | 注意力基础 |
| Day 2 | 多头与位置编码 |
| Day 3 | Transformer 架构 |
| Day 4 | PyTorch 张量与 HF 推理 |
| Day 5 | API 调用、流式、重试入门 |
| Day 6 | 成本/上下文/可观测 CLI |
| **Day 7** | **FastAPI + SSE 服务化** |

**本日回顾建议**

1. 我的 API 契约是否清晰（请求/响应字段固定）？  
2. 我的流式接口是否可稳定结束（有 done 事件）？  
3. 我的日志是否足够支持排障？

---

- 学习日期：________  
- 学习时长：约 120～180 分钟  
- 掌握程度（1–5）：___  
- 今日脚本路径：________________  
- 日志路径：________________  
- 疑问 / 笔记：  
  -  
  -  

---

## 9. 环境备忘（通用）

```text
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install fastapi uvicorn openai tenacity pydantic
```

启动服务：

```text
uvicorn day7_service:app --reload --port 8000
```

测试：

```text
curl http://127.0.0.1:8000/health
```

完成 day7 后，你已经有“模型调用 -> 工程控制 -> 服务接口”完整链路。接下来进 RAG 或 Agent，都可以直接复用这层服务能力。
