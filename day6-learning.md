# 第 6 天：从“能调用”到“能运营” — Token 成本、上下文管理与可观测 CLI 聊天器

> **学习时间：约 1.5～2.5 小时（100～150 分钟）**（概念 45～60 min + 动手 55～90 min）  
> **目标**：延续 day5，把 API 脚本升级成“可长期使用”的小工具：有会话管理、可观测指标、基础容错与成本意识。你将能解释：为什么慢、为什么贵、如何改。

---

## 1. 今日学习目标

完成今天学习后，你应能：

- 解释输入/输出 token 与延迟、费用之间的关系（prefill 与 decode）  
- 在多轮聊天中实现最小上下文管理（保留 system + 最近 N 轮）  
- 记录并解读关键指标：`ttft_ms`、`elapsed_ms`、重试次数、输入/输出长度  
- 写一个可运行 CLI 聊天器：支持 `stream`、`/clear`、`/exit`、`/stats`（可选）  
- 说明“该缩 prompt / 换模型 / 调参数”的典型触发条件

---

## 2. 核心概念讲解

### 2.0 为什么 day6 重要？

day5 解决的是“能跑通请求”；day6 解决的是“能稳定用起来”。  
工程里最常见问题不是 API 不会调，而是：

1. **越聊越慢**：历史消息不断累积；
2. **越聊越贵**：每轮都把旧上下文重复计费；
3. **偶发失败难定位**：没有指标、没有日志，问题复现不了。

day6 的核心就是把这三件事做成“可见、可控、可解释”。

---

### 2.1 Token、延迟与成本：从“概念”到“可操作”

先把一个常见误区说清楚：  
很多人以为“费用只和输出长度有关”，其实输入长度同样关键，尤其在多轮聊天里。

一次对话请求可以拆成两个阶段：

1. **Prefill（处理输入上下文）**
  - 模型先读取并编码你给的全部 `messages`；  
  - 输入越长，prefill 越慢，且会增加输入侧成本。
2. **Decode（逐步生成输出）**
  - 模型逐 token 生成回答；  
  - 输出越长，decode 越慢，且会增加输出侧成本。

所以在工程里要同时管两件事：

- **输入预算**：历史消息不能无限增长；  
- **输出预算**：`max_tokens` 要根据任务设上限。

一个实用判断法：

- 用户说“首字出来慢” -> 先查输入是否过长（prefill 压力）；  
- 用户说“总是等很久才结束” -> 再查输出是否太长（decode 压力）。

这和你在 `llm-terminology.md` 里学的 KV Cache / Prefill-Decode 是同一原理在应用层的落地。

**对应代码（2.1）**

```python
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = 0.3
MAX_TOKENS = 256

resp = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)
```

---

### 2.2 上下文管理：为什么必须做，而不是“可选优化”

如果你做的是单轮问答，不管理上下文也能凑合；  
但只要进入多轮聊天，不做上下文策略通常会出现三连问题：

- 第 1 个问题：第 8～10 轮开始明显变慢；  
- 第 2 个问题：费用增长超预期；  
- 第 3 个问题：模型被旧信息干扰，回答开始跑偏。

#### day6 推荐的“先易后难”策略

**第一层（必做）**：按轮数裁剪

1. 固定保留 1 条 `system`；
2. `history` 仅保留最近 N 轮（例如 6 轮）；
3. 每次请求前先裁剪再发送。

优点：实现简单、收益立竿见影、最适合新手。

**第二层（选做）**：按 token 预算裁剪

- 设定 `max_input_tokens`，超过就从最旧轮开始移除；  
- 精细度更高，但需要接入 token 估算（如 `tiktoken`）。

**第三层（进阶）**：上下文总结（Summary Memory）

- 当历史过长时，不再保留全部原文，而是把“较早轮次”压缩成一段结构化摘要；  
- 后续请求只携带：`system + 摘要记忆 + 最近 N 轮原文`；  
- 常见触发条件：超过指定轮数、超过 token 预算、或 TTFT 连续升高。

建议摘要包含 4 类信息（尽量固定字段）：

1. 用户长期目标（用户到底要完成什么）；
2. 已确认事实（不可丢的关键信息）；
3. 约束与偏好（语气、格式、边界条件）；
4. 未解决事项（下一轮优先处理什么）。

这样做的优势是：在控制成本的同时，减少“裁剪后完全失忆”的问题。

#### 常见误区

1. **把 system 也裁掉**：会导致风格/规则漂移；
2. **每轮重复塞长模板**：输入膨胀极快；
3. **摘要写成随意自然语言**：下次难以稳定复用，建议结构化字段；
4. **不区分“记忆”与“上下文”**：长期记忆应交给摘要或外部存储，不应全塞对话窗口。

**对应代码（2.2）**

```python
import json

def trim_history_by_turns(history: list[dict], n_turns: int) -> list[dict]:
    max_msgs = n_turns * 2
    return history[-max_msgs:] if len(history) > max_msgs else history

def estimate_tokens_rough(messages: list[dict]) -> int:
    # 简化估算：中文按约 1 字 = 1 token（仅做预算闸门，精确版可换 tiktoken）
    return sum(len(m.get("content", "")) for m in messages)

def trim_history_by_token_budget(
    system_prompt: str,
    summary_json: str,
    history: list[dict],
    user_text: str,
    max_input_tokens: int = 2400,
) -> list[dict]:
    # 从最新消息开始保留，直到达到输入预算
    kept_reversed = []
    for msg in reversed(history):
        candidate_history = list(reversed(kept_reversed + [msg]))
        candidate_messages = build_messages(system_prompt, summary_json, candidate_history, user_text)
        if estimate_tokens_rough(candidate_messages) > max_input_tokens:
            break
        kept_reversed.append(msg)
    return list(reversed(kept_reversed))

def summarize_old_history_structured(client, model: str, old_part: list[dict]) -> str:
    schema = {
        "user_goal": "string",
        "confirmed_facts": ["string"],
        "constraints_preferences": ["string"],
        "open_items": ["string"],
    }
    text = "\n".join([f"{m['role']}: {m['content']}" for m in old_part])
    prompt = (
        "请将历史对话总结为 JSON，且必须严格符合以下 schema，不要输出额外文字：\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "要求：简洁、可复用、不编造。"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是对话记忆压缩器。"},
            {"role": "user", "content": f"{prompt}\n\n历史对话：\n{text}"},
        ],
        temperature=0.0,
        max_tokens=280,
    )
    return resp.choices[0].message.content or "{}"

def maybe_use_summary(
    client,
    model: str,
    history: list[dict],
    summary_json: str,
    trigger_turns: int = 8,
) -> tuple[list[dict], str]:
    # 超过阈值时，把旧消息压缩为结构化 summary，保留最近 4 轮原文
    if len(history) // 2 <= trigger_turns:
        return history, summary_json
    old_part = history[:-8]
    if old_part:
        summary_json = summarize_old_history_structured(client, model, old_part)
    return history[-8:], summary_json



 

# 用法示例：
# history = trim_history_by_turns(history, 6)
# history, summary_json = maybe_use_summary(client, MODEL, history, summary_json, trigger_turns=8)
# history = trim_history_by_token_budget(SYSTEM_PROMPT, summary_json, history, user_text, max_input_tokens=2400)
# messages = build_messages(SYSTEM_PROMPT, summary_json, history, user_text)
```

---

### 2.3 可观测性：指标怎么定义，怎么解读

“记录指标”如果只停留在口号，价值不大。  
day6 你至少要把以下指标定义清楚，并且知道异常时怎么判断。

1. `**ttft_ms`（Time To First Token）**
  - 定义：发请求到收到第一个非空 token 的时间；  
  - 意义：用户体感“快不快”的第一指标。
2. `**elapsed_ms`（总耗时）**
  - 定义：发请求到本轮结束的总时间；  
  - 意义：吞吐与等待成本的核心指标。
3. `**retries`（重试次数）**
  - 定义：本轮请求自动重试了几次；  
  - 意义：稳定性预警指标，升高通常说明服务波动或限流。
4. `**input_chars/output_chars`（或 token）**
  - 定义：输入输出文本长度代理值；  
  - 意义：用于解释为何费用和延迟变化。

#### 指标联动判断示例

- `ttft_ms` 高、`output_chars` 正常 -> 常见是输入过长或网络慢；  
- `elapsed_ms` 高、`ttft_ms` 正常 -> 常见是输出太长；  
- `retries` 持续升高 -> 常见是限流或服务端不稳定。

做到这一步，你才真正具备“定位性能问题”的能力。

**对应代码（2.3）**

```python
t0 = time.perf_counter()
t_first = None
# ... 流式循环 ...
if delta and t_first is None:
    t_first = time.perf_counter()
t1 = time.perf_counter()

ttft_ms = None if t_first is None else (t_first - t0) * 1000
elapsed_ms = (t1 - t0) * 1000
print(f"[stats] ttft_ms={ttft_ms} elapsed_ms={elapsed_ms:.1f} retries={retries}")
```

---

### 2.4 错误分级与重试：稳定性的底层机制

重试策略的核心不是“出错就再来一次”，而是“只对临时性错误再试”。

#### 推荐错误分级

- **可重试（临时错误）**  
  - 429（限流）  
  - 500/502/503/504（服务端短时异常）  
  - timeout（网络抖动）
- **不可重试（确定性错误）**  
  - 401/403（鉴权、权限）  
  - 400（参数不合法、请求格式错误）

#### 推荐重试策略

1. 最大 3 次（避免无限重试）；
2. 指数退避（如 1s/2s/4s）；
3. 每次重试写日志（错误类型、当前第几次）；
4. 超过上限后返回可读错误信息，而不是静默失败。

#### 新手常见反模式

- 任何错误都重试 -> 401 会白白浪费时间；  
- 不限制重试次数 -> 可能导致卡死；  
- 不记录错误类型 -> 复盘时完全无从下手。

**对应代码（2.4）**

```python
from openai import APIError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((RateLimitError, APIError, APITimeoutError)),
    reraise=True,
)
def call_with_retry(client, model, messages):
    return client.chat.completions.create(model=model, messages=messages, stream=True)
```

---

### 2.5 CLI 聊天器设计：从脚本到“可演示作品”

day6 的 CLI 不是为了炫技，而是为了把前面知识串起来形成交付物。  
推荐文件：

- 主程序：`day6_cli_chat.py`  
- 日志：`logs/day6_chat.jsonl`

#### 功能分层（建议按顺序实现）

**基础层（必做）**

1. 启动后循环输入；
2. 普通输入触发模型调用；
3. `/exit` 正常退出；
4. `/clear` 清空历史但保留 system。

**增强层（必做）**

1. `stream=True` 实时输出；
2. 每轮打印 `ttft_ms/elapsed_ms/retries/chars`；
3. 写一行 jsonl 日志用于复盘。

**展示层（选做）**

1. `/stats` 展示最近 5 轮平均指标；
2. 支持“稳定/创意”参数档位切换（如 `temperature`）。

这样你不仅“能写代码”，还能解释设计取舍：  
为什么先做按轮裁剪、为什么指标先记这四项、为什么错误要分级处理。

**对应代码（2.5）**

```python
while True:
    user_text = input("你: ").strip()
    if user_text == "/exit":
        break
    if user_text == "/clear":
        history.clear()
        summary = ""
        print("已清空历史。")
        continue
    if not user_text:
        continue

    messages = build_messages(SYSTEM_PROMPT, summary, history, user_text)
    print("助手: ", end="", flush=True)
    answer, ttft_ms, elapsed_ms = stream_with_retry(client, MODEL, messages)
    history.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}])
```

---

### 2.6 与 `learning-guide.md` 的衔接

- 对 W1：你把“调通 API”升级成“可演示的小应用”；  
- 对 W2：你已经具备上下文预算意识，做 RAG 拼接时更稳；  
- 对 W8：你提前具备评测所需的数据记录基础。

---

## 3. 总整合参考代码（`day6_cli_chat.py`）

```python
import os
from typing import Optional

# 依赖前面的函数：
# build_client, trim_history_by_turns, summarize_old_history
# stream_with_retry, append_log, build_log_row

SYSTEM_PROMPT = "你是一个简洁、可靠的中文助教。"
MAX_TURNS = 6
SUMMARY_TRIGGER_TURNS = 8


def build_messages(system_prompt: str, summary: str, history: list[dict], user_text: str) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt}]
    if summary:
        msgs.append({"role": "system", "content": f"历史摘要:\n{summary}"})
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_text})
    return msgs


def main() -> None:
    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise ValueError("请先设置 OPENAI_MODEL")

    client = build_client()
    history: list[dict] = []
    summary = ""
    stats: list[dict] = []

    print("输入 /exit 退出，/clear 清空历史，/stats 查看最近统计。")
    while True:
        user_text = input("\n你: ").strip()
        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/clear":
            history.clear()
            summary = ""
            print("已清空历史。")
            continue
        if user_text == "/stats":
            if not stats:
                print("暂无统计。")
            else:
                last = stats[-5:]
                avg_elapsed = sum(x["elapsed_ms"] for x in last) / len(last)
                print(f"最近{len(last)}轮 avg_elapsed_ms={avg_elapsed:.1f}")
            continue

        if len(history) // 2 > SUMMARY_TRIGGER_TURNS:
            old_part = history[:-8]
            summary = summarize_old_history(client, model, old_part)
            history = history[-8:]

        history = trim_history_by_turns(history, MAX_TURNS)
        messages = build_messages(SYSTEM_PROMPT, summary, history, user_text)

        print("助手: ", end="", flush=True)
        try:
            answer, ttft_ms, elapsed_ms = stream_with_retry(client, model, messages)
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": answer})

            row = build_log_row(model, user_text, answer, ttft_ms, elapsed_ms, retries=0, ok=True)
            append_log(row)
            stats.append(row)

            ttft_text = "None" if ttft_ms is None else f"{ttft_ms:.1f}"
            print(f"[stats] ttft_ms={ttft_text} elapsed_ms={elapsed_ms:.1f}")
        except Exception as e:
            print(f"\n请求失败: {type(e).__name__}")
            row = build_log_row(model, user_text, "", None, 0.0, retries=0, ok=False, err=type(e).__name__)
            append_log(row)
            stats.append(row)


if __name__ == "__main__":
    main()
```

---

## 4. 推荐阅读（可选）

1. 你当前服务商的计费与限流文档（理解 429 策略）
2. SDK 关于 `stream`、timeout、异常类型说明
3. `learning-guide.md` W2（提前感受上下文预算在 RAG 中的作用）

---

## 5. 100～150 分钟时间安排


| 时段  | 时长        | 内容                            |
| --- | --------- | ----------------------------- |
| A   | 20 min    | 通读 §2，画出“请求 -> 统计 -> 日志”流程    |
| B   | 30 min    | 完成 CLI 基础循环与 `/clear`、`/exit` |
| C   | 25 min    | 接入流式输出 + TTFT/总耗时统计           |
| D   | 25 min    | 加上历史裁剪（最近 N 轮）与错误分级重试         |
| E   | 10～20 min | 记录一次实验对比 + 完成 §6 自测           |


---

## 6. 练习题

### 必做 1：实现最小 CLI 循环

要求：

1. 读取用户输入并调用模型；
2. 输入 `/exit` 时退出；
3. 至少完成连续 3 轮问答。

参考答案代码

```python
def cli_loop():
    while True:
        text = input("你: ").strip()
        if text == "/exit":
            break
        if text == "/clear":
            print("已清空")
            continue
        if not text:
            continue
        print("这里调用模型并打印回答")
```



---

### 必做 2：流式输出与 TTFT

要求：

1. `stream=True` 增量打印；
2. 首个非空 token 到达时记录 `t_first`；
3. 结束打印 `ttft_ms` 与 `elapsed_ms`。

参考答案代码

```python
import time

def run_stream(client, model, messages):
    t0 = time.perf_counter()
    t_first = None
    full = ""
    stream = client.chat.completions.create(model=model, messages=messages, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if t_first is None:
                t_first = time.perf_counter()
            print(delta, end="", flush=True)
            full += delta
    print()
    t1 = time.perf_counter()
    ttft_ms = None if t_first is None else (t_first - t0) * 1000
    elapsed_ms = (t1 - t0) * 1000
    return full, ttft_ms, elapsed_ms
```



---

### 必做 3：历史裁剪（最近 N 轮）

要求：

1. 始终保留 system；
2. 保留最近 `N` 轮 user/assistant；
3. 连续对话 10 轮后观察输入长度变化。

参考答案代码

```python
def trim_history_by_turns(history: list[dict], n_turns: int) -> list[dict]:
    max_msgs = n_turns * 2
    return history[-max_msgs:] if len(history) > max_msgs else history

# 用法
history = trim_history_by_turns(history, 6)
messages = [{"role": "system", "content": "你是助教。"}] + history + [
    {"role": "user", "content": user_text}
]
```



---

### 必做 4：重试与错误分级

要求：

1. 429/5xx/timeout 自动重试；
2. 401/403/400 直接报错；
3. 每轮打印 `retries` 值。

参考答案代码

```python
from openai import APIError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((RateLimitError, APIError, APITimeoutError)),
    reraise=True,
)
def call_with_retry(client, model, messages):
    return client.chat.completions.create(model=model, messages=messages, stream=True)
```



---

### 选做 5：`/stats` 命令

显示最近 5 轮：

- 平均 TTFT  
- 平均总耗时  
- 平均输出长度  
- 失败率

---

### 选做 6：jsonl 结构化日志

写入 `logs/day6_chat.jsonl`，每轮一行，字段建议：

- `ts`, `model`, `ttft_ms`, `elapsed_ms`, `retries`, `input_chars`, `output_chars`, `ok`, `error_type`

---

### 选做 7：上下文总结（Summary Memory）实验

要求：

1. 设定触发条件（如历史超过 8 轮）；
2. 将较早轮次总结为结构化摘要（目标/事实/约束/待办）；
3. 后续请求改为 `system + summary + 最近 4 轮`；
4. 对比“仅按轮裁剪”与“摘要+最近轮”的质量与耗时差异。

参考答案代码

```python
def summarize_old_history(client, model, old_msgs):
    prompt = (
        "请按四段输出摘要：用户长期目标/已确认事实/约束与偏好/未解决问题。"
        " 不要编造。"
    )
    text = "\n".join([f"{m['role']}: {m['content']}" for m in old_msgs])
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是摘要器。"},
            {"role": "user", "content": f"{prompt}\n\n{text}"},
        ],
        temperature=0.0,
        max_tokens=220,
    )
    return resp.choices[0].message.content or ""

# 触发逻辑示例
if len(history) // 2 > 8:
    summary = summarize_old_history(client, model, history[:-8])
    history = history[-8:]
```



---

## 7. 自测检查点

- 我能解释为什么“多轮越多越慢越贵”。  
- 我能实现 `/clear` 且不会破坏后续会话。  
- 我能在流式调用里正确测 TTFT。  
- 我知道哪些错误应该重试，哪些不该重试。  
- 我能说清“按轮裁剪”和“上下文总结”分别适合什么场景。  
- 我能用日志证明优化是否有效（而不是只凭感觉）。

自测要点（先闭卷）

- 成本由输入和输出共同决定，输入历史是常见隐性成本  
- 上下文裁剪是多轮应用基础能力；摘要记忆是常见进阶方案  
- 指标和日志是优化前提：没有观测就没有优化



---

## 8. 今日学习总结与记录

**六天串联**


| 天         | 主题                        |
| --------- | ------------------------- |
| Day 1     | 注意力基础与形状直觉                |
| Day 2     | 多头与位置编码                   |
| Day 3     | 完整 Transformer 架构与掩码      |
| Day 4     | PyTorch 张量对齐 + HF 最小推理    |
| Day 5     | OpenAI 兼容 API、流式与重试入门     |
| **Day 6** | **成本/延迟观测、上下文管理、CLI 工程化** |


**本日一页回顾（建议写）**

1. 我的默认参数是什么？为什么？
2. 我的裁剪策略是什么？何时会升级为 token 裁剪或摘要记忆？
3. 我今天拿到的关键指标均值是多少（至少 5 轮）？
4. 我下一步最想优化的一件事是什么？

---

- 学习日期：________  
- 学习时长：约 100～150 分钟  
- 掌握程度（1–5）：___  
- 今日脚本路径：________________  
- 日志路径：________________  
- ## 疑问 / 笔记：  
  - 

---

## 9. 环境备忘（通用）

```text
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install openai python-dotenv tenacity tiktoken
```

建议目录：

```text
LLM-Learning/
  day5_api_chat.py
  day6_cli_chat.py
  logs/
    day6_chat.jsonl
```

完成 day6 后，你已经具备“可调用 + 可观察 + 可优化”的最小工程闭环，进入 W2（RAG）会轻松很多。