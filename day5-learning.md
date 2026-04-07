# 第 5 天：从零开始调用 LLM API（新手版）— 能跑通、能看懂、能排错

> **学习时间：约 2～2.5 小时（120～150 分钟）**  
> **定位**：这是给新手的 day5。你不需要先懂很多工程细节，跟着步骤做就能从 0 到 1 跑通「非流式 + 流式 + 基础重试」。

---

## 0. 先说清楚：今天学完你能做到什么？

你会得到一个可运行脚本 `day5_api_chat.py`，并且你能：

- 用 OpenAI 兼容方式发送一条问题并拿到回答；
- 把回答改成「边生成边打印」（stream）；
- 遇到常见错误（如 429、超时）时不慌，知道怎么重试；
- 看懂最核心参数：`model`、`messages`、`temperature`、`max_tokens`。

如果你之前只学了 day1～day4，这个衔接是足够平滑的。

---

## 1. 前置准备（一定先做）

### 1.1 你需要准备什么

1. 一台能联网的电脑；  
2. Python 3.10+（建议 3.11）；  
3. 一个可用的 API Key；  
4. 一个兼容 OpenAI 的 `base_url` 与可用模型名（例如某服务商提供的 `xxx-chat`）。

### 1.2 新建环境（不要跳）

```text
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install openai python-dotenv tenacity
```

### 1.3 配置环境变量（推荐）

macOS / Linux:

```text
export OPENAI_API_KEY="你的key"
export OPENAI_BASE_URL="你的base_url"
export OPENAI_MODEL="你的模型名"
```

Windows（PowerShell）:

```text
$env:OPENAI_API_KEY="你的key"
$env:OPENAI_BASE_URL="你的base_url"
$env:OPENAI_MODEL="你的模型名"
```

---

## 2. 概念先讲人话（5 分钟版）

### 2.1 `messages` 是什么？

`messages` 就是一段「聊天历史」。最小一般两条：

- `system`：告诉模型你希望它怎么回答（风格/边界）  
- `user`：用户输入的问题

例子：

```python
messages = [
    {"role": "system", "content": "你是一个耐心的 Python 助教。"},
    {"role": "user", "content": "请解释什么是列表推导式，并给一个例子。"},
]
```

### 2.2 非流式和流式有啥区别？

- 非流式：一次性返回完整答案（实现简单）  
- 流式：一小段一小段返回（体验更好，像打字）

### 2.3 最常用参数（先记 4 个）

- `model`：用哪个模型  
- `messages`：你给模型的上下文  
- `temperature`：随机性，低更稳，高更发散  
- `max_tokens`：最多生成多长

---

## 3. 第一步：先跑通非流式（最小成功）

创建文件 `day5_api_chat.py`，先放下面这版最小代码：

```python
import os
from openai import OpenAI


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise ValueError("缺少环境变量 OPENAI_API_KEY")
    if not base_url:
        raise ValueError("缺少环境变量 OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def chat_once(client: OpenAI, model: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "你是一个耐心的中文助教。"},
        {"role": "user", "content": user_text},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=256,
    )
    return resp.choices[0].message.content or ""


def main() -> None:
    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise ValueError("缺少环境变量 OPENAI_MODEL")

    client = build_client()
    answer = chat_once(client, model, "用 3 句话解释什么是 Transformer。")
    print("\n[模型回答]\n")
    print(answer)


if __name__ == "__main__":
    main()
```

运行：

```text
python day5_api_chat.py
```

### 你应该看到什么？

- 终端输出一段完整回答。  
- 如果报错，先看第 7 节「常见报错排查」。

---

## 4. 第二步：改成流式输出（体验升级）

把 `chat_once` 换成流式函数（或新增一个）：

```python
def chat_stream(client: OpenAI, model: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "你是一个耐心的中文助教。"},
        {"role": "user", "content": user_text},
    ]

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=256,
        stream=True,
    )

    full_text = ""
    print("\n[流式输出]\n")
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_text += delta
    print()
    return full_text
```

然后在 `main()` 里改成：

```python
answer = chat_stream(client, model, "用 3 句话解释什么是 Transformer。")
print(f"\n[最终长度] {len(answer)} 字符")
```

---

## 5. 第三步：加上重试（新手最容易忽略）

很多报错不是你写错代码，而是网络波动或服务限流。  
先用 `tenacity` 做一个安全版本：

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, RateLimitError, APITimeoutError


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    retry=retry_if_exception_type((RateLimitError, APIError, APITimeoutError)),
    reraise=True,
)
def chat_once_with_retry(client: OpenAI, model: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "你是一个耐心的中文助教。"},
        {"role": "user", "content": user_text},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=256,
    )
    return resp.choices[0].message.content or ""
```

> 说明：不同 SDK 版本异常类名可能略有区别。若导入报错，先 `pip show openai` 看版本，再按官方文档调整异常类型。

---

## 6. 今天你真正要理解的工程点

### 6.1 什么情况要重试？

通常可重试：

- 429（限流）  
- 5xx（服务端临时异常）  
- timeout（超时）

通常不要重试：

- 401/403（密钥或权限问题）  
- 400（请求参数明显错）

### 6.2 参数怎么设才适合新手练习？

推荐默认值：

- `temperature=0.2~0.4`（更稳定）  
- `top_p=1`（先不折腾联调）  
- `max_tokens=256`（先够用）

### 6.3 day1～day4 和今天到底怎么连起来？

- day1～day3 让你知道模型在内部怎么「看」文本；  
- day4 让你会看输入输出张量形状；  
- day5 让你把这些能力接到真实 API，开始做可演示应用。

---

## 7. 新手常见报错排查（实用）

| 现象 | 常见原因 | 你该做什么 |
|------|----------|------------|
| `401 Unauthorized` | key 错 / 过期 | 检查 `OPENAI_API_KEY` |
| `404 model not found` | 模型名写错 | 检查 `OPENAI_MODEL` |
| `Connection error` | 网络或 `base_url` 错 | 检查网络和 `OPENAI_BASE_URL` |
| `429` | 限流 | 加重试、降低并发、稍后再试 |
| 一直空输出 | 读错流式字段 | 检查 `delta.content` 是否判空 |

排查顺序建议：

1. 先看环境变量是否存在；  
2. 再看模型名和 `base_url`；  
3. 最后看代码逻辑（尤其流式字段读取）。

---

## 8. 120～150 分钟学习安排（新手节奏）

| 时段 | 时长 | 任务 |
|------|------|------|
| A | 20 min | 做完环境准备与变量配置 |
| B | 30 min | 跑通非流式最小脚本 |
| C | 30 min | 改成流式输出并验证 |
| D | 25 min | 加入重试逻辑并做一次失败演练 |
| E | 15 min | 按 §9 自测 + 记录 |

---

## 9. 练习题（对新手更友好版）

### 必做 1：复制最小脚本并跑通

标准：你能看到一段完整回答。

### 必做 2：切换为流式

标准：你能看到回答一段段出现，而不是一次性打印。

### 必做 3：故意写错模型名再修复

标准：你能看懂错误并改回正确模型。

### 必做 4：重试演练

标准：出现临时错误时，程序会自动再试，不会立刻崩溃。

### 选做 5：做一个最小问答循环

让用户输入问题，回答后继续下一轮，输入 `exit` 退出。

---

## 10. 自测检查点

- [ ] 我知道 `messages` 至少要包含什么。  
- [ ] 我能说出非流式和流式的区别。  
- [ ] 我能解释为什么 429 应该重试。  
- [ ] 我遇到 401 时知道先查 key，而不是瞎改代码。  
- [ ] 我能在 5 分钟内重新从空文件写出最小调用骨架。

---

## 11. 与 `learning-guide.md`（W1）对齐

W1 要求里最关键的 3 条，你今天都覆盖了：

1. `chat.completions` 调通；  
2. `stream=True` 能跑；  
3. 429/5xx 有重试思路。

所以这份 day5 不是“了解”，而是可交付的第一步。

---

## 12. 今日记录模板

- 学习日期：________  
- 学习时长：约 120～150 分钟  
- 掌握程度（1–5）：___  
- 今日脚本路径：________________  
- 我踩过的 1 个坑：________________  
- 我解决它的方法：________________

---

把今天脚本保存到仓库，明天 day6 再补「成本/Token/上下文管理」，你就会从“能调接口”走向“能做小应用”。
