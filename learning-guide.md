# 十二周路线：LLM 应用开发 + OpenClaw 专题月

> **定位**：约 **12 周 / 3 个月**，前 8 周侧重 **应用开发**（RAG、Agent、API、服务、评测与安全），后 4 周 **面向 OpenClaw**（架构、Skill、Gateway、部署与综合项目）。  
> **节奏**：建议 **工作日 2～2.5 小时**（概念 45～60 min + 工程 60～90 min）+ **周末项目块**。时间紧时每周删「选做」，但 **OpenClaw 月**建议保留最低限度的可演示交付。  
> **术语表**：专用名词见同目录 **`llm-terminology.md`**（按主题分类，与本周计划交叉背诵）。

---

## 为什么这样排

- **前八周**：覆盖岗位高频考点（RAG 全链路、工具与 Agent、流式 API、微调取舍、推理与服务、评测与安全）。  
- **后四周**：把同一套能力 **收口到 OpenClaw**（配置、Gateway、Channel、Skill、定时与多通道），形成可展示的 **个人 Agent 平台** 叙事。

---

## 十二周总览

### 阶段 A：应用开发与工程（第 1～8 周）

| 周 | 主题 | 必交付（面试可讲） |
|----|------|-------------------|
| **W1** | Transformer 要点 + **LLM API 与流式** | 能画 Attention；调通 **流式 Chat**（OpenAI 兼容或国内 API） |
| **W2** | **RAG 主干**：切分、Embedding、向量库、拼接 | **可运行最小 RAG**（含引用片段 id）+ 1 个失败案例分析 |
| **W3** | **RAG 进阶**：Hybrid、Rerank、Query 改写、元数据 | **baseline vs +rerank** 对比表（指标可简） |
| **W4** | **Agent**：Function Calling、状态机、超时/重试 | **2～3 个工具** 的闭环 + 降级策略说明 |
| **W5** | **框架 + 结构化输出**：LangChain / LlamaIndex / **LangGraph 选一** | 一条可运行 **Chain 或 Graph**；**Pydantic** 约束输出示例 |
| **W6** | **微调应用向**：SFT 数据、LoRA/QLoRA、与 RAG 取舍 | **LoRA 或 QLoRA** 极小任务跑通 + 书面取舍话术 |
| **W7** | **推理与服务**：KV/上下文、**FastAPI**、**vLLM/Ollama**、Docker | **SSE** 推理服务 + **Dockerfile** 可构建 |
| **W8** | **评测、安全、成本、作品集** | **Ragas 或自研 faithfulness 检查**；注入防御清单；**架构图 + 2×STAR** |

### 阶段 B：OpenClaw 专题月（第 9～12 周）

| 周 | 主题 | 必交付（可演示） |
|----|------|------------------|
| **W9** | **OpenClaw 架构与安装**：组件关系、`openclaw.json`、Gateway | 本地/目标环境 **成功启动**；手绘 **Agent–Gateway–Channel–Skill** 关系图 |
| **W10** | **Channel 与消息流**：路由、多通道概念、调试路径 | 至少 **一种通道** 打通（按官方支持选）；文档记录消息从进入到回复的路径 |
| **W11** | **Skill 开发**：目录结构、`SKILL.md`、脚本与工具集成 | **1 个自定义 Skill**（可调用你前八周的 API/RAG/小工具之一） |
| **W12** | **综合 Agent + 运维向**：Cron/定时、记忆或工具二选一、复盘 | **日常可用助手场景** 1 个；README：配置、密钥管理原则、失败降级；**模拟答辩** 15 min 录音 |

**阶段 B 说明**：具体键名、目录以 **OpenClaw 当前官方文档与仓库为准**；本表只固定「每周能力结果」，避免与版本细节冲突。

---

## 每日时间分配（建议）

| 块 | 时长 | 内容 |
|----|------|------|
| 概念 | 45～60 min | 文档/博客 + **llm-terminology.md** 对应类 |
| 工程 | 60～90 min | 本周「必交付」相关代码与配置 |
| 复盘 | 10～15 min | 口述 3 个要点；OpenClaw 月可改为「演示一条用户路径」 |

---

## 按周要点（前八周与原版一致，术语详表见 `llm-terminology.md`）

### W1：原理速成 + API

- **必做**：Illustrated Transformer + `day1–4` 笔记（第 4 天 PyTorch/HF 形状对齐）；手写 attention 与 \((B,L,L)\) 形状。  
- **必做**：OpenAI 兼容 SDK：`chat.completions`、**stream=True**、429/5xx 重试。  
- **术语**：背 **§1**（架构）核心词 + §3 中 KV Cache / Prefill-Decode。  
- **选做**：tiktoken 粗算费用。

### W2：RAG 主干

- **必做**：加载 → chunk → embed → 向量库 → top-k → prompt → **带 source id 的回答**。  
- **术语**：背 **§5** RAG 全表。  
- **选做**：Markdown 结构切分。

### W3：RAG 进阶

- **必做**：Hybrid + Rerank；Query rewrite 或 HyDE 任选一种。  
- **术语**：MRR/nDCG/Hit@k 能举例说明用途。  
- **选做**：多租户 metadata。

### W4：Agent

- **必做**：Tool calling 解析–执行–回填；超时与异常 JSON。  
- **术语**：背 **§4** Agent / ReAct / Tool Calling。  
- **选做**：纯文本 ReAct。

### W5：框架与结构化输出

- **必做**：LangChain / LlamaIndex / LangGraph **选一** 官方 tutorial 改版；Pydantic 输出。  
- **术语**：§9 中 MCP、框架角色（能对比「不用框架怎么写」）。  
- **选做**：LangGraph checkpoint。

### W6：微调

- **必做**：PEFT LoRA；TRL SFT 或小脚本；**何时 RAG / 何时 LoRA** 写一页。  
- **术语**：§2 对齐与 PEFT。  
- **选做**：DPO 流程口述。

### W7：推理与服务

- **必做**：KV/上下文显存量级；FastAPI + SSE；Docker；vLLM 或 Ollama。  
- **术语**：§3 推理与量化名词。  
- **选做**：Triton Python Backend。

### W8：评测、安全、面试材料

- **必做**：Ragas 或自建 grounded 检查；Prompt 注入与最小权限；成本与 token 统计；**GitHub README + 架构图 + STAR×2**。  
- **术语**：§6 安全 + §7 评测。  
- **选做**：模拟面试录音。

---

## OpenClaw 四周（第 9～12 周）细化

### W9：入门

- 阅读官方：**整体架构、配置入口、Gateway 职责**。  
- 完成：**可运行实例** + 自己的 **架构草图**（与 `llm-terminology.md` 里 Agent/工具概念对照）。  
- **术语**：把 OpenClaw 专有名词（以文档为准）补进你的卡片盒「第 10 类：平台」。

### W10：通道与消息

- 理解：**消息从 Channel 进入 → Gateway/Agent 处理 → 回复路径**。  
- 完成：**一条通道** 端到端；**问题排查笔记**（日志位置、常见配置错误 1 条真实案例）。  

### W11：Skill

- 完成：**SKILL.md** 规范阅读 + **1 个 Skill**：可封装 W7 的小服务、W2 的检索脚本或 W4 的工具调用。  
- 标准：**别人按 README 能装能跑**。  

### W12：综合与答辩

- 完成：**个人助手场景**（问答 + 检索或工具至少其一）；**定时或提醒**若官方支持则做最小配置。  
- 文档：**密钥不放仓库**、配置分层、**失败时行为**（模型挂了/工具超时怎么说）。  
- **必做**：**15 min 自述录屏或录音**（架构 + 难点 + 下一步）。  

---

## 实践项目阶梯（十二周对齐）

| 优先级 | 项目 | 建议周次 |
|--------|------|----------|
| P0 | 最小 RAG + 引用 + 评测 | W2–W3 |
| P0 | Tool-calling Agent | W4 |
| P1 | FastAPI 流式 + Docker + vLLM/Ollama | W7 |
| P1 | 框架 Chain/Graph + 结构化输出 | W5 |
| P2 | LoRA/QLoRA 小任务 | W6 |
| **P2** | **OpenClaw + 自定义 Skill** | **W9–W11** |
| **P3** | **OpenClaw 综合助手** | **W12** |

---

## 模拟面试题库（简版）

**应用岗**：RAG 全流程与坑；Hybrid/rerank 取舍；Agent 终止条件与权限；TTFT/TPS；Prompt 注入；faithfulness 怎么查。  
**OpenClaw 月**：你说清 **配置从哪读、消息从哪进、Skill 怎么加载、如何加一个新工具**。

每条准备 **30 秒版 + 2 分钟版**。

---

## 推荐资源（短名单）

- Illustrated Transformer；Attention Is All You Need（§3）  
- OpenAI / 国内 API 文档（流式、tools、错误码）  
- LangChain / LlamaIndex / LangGraph 官方 Docs  
- Hugging Face PEFT、TRL；vLLM、Ollama；FastAPI  
- OWASP LLM Top 10  
- **OpenClaw 官方文档与示例仓库**（以最新版为准）  

---

## 学习检查清单

**前八周**：`llm-terminology.md` 中 §1–§7 **各能闭卷解释 70% 词条**；P0/P1 项目可演示。  
**OpenClaw 月**：能独立 **新增 Skill**、说清 **Gateway 与 Channel**、有一份 **可交付 README**。  
**表达**：1 页总架构图 + 2 个 STAR + OpenClaw 项目「用户故事」一条。  

---

## 与 `day1–4` 的关系

Transformer 日课（含 **`day4-learning.md`** 代码对齐张量）压在 **W1**；不要因手写 attention 拖延 **W2 RAG**。  

---

把本指南当作 **backlog**：每周五调整下周必做/选做，**保证始终有可演示的 commit**。祝学习顺利。
