# 第 8 天：微调起步（LoRA/QLoRA）— 从数据格式到最小 SFT 跑通

> **学习时间：约 2～3 小时（120～180 分钟）**（概念 55～75 min + 动手 65～105 min）  
> **目标**：按你当前路线（W2 先微调）完成第一版微调闭环：准备最小数据、加载基座模型、挂 LoRA、跑 1 次 SFT，并能做基础推理对比。

---

## 1. 今日学习目标

完成今天学习后，你应能：

- [ ] 说清 LoRA/QLoRA 与全参微调的差异  
- [ ] 准备最小 SFT 数据（instruction/input/output）  
- [ ] 跑通一次 PEFT LoRA 训练（哪怕很小步数）  
- [ ] 保存并加载 adapter 做推理  
- [ ] 写出“何时微调、何时 RAG”的第一版判断

---

## 2. 核心概念讲解

### 2.0 为什么先学微调？

你已经会 API、流式、可观测、服务化。现在先做微调有两个好处：

1. 能快速建立“模型行为可被数据定向改变”的直觉；  
2. 后面学 RAG 时，你会更清楚“该检索还是该训练”。

**对应代码（2.0）**

```python
decision = {
    "want_new_knowledge_fast_update": "RAG 优先",
    "want_stable_style_or_task_behavior": "微调优先",
}
print(decision)
```

---

### 2.1 LoRA / QLoRA：你到底在训练什么

- **全参微调**：改所有参数，资源重  
- **LoRA**：冻结基座，只训练低秩增量矩阵，资源轻  
- **QLoRA**：在低比特量化基座上做 LoRA，进一步省显存

新手阶段要点：先跑通 **LoRA**，QLoRA 后续加。

**对应代码（2.1）**

```python
from peft import LoraConfig

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # 具体模块名依模型结构
    task_type="CAUSAL_LM",
)
```

---

### 2.2 SFT 数据格式：比模型更容易出错的环节

最小推荐格式（jsonl 每行一条）：

```json
{"instruction":"...", "input":"...", "output":"..."}
```

常见坑：

- instruction 太模糊  
- output 质量不一致  
- 样本太少且重复

**对应代码（2.2）**

```python
import json

def to_prompt(x: dict) -> str:
    inp = x.get("input", "").strip()
    if inp:
        return f"指令：{x['instruction']}\n输入：{inp}\n回答：{x['output']}"
    return f"指令：{x['instruction']}\n回答：{x['output']}"

with open("data/sft_small.jsonl", "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]
texts = [to_prompt(s) for s in samples]
```

---

### 2.3 训练最小闭环：先“能跑”，再“变好”

day8 目标不是练高分，而是跑通这条链：

1. tokenizer + base model  
2. 挂 LoRA adapter  
3. tokenization  
4. Trainer/TRL 跑若干 step  
5. 保存 adapter

**对应代码（2.3）**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # 示例，可按环境替换

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
```

---

### 2.4 评估与对比：先做“前后对比”就够了

新手最实用评估：

- 同一组 prompt，比较基座 vs adapter 输出  
- 看三件事：格式稳定性、任务贴合度、错误率

**对应代码（2.4）**

```python
def compare_outputs(base_answer: str, lora_answer: str) -> dict:
    return {
        "base_len": len(base_answer),
        "lora_len": len(lora_answer),
        "changed": base_answer != lora_answer,
    }
```

---

### 2.5 与 `learning-guide.md` 的衔接

你现在路线是：

- W2：微调先行（day8 起步）  
- W3：RAG 主干

所以 day8 的交付重点是“可复现的最小 LoRA 实验”，不是追 benchmark。

---

## 3. 总整合参考代码（`day8_lora_sft_minimal.py`）

```python
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/sft_small.jsonl"
OUT_DIR = "outputs/day8-lora"


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def to_text(x: dict) -> str:
    ins = x["instruction"].strip()
    inp = x.get("input", "").strip()
    out = x["output"].strip()
    if inp:
        return f"指令：{ins}\n输入：{inp}\n回答：{out}"
    return f"指令：{ins}\n回答：{out}"


def main():
    rows = load_jsonl(DATA_PATH)
    texts = [to_text(x) for x in rows]
    ds = Dataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    def tok(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=512)
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = ds.map(tok, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=50,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"adapter saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
```

---

## 4. 推荐阅读（可选）

1. PEFT 官方 LoRA 入门  
2. Hugging Face Transformers Trainer 基础  
3. 你模型提供方对 `target_modules` 的建议配置

---

## 5. 120～180 分钟时间安排

| 时段 | 时长 | 内容 |
|------|------|------|
| A | 25 min | 通读 §2，准备 `sft_small.jsonl` |
| B | 35 min | 跑通模型加载 + LoRA 挂载 |
| C | 45 min | 跑最小训练并保存 adapter |
| D | 30 min | 做基座 vs LoRA 的 3 组对比 |
| E | 10～20 min | 记录失败点 + §7 自测 |

---

## 6. 练习题

### 必做 1：准备 20～50 条最小 SFT 数据

要求：

1. 使用 `instruction/input/output` 三字段；  
2. 至少覆盖 2 类任务（如摘要 + 改写）；  
3. 存成 `data/sft_small.jsonl`。

<details>
<summary>参考答案代码</summary>

```python
import json

rows = [
    {"instruction": "把句子改成更礼貌语气", "input": "你快点发我文件", "output": "请问方便尽快把文件发给我吗？"},
    {"instruction": "总结一句话", "input": "RAG 先检索再生成，可降低幻觉并提升可追溯性。", "output": "RAG 通过检索增强生成，提高可靠性与可追溯性。"},
]
with open("data/sft_small.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

</details>

---

### 必做 2：打印可训练参数比例

要求：挂载 LoRA 后打印 `trainable params`，确认确实是参数高效训练。

<details>
<summary>参考答案代码</summary>

```python
from peft import get_peft_model, LoraConfig

lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
```

</details>

---

### 必做 3：跑 1 次最小训练

要求：

1. `num_train_epochs=1`；  
2. 能完成 `trainer.train()`；  
3. 能保存 adapter。

<details>
<summary>参考答案代码</summary>

```python
trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator)
trainer.train()
model.save_pretrained("outputs/day8-lora")
```

</details>

---

### 必做 4：做前后输出对比

要求：同一 prompt 比较基座和 LoRA 输出，写 3 行结论。

<details>
<summary>参考答案代码</summary>

```python
def quick_eval(base_text, lora_text):
    print("base:", base_text)
    print("lora:", lora_text)
    print("changed:", base_text != lora_text)
```

</details>

---

### 选做 5：尝试 QLoRA

在硬件允许下引入 4bit/8bit 量化，观察显存占用差异。

---

### 选做 6：一页取舍文档

写一页《何时微调，何时 RAG》：从数据更新频率、成本、上线速度、可解释性比较。

---

## 7. 自测检查点

- [ ] 我能解释 LoRA 为什么比全参微调省资源。  
- [ ] 我能独立准备最小 SFT 数据格式。  
- [ ] 我能跑通一次 LoRA 训练并保存 adapter。  
- [ ] 我能做基座 vs LoRA 的基础对比。  
- [ ] 我能说出至少 2 条“微调不适合”的场景。

<details>
<summary>自测要点</summary>

- 微调强在“行为与风格固化”，弱在“知识快速更新”  
- RAG 强在“知识更新快与可追溯”，弱在“复杂行为长期稳定”  
- 真正工程里常常是“RAG + 轻量微调”组合

</details>

---

## 8. 今日学习总结与记录

**八天串联（新版）**

| 天 | 主题 |
|----|------|
| Day 1 | 注意力基础 |
| Day 2 | 多头与位置编码 |
| Day 3 | Transformer 架构 |
| Day 4 | PyTorch 张量与 HF 推理 |
| Day 5 | API 调用与流式 |
| Day 6 | 上下文管理与可观测 |
| Day 7 | FastAPI 服务化 |
| **Day 8** | **LoRA/QLoRA 微调起步** |

**本日回顾建议**

1. 我的数据质量问题主要在哪（格式、覆盖、一致性）？  
2. LoRA 后输出变化是否符合预期？  
3. 下一步我更该补“数据质量”还是“训练策略”？

---

- 学习日期：________  
- 学习时长：约 120～180 分钟  
- 掌握程度（1–5）：___  
- 今日脚本路径：________________  
- 训练输出目录：________________  
- 疑问 / 笔记：  
  -  
  -  

---

## 9. 环境备忘（通用）

```text
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch transformers datasets peft accelerate
```

建议目录：

```text
LLM-Learning/
  data/
    sft_small.jsonl
  day8_lora_sft_minimal.py
  outputs/
    day8-lora/
```

day8 跑通后，你的 W2 微调主线就站住了。接下来可以继续做 day9（微调评估与稳定性）或按计划切到 W3 开始 RAG 主干。
