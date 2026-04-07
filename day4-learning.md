# 第 4 天：用代码对齐张量 — PyTorch 单层 Encoder 与 Hugging Face 推理入门

> **学习时间：约 1.5～2 小时（90～120 分钟）**（概念 45 min + 动手 45～75 min）  
> **目标**：能在 PyTorch 里说出 **张量形状在每一子层如何变化**；能手写或读懂 **一层 Encoder 风格模块** 的骨架；能用 **Transformers** 完成「分词 → 前向 → 隐藏状态形状」的最小流程，为 **W1 API 与下周 RAG** 打底。

---

## 1. 今日学习目标

完成今天学习后，你应能：

- [ ] 默写 batch 维下常见形状：**输入** \((B,L,d)\)、**注意力分数** \((B,L,L)\) 或多头 \((B,h,L,L)\)  
- [ ] 说明 **`nn.Linear(d_in, d_out)`** 在最后一维上做变换，因此 **\(X\) 为 \((B,L,d_{\mathrm{model}})\)** 时与 **\(W^Q\)** 的对应关系  
- [ ] 写出 **Pre-LN Encoder 一层** 的逻辑顺序（与 day3 一致，但落到「先 Norm 再子层」的代码顺序）  
- [ ] 用 **Hugging Face** 加载一个小模型，打印 **`input_ids`**、**`last_hidden_state`** 的 **`.shape`**  
- [ ] 解释 **`attention_mask`** 在填充序列（padding）时为什么必要  

---

## 2. 核心概念讲解

### 2.0 为什么在第 4 天写代码？

前三天你在**数学与架构**上建立了骨架；第 4 天用**同一张形状表**在 PyTorch 里走一遍，会固定住「哪里该 transpose、哪里是 batch 维」的肌肉记忆。不必追求一次写满生产级实现，**形状对、能 forward、能对照 day1–3**，即达标。

---

### 2.1 PyTorch 里和 Transformer 最相关的几块

| 组件 | 作用（与 day1–3 对应） |
|------|------------------------|
| **`torch.Tensor`** | 多维数组；注意 **`.shape`**，默认 **float32** 训练。 |
| **`nn.Embedding(vocab, d_model)`** | 词 id → 向量；输出 \((B,L,d_{\mathrm{model}})\)。 |
| **`nn.Linear(d_in, d_out)`** | 对**最后一维**做 \(xW^\top+b\)；故 \(Q=XW^Q\) 常写成 `self.W_q(x)`。 |
| **`nn.LayerNorm(d_model)`** | 在最后一维 \(d_{\mathrm{model}}\) 上归一化（每个 token 一行）。 |
| **`F.softmax(..., dim=-1)`** | 注意力在 **key 维**归一：对最后一维是 key 时，常 `dim=-1`（实现细节依张量布局而定）。 |

**约定**：下文用 \(B\) = batch，\(L\) = 序列长度，\(d\) = \(d_{\mathrm{model}}\)，\(h\) = 头数，\(d_k=d/h\)。

---

### 2.2 形状追踪（本日核心技能）

- 嵌入后：**\(X \in \mathbb{R}^{B\times L\times d}\)**。  
- 单头注意力（为清晰先不写多头拆分）：  
  - `Q, K, V = linear_q(x), linear_k(x), linear_v(x)` → 均为 \((B,L,d_k)\) 或保持 \((B,L,d)\) 视实现而定。  
  - **scores** \(= Q K^\top\)：\((B,L,L)\)。  
  - **attn @ V**：\((B,L,d_k)\)。  
- **多头**：常见实现为 \((B,L,d)\) → 线性到 \((B,L,d)\) → **reshape** 为 \((B,L,h,d_k)\) → **transpose** 为 \((B,h,L,d_k)\)，使注意力在 **\(h\)** 个头上并行，scores 为 \((B,h,L,L)\)。

**自检一句话**：出现维度对不上时，先问：**batch 维有没有丢？多头维 \(h\) 在哪一维？\(L\times L\) 是「谁看谁」？**

---

### 2.3 手写「一层 Encoder 风格」模块（逻辑骨架）

与 day3 **Encoder 一层**对应，**Pre-LN** 常见写法如下（伪代码，重在顺序）：

```text
x  # (B, L, d)
x = x + sublayer_mha( layer_norm(x) )   # 残差 + 多头自注意力
x = x + sublayer_ffn( layer_norm(x) )   # 残差 + FFN
return x  # 仍为 (B, L, d)
```

- **`sublayer_mha`** 内部：`Linear` 得到 QKV → 注意力 → 输出投影 \(W^O\) → 输出 \((B,L,d)\)。  
- **`sublayer_ffn`**：`d → 4d → d`（或你设定 `d_ff`），中间 GELU/ReLU。

**本日最低交付**：你能在一个 `.py` 里定义上述结构，**随机初始化** `x`，`print(x.shape)` 进出一致即可；不必训练。

---

### 2.4 Hugging Face Transformers：最小推理心智模型

1. **`AutoTokenizer.from_pretrained("某模型")`**  
   - `tokenizer(text, return_tensors="pt", padding=True)` → **`input_ids`** \((B,L)\)，**`attention_mask`** \((B,L)\)（1 表示真实 token，0 表示 padding）。  

2. **`AutoModel.from_pretrained(...)`**  
   - `outputs = model(**inputs)`  
   - **`outputs.last_hidden_state`**：\((B,L,d_{\mathrm{model}})\)，即最后一层每个位置的向量。  

3. **为什么要 `attention_mask`**：批量里句子长短不一要 pad；模型内部用 mask **屏蔽 pad 位置**对注意力的影响（否则 pad 会参与「看谁」）。  

4. **`generate`**：自回归解码，涉及 **KV Cache**（`llm-terminology.md` §3）；第 4 天只需知道：**生成比单次前向多一步缓存与逐步采样**，形状直觉仍以 \((B,L,d)\) 为锚。

---

### 2.5 常见踩坑（提前避雷）

| 现象 | 可能原因 |
|------|----------|
| `matmul` 维度报错 | \(QK^\top\) 时 \(K\) 应是 \((B,L,d_k)\) 与转置配合；多头时先统一成 \((B,h,L,d_k)\)。 |
| softmax 后权重不对 | **dim** 取错；应对 **key 所在维** softmax。 |
| 与论文公式差一个转置 | PyTorch `Linear` 是 **行向量 × 权重^T**；读别人代码时盯 **形状** 而非符号。 |
| HF 模型输出 NaN | 半精度/梯度问题多见于训练；推理小模型先试 **`float32`**、`model.eval()`。 |

---

### 2.6 与 `learning-guide.md`（W1）的衔接

- **W1 必做**：OpenAI 兼容 **流式 Chat**；第 4 天练的是 **张量与库**，不冲突。  
- **下周 RAG**：你会把 **同一套**「向量维度 \(d\)」与 **批处理形状** 接到 **Embedding 与向量库**；今天把 **\(B,L,d\)** 刻进脑子，后面少一半低级 bug。

---

## 3. 推荐阅读（可选）

1. **PyTorch 官方**：`nn.TransformerEncoderLayer` 文档 — 只看 **参数表** 与 **forward 输入输出形状**（不必直接用，可对照你自己手写）。  
2. **Hugging Face**：*Quicktour* — `AutoTokenizer` / `AutoModel` 一节：  
   https://huggingface.co/docs/transformers/quicktour  
3. **The Annotated Transformer**（Harvard）— 若第 3 天没扫过，今天可对照 **一行代码一行形状**（选读）。

---

## 4. 90～120 分钟时间安排

| 时段 | 时长 | 内容 |
|------|------|------|
| A | 20 min | 通读 **§2**；在纸上画 \((B,L,d)\) → MHA → \((B,L,d)\) |
| B | 25 min | 新建 `venv`，安装 `torch`、`transformers`（CPU 即可）；跑通 HF 官方 Quicktour 最小例子 |
| C | 35 min | **§5 必做 1–3**（形状 + 一段读代码/自写骨架） |
| D | 20 min | **§5 选做 4–5** 或 Pre-LN 顺序默写 + **§6 自测** |
| E | 10 min | 学习记录；写明 1 个仍模糊的形状（下周对照 RAG 再杀） |

---

## 5. 练习题

### 必做 1：形状填空

\(B=2,\; L=16,\; d_{\mathrm{model}}=768,\; h=12,\; d_k=64\)。

1. 嵌入层输出 `x` 的形状？  
2. 若 `scores` 为单头 \(QK^\top\)，形状？  
3. 若多头合并为 \((B,h,L,d_k)\)，则 `scores` 形状？  
4. 该 Encoder 子层输出（未接分类头）形状？

<details>
<summary>参考答案</summary>

1. \((2,\,16,\,768)\)  
2. \((2,\,16,\,16)\)  
3. \((2,\,12,\,16,\,16)\)  
4. \((2,\,16,\,768)\)

</details>

---

### 必做 2：代码与数学对应

写出下面每一行对应的符号（来自 day1–3）：

```python
# x: (B, L, d_model)
u = layer_norm(x)
q = Wq(u)   # nn.Linear(d_model, d_model) 或分头时到 h*d_k
```

1. `u = layer_norm(x)` 在 Pre-LN 里对应「先 Norm 再进子层」的哪一部分？  
2. `q = Wq(u)` 对应 \(Q = XW^Q\) 还是 \(Q = \mathrm{LN}(X)W^Q\)？

<details>
<summary>参考答案</summary>

1. Pre-LN 中，子层输入是 **LayerNorm 后的张量**。  
2. 对应 **\(Q = \mathrm{LN}(X)\,W^Q\)**（若整体结构是 Pre-LN；与 Post-LN 写法不同，以你采用的块为准）。

</details>

---

### 必做 3：Hugging Face 实操（动手）

用任意**小模型**（如 `distilbert-base-uncased` 或官方 quicktour 示例模型）完成：

1. `tokenizer` 两段不同长度句子，`padding=True`，打印 `input_ids.shape` 与 `attention_mask`。  
2. `model(**inputs)`，打印 `last_hidden_state.shape`。  
3. 一句话解释：`attention_mask` 里 **0** 的位置在算什么时通常要被忽略？

<details>
<summary>参考答案要点</summary>

1. `input_ids` 与 `attention_mask` 均为 \((B,L)\)；\(L\) 为 pad 后统一长度。  
2. `last_hidden_state` 为 \((B,L,d_{\mathrm{model}})\)（具体 \(d\) 依模型 config）。  
3. **padding 位置**不应参与注意力聚合（实现上通过 mask 将对应 logits 置为极大负值或跳过）。

</details>

---

### 选做 4：一行参数量（复习 day3）

仅 **一层**、**Encoder 风格**，\(d=512,\; d_{\mathrm{ff}}=2048\)。忽略 bias 与 LayerNorm，估算 **MHA 四个大矩阵 + FFN 两层** 的参数量级（与 day3 选做 5 同思路）。

<details>
<summary>参考答案</summary>

与 day3：**\(4\times512^2 + 512\times2048 + 2048\times512\)**（数量级约数百万）；本日重点在 **代码里 `nn.Linear` 的 `in_features/out_features` 与之一致**。

</details>

---

### 选做 5：最小「随机张量走通」

不写训练，仅：

```python
import torch
import torch.nn as nn

B, L, d = 2, 8, 64
x = torch.randn(B, L, d)
ln = nn.LayerNorm(d)
# 自行补：两个 Linear 当作简化「注意力+输出投影」或直接用 nn.MultiheadAttention(embed_dim=d, num_heads=4, batch_first=True)
```

目标：**`forward` 后张量仍为 \((B,L,d)\)**。把最终 `y.shape` 与一两句注释写进你的 `practice.py` 即可。

<details>
<summary>提示</summary>

可直接使用 `nn.MultiheadAttention(..., batch_first=True)` 验证形状；再对照 day2「分头」手写拆分留作进阶。

</details>

---

## 6. 自测检查点

- [ ] 说出 **Pre-LN** 下一层 Encoder 的两个残差块各包什么。  
- [ ] **\(B,L,d\)** 在 `nn.Linear` 下哪一维被变换？  
- [ ] **多头** 时 **scores** 常见是 \((B,h,L,L)\) 还是 \((B,L,h,L)\)？（以你使用的 API 为准，能自圆其说即可。）  
- [ ] **`attention_mask` 与 `causal mask`** 分别解决什么问题？（第 3 天 + 今日 HF。）

<details>
<summary>自测要点</summary>

- Pre-LN：**LN → MHA → 残差**；**LN → FFN → 残差**。  
- `nn.Linear`：**最后一维** `d_in → d_out`。  
- `nn.MultiheadAttention(batch_first=True)` 内部通常 head 维在 batch 后；读 `shape` 确认。  
- `attention_mask`：**padding**；causal：**自回归不看未来**。

</details>

---

## 7. 今日学习总结与记录

**四天串联**

| 天 | 主题 |
|----|------|
| Day 1 | Scaled dot-product、Q/K/V、\((B,L,L)\) |
| Day 2 | 多头、位置编码 |
| Day 3 | Encoder/Decoder、掩码、交叉注意力、FFN、BERT/GPT |
| **Day 4** | **PyTorch 形状、单层骨架、HF 推理与张量** |

**明日 / 下周**：按 `learning-guide.md` **W1**，并行推进 **Chat API 流式** 与 **术语表 §3**（KV Cache 等）；RAG 周会复用今日的 **\(B,L,d\)** 直觉。

---

- 学习日期：________  
- 学习时长：约 90～120 分钟  
- 掌握程度（1–5）：___  
- 疑问 / 笔记：  
  -  
  -  

---

## 8. 环境备忘（Windows / 通用）

```text
python -m venv .venv
.venv\Scripts\activate
pip install torch transformers --index-url https://download.pytorch.org/whl/cpu
```

有 NVIDIA GPU 时可换官方 CUDA 安装指引；**CPU 足够完成第 4 天目标**。

---

把今天写的 **最小可运行脚本** 存进仓库或学习目录，下周 RAG 切分时可以回看：**同一套张量纪律**。
