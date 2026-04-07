# 第 3 天：Transformer 完整架构（Encoder–Decoder）与变体对照

> **学习时间：约 1.5～2 小时**（§2 精读约 50～70 min + 练习）  
> **目标**：能画出一层 Encoder / 一层 Decoder 的数据流；理解**因果掩码与 padding 掩码**的差别；掌握**交叉注意力**里 Q/K/V 从哪来、张量形状如何对齐；理解残差、**Pre/Post-Norm**、FFN 的分工；能对照 **BERT / GPT / 原始 Transformer**。

---

## 1. 今日学习目标

完成今天学习后，你应能：

- [ ] 说出 **Encoder** 与 **Decoder** 各有哪些子层，Decoder 多出来的块是什么  
- [ ] 解释 **因果掩码**（causal mask）在训练解码器时的必要性  
- [ ] 写出 **Pre-Norm / Post-Norm** 常见写法（知道现代实现多为 Pre-LN 即可）  
- [ ] 说明 **FFN** 在「每个位置」上的作用及典型扩展比（如 4×）  
- [ ] 完成 **形状追踪** 与 **掩码矩阵** 小练习  
- [ ] 用表格对比 **原始 Transformer / BERT / GPT** 的结构差异

---

## 2. 核心概念讲解

### 2.0 与第 1～2 天衔接：今天多出来的三件事

你已经会：**缩放点积注意力**、**多头**、**位置编码**。在**单层**里，隐藏状态形状始终是 \((B,L,d_{\mathrm{model}})\)。今天要解决的是把它们**堆成网络**之后的问题：

1. **堆叠**：\(N\) 层 Encoder / Decoder，每层之间用 **残差** 把梯度与原始信号「短路」传下去。  
2. **两种掩码**：**因果掩码**（decoder 不能看未来）；**padding 掩码**（batch 里短句补零处不应参与注意力）。二者常**同时**出现，实现上要分清维度。  
3. **第三套 Q/K/V（交叉注意力）**：Decoder 自己用 **掩码自注意力** 建模目标语言；再用 **Cross-Attention** 把 **Encoder 输出的源语言整句** 拉进来——**Query 来自 Decoder，Key/Value 来自 Encoder**。

---

### 2.1 任务设定：Seq2Seq 与 Teacher Forcing（训练时）

以**机器翻译**为例：源序列 \(x\)（如英文），目标序列 \(y\)（如中文）。**Encoder** 读入 \(x\)，得到每个源词位置的上下文向量；**Decoder** 在每一步生成目标词时，既要参考**已生成的前缀**，又要对齐**源句**。

**训练**时常用 **Teacher Forcing**：Decoder 的输入是 **真实目标序列右移一位**（前面加 BOS），模型在位置 \(t\) 预测 **\(y_t\)**（或下一词）。这样每个时间步都有**标准答案**可算交叉熵。**推理/生成**时则没有未来真值，只能**自回归**地用自己刚生成的 token 作为下一步输入——因此必须在架构上保证：训练时 Decoder 也**看不到** \(t\) 之后的真值 → **因果掩码**。

---

### 2.2 整体结构：数据流与张量形状（鸟瞰）

记号：\(L_s\) = 源序列长度，\(L_t\) = 目标序列长度，\(B\) = batch，\(d=d_{\mathrm{model}}\)。

```
源 token ids  →  Embedding + PE  →  Encoder × N  →  H_enc  (B, L_s, d)
                                                      ↓ K, V
目标 token ids → Embedding + PE  →  Decoder × N  →  H_dec  (B, L_t, d)
                                                      ↓ 线性层
                                              logits (B, L_t, |V|) → softmax → 下一词分布
```

- **Encoder 输出 \(H_{\mathrm{enc}}\)**：源语言每个位置一个 \(d\) 维向量，**整句上下文**已编码在内。  
- **Decoder** 在每一层都会通过 **Cross-Attention** 用 \(H_{\mathrm{enc}}\) 作为 **K、V**（见 §2.5）。  
- 最后 **`nn.Linear(d, |V|)`** 把每个目标位置映射到词表大小，得到 **logits**；与标签算 **交叉熵**。

---

### 2.3 Encoder 单层：双向自注意力 + FFN

**作用**：对**源序列**做**双向**自注意力——每个源位置可以看**整句**所有源位置（无因果约束），适合「充分理解源句」。这与 Decoder 的「只能看已生成前缀」正交。

**子层顺序（逻辑）**：

1. **多头自注意力**（Self-Attention）：\(Q,K,V\) 都来自本层输入 \(X\)（形状 \((B,L_s,d)\)）。  
2. **残差 + Norm**（见 §2.7 的 Pre/Post 两种写法）。  
3. **FFN**：对每个位置独立做同一套 MLP（见 §2.8）。  
4. **残差 + Norm**。

**输出**：仍为 \((B,L_s,d)\)，作为下一层 Encoder 输入，并供 **所有 Decoder 层** 的 Cross-Attention 取 **K、V**（实现上常对 \(H_{\mathrm{enc}}\) 做线性投影得到该层的 \(K,V\)，但**信息来源**是 Encoder）。

---

### 2.4 Decoder 单层：三个子层（顺序必背）

标准顺序：

1. **Masked Multi-Head Self-Attention**  
   - \(Q,K,V\) 均来自 **Decoder 当前层输入**（目标语言一侧）。  
   - **因果掩码**：位置 \(i\) 的 query **只能**与 key 位置 \(j\le i\) 计算注意力；\(j>i\) 的 logits 置为 \(-\infty\)，softmax 后为 0。  
   - **目的**：模拟推理时「看不到未来词」；否则训练会**泄露**答案，生成任务会崩。

2. **Encoder–Decoder（Cross）Attention**  
   - **Query**：来自 **Decoder** 在子层 1 之后的表示（「我现在要生成到这一步，该对齐源句哪里？」）。  
   - **Key、Value**：来自 **Encoder 输出** \(H_{\mathrm{enc}}\)（同一套 \(K,V\) 供当前步对所有源位置的加权）。  
   - **形状**：\(Q\in\mathbb{R}^{B\times L_t\times d}\)，\(K,V\in\mathbb{R}^{B\times L_s\times d}\)，注意力分数矩阵为 **\((B,L_t,L_s)\)**——第 \(t\) 个目标位置对每个**源位置**的权重，即**对齐/注意力图**的常见来源。

3. **FFN**  
   - 与 Encoder 中 FFN 形式相同，作用在 **每个目标位置** 上。

每一步子层后都有 **残差 + Norm**（具体 Pre 或 Post 见 §2.7）。

---

### 2.5 交叉注意力：为什么 Q 与 K/V 来源不同？

**直觉**：Decoder 每一步在问：「基于**目前已解码的目标前缀**，下一步应该关注**源句的哪些片段**？」——所以 **Query 必须来自 Decoder（目标侧）**；而「被检索的条目」是**源句各位置的表示**，故 **Key/Value 来自 Encoder（源侧）**。

**与自注意力的对比**：

| 类型 | Self-Attn（Encoder） | Self-Attn（Decoder，带掩码） | Cross-Attn |
|------|----------------------|------------------------------|------------|
| Q 来自 | 本序列 | 本序列（目标） | **目标序列（Decoder）** |
| K,V 来自 | 本序列 | 本序列（目标，掩码） | **源序列（Encoder）** |
| 分数矩阵形状（单头） | \((L_s,L_s)\) | \((L_t,L_t)\) | **\((L_t,L_s)\)** |

若去掉 Cross-Attention，Decoder 只能根据目标语言前缀自洽生成，**无法以源句为条件**，翻译等条件生成任务会失去核心能力。

---

### 2.6 两种掩码：因果 vs Padding（不要混）

**因果掩码（Causal / Look-ahead mask）**  
- **用在**：Decoder 的自注意力。  
- **形式**：\(L_t\times L_t\) 下三角允许、上三角禁止（见 §5 必做 2）。  
- **解决**：防止看见**未来目标词**。

**Padding 掩码**  
- **用在**：Encoder 自注意力、以及 Cross-Attention 里对 **源序列** 的 key（若 batch 内 \(L_s\) 不一致、短句右侧为 pad）。  
- **目的**：pad 位置不应贡献有效 key/value，否则模型会向「空」位置分配权重。  
- **实现**：对 pad 位置的 key 在 softmax 前加 \(-\infty\)，或直接在注意力实现里忽略这些位置。

**同一段 Decoder 自注意力里**：可以同时有 **因果** + **目标侧 padding** 两种约束（实现上常合并成一个 mask 矩阵）。

---

### 2.7 残差与 LayerNorm：Post-Norm 与 Pre-Norm

**残差**（以子层 \(\mathcal{F}\) 为例）：

\[
x_{\mathrm{out}} = x + \mathcal{F}(x).
\]

**动机**：让深层网络里存在**恒等映射**附近的通路，梯度可以更直接地回传；与 **LayerNorm** 一起显著稳定训练。

**Post-LN（原始论文常见写法）**：

\[
x' = \mathrm{LayerNorm}\bigl(x + \mathcal{F}(x)\bigr).
\]

**Pre-LN（许多现代实现，含 GPT-2/3 系、不少 Encoder 实现）**：

\[
x' = x + \mathcal{F}\bigl(\mathrm{LayerNorm}(x)\bigr).
\]

**Pre-LN 常被提及的优点**：归一化在子层**之前**，子层输入尺度更稳定；**梯度路径**更接近「每步加一项」，**深层堆叠**时训练更稳、更易调大学习率（具体仍依赖实现与任务）。你要能在面试里**写出两种公式**，并说明「我读代码时会先看是 Pre 还是 Post」。

---

### 2.8 FFN（Feed-Forward Network）：逐位置、扩维再压回

典型形式（对每个位置 \(x\in\mathbb{R}^d\) 独立应用同一组权重）：

\[
\mathrm{FFN}(x)=\mathrm{Act}(xW_1+b_1)W_2+b_2,\quad
W_1\in\mathbb{R}^{d\times d_{\mathrm{ff}}},\; W_2\in\mathbb{R}^{d_{\mathrm{ff}}\times d}.
\]

- **\(d_{\mathrm{ff}}\)** 常取 **\(4d\)**（如 \(d=512\) 时 \(d_{\mathrm{ff}}=2048\)）。  
- **激活**：原论文 ReLU；BERT 等用 **GELU**；与任务/预训练惯例有关。  

**分工一句话**（与 day2 一致）：**注意力**在 token 之间**路由信息**；**FFN** 在每个位置做**非线性特征变换**（类似把每个位置的向量单独送入宽 MLP）。两者交替堆叠，一层层抽象。

---

### 2.9 输出层与训练目标

Decoder 顶层输出 \(H_{\mathrm{dec}}\in\mathbb{R}^{B\times L_t\times d}\) 后，共享权重矩阵 \(W_{\mathrm{vocab}}\in\mathbb{R}^{d\times |V|}\)（或等价写法）得到 **logits** \(\in\mathbb{R}^{B\times L_t\times |V|}\)。

**损失**：对每个预测位置（常 mask 掉 padding）做 **交叉熵**，目标为 **下一个真实 token**（配合 teacher forcing 的右移标签）。工程上常见 **label smoothing**、**学习率 warmup** 等，本日只需建立「**逐位置多分类**」图景。

---

### 2.10 与 BERT / GPT 及 Encoder-only、Decoder-only 的对应

| | **原始 Transformer** | **BERT** | **GPT** |
|---|------------------------|----------|---------|
| **堆叠结构** | Encoder + Decoder | **仅 Encoder**（\(N\) 层） | **仅 Decoder**（\(N\) 层） |
| **注意力** | Encoder：双向自注意力；Decoder：因果 + Cross | **双向**自注意力 | **因果**自注意力 |
| **预训练目标（典型）** | Seq2Seq（翻译等） | **MLM**（掩码词重建）+ 句对任务（如 NSP，视版本而定） | **下一词预测**（因果 LM） |
| **为何不需要另一半** | 翻译需要「编码源 + 生成目标」 | 理解/句向量：**不需**自回归解码器 | 生成：只需**单向**堆叠即可从左到右建模 |

**记法**：  
- **BERT** ≈ Transformer **Encoder 塔**；适合**分类、抽取、句向量**，不自带自回归生成头（后来可加）。  
- **GPT** ≈ **Decoder 塔**（无 Encoder、无 Cross-Attention）；适合**续写、对话**。  
- **T5、BART** 等：仍是 Encoder–Decoder，与原始论文最像，预训练目标多为 span 去噪等（细节可查论文，本日不展开）。

---

## 3. 推荐阅读（可选加深）

**说明**：§2 已覆盖当日主干；下列材料用于大图与代码对照。

1. **The Illustrated Transformer** — 全文架构图：  
   https://jalammar.github.io/illustrated-transformer/
2. **The Annotated Transformer**（Harvard）— 浏览目录与 Encoder/Decoder 代码对应关系（不必读完）：  
   http://nlp.seas.harvard.edu/annotated-transformer/

---

## 4. 时间安排（约 90～120 分钟）

| 时段 | 时长 | 内容 |
|------|------|------|
| A | 20 min | 通读 **§2**；白纸默画 Encoder 一层 + Decoder 三层子层 |
| B | 20 min | 可选：Illustrated 大图；对照 **§2.4～2.5** 核对 Cross-Attention |
| C | 35 min | **§5 必做 1–4**（含交叉注意力形状题） |
| D | 20 min | **§5 选做 5–7** + **§6 自测** |
| E | 10 min | 学习记录；预告 **day4** 手写张量 |

---

## 5. 练习题

### 必做 1：填空——Decoder 三个子层顺序

按 **标准教学顺序** 填写：

1. ________________  
2. ________________（Q 来自 Decoder，K/V 来自 Encoder）  
3. ________________

<details>
<summary>参考答案</summary>

1. 掩码多头自注意力（Masked Self-Attention）  
2. 交叉注意力（Encoder-Decoder / Cross-Attention）  
3. 前馈网络（FFN）

</details>

---

### 必做 2：因果掩码矩阵

序列长度 \(L=4\)，行 = query 位置，列 = key 位置。允许看自己和过去：上三角（不含对角）为禁止。

1. 写出 4×4 的 **0/1 允许矩阵**（1 表示允许 attend）。  
2. 预测第 2 个 token（索引 1）时，允许哪些 key 索引？

<details>
<summary>参考答案</summary>

1. 下三角全 1，上三角严格上三角为 0：

```
     k0 k1 k2 k3
q0    1  0  0  0
q1    1  1  0  0
q2    1  1  1  0
q3    1  1  1  1
```

2. 索引 1 允许 key 索引 **0 和 1**。

</details>

---

### 必做 2b：交叉注意力形状（对照 §2.5）

设 \(B=2,\; L_s=10\)（源）,\( L_t=8\)（目标）,\( d=256\)。Encoder 输出 \(H_{\mathrm{enc}}\)，Decoder 当前层得到 \(Q\in\mathbb{R}^{B\times L_t\times d}\)，由 \(H_{\mathrm{enc}}\) 得到 \(K,V\in\mathbb{R}^{B\times L_s\times d}\)。

1. 单头下，注意力 **score** 矩阵（未 softmax）形状？  
2. 与 Decoder **自注意力**的 \((L_t,L_t)\) 相比，多了哪一维语义上的含义？

<details>
<summary>参考答案</summary>

1. \((B,\,L_t,\,L_s)=(2,\,8,\,10)\)（若实现为 batch 内矩阵乘，等价理解为每个 query 位置对每个 **源位置** 一条分数）。多头时常见 \((B,\,h,\,L_t,\,L_s)\)。  
2. **目标位置 × 源位置** 的对齐权重，即「当前解码步关注源句哪里」；自注意力则是「目标序列内部谁看谁」。

</details>

---

### 必做 3：形状追踪

\(d_{\mathrm{model}}=512,\; L_{\mathrm{src}}=20,\; L_{\mathrm{tgt}}=30,\; B=16,\; \mathrm{vocab}=32000\)。

Decoder 最后一层输出经线性层到 logits：

- 解码器输出张量形状？  
- logits 形状？

<details>
<summary>参考答案</summary>

- 解码器输出：\((B,\,L_{\mathrm{tgt}},\,d_{\mathrm{model}})=(16,\,30,\,512)\)  
- logits：\((16,\,30,\,32000)\)

</details>

---

### 必做 4：简答

1. 交叉注意力若 **去掉**，Decoder 仅保留掩码自注意力，翻译任务会缺什么能力？  
2. FFN 隐藏层通常为 \(d_{\mathrm{model}}\) 的几倍？注意力与 FFN 分工一句话区分。

<details>
<summary>参考答案</summary>

1. 无法有效利用 **源语言** 整句表示，跨语言对齐困难，翻译质量大幅下降。  
2. 常 **4 倍**；注意力做 **token 间** 信息路由，FFN 做 **每个位置** 的非线性特征变换。

</details>

---

### 选做 5：参数量估算（一层 Encoder，简化）

\(d_{\mathrm{model}}=512,\; h=8,\; d_k=64,\; d_{\mathrm{ff}}=2048\)。**忽略 bias 与 LayerNorm 参数**，估算：

- 多头注意力中 \(W^Q,W^K,W^V,W^O\) 总参数量（把多头等价成大矩阵亦可）  
- 该层 FFN 中 \(W_1,W_2\) 参数量  
- 两者合计

<details>
<summary>参考答案</summary>

- 四个 \(512\times512\) 矩阵：\(4\times512^2 = 1{,}048{,}576\)  
- FFN：\(512\times2048 + 2048\times512 = 2{,}097{,}152\)  
- 合计约 **314.6 万**（与旧版 day3 同量级；此处忽略 bias/LN）

</details>

---

### 选做 6：架构选择题

以下哪项最贴切？

1. **BERT-base** 更接近：A) 仅 Encoder  B) 仅 Decoder  C) Encoder+Decoder  
2. **GPT** 预训练：A) MLM  B) 下一词预测  C) 翻译对齐

<details>
<summary>参考答案</summary>

1. **A**  
2. **B**

</details>

---

### 进阶 7：Post-Norm vs Pre-Norm（概念）

原始论文常用 **Post-LN**：\( \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x)) \)。  
很多深层实现改为 **Pre-LN**：\( x + \mathrm{Sublayer}(\mathrm{LayerNorm}(x)) \)。

用一句话说明 **Pre-LN** 常被报告的优点（训练稳定性相关）。

<details>
<summary>参考答案</summary>

Pre-LN 使梯度路径更直接、深层训练更稳定，常更易缩放层数与学习率（具体仍依赖实现与任务）。

</details>

---

## 6. 自测检查点

- [ ] Encoder 与 Decoder 各几个子层？Decoder 多哪一类注意力？  
- [ ] **因果掩码**与 **padding 掩码**分别防什么？能否同时用在 Decoder 自注意力里？  
- [ ] 残差 + Norm 的动机？**Post-LN** 与 **Pre-LN** 公式各写一种。  
- [ ] 为什么 BERT 不需要 Decoder？为什么 GPT 不需要 Encoder？  
- [ ] 画出：Encoder 一层；Decoder 一层（掩码 MHA → Cross-Attn → FFN，并标残差）。  
- [ ] Cross-Attention 的 score 矩阵是 \((L_t,L_s)\) 还是 \((L_s,L_t)\)？（以 **query 行 × key 列** 约定说明。）

<details>
<summary>自测要点（先闭卷再展开）</summary>

- Encoder：MHA + FFN，各 **残差+Norm**；Decoder：**掩码 MHA + Cross-Attn + FFN**。多出来的是 **Cross-Attention**。  
- **因果**：防看见**未来目标词**；**padding**：防 **pad token** 参与有效注意力；Decoder 自注意力可同时施加二者。  
- 残差：梯度/信号短路；LN：稳定尺度。Post：\(\mathrm{LN}(x+\mathcal{F}(x))\)；Pre：\(x+\mathcal{F}(\mathrm{LN}(x))\)。  
- BERT 做理解/MLM，只需 **双向 Encoder**；GPT 做因果 LM，只需 **Decoder 堆叠**。  
- Cross-Attn：通常 **\(Q\) 行对应 \(L_t\)**，**\(K\)** 列对应 \(L_s\)，故分数为 **\(L_t\times L_s\)**（与实现 batch/head 维顺序无关，语义如此）。

</details>

---

## 7. 今日学习总结与记录

**要点回顾**

1. **Seq2Seq**：Encoder 编码源句 \(H_{\mathrm{enc}}(B,L_s,d)\)；Decoder **因果自注意力 + Cross（\(Q\) 目标，\(K,V\) 源）+ FFN**，输出 \((B,L_t,d)\) → logits \((B,L_t,|V|)\)。  
2. **Teacher Forcing**：训练用右移目标；**因果掩码**保证与推理一致（不看未来）。  
3. **两种掩码**：因果（Decoder 自注意力）+ padding（Encoder / Cross 的源侧 key 等）。  
4. **Pre/Post-LN**：会写公式；现代实现多 Pre-LN。  
5. **BERT ≈ Encoder 塔**；**GPT ≈ Decoder 塔**；全栈翻译仍要 **Encoder–Decoder**。

**四天串联**

| 天 | 主题 |
|----|------|
| Day 1 | Scaled dot-product、Q/K/V、形状 \((B,L,L)\) |
| Day 2 | 多头、位置编码、RoPE 了解 |
| Day 3 | 堆叠、双掩码、Cross-Attn、Pre/Post-Norm、FFN、BERT/GPT |
| Day 4 | PyTorch 形状、单层 Encoder 骨架、HF 推理（见 `day4-learning.md`） |

**后续建议**：第 4 天见 **`day4-learning.md`**（PyTorch 张量、单层 Encoder 骨架、Hugging Face 最小推理），与 `learning-guide.md` 中 **W1** 工程线对齐。

---

- 学习日期：________  
- 学习时长：约 90～120 分钟  
- 掌握程度（1–5）：___  
- 疑问 / 笔记：  
  -  
  -  

---

## 8. 附加：架构手绘检查表

在纸上完成并打勾：

- [ ] Encoder：自注意力 + FFN，两处残差与 Norm  
- [ ] Decoder：掩码自注意力 + 交叉注意力 + FFN  
- [ ] 标注 Cross-Attention 的 Q / K / V 来源  
- [ ] 标注 logits 与 vocab 维

---

前三天结束后，你具备阅读 **Attention Is All You Need** 全文与 **Illustrated Transformer** 的骨架；**第 3 天 §2** 可与论文图 1、图 2 对照。接下来用 **`day4-learning.md`** 把张量形状钉进代码。
