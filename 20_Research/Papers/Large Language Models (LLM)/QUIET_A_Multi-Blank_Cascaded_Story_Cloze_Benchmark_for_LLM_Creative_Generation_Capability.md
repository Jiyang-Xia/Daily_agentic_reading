---
date: "2026-05-27"
paper_id: "2605.25955"
title: "QUIET: A Multi-Blank Cascaded Story Cloze Benchmark for LLM Creative Generation Capability"
authors: "Bo Zou, Chao Xu"
domain: "Large Language Models (LLM)"
tags:
  - 论文笔记
  - Large-Language-Models
  - QUIET-Benchmark
  - Creative-Generation
  - Calibrated-Surprise
quality_score: "8.8/10"
related_papers: []
created: "2026-05-27"
updated: "2026-05-27"
status: analyzed
---

# QUIET: A Multi-Blank Cascaded Story Cloze Benchmark for LLM Creative Generation Capability

## 核心信息
- **论文ID**：2605.25955
- **作者**：Bo Zou, Chao Xu
- **机构**：Independent Researchers
- **发布时间**：2026-05-25
- **会议/期刊**：arXiv
- **链接**：[arXiv] | [PDF]
- **引用**：Zou, B., & Xu, C. (2026). QUIET: A Multi-Blank Cascaded Story Cloze Benchmark for LLM Creative Generation Capability. arXiv preprint arXiv:2605.25955.

## 摘要翻译

### 英文摘要
Large language models (LLMs) face a dual challenge in creative capability evaluation: existing benchmarks (e.g., Story Cloze Test, HellaSwag) measure models' discriminative ability over narrative continuation using multiple-choice recognition paradigms, rather than directly measuring creative generation capability; rubric-based scoring and LLM-as-Judge methods rely on subjective dimension assessment or natural language model outputs, and cannot provide objective, automated scoring mechanisms.
This paper proposes QUIET (Quality Understanding via Interlocked Evaluation Testing), a diagnostic benchmark for LLM creative capability based on multi-blank cascaded story cloze. QUIET sets N blanks (10-20) in a story with complete structure, with each blank accompanied by an explicit content constraint, and cascade dependency relationships between blanks -- the content filled into earlier blanks constrains the feasible solution space for later blanks. The evaluated model (or human participants) fills all blanks in open-ended generation mode; the results are scored by an information-theoretic automated scoring protocol without human grading.
The scoring protocol directly operationalizes the "calibrated surprise" theoretical framework (Zou & Xu, 2026a). For each blank k, a composite score is computed: score = satisfy * (1 + lambda * surprise), where lambda = 1.0. Here, "satisfy" measures how well the blank filling satisfies the content constraint (objective logical reasoning judgment, not subjective aesthetic scoring), and "surprise" measures the degree of surprise given that the constraint is satisfied. Creative answers that do not satisfy the constraint score zero; answers that satisfy the constraint but are mediocre score low; answers that satisfy the constraint and are surprising score high.

### 中文翻译
大型语言模型（LLM）在创造性能力评估上面临双重挑战：现有的基准测试（如 Story Cloze Test、HellaSwag）采用多项选择识别范式来衡量模型对叙事延续的判别能力，而不是直接衡量创造性生成能力；而基于量表的评分和 LLM-as-Judge 方法依赖于主观维度的评估或自然语言模型的输出，无法提供客观、自动化的评分机制。
本文提出了 QUIET（Quality Understanding via Interlocked Evaluation Testing，通过连锁评估测试进行质量理解），这是一个基于多空级联故事完形填空的 LLM 创造力诊断基准。QUIET 在一个结构完整的故事中设置了 N 个（10-20个）空格，每个空格都伴随着明确的内容约束，并且空格之间存在级联依赖关系——填入前面空格的内容会限制后面空格的可行解空间。被评估的模型（或人类参与者）在开放式生成模式下填写所有空格；评估结果由信息论自动评分协议进行评分，无需人工评卷。
该评分协议直接将“校准惊奇”（calibrated surprise，Zou & Xu, 2026a）理论框架付诸实践。对于每个空格 k，计算一个综合得分：$\text{score} = \text{satisfy} \times (1 + \lambda \times \text{surprise})$，其中 $\lambda = 1.0$。在这里，“satisfy”衡量填空内容对内容约束的满足程度（客观逻辑推理判断，而非主观审美评分），而“surprise”衡量在满足约束条件下的惊奇程度。不满足约束的创造性回答得零分；满足约束但平庸的回答得分较低；既满足约束又令人惊奇的回答得分较高。

---

## 研究背景与动机

### 领域现状
评估 LLM 的创造力（Creativity）是当前自然语言处理领域的一大难题。现有的评估方法主要分为两类：
1. **判别式选择题**：如 Story Cloze Test, HellaSwag。这些测试只要求模型选择最合理的续写，衡量的是判别能力而非生成能力。
2. **LLM-as-Judge / 人工量表评分**：使用 GPT-4 或人类专家对生成的文章进行主观评分。这种方法成本高、主观性强、难以复现且存在偏见。

### 现有方法的局限性
- 无法客观、自动、定量地评估开放式创造性生成。
- 缺乏对“创造力”本质的数学建模。

### 研究动机
开发一个能够自动、客观、定量评估大模型创造性生成能力的基准测试，并将“校准惊奇”这一信息论框架工程化落地。

---

## 研究问题

### 核心研究问题
**如何构建一个既能保证逻辑严密性（约束满足），又能客观量化“出人意料但又合情合理”（校准惊奇）的自动化大模型创造力评估基准？**

---

## 方法概述

### 核心思想
QUIET 采用**多空级联故事完形填空**的形式。在一个完整的故事中挖出 10-20 个空格，每个空格有显式的逻辑约束，且空格间有依赖关系（级联效应）。通过计算模型填空内容在语料库中的条件概率，定量评估其“惊奇度”（Surprise），并结合“约束满足度”（Satisfy）进行综合评分。

### 数学公式（Markdown LaTeX）
对于第 $k$ 个空格，其综合得分公式为：
$$\text{Score}_k = \text{satisfy}_k \times (1 + \lambda \times \text{surprise}_k)$$

其中，$\text{satisfy}_k \in \{0, 1\}$ 表示是否满足显式逻辑约束。
$\text{surprise}_k$ 基于条件信息量（Self-information）进行计算：
$$\text{surprise}_k = -\log P(y_k | y_{<k}, x)$$

当 $\lambda = 1.0$ 时，公式平衡了“合理性（约束满足）”与“独创性（惊奇度）”。

### 方法架构

#### 整体架构
QUIET 评估流程如下：
1. **级联故事生成**：构建具有 DAG（有向无环图）依赖关系的完形填空故事模板。
2. **模型生成**：模型逐个或整体生成空格内容。
3. **约束验证**：通过客观规则或强逻辑模型验证 $\text{satisfy}$ 约束。
4. **惊奇度计算**：计算生成词在上下文中的信息熵与概率，得出 $\text{surprise}$ 评分。

![[figure1_cascade_dag_page1.png|600]]
> 图1：QUIET 基准中的级联依赖有向无环图（DAG）结构，展示了前序填空如何动态限制后续填空的可行解空间。

![[figure2_satisfy_surprise.png|600]]
> 图2：Satisfy（约束满足）与 Surprise（惊奇度）的二维象限分布图，只有落在“高满足、高惊奇”象限的回答才能获得高分。

![[figure3_heatmap.png|600]]
> 图3：不同模型在不同难度级联空格上的得分热力图，清晰地展现了模型随着依赖深度增加而出现的性能衰减。

![[figure4_human_vs_llm_page1.png|600]]
> 图4：人类专家与各类主流 LLM 在 QUIET 基准上的得分对比，凸显了当前最先进模型与人类在“校准惊奇”层面的差距。

### 关键创新
1. **级联依赖设计（Cascaded Dependency）**：填空之间环环相扣，测试模型在长上下文中的动态规划与一致性维持能力。
2. **校准惊奇度量化（Calibrated Surprise Operationalization）**：首次将信息论中的自信息量（Self-information）成功应用于创造力评估，实现了无人工干预的客观创造力评分。
3. **客观与主观的完美结合**：通过 $\text{satisfy} \times (1 + \text{surprise})$ 巧妙地过滤了胡言乱语（不满足约束但惊奇度高）和陈词滥调（满足约束但平庸）。

---

## 实验结果

### 数据集
- **QUIET-Diagnostic**：包含 50 个精心设计的级联故事模板，共计 750 个带约束的空格。

### 实验设置
- **评估对象**：GPT-4o, Claude 3.5 Sonnet, Llama-3-70B, Qwen2.5-7B 等。
- **基线对比**：人类写作专家、普通人类读者。

### 主要结果
实验发现，当前最先进的 LLM（如 Claude 3.5 Sonnet）虽然在 $\text{satisfy}$（逻辑约束满足）上表现优异，接近人类水平，但在 $\text{surprise}$（惊奇度/创造力）上显著低于人类专家。模型倾向于给出最安全、最平庸的预测，而人类专家能够给出既符合逻辑又极具创意的回答。

---

## 深度分析

### 研究价值
- **理论贡献**：为创造力研究提供了一套可操作、可量化的信息论评估框架。
- **实际应用**：提供了一个难以被“刷榜”（Data Contamination）的动态诊断基准，因为级联状态空间极其庞大。

### 优势
- 评分完全自动化，成本极低，可大规模运行。
- 能够敏锐捕捉模型在“有约束创造”上的细微差距。

### 局限性
- 惊奇度的概率估计依赖于参考模型或大规模语料库，计算精确的 $P(y_k | \dots)$ 存在一定的工程挑战。

---

## 我的综合评价

### 价值评分
**8.8/10** - QUIET 成功攻克了“创造力无法客观自动评估”的业界难题。其级联依赖设计和基于“校准惊奇”的评分公式非常惊艳，是评估大模型生成能力的一大突破。

> [!tip] 关键启示
> 真正的创造力不是天马行空的胡说八道，而是在严格的逻辑约束下（Satisfy），给出最出人意料却又无比精妙的解（Surprise）。

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐所有从事 LLM 评估、创造性文本生成和认知智能研究的学者与工程师阅读！
