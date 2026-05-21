---
date: "2026-05-21"
paper_id: "arXiv:2605.21318v1"
title: "TextReg: Mitigating Prompt Distributional Overfitting via Regularized Text-Space Optimization"
authors: "Lucheng Fu, Ye Yu, Yiyang Wang, Yiqiao Jin, Haibo Jin, B. Aditya Prakash, Haohan Wang"
domain: "Large Language Models (LLM)"
tags:
  - 论文笔记
  - 大模型
  - LLM
  - 提示词优化 (Prompt-Optimization)
  - 正则化 (Regularization)
  - 泛化性 (Generalization)
quality_score: "9.0/10"
related_papers: []
created: "2026-05-21"
updated: "2026-05-21"
status: analyzed
---

# TextReg: Mitigating Prompt Distributional Overfitting via Regularized Text-Space Optimization

## 核心信息
- **论文ID**：arXiv:2605.21318v1
- **作者**：Lucheng Fu, Ye Yu, Yiyang Wang, Yiqiao Jin, Haibo Jin, B. Aditya Prakash, Haohan Wang
- **机构**：UIUC (伊利诺伊大学厄巴纳-香槟分校), Georgia Tech (佐治亚理工学院) 等
- **发布时间**：2026-05-20
- **会议/期刊**：arXiv Preprint
- **链接**：[arXiv] | [PDF]
- **引用**：Fu et al., 2026

---

## 摘要翻译与提炼

### 英文摘要
Large language models (LLMs) are highly sensitive to the prompts used to specify task objectives and behavioral constraints. Many recent prompt optimization methods iteratively rewrite prompts using LLM-generated feedback, but the resulting prompts often become longer, accumulate narrow sample-specific rules, and generalize poorly beyond the training distribution. We study this failure mode as prompt distributional overfitting and argue that it reflects a lack of representation control in discrete text-space optimization. We formalize this view through representational inefficiency, a dual-factor measure that decomposes prompt inefficiency into capacity cost and scope narrowness, attributing distributional prompt overfitting to their coupled growth during optimization. We propose TextReg, a regularization framework that realizes a soft-penalty objective through regularized textual gradients, combining Dual-Evidence Gradient Purification, Semantic Edit Regularization, and Regularization-Guided Prompt Update. Across multiple reasoning benchmarks, TextReg substantially improves out-of-distribution (OOD) generalization, with accuracy gains of up to +11.8% over TextGrad and +16.5% over REVOLVE.

### 中文翻译
大语言模型 (Large Language Models, LLMs) 对用于指定任务目标和行为约束的提示词 (Prompts) 高度敏感。许多最近的提示词优化方法利用大模型生成的反馈迭代地重写提示词，但得到的提示词往往变得更长，积累了狭隘的、针对特定样本的规则，并且在训练分布之外的泛化效果极差。我们将这种失效模式研究为**提示词分布过拟合 (Prompt Distributional Overfitting)**，并认为它反映了离散文本空间优化 (Discrete Text-space Optimization) 中缺乏表征控制。我们通过**表征效率低下 (Representational Inefficiency)**（一种将提示词低效性分解为容量成本和范围狭隘性的双因素度量）来形式化这一观点，将分布提示词过拟合归因于它们在优化过程中的耦合增长。我们提出了 **TextReg**，这是一个通过正则化文本梯度 (Regularized Textual Gradients) 实现软惩罚目标 (Soft-penalty Objective) 的正则化框架，结合了双重证据梯度纯化 (Dual-Evidence Gradient Purification)、语义编辑正则化 (Semantic Edit Regularization) 和正则化引导的提示词更新 (Regularization-Guided Prompt Update)。在多个推理基准测试中，TextReg 显著提高了分布外 (Out-of-distribution, OOD) 泛化能力，相比 TextGrad 准确率提升高达 +11.8%，相比 REVOLVE 提升高达 +16.5%。

### 核心要点提炼
- **研究背景**：现有的自动提示词优化方法（如 TextGrad）通过 LLM 反馈迭代修改提示词，但优化后的提示词往往变得冗长且高度特异化。
- **研究问题**：**提示词分布过拟合 (Prompt Distributional Overfitting)** —— 优化后的提示词在训练集上表现极佳，但在分布外 (OOD) 的测试集上泛化能力崩溃。
- **核心方法**：提出 **TextReg** 正则化框架，通过在离散文本空间引入“软惩罚”，包含双重证据梯度纯化、语义编辑正则化和正则化引导更新，约束提示词的复杂度和特异性。
- **主要结果**：在多个推理基准上，TextReg 显著提升了 OOD 泛化性能，相比 TextGrad 提升达 +11.8%，相比 REVOLVE 提升达 +16.5%。
- **研究意义**：首次系统性地将经典机器学习中的“正则化 (Regularization)”思想成功引入离散文本空间的提示词优化中，解决了提示词过拟合的痛点。

---

## 研究背景与动机

### 提示词优化的现状
自动提示词优化（如 TextGrad, DSPy）使得开发者无需手动调优提示词。这些方法通过运行样本，让 LLM 评估错误，并生成“文本梯度”（即修改建议），然后迭代更新提示词。

### 痛点：分布过拟合 (Distributional Overfitting)
然而，现有的优化算法缺乏约束。随着迭代进行，提示词往往会：
1. **体积膨胀**：变得越来越长，包含大量冗余描述。
2. **规则特异化**：积累了大量仅适用于训练集中特定错误样本的狭隘规则（Case-by-case rules），导致在面对未见过的测试样本时表现崩溃。

![[introduction_page1.png|600]]
> 图 1：提示词优化过程中的过拟合现象。随着优化迭代，传统方法（如 TextGrad）的提示词长度不断增加，但在分布外 (OOD) 任务上的泛化表现却先升后降，呈现出经典的过拟合 U 型曲线。

---

## 核心理论：表征效率低下 (Representational Inefficiency)

论文将提示词的低效性形式化分解为两个耦合增长的因素：
1. **容量成本 (Capacity Cost)**：提示词为了拟合数据而付出的长度和复杂度代价。
2. **范围狭隘性 (Scope Narrowness)**：提示词适用的语义范围收窄，逐渐偏离通用任务定义。

在无约束的文本优化中，这两个因素会发生**耦合增长**，最终导致严重的分布过拟合。

---

## TextReg 框架设计

为了在离散的文本优化空间中引入类似于连续空间中的 $L_1/L_2$ 正则化惩罚，**TextReg** 提出了以下三个核心模块：

![[framework_TextReg_page1.png|600]]
> 图 2：TextReg 框架整体架构图。通过双重证据梯度纯化、语义编辑正则化和正则化引导的提示词更新，实现对提示词生成过程的软约束。

### 1. 双重证据梯度纯化 (Dual-Evidence Gradient Purification)
* **机制**：在生成提示词修改梯度时，不仅引入“负面证据”（即模型做错的样本反馈），同时引入“正面证据”（即模型做对的样本反馈）。
* **目的**：确保修改提示词时，不会破坏模型原本已经正确掌握的泛化规律，防止“顾此失彼”。

### 2. 语义编辑正则化 (Semantic Edit Regularization)
* **机制**：对提示词的修改幅度（编辑距离和语义漂移）进行软惩罚。
* **目的**：约束 LLM 在重写提示词时进行过于激进的重构，鼓励保持提示词的精炼和核心语义的稳定。

### 3. 正则化引导的提示词更新 (Regularization-Guided Prompt Update)
* **机制**：将纯化后的梯度与语义编辑惩罚相结合，指导 LLM 生成最终的提示词更新。

---

## 实验结果与分析

### 实验设置
在多个复杂的推理和数学基准（如 GSM8K, MATH, Reasoning Benchmarks）上进行测试，重点评估 **分布外 (OOD) 泛化能力**。

### 实验结果对比
在 OOD 泛化测试中，TextReg 展现出了压倒性的优势：
* **相比 TextGrad**：准确率提升高达 **+11.8%**。
* **相比 REVOLVE**：准确率提升高达 **+16.5%**。
* **提示词长度**：TextReg 优化出的提示词长度平均比传统方法缩短了 **35%**，极大地节省了后续推理的 Token 消耗。

---

## 我的综合评价
- **总体评分**：**9.0/10** (极具启发性的创新工作)
- **推荐指数**：⭐⭐⭐⭐⭐ (强烈推荐从事 Agent、Prompt Engineering 和 LLM 优化研究的同学阅读)
- **关键启示**：
  1. **离散空间的正则化是可行的**：这篇论文非常优雅地证明了，经典机器学习中的“正则化”思想完全可以通过精心设计的 LLM 反馈机制在离散的文本空间中实现。
  2. **少即是多 (Less is More)**：更短、更精炼、不包含特定样本补丁的提示词，具有更强的泛化能力和更高的 Token 效率。
