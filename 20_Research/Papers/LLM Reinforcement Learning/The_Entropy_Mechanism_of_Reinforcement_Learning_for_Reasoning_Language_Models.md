---
date: "2026-05-26"
paper_id: "arXiv:2505.22617"
title: "The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models"
authors: "Peking University, Tsinghua University"
domain: "LLM Reinforcement Learning"
tags:
  - "论文笔记"
  - "LLM-Reinforcement-Learning"
  - "Entropy-Collapse"
  - "Policy-Gradient"
  - "Reasoning-LLMs"
quality_score: "9.2/10"
created: "2026-05-26"
updated: "2026-05-26"
status: "analyzed"
---

# The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models

## 核心信息
- **论文ID**：arXiv:2505.22617
- **作者**：Peking University, Tsinghua University (北京大学、清华大学等)
- **机构**：Peking University, Tsinghua University
- **发布时间**：2025-05-28
- **会议/期刊**：arXiv Preprint
- **链接**：[arXiv](https://arxiv.org/abs/2505.22617) | [PDF](https://arxiv.org/pdf/2505.22617)
- **引用**：--

## 摘要翻译

### 英文摘要
This paper aims to overcome a major obstacle in scaling RL for reasoning with LLMs, namely the collapse of policy entropy. Such phenomenon is consistently observed across vast RL runs without entropy intervention, where the policy entropy dropped sharply at the early training stage, this diminished exploratory ability is always accompanied with the saturation of policy performance. In practice, we establish a transformation equation $R = -a \cdot e^H + b$ between entropy $H$ and downstream performance $R$. This empirical law strongly indicates that, the policy performance is traded from policy entropy, thus bottlenecked by its exhaustion, and the ceiling is fully predictable $H=0, R=-a+b$. Our finding necessitates entropy management for continuous exploration toward scaling compute for RL. To this end, we investigate entropy dynamics both theoretically and empirically. Our derivation highlights that, the change in policy entropy is driven by the covariance between action probability and the change in logits, which is proportional to its advantage when using Policy Gradient-like algorithms. Empirical study shows that, the values of covariance term and entropy differences matched exactly, supporting the theoretical conclusion. Moreover, the covariance term stays mostly positive throughout training, further explaining why policy entropy would decrease monotonically. Through understanding the mechanism behind entropy dynamics, we motivate to control entropy by restricting the update of high-covariance tokens. Specifically, we propose two simple yet effective techniques, namely Clip-Cov and KL-Cov, which clip and apply KL penalty to tokens with high covariances respectively. Experiments show that these methods encourage exploration, thus helping policy escape entropy collapse and achieve better downstream performance.

### 中文翻译
本文旨在克服大语言模型（LLM）推理强化学习（Reinforcement Learning, RL）扩展过程中的一个主要障碍，即**策略熵崩溃 (Policy Entropy Collapse)**。在没有熵干预的大规模 RL 运行中，这一现象被普遍观察到：策略熵在训练早期急剧下降，而这种探索能力的减弱总是伴随着策略性能的饱和。在实践中，我们建立了熵 $H$ 与下游性能 $R$ 之间的转换方程 $R = -a \cdot e^H + b$。这一经验法则强烈表明，策略性能是通过消耗策略熵换取的，因此其上限受限于熵的枯竭，且最大性能上限是完全可预测的（即当 $H=0$ 时，$R = -a + b$）。我们的发现表明，为了在 RL 中实现持续的计算量扩展（Scaling Compute），必须进行熵管理以维持持续的探索。为此，我们从理论和实证两方面研究了熵动力学（Entropy Dynamics）。我们的推导表明，策略熵的变化是由**动作概率与 Logits 变化之间的协方差 (Covariance)** 驱动的，在使用类策略梯度（Policy Gradient）算法时，该协方差与优势函数（Advantage）成正比。实证研究表明，协方差项的值与熵差完全匹配，支持了理论结论。此外，在整个训练过程中，协方差项大部分时间保持为正，进一步解释了为什么策略熵会单调下降。通过理解熵动力学背后的机制，我们提出通过限制高协方差 Token 的更新来控制熵。具体而言，我们提出了两种简单而有效的技术，即 **Clip-Cov** 和 **KL-Cov**，它们分别对高协方差的 Token 进行裁剪和施加 KL 惩罚。实验表明，这些方法能够鼓励探索，从而帮助策略摆脱熵崩溃并获得更好的下游性能。

### 核心要点提炼
- **研究背景**：强化学习（RL）已成为提升大语言模型（LLM）推理能力的核心技术，但在大规模训练中普遍存在策略熵急剧下降、探索能力丧失的现象。
- **研究动机**：揭示策略熵与模型性能之间的定量关系，并从数学上推导策略熵衰减的根本动力学机制，进而设计有效的熵管理算法。
- **核心方法**：推导了熵动力学方程，发现 Logits 变化与动作概率的协方差驱动了熵衰减；提出了 **Clip-Cov** 和 **KL-Cov** 算法，通过限制高协方差 Token 的更新来防止熵崩溃。
- **主要结果**：在数学和代码推理任务上，所提方法成功延缓了熵崩溃，使模型在长序列探索中取得显著优于标准 PPO/GRPO 的下游性能。
- **研究意义**：为 LLM 推理强化学习的探索机制提供了坚实的理论基础，是实现 RL 训练 Scaling Laws 的关键一步。

---

## 研究背景与动机

### 领域现状
在以 OpenAI o1/o3、DeepSeek-R1 为代表的推理模型中，强化学习（RL）被广泛用于通过“思考时间”（Inference-time Compute）来提升模型的推理能力。通过大规模 RL 训练，模型能够自主学会回溯、自我纠错、多路径规划等高级推理行为。

### 现有方法的局限性
然而，在大规模 RL 训练（如 PPO、GRPO）中，研究人员经常观察到**策略熵崩溃 (Policy Entropy Collapse)** 的现象。模型在训练极早期就会迅速锁定在某些特定的 Token 输出路径上，导致策略熵迅速归零。一旦熵崩溃发生，模型就会丧失探索新路径的能力，下游性能（如数学、代码准确率）也随之停止增长，进入平台期。

### 研究动机
作者希望回答两个核心问题：
1. **策略熵与下游性能之间是否存在定量的数学关系？**
2. **策略熵衰减的底层数学机制是什么？我们如何主动控制它？**

---

## 研究问题

### 核心研究问题
如何定量刻画 LLM 强化学习中的策略熵演化，并设计无损于性能的熵保持机制，以实现持续的强化学习探索？

---

## 方法概述

### 核心思想
作者首先通过实证发现，策略熵 $H$ 与下游性能 $R$ 之间满足极其精确的指数转换方程：
$$R = -a \cdot e^H + b$$
这表明，性能的提升本质上是“消耗”策略熵（即从不确定到确定）的过程。

为了阻止熵的无节制衰减，作者对策略梯度算法下的熵变化进行了微分推导，发现策略熵的变化率 $\frac{dH}{dt}$ 与 **Logit 变化量与动作概率之间的协方差** 负相关：
$$\Delta H \approx - \sum_{x} \text{Cov}(\Delta \text{logit}(x, a), \pi(a|x))$$
由于在策略梯度更新中，高 Advantage 的动作其 Logit 增加，这导致协方差在训练中几乎恒为正，从而导致熵单调下降。

基于此，作者提出限制高协方差 Token 的更新：
1. **Clip-Cov**：对协方差过高的 Token 更新进行截断。
2. **KL-Cov**：在高协方差 Token 上施加局部的 KL 散度惩罚。

### 方法架构图

![[train_entropy_vs_covariance_page1.png|800]]
> 图1：策略熵变化与 Logits-概率协方差之间的精确匹配关系（实证验证了理论推导）。

---

## 实验结果

### 数据集
- **MATH**：经典数学推理数据集。
- **Codeforces / APPS**：代码生成与算法推理数据集。

### 主要结果

![[test_accuracy_klcov_page1.png|800]]
> 图2：使用 KL-Cov 算法后，模型在训练后期的测试准确率显著超越了标准 PPO 基线，成功突破了熵崩溃带来的性能瓶颈。

---

## 深度分析

### 研究价值评估
- **理论贡献**：首次推导出了策略梯度下策略熵演化的动力学公式，将熵崩溃问题归结为 Logits 与概率的协方差演化，具有极高的理论美感和学术价值。
- **实际应用价值**：针对大规模 RL 训练（如 GRPO/PPO），Clip-Cov 和 KL-Cov 提供了开箱即用的即插即用模块，能显著提升长程 RL 训练的稳定性。

### 局限性分析
- **计算开销**：计算每个 Token 的协方差需要额外的统计开销，在大规模分布式训练中可能会带来轻微的通信和计算延迟。

---

## 我的综合评价

### 价值评分
**9.2/10** - 这是一篇极具启发性的学术佳作，深刻揭示了大模型强化学习中“探索-利用”天平的底层数学机制，对于研究 Scaling Laws 和推理模型（Reasoning Models）的学者来说是必读文献。

> [!tip] 关键启示
> 模型的性能提升本质上是通过消耗策略熵换取的。通过控制高协方差 Token 的更新，我们能够优雅地延缓熵崩溃，从而让模型在更长的训练周期中保持探索能力。

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！这是大语言模型强化学习与推理优化领域的里程碑式工作。
