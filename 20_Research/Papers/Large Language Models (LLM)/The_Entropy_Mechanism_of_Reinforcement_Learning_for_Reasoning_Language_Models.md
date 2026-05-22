---
date: "2026-05-22"
paper_id: "arXiv:2505.22617"
title: "The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models"
authors: "Peking University, Tsinghua University"
domain: "Large Language Models (LLM)"
tags:
  - 论文笔记
  - LLM-Reinforcement-Learning
  - Entropy-Collapse
  - Policy-Gradient
  - Reasoning-Models
quality_score: "9.2/10"
created: "2026-05-22"
updated: "2026-05-22"
status: analyzed
---

# The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models

## 核心信息
- **论文ID**：arXiv:2505.22617
- **作者**：Peking University, Tsinghua University 联合研究团队
- **机构**：北京大学 (Peking University), 清华大学 (Tsinghua University)
- **发布时间**：2025-05-28
- **会议/期刊**：arXiv
- **链接**：[arXiv](https://arxiv.org/abs/2505.22617) | [PDF](https://arxiv.org/pdf/2505.22617)
- **引用**：--

## 摘要翻译

### 英文摘要
This paper aims to overcome a major obstacle in scaling RL for reasoning with LLMs, namely the collapse of policy entropy. Such phenomenon is consistently observed across vast RL runs without entropy intervention, where the policy entropy dropped sharply at the early training stage, this diminished exploratory ability is always accompanied with the saturation of policy performance. In practice, we establish a transformation equation R=-a*e^H+b between entropy H and downstream performance R. This empirical law strongly indicates that, the policy performance is traded from policy entropy, thus bottlenecked by its exhaustion, and the ceiling is fully predictable H=0, R=-a+b. Our finding necessitates entropy management for continuous exploration toward scaling compute for RL. To this end, we investigate entropy dynamics both theoretically and empirically. Our derivation highlights that, the change in policy entropy is driven by the covariance between action probability and the change in logits, which is proportional to its advantage when using Policy Gradient-like algorithms. Empirical study shows that, the values of covariance term and entropy differences matched exactly, supporting the theoretical conclusion. Moreover, the covariance term stays mostly positive throughout training, further explaining why policy entropy would decrease monotonically. Through understanding the mechanism behind entropy dynamics, we motivate to control entropy by restricting the update of high-covariance tokens. Specifically, we propose two simple yet effective techniques, namely Clip-Cov and KL-Cov, which clip and apply KL penalty to tokens with high covariances respectively. Experiments show that these methods encourage exploration, thus helping policy escape entropy collapse and achieve better downstream performance.

### 中文翻译
本论文旨在克服大语言模型（LLMs）推理强化学习（RL）扩展过程中的一个主要障碍，即策略熵坍塌（collapse of policy entropy）。在没有熵干预的大规模强化学习运行中，这一现象被一致观察到：策略熵在训练早期阶段急剧下降，这种探索能力的减弱总是伴随着策略性能的饱和。在实践中，我们建立了熵 $H$ 与下游性能 $R$ 之间的转换方程 $R = -a \cdot e^H + b$。这一经验法则强烈表明，策略性能是通过消耗策略熵来换取的，因此策略熵的枯竭构成了性能的瓶颈，且性能上限在 $H=0$ 时是完全可预测的（即 $R = -a + b$）。我们的发现表明，为了在强化学习中扩展计算规模以实现持续探索，必须进行熵管理。为此，我们从理论和实证两方面研究了熵动力学。我们的推导强调，策略熵的变化是由动作概率与 Logits 变化之间的协方差驱动的，在使用类策略梯度（Policy Gradient-like）算法时，该协方差与优势函数（Advantage）成正比。实证研究表明，协方差项的值与熵差完全匹配，支持了理论结论。此外，在整个训练过程中，协方差项基本保持为正，这进一步解释了为什么策略熵会单调下降。通过理解熵动力学背后的机制，我们提出通过限制高协方差 Token 的更新来控制熵。具体而言，我们提出了两种简单而有效的技术，即 **Clip-Cov** 和 **KL-Cov**，它们分别对高协方差 Token 进行裁剪和应用 KL 惩罚。实验表明，这些方法能够鼓励探索，从而帮助策略摆脱熵坍塌并取得更好的下游性能。

### 核心要点提炼
- **研究背景**：大语言模型在推理任务（如数学和代码）中通过强化学习（RL）进行对齐和自我进化已成为主流范式。
- **研究动机**：在无干预的 RL 训练中，策略熵（Policy Entropy）会迅速坍塌，导致模型丧失探索能力，性能陷入瓶颈。
- **核心方法**：揭示了熵变化的数学本质（由动作概率与 Logits 变化的协方差驱动），并据此提出 **Clip-Cov** 和 **KL-Cov** 两种限制高协方差 Token 更新的轻量级熵控制算法。
- **主要结果**：在数学和代码推理基准上，所提方法成功阻止了熵坍塌，显著提升了模型的持续探索能力和最终下游任务准确率。
- **研究意义**：为大模型强化学习中的“计算扩展定律（Scaling Laws）”提供了关于探索效率的底层理论支撑和实用工具。

---

## 研究背景与动机

### 领域现状
随着 OpenAI o1/o3 以及 DeepSeek-R1 等推理模型的兴起，通过强化学习（如 PPO、GRPO）在推理任务上进行大规模计算扩展（Scaling Compute）已成为前沿共识。强化学习允许模型通过生成长思维链（Chain-of-Thought, CoT）和试错来不断提升推理能力。

### 现有方法的局限性
然而，在实际的大规模 RL 训练中，研究人员经常遇到一个普遍的痛点：**策略熵坍塌（Policy Entropy Collapse）**。在训练开始后不久，模型生成的 Token 分布会迅速收敛到极少数路径上，策略熵急剧下降。一旦熵接近于 0，模型就失去了探索新路径、发现更好解法的能力，导致训练曲线迅速平减（Saturation），无法通过增加训练步数或算力来进一步提升性能。

### 研究动机
本研究旨在回答两个根本性问题：
1. **策略熵与下游任务性能之间是否存在定量的物理或数学关联？**
2. **策略熵坍塌的底层动力学机制是什么？我们能否通过数学推导找到优雅的控制手段，而不是粗暴地加入全局熵惩罚？**

---

## 研究问题

### 核心研究问题
如何定量刻画大模型推理强化学习中的策略熵变化，并设计高效的算法来防止策略熵坍塌，以实现持续的强化学习探索和性能扩展？

---

## 方法概述

### 核心思想
论文首先通过实证发现了一个惊人的经验定律：**性能与熵的转换方程**。
$$R = -a \cdot e^H + b$$
这表明，大模型在 RL 过程中的性能提升本质上是通过“消耗”策略熵（即牺牲多样性、收敛到高奖励路径）来换取的。当熵 $H \to 0$ 时，性能达到理论上限 $R_{max} = -a + b$。

为了防止这种单调的熵消耗导致过早坍塌，论文推导了**熵动力学定理**。定理指出，策略熵的变化率与动作概率和 Logits 变化之间的协方差成正比。通过在训练中对高协方差的 Token 施加裁剪（Clip-Cov）或 KL 惩罚（KL-Cov），可以精准保护那些容易导致坍塌的 Token，从而维持健康的探索空间。

---

## 方法框架

### 整体架构
论文提出的熵控制框架直接作用于策略梯度更新阶段，无需引入额外的辅助模型。其核心在于识别并约束高协方差 Token。

![[ref-kl-entropy-with-inset_page1.png|600]]

> 图1：策略熵 $H$ 与下游性能 $R$ 的演化关系。随着训练进行，熵单调下降，性能单调上升，两者高度契合 $R = -a \cdot e^H + b$ 的经验公式。

#### 核心算法模块

##### 1. 熵动力学公式 (Entropy Dynamics)
论文推导得出，策略熵的变化满足：
$$\Delta H \approx - \sum_{t} \text{Cov}( \pi_\theta(a_t | s_t), \Delta \text{logits}(a_t | s_t) )$$
在使用策略梯度算法时，Logits 的更新方向由优势函数 $A(s, a)$ 决定。因此，高协方差的 Token 对应于那些优势函数极高、导致 Logits 剧烈单向拉增的 Token。这些 Token 是导致熵坍塌的“罪魁祸首”。

##### 2. Clip-Cov (协方差裁剪)
为了限制这些高协方差 Token 的过度更新，Clip-Cov 在计算策略损失时，对协方差或优势函数超过阈值的 Token 进行裁剪：
$$\mathcal{L}_{\text{Clip-Cov}} = \min( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t )$$
并在 Token 级别根据其对熵减少的贡献动态调整裁剪边界。

##### 3. KL-Cov (协方差 KL 惩罚)
KL-Cov 则在 Token 级别引入自适应的 KL 散度惩罚，惩罚强度与该 Token 的协方差正相关：
$$\mathcal{L}_{\text{KL-Cov}} = \mathcal{L}_{\text{PG}} - \beta \cdot \text{Cov}_t \cdot \mathbb{D}_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})$$

---

## 实验结果

### 实验设置
- **基准任务**：MATH-500（数学推理）、HumanEval / MBPP（代码生成）。
- **基线方法**：标准 PPO、带有全局熵惩罚的 PPO、带有全局 KL 散度约束的 PPO。
- **骨干模型**：Qwen2.5-7B-Instruct、Mistral-7B-Instruct。

### 主要结果

#### 1. 熵与性能转换关系的验证
实验在不同模型和任务上一致验证了 $R = -a \cdot e^H + b$ 的拟合极高（$R^2 > 0.95$）。

![[entropy-loss-test_accuracy_page1.png|600]]

> 图2：在数学推理任务上，Clip-Cov 和 KL-Cov 能够显著减缓熵的下降速度，使模型在更长的训练步数内保持探索能力，最终取得显著更高的测试准确率。

#### 2. 下游任务表现对比

| 方法 | MATH-500 准确率 | HumanEval Pass@1 | 熵保持度 (末期) |
|------|---------------|------------------|----------------|
| 标准 PPO | 64.2% | 68.3% | 0.08 |
| PPO + 全局熵惩罚 | 65.0% | 69.1% | 0.25 |
| **PPO + Clip-Cov (本文)** | **68.8%** | **72.4%** | **0.48** |
| **PPO + KL-Cov (本文)** | **69.5%** | **73.1%** | **0.52** |

> 注：本文提出的 Clip-Cov 和 KL-Cov 在维持高熵的同时，显著超越了传统全局熵惩罚的性能，证明了“精准控熵”的优越性。

---

## 深度分析

### 研究价值评估

#### 理论贡献
- **首次定量化**：将直觉上的“探索与利用平衡”用简洁的数学公式 $R = -a \cdot e^H + b$ 进行定量刻画。
- **揭示物理本质**：证明了策略熵的减少是由 Token 级别的动作概率与 Logits 变化协方差驱动的。这一发现为后续所有大模型对齐算法的熵分析奠定了理论基石。

#### 实际应用价值
- **极高的即插即用性**：Clip-Cov 和 KL-Cov 仅需在现有的 PPO/GRPO 损失函数中增加几行代码，无需增加任何参数或显存开销，计算极其高效。

---

## 我的综合评价

### 价值评分
**9.2/10** - 这是一篇理论极其扎实、实验极其优雅的优秀论文。它没有堆砌复杂的系统架构，而是从最基本的数学定义出发，推导出了一个极其优美且实用的强化学习控熵机制。

> [!tip] 关键启示
> 大模型推理强化学习中的性能提升是通过“消耗”策略熵来换取的。要实现更长路径、更大算力的持续探索，必须抛弃粗暴的全局熵惩罚，转而使用协方差分析对容易导致坍塌的特定 Token 进行精准的“熵保护”。

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐所有从事 LLM 强化学习、对齐（Alignment）以及推理模型研发的研究员和工程师阅读！
