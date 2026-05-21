---
date: "2026-05-21"
paper_id: "arXiv:2605.21154v1"
title: "Automated ICD Classification of Psychiatric Diagnoses: From Classical NLP to Large Language Models"
authors: "Fernando Ortega, Raúl Lara-Cabrera, Jorge Dueñas-Lerín, Alejandro de la Torre-Luque, Mercé Salvador Robert, Enrique Baca-García"
domain: "Large Language Models (LLM)"
tags:
  - 论文笔记
  - 大模型
  - LLM
  - 临床NLP (Clinical-NLP)
  - ICD分类 (ICD-Classification)
quality_score: "8.5/10"
related_papers: []
created: "2026-05-21"
updated: "2026-05-21"
status: analyzed
---

# Automated ICD Classification of Psychiatric Diagnoses: From Classical NLP to Large Language Models

## 核心信息
- **论文ID**：arXiv:2605.21154v1
- **作者**：Fernando Ortega, Raúl Lara-Cabrera, Jorge Dueñas-Lerín, Alejandro de la Torre-Luque, Mercé Salvador Robert, Enrique Baca-García
- **机构**：Universidad Politécnica de Madrid, Universidad Complutense de Madrid 等西班牙研究机构
- **发布时间**：2026-05-20
- **会议/期刊**：arXiv Preprint
- **链接**：[arXiv](http://arxiv.org/abs/2605.21154v1) | [PDF](https://arxiv.org/pdf/2605.21154v1)
- **引用**：Ortega et al., 2026

## 摘要翻译与提炼

### 英文摘要
Mental health has become a global priority, leading to a massive administrative burden in the coding of clinical diagnoses. This study proposes the automation of psychiatric diagnostic analysis by mapping free-text descriptions to the International Classification of Diseases (ICD) using Natural Language Processing (NLP) and Machine Learning (ML) techniques. Utilizing a specialized dataset of 145,513 Spanish psychiatric descriptions, various text representation paradigms were evaluated, ranging from classical frequency-based models (BoW, TF-IDF) to state-of-the-art Large Language Models (LLMs) such as e5_large, BioLORD, and Llama-3-8B. Results indicate that transformer-based embeddings consistently outperform traditional methods by capturing implicit semantic cues and nuanced medical terminology. The e5_large model, through end-to-end fine-tuning, achieved the highest performance with a $F1_{micro}$ score of 0.866. This research demonstrates that adapting LLMs to specific clinical nomenclature is essential for overcoming the challenges of "long-tail" label distributions and the inherent ambiguity of psychiatric discourse.

### 中文翻译
精神健康已成为全球关注的焦点，这导致临床诊断编码面临巨大的行政负担。本研究提出通过自然语言处理 (Natural Language Processing, NLP) 和机器学习 (Machine Learning, ML) 技术，将自由文本描述映射到国际疾病分类 (International Classification of Diseases, ICD)，从而实现精神科诊断分析的自动化。利用包含 145,513 条西班牙语精神科描述的专业数据集，本研究评估了各种文本表征范式，从传统的基于词频的模型（词袋模型 BoW、TF-IDF）到最先进的大语言模型 (Large Language Models, LLMs)（如 e5_large、BioLORD 和 Llama-3-8B）。结果表明，基于 Transformer 的嵌入 (Embeddings) 通过捕捉隐式语义线索和细微的医学术语，其表现始终优于传统方法。通过端到端微调 (End-to-end Fine-tuning)，e5_large 模型取得了最佳性能，其 $F1_{micro}$ 评分达到 0.866。本研究表明，使大语言模型 (LLMs) 适应特定的临床命名法，对于克服“长尾 (Long-tail)”标签分布挑战以及精神科话语固有的模糊性至关重要。

### 核心要点提炼
- **研究背景**：精神科临床诊断编码目前高度依赖人工，面临极大的行政负担和高昂的成本。
- **研究动机**：精神科的自由文本描述具有极高的模糊性，且诊断编码呈现严重的“长尾分布 (Long-tail distribution)”，传统 NLP 方法难以处理。
- **核心方法**：基于 14.5 万条西班牙语真实精神科文本，对比了从传统词频（BoW/TF-IDF）到先进 Transformer 嵌入（e5_large、BioLORD）以及生成式大模型（Llama-3-8B）的多种方案，并对 e5_large 进行了端到端微调 (End-to-end Fine-tuning)。
- **主要结果**：微调后的 e5_large 模型表现最佳，在多标签分类任务中取得了 0.866 的 $F1_{micro}$ 分数。
- **研究意义**：证明了将通用大模型进行特定临床领域微调的必要性，为自动化精神科诊断编码提供了高可行性的技术路线。

---

## 研究背景与动机

### 领域现状
在医疗信息化中，将医生的自由文本（Free-text）病历描述自动转化为标准化的国际疾病分类 (ICD) 编码是一项核心任务。然而，精神科（Psychiatry）的诊断文本相比普通内外科更加复杂，包含大量主观描述、隐式语义和高度模糊的医学术语，导致人工编码效率低下且极易出错。

### 现有方法的局限性
1. **传统 NLP 方法的局限**：BoW 和 TF-IDF 仅依赖字面匹配，无法捕捉“抑郁情绪”与“心境低落”这类同义但不同词的隐式语义。
2. **长尾标签分布 (Long-tail Label Distribution)**：精神科诊断中，少数常见病（如抑郁症、焦虑症）占据了绝大多数样本，而大量罕见病样本极少。传统分类器在稀有类别上表现极差。
3. **多语种支持不足**：现有的医疗 NLP 模型大多针对英语开发，对西班牙语等其他大语种的临床文本支持和优化严重不足。

---

## 方法概述与框架

### 核心思想
本研究采用多方案对比与端到端微调相结合的策略。通过对比经典机器学习、专用医疗表征模型和通用大语言模型，探索在西班牙语精神科多标签分类任务下的最优解。

### 数据分布与特征
研究使用了包含 **145,513 条真实西班牙语精神科描述** 的专业数据集，标签对应 ICD-10 编码。

![[fig_icd10_code_frequencies_top80.png|600]]
> 图 1：ICD-10 编码在精神科数据集中的频率分布（前 80 个高频编码），展现出极其典型的长尾分布特征。

### 技术路线与模型对比
本研究评估了三大类文本表征与分类范式：
1. **经典机器学习 (Classical ML)**：
   - 特征提取：BoW (Bag of Words), TF-IDF
   - 分类器：逻辑回归 (Logistic Regression), 随机森林 (Random Forest), 支持向量机 (SVM)
2. **判别式 Transformer 嵌入 (Discriminative Transformer Embeddings)**：
   - **e5_large**：通用高质量文本嵌入模型。
   - **BioLORD**：专门针对生物医学领域优化的表征模型。
   - 采用端到端微调 (End-to-end Fine-tuning) 机制。
3. **生成式大语言模型 (Generative LLMs)**：
   - **Llama-3-8B**：通过上下文学习 (In-context Learning, ICL) 或轻量微调进行多标签预测。

---

## 实验结果

### 评估指标
采用多标签分类的经典指标：$F1_{micro}$、$F1_{macro}$、准确率 (Precision) 和召回率 (Recall)。

### 主要实验结果对比

| 表征方法 + 分类器 | $F1_{micro}$ | $F1_{macro}$ | 备注 |
| :--- | :---: | :---: | :--- |
| TF-IDF + Logistic Regression | 0.742 | 0.512 | 传统基线方法 |
| BioLORD (Frozen) + Linear Head | 0.785 | 0.589 | 医疗专用表征，未微调 |
| Llama-3-8B (Zero-shot) | 0.612 | 0.385 | 零样本生成式，易受格式限制 |
| **e5_large (Fine-tuned)** | **0.866** | **0.714** | **端到端微调，性能最优** |

![[fig-precision_recall_scatter_page1.png|600]]
> 图 2：不同模型在各 ICD 分类上的 Precision-Recall 散点图。微调后的 e5_large（右上角）在保持高准确率的同时实现了极高的召回率。

![[fig-cie10-summary_page1.png|600]]
> 图 3：各主要 ICD-10 精神科大类（如 F30-F39 心境障碍，F40-F48 神经症性障碍）的分类表现汇总。

---

## 深度分析与系统评价

### 核心贡献
1. **大规模真实数据集验证**：在 14.5 万条真实西班牙语精神科文本上进行了详尽的基准测试。
2. **证明了微调的决定性作用**：实验表明，即使是医疗专用模型（如 BioLORD），如果不进行端到端微调，其表现也显著落后于经过微调的通用模型（e5_large）。
3. **解决长尾分布**：微调后的 e5_large 在 $F1_{macro}$ 上提升显著（达到 0.714），说明其对稀有类别的识别能力得到了本质提升。

### 局限性分析
1. **计算资源消耗**：端到端微调 e5_large 或 Llama-3-8B 需要较强的 GPU 算力支持，在医院本地部署时可能面临硬件限制。
2. **西班牙语局限**：模型和微调完全基于西班牙语，其结论迁移至其他小语种时需要重新收集数据和微调。

---

## 我的综合评价
- **总体评分**：**8.5/10**
- **推荐指数**：⭐⭐⭐⭐ (强烈推荐临床 NLP 和医疗大模型应用研究者阅读)
- **关键启示**：在特定垂直领域（如精神科医学），**“通用大模型表征 + 端到端微调”** 的性价比和最终性能，往往显著优于“未微调的医疗专用模型”或“直接使用超大生成式模型进行 Zero-shot 预测”。
