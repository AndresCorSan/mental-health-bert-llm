# Mental Health Disorders Detection using BERT and LLMs 

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/BERT-Fine--Tuning-blue?style=for-the-badge&logoColor=white" alt="BERT Fine-Tuning">
  <img src="https://img.shields.io/badge/LLMs-Zero--Shot-green?style=for-the-badge&logo=openai&logoColor=white" alt="LLMs Zero-Shot">
  <img src="https://img.shields.io/badge/NLP-Mental%20Health-red?style=for-the-badge&logo=health&logoColor=white" alt="Mental Health NLP">
</p>

This repository contains the research and technical development for the paper:  
**"Detection of Anxiety, Depression, and Eating Disorders Using Transformer-Based Language Models: A Comparative Study from BERT Encoders to Frontier LLMs"**.

> 📄 The preprint of the paper is available [here](paper/paper.pdf).

---

## 🎯 Project Overview

The main objective is to evaluate and compare the effectiveness of two leading approaches in Natural Language Processing (NLP) for the early detection of mental health disorders in Spanish:

*   **BERT Models:** Adjusted through supervised fine-tuning (Task-specialized models).
*   **State-of-the-art LLMs:** Used via zero-shot prompting (Leveraging broad general knowledge without any retraining).

The study focuses on three mental health disorders: **Anxiety, Depression, and Eating Disorders (ED)**, analyzing not only the predictive accuracy but also the stability, cost, and feasibility of applying these models in real-world environments.

---

## 🛠️ Evaluated Models

| Category | Models | Technique |
| :--- | :--- | :--- |
| **BERT Encoders** | DistilBETO, RoBERTa-large-BNE, mDeBERTa-v3 | Supervised Fine-tuning |
| **Open-Source LLMs** | Llama 3.1 8B, GPT-OSS-20B, DeepSeek-chat | Zero-shot Prompting |
| **Proprietary Frontier LLMs** | GPT-5.2, GPT-5 Mini, Claude Sonnet 4.6, Claude Haiku 4.5, Gemini 3.1 Pro, Gemini 3 Flash | Zero-shot Prompting |

---

## 📂 Repository Structure

All experimental development is contained within the **`code/`** directory. It includes Jupyter notebooks (`.ipynb`) used for the development of the project, each corresponding to a specific phase of the experimental development:

1.  **Exploratory Data Analysis** (`1_EDA_mentalriskes.ipynb`)
    *   Initial analysis of the MentalRiskES dataset.
    *   Study of class distribution, number of messages, and text length.
    *   Identification of imbalance and relevant features.

2.  **Data Engineering** (`2_data_preparation.ipynb`)
    *   Loading from JSON and advanced preprocessing.
    *   Generation of datasets for binary and multiclass classification.
    *   Division into training, validation, and test sets (BERT).
    *   Visualization of embeddings using **PCA** and **t-SNE**.

3.  **BERT Models Training** (`3_BERT_fine_tuning.ipynb`)
    *   Tokenization, model configuration, and supervised fine-tuning.
    *   Evaluation on the test set and generation of predictions.

4.  **LLM Inference (Zero-shot)** (`4_LLMs_ZeroShot/`)
    *   Notebooks: `OpenAI.ipynb`, `Claude.ipynb`, `Gemini.ipynb`, `DeepSeek.ipynb`, `OpenRouter.ipynb`.
    *   Dynamic prompt construction.
    *   API call management and response processing.
    *   Structured storage of predictions.

5.  **Results and Evaluation** (`5_results_evaluation/`)
    *   `5_1_metrics.ipynb`: Calculation of **Macro F1-score, Precision, Recall, MCC**, stability (**TARa@5**), and bootstrap confidence intervals.
    *   `5_2_confusion_matrix.ipynb`: Generation of confusion matrices.
    * `5_3_model_comparison.ipynb`: Statistical comparison between top-performing models and intra-family comparisons of proprietary LLMs (flagship vs. lightweight models) using paired bootstrap confidence intervals, the Approximate Randomization Test (ART), and the McNemar test.
    *   `5_4_cost_performance.ipynb`: Analysis of the **cost-performance** relationship.

---

## 📈 Key Results

### 🏆 Best Models by Task
* **Anxiety:** **GPT-OSS-20B** (**Macro F1 = 0.900**), substantially outperforming the best BERT encoder (RoBERTa-large-BNE, **0.680**).
* **Depression:** **Gemini 3.1 Pro** (**Macro F1 = 0.842**), tied with Gemini 3 Flash.
* **Eating Disorders (ED):** **GPT-5.2** (**Macro F1 = 0.979**), achieving the highest performance among all evaluated models.
* **Multiclass:** **Gemini 3 Flash** (**Macro F1 = 0.849**), narrowly outperforming Gemini 3.1 Pro and RoBERTa-large-BNE.

### 💡 Key Findings
* **LLMs achieved the highest predictive performance** across all classification tasks, although the performance gap over the best BERT encoder was generally small except for anxiety detection.
* **RoBERTa-large-BNE remained highly competitive**, matching or outperforming several proprietary frontier LLMs in the multiclass task.
* **Eating disorder detection was consistently the easiest task**, with all top-performing models achieving Macro F1-scores above 0.94.
* **Prediction stability was high across all evaluated LLMs**, with TARa@5 values consistently above **0.85** despite their non-deterministic inference.
* **Most differences between the strongest models were not statistically significant**, highlighting the importance of complementing performance metrics with statistical testing.
* **Cost-performance analysis showed that RoBERTa-large-BNE and Gemini 3 Flash offer the most favorable trade-offs** between predictive performance and computational cost.

---

## 📊 Dataset

This work uses the **MentalRiskES** dataset (SINAI Group, University of Jaén), composed of Spanish texts from public Telegram groups, annotated for the detection of:
- Anxiety
- Depression
- Eating Disorders (ED)

⚠️ *For privacy and licensing reasons, the dataset is not included. You can request access in the [official SINAI repository](https://github.com/sinai-uja/corpusMentalRiskES.git).*

---

## 🚀 Execution Requirements

The project is designed to run primarily on **Google Colab**, taking advantage of its GPU infrastructure.

### 💻 Environment and Hardware
*   **Language:** Python 3.9+
*   **Hardware:** An **NVIDIA T4 GPU or better** is recommended for training and evaluating BERT models.

### 📦 Main Libraries
Most of the required libraries are already pre-installed in Google Colab. Where additional installation is required, installation cells (`!pip install`) are included at the beginning of each notebook.

### 🔑 Model APIs (LLMs)
To replicate the inference phase with large-scale models, you need to configure the API keys from the following providers:
- **OpenAI** | **Anthropic** | **Google AI Studio** | **DeepSeek** | **OpenRouter**

---

## 👥 Authors

* **Andrés Corbacho-Sánchez** ([GitHub Profile](https://github.com/AndresCorSan))
  * *Conceptualization, Methodology, Software, Investigation, Data curation, Formal analysis, Validation, Visualization, Writing – original draft, Writing – review & editing.*
* **Adrián Domínguez-Díaz**
  * *Conceptualization, Methodology, Supervision, Validation, Writing – review & editing.*
* **Luis de-Marcos**
  * *Conceptualization, Methodology, Supervision, Validation.*

---

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**. 
See the `LICENSE` file for details.