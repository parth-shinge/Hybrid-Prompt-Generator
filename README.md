# 🚀 Hybrid Prompt Generator

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Tests](https://img.shields.io/badge/tests-111%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Conference](https://img.shields.io/badge/ICTCS-2025-orange)
![GitHub stars](https://img.shields.io/github/stars/parth-shinge/Hybrid-Prompt-Generator?style=social)

> **Official Implementation of the Research Paper**
> *A Hybrid Framework for Adaptive Prompt Generation Using Templates, LLMs, and Learned Rankers*

📖 Published in **ICT: Applications and Social Interfaces — Proceedings of ICTCS 2025, Volume 3 (Springer LNNS)**

🔗 Repository:
https://github.com/parth-shinge/Hybrid-Prompt-Generator

---

# 🧠 Overview

Hybrid Prompt Generator is a research-driven framework that combines **template-based prompt generation**, **LLM augmentation**, and **machine learning ranking models** to generate high-quality prompts for creative tools such as **Canva, Gamma, and other AI design platforms**.

The system integrates **Human-in-the-Loop learning**, enabling the model to continuously improve prompt quality based on user selections.

---

# 🏗 System Architecture

```
User Input
   │
   ├── Template Generator
   │
   ├── Gemini Generator
   │
   └── Hybrid Mode
         │
         ▼
   🧠 Ensemble Prompt Synthesis
      (Slot Coverage + Fluency)
         │
         ▼
   🔎 Neural Ranker
   (384 → 128 → 64 → 1)
         │
         ▼
   👤 User Choice Logging
         │
         ▼
   📊 Dataset Creation
         │
         ▼
   📈 Evaluation + Statistical Testing
         │
         ▼
   🔍 SHAP Interpretability
```

---

# ✨ Key Features

### 🧩 Template Prompt Generator

Deterministic prompt construction using **7 structured design parameters**.

### 🤖 Gemini Integration

LLM-powered prompt generation using **Google Gemini API**.

### ⚡ Hybrid Generation Mode

Generates prompts from both systems and selects the best automatically.

### 🧠 Ensemble Prompt Synthesis

Prompt quality scoring:

```
Final Score = α × SlotCoverage + β × Fluency
```

### 📊 Neural Ranker

Binary classifier trained on **user choice data**.

Architecture:

```
Embedding (384)
 → Linear(128)
 → ReLU
 → Dropout(0.2)
 → Linear(64)
 → ReLU
 → Linear(1)
 → Sigmoid
```

Embedding model: **all-MiniLM-L6-v2**

---

# 👤 Human-in-the-Loop Learning

User selections are logged into a **SQLite database**, which is converted into a dataset for training the neural ranker.

The system continuously improves as more user feedback is collected.

---

# 📊 Evaluation Protocol

The repository includes a full ML evaluation pipeline.

### Models Compared

• Random Baseline
• Popularity Baseline
• TF-IDF + Logistic Regression
• Embedding + Logistic Regression
• Neural Ranker

### Metrics

• Accuracy
• Precision
• Recall
• F1 Score
• ROC-AUC

Evaluation uses:

• **5-Fold Stratified Cross Validation**
• **Held-out Test Set**

---

# 📈 Statistical Significance Testing

To validate experimental results, the following statistical tests are implemented:

🧪 McNemar Test
🧪 Wilcoxon Signed-Rank Test
🧪 Bootstrap Confidence Intervals

Results saved to:

```
results/statistical_tests.json
```

---

# 🔍 SHAP Interpretability

To improve transparency, the neural ranker supports explainability using **SHAP**.

### Global Explanations

Feature importance across the dataset.

### Local Explanations

Explains **why the model preferred one prompt over another**.

Accessible via the **Admin Dashboard**.

---

# 🖥 Admin Dashboard

The admin panel provides:

📊 System analytics
🧠 Ranker retraining
🔍 SHAP visualization
🗃 Dataset inspection

---

# ⚙️ Installation

```bash
git clone https://github.com/parth-shinge/Hybrid-Prompt-Generator

cd Hybrid-Prompt-Generator

python -m venv .venv

source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

# ▶️ Running the Application

```bash
streamlit run prompt_generator.py
```

---

# 🧠 Training the Neural Ranker

```python
from neural_ranker import train_ranker
from database import get_choice_dataset

pairs = get_choice_dataset()

texts = [t for t,l in pairs]
labels = [l for t,l in pairs]

train_ranker(texts, labels)
```

---

# 📂 Project Structure

```
Hybrid-Prompt-Generator/
│
├── prompt_generator.py
├── ensemble_synthesis.py
├── neural_ranker.py
├── eval_protocol.py
├── statistical_tests.py
├── shap_explain.py
├── database.py
├── config.yaml
│
├── tests/
├── results/
├── models/
└── utils/
```

---

# 🔁 Reproducibility

The project ensures reproducible experiments through:

• Deterministic seeding
• Dataset hashing
• Experiment tracking
• Git commit logging
• Config-based hyperparameters

Each experiment logs:

```
dataset hash
git commit
random seed
config snapshot
timestamp
```

---

# 📜 Citation

If you use this work in your research, please cite the following:

### BibTeX

```bibtex
@inproceedings{shinge2026hybridprompt,
  title     = {A Hybrid Framework for Adaptive Prompt Generation Using Templates, LLMs, and Learned Rankers},
  author    = {Parth Shinge},
  booktitle = {ICT: Applications and Social Interfaces},
  series    = {Lecture Notes in Networks and Systems},
  publisher = {Springer Nature Switzerland AG},
  year      = {2026},
  note      = {Proceedings of the 10th International Conference on Information and Communication Technology for Competitive Strategies (ICTCS-2025)}
}
```

### Author

Parth Shinge
Vishwakarma Institute of Technology, Pune, India

ORCID: https://orcid.org/0009-0007-3790-2373

---

# ⭐ Acknowledgement

This work was presented at:

**10th International Conference on Information and Communication Technology for Competitive Strategies (ICTCS-2025)**

and published in **Springer Lecture Notes in Networks and Systems (LNNS)**.

---

# 📜 License

This repository is released for **academic and research purposes**.
