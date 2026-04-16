# 🚀 Hybrid Prompt Generator (Hackathon Prototype)

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Tests](https://img.shields.io/badge/tests-111%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![GitHub stars](https://img.shields.io/github/stars/parth-shinge/Hybrid-Prompt-Generator?style=social)

> **AI-powered system for generating high-quality prompts using hybrid intelligence and user feedback**

🔗 Repository:
https://github.com/parth-shinge/Hybrid-Prompt-Generator

---

## 🏆 Project Context

This project is being developed and presented as part of a hackathon.

The goal is to build a smart prompt generation system that:

* Improves AI output quality
* Learns from user behavior
* Adapts over time

This version focuses on:

* A working prototype
* Real-time prompt generation
* Intelligent ranking and selection
* Scalability and real-world application

---

## 🧠 Overview

Hybrid Prompt Generator is an AI-based system that combines:

* Template-based prompt generation
* LLM-powered generation (Gemini)
* Machine learning-based ranking

to generate high-quality prompts for tools like **Canva, Gamma, and ChatGPT**.

The system uses **Human-in-the-Loop learning**, meaning it improves continuously based on user selections.

---

## 🏗 System Architecture

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
   🔁 Continuous Learning Loop
         │
         ▼
   🔍 Explainability Layer (SHAP)
```

---

## ✨ Key Features

### ⚡ Hybrid Prompt Generation

Combines template-based and AI-generated prompts to produce better results.

### 🧠 Neural Ranking Model

Learns user preferences and selects the best prompt automatically.

### 👤 Human-in-the-Loop Learning

Every user choice improves the system over time.

### 📊 Ensemble Scoring

Evaluates prompt quality using:

```
Final Score = α × SlotCoverage + β × Fluency
```

### 🔍 Explainability

Provides insights into why a prompt was selected.

---

## 🖥 Working Prototype

* ✅ Fully functional Streamlit application
* ✅ Supports multiple tools (Canva, Gamma, etc.)
* ✅ Hybrid generation (Template + AI)
* ✅ Intelligent prompt selection
* ✅ Real-time user interaction

---

## 📊 Technical Approach

* **Frontend:** Streamlit
* **Backend:** Python
* **LLM Integration:** Gemini API
* **Embeddings:** Sentence Transformers
* **Model:** Neural Network (MLP)
* **Database:** SQLite
* **Explainability:** SHAP

---

## 💼 Business & Impact

### 🚀 Use Cases

* Students & educators
* Content creators
* Marketers & designers

### 💡 Value

* Reduces time spent writing prompts
* Improves AI-generated content quality
* Makes AI tools easier to use for non-experts

### 💰 Potential

* SaaS platform
* API for AI tools
* Integration with design and productivity tools

---

## 🌍 SDG Alignment

* 🎓 **Quality Education (SDG 4)** — helps students create better content
* 💼 **Decent Work (SDG 8)** — empowers creators and freelancers
* 🏭 **Industry & Innovation (SDG 9)** — builds smarter AI systems
* 🌐 **Reduced Inequalities (SDG 10)** — makes AI accessible to everyone

---

## ▶️ Running the Application

```bash
git clone https://github.com/parth-shinge/Hybrid-Prompt-Generator

cd Hybrid-Prompt-Generator

python -m venv .venv

source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt

streamlit run prompt_generator.py
```

---

## 📂 Project Structure

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
├── utils/
```

---

## 🔁 Continuous Improvement

The system improves through:

* User interaction data
* Model retraining
* Adaptive ranking

Making it a **learning system, not a static tool**.
