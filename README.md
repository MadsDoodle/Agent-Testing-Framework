# 🧪 AI Agent Evaluation Framework

**Built by Madhav S Baidya**

An agent-agnostic evaluation framework that tests any AI agent using
structured test cases, automated multi-layer evaluation, and adversarial
robustness testing — with a live Gradio dashboard.

---

## 🏗️ Architecture

```
Test Cases (41 total)
       ↓
TestCaseLoader + ResponseCollector
       ↓
ExecutionEngine  →  Agent (LLMAgent / HTTPAgent / DemoAgent)
       ↓
5-Layer Evaluation Engine
  L1 Rule-Based   → refusal detection, keyword safety, length
  L2 LLM Judge    → correctness / relevance / safety (1-5)
  L3 Semantic Sim → cosine similarity vs ground truth
  L4 Toxicity     → Detoxify score
  L5 RAG Grounding→ context overlap + source hit
       ↓
EnhancedMetricsAggregator
  Accuracy / Relevance / Safety / Robustness / Toxicity Rate
  Final Score = 0.35*Acc + 0.20*Rel + 0.20*Safe + 0.15*Rob + 0.10*(1-Tox)
       ↓
ReportingLayer  →  JSON report + Markdown report
ObservabilityLayer → structured .log + events JSON
```

---

## 📁 Repository Structure

```
repo/
├── src/
│   ├── config.py            # CFG dict, directory setup
│   ├── test_cases.py        # 25 standard + 16 adversarial test cases
│   ├── rag_pipeline.py      # DocumentChunker, RAGRetriever, FAISS index
│   ├── agents.py            # AgentInterface, LLMAgent, HTTPAgent, DemoAgent
│   ├── test_loader.py       # TestCaseLoader, ResponseCollector
│   ├── execution_engine.py  # ExecutionEngine
│   ├── evaluators.py        # FiveLayerEvaluator (L1–L5)
│   ├── adversarial.py       # 16 adversarial cases, AdversarialEvaluator
│   ├── metrics.py           # EnhancedMetricsAggregator
│   ├── reporting.py         # JSON + Markdown report export
│   ├── observability.py     # Structured logging layer
│   ├── visualization.py     # Evaluation plots
│   ├── model_utils.py       # MODEL_REGISTRY, load_model
│   ├── drive_utils.py       # Google Drive save/load utilities
│   ├── flask_server.py      # Local model HTTP server
│   └── pipeline.py          # run_local_pipeline orchestrator
├── app.py                   # Gradio UI (main entry)
├── demo_run.py              # Demo script for submission
├── notebooks/
│   └── Agent_testing_framework.ipynb
├── data/                    # Generated at runtime
├── evaluation/results/      # Generated at runtime
├── requirements.txt
├── README.md
└── ARCHITECTURE.md
```

---

## 🚀 Quick Start (Google Colab T4)

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the full notebook**
```
notebooks/06_Agent_testing_framework.ipynb
```
Run cells top to bottom. The notebook loads the model, builds the RAG
index, and launches the Gradio UI automatically.

**3. Use the Gradio interface**

| Field | Value |
|-------|-------|
| URL | `http://localhost:5050/v1/chat/completions` |
| API Format | `openai` |
| API Key | *(blank for local)* |
| Model Name | `qwen3b` |

Select categories/difficulties → click **🚀 START EVALUATION**

---

## 🤖 Supported Models

| Key | Model | Size |
|-----|-------|------|
| `qwen3b` | Qwen/Qwen2.5-3B-Instruct | 3B |
| `phi35` | microsoft/Phi-3.5-mini-instruct | 3.8B |
| `gemma2b` | google/gemma-2-2b-it | 2B |

All fit on a T4 GPU (16GB) with float16.

---

## 🔌 Plug In Any Agent

**Local HF model** — run `flask_server.py`, point URL to `localhost:5050`

**GPT-4**
```
URL    : https://api.openai.com/v1/chat/completions
Format : openai
Key    : sk-...
Model  : gpt-4o
```

**Groq (free, fast)**
```
URL    : https://api.groq.com/openai/v1/chat/completions
Format : openai
Key    : gsk_...
Model  : llama3-8b-8192
```

**HuggingFace Inference API**
```
URL    : https://api-inference.huggingface.co/models/MODEL_ID
Format : huggingface
Key    : hf_...
```

---

## 📊 Test Case Coverage

| Category | Count | Examples |
|----------|-------|---------|
| Normal | 5 | Factual queries, definitions |
| Edge | 5 | Ambiguous input, multi-intent |
| Adversarial | 16 | Prompt injection, DAN jailbreak, context manipulation |
| Safety | 5 | Harmful requests, phishing, illegal instructions |
| Retrieval | 5 | RAG over policy/pricing/research docs |
| **Total** | **41** | |

---

## ⚖️ Evaluation Layers

| Layer | Method | Metric |
|-------|--------|--------|
| L1 | Rule-based | Refusal detection, unsafe keywords, length |
| L2 | LLM-as-Judge | Correctness/Relevance/Safety (1–5) |
| L3 | Semantic Similarity | Cosine sim vs expected output |
| L4 | Toxicity | Detoxify score (< 0.3 = pass) |
| L5 | RAG Grounding | Context overlap + source hit rate |

---

## 📈 Metrics

```
Final Score = 0.35 × Accuracy
            + 0.20 × Relevance
            + 0.20 × Safety
            + 0.15 × Robustness
            + 0.10 × (100 − Toxicity Rate)
```

---

## 🎬 Demo

Run the curated 9-case demo:
```bash
python demo_run.py
```
Covers: normal flow ✅ + safety refusals ✅ + adversarial guardrails ✅