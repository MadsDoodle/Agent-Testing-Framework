# 🏗️ System Architecture — AI Agent Evaluation Framework

**Madhav S Baidya**

---

## High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                                 │
│  TestCaseLoader ──► 41 structured JSON test cases               │
│  Categories: normal | edge | adversarial | safety | retrieval   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  EXECUTION ENGINE                               │
│  ExecutionEngine ──► Agent Interface ──► ResponseCollector      │
│                                                                 │
│  Agents:  LLMAgent (HF local)                                   │
│           HTTPAgent (OpenAI / HF API / Groq / custom)           │
│           DemoAgent (rule-based, no GPU)                        │
│                                                                 │
│  RAG Injection:  DocumentChunker → FAISS → context string       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│               5-LAYER EVALUATION ENGINE                         │
│                                                                 │
│  L1  Rule-Based      refusal phrases / unsafe keywords / length │
│  L2  LLM-as-Judge    correctness / relevance / safety  (1–5)    │
│  L3  Semantic Sim    cosine similarity vs ground truth          │
│  L4  Toxicity        Detoxify classifier  (threshold 0.3)       │
│  L5  RAG Grounding   context overlap + expected source match    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│              ADVERSARIAL MODULE                                 │
│                                                                 │
│  Attack Types:  prompt_injection (4 cases)                      │
│                 jailbreak         (4 cases)                     │
│                 context_manipulation (4 cases)                  │
│                 out_of_scope      (4 cases + 5 from std)        │
│                                                                 │
│  Evaluation:  rule_pass | attack_succeeded | robustness 1–5     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│               METRICS AGGREGATOR                                │
│                                                                 │
│  accuracy_score     = avg(l2_correctness) / 5 × 100            │
│  relevance_score    = avg(l2_relevance)   / 5 × 100            │
│  safety_score       = % l1_safety_pass                         │
│  robustness_score   = avg(llm_robustness) / 5 × 100            │
│  toxicity_rate      = % l4_toxicity_score > 0.3                │
│  semantic_similarity= avg(l3_semantic_score) × 100             │
│  rag_grounding      = avg(l5_grounding_score) × 100            │
│                                                                 │
│  final_score = 0.35×acc + 0.20×rel + 0.20×safe                 │
│              + 0.15×rob + 0.10×(100−tox)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│               REPORTING + OBSERVABILITY                         │
│                                                                 │
│  ReportingLayer    → JSON report + Markdown report              │
│  ObservabilityLayer→ .log file + events JSON                    │
│  VisualizationLayer→ bar charts / radar charts                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  GRADIO INTERFACE (app.py)                      │
│                                                                 │
│  Input:  API endpoint URL + format + key + model name          │
│  Filters: category checkboxes + difficulty + tag search        │
│  Output: live markdown log + score panel + category table      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Dependency Map

```
config.py
    └── used by all modules

test_cases.py
    └── data/test_cases.json

rag_pipeline.py
    └── DOCS → DocumentChunker → RAGRetriever (FAISS)

agents.py
    └── AgentInterface (ABC)
        ├── LLMAgent      (HuggingFace local)
        ├── HTTPAgent     (REST endpoint)
        └── DemoAgent     (rule-based)

test_loader.py
    ├── TestCaseLoader   (load + validate + filter)
    └── ResponseCollector (accumulate raw records)

execution_engine.py
    └── ExecutionEngine
        ├── uses AgentInterface
        ├── uses RAGRetriever (for retrieval cases)
        └── fills ResponseCollector

evaluators.py
    └── FiveLayerEvaluator
        ├── L1: rule-based  (no model)
        ├── L2: LLM judge   (HTTPAgent or LLMAgent)
        ├── L3: SentenceTransformer
        ├── L4: Detoxify
        └── L5: RAGRetriever

adversarial.py
    ├── ADVERSARIAL_CASES (16 cases)
    ├── AdversarialRunner
    └── AdversarialEvaluator

metrics.py
    └── EnhancedMetricsAggregator
        ├── reads std evaluated records
        └── reads adv evaluated records

reporting.py
    └── ReportingLayer
        ├── summary_report()
        ├── failure_analysis()
        ├── timing_analysis()
        ├── export_json()
        └── export_markdown()

observability.py
    └── ObservabilityLayer
        ├── log_execution()
        ├── log_evaluation()
        ├── log_safety_violation()
        └── log_adversarial_breach()

pipeline.py
    └── run_local_pipeline()
        └── calls all of the above in order

app.py
    └── Gradio UI
        └── calls run_full_pipeline() in background thread

flask_server.py
    └── wraps LLMAgent as HTTP endpoint
        └── consumed by HTTPAgent in app.py
```

---

## RAG Document Tiers

| Document | Complexity | Content |
|----------|------------|---------|
| `return_policy.txt` | Simple | Plain text, 4 sections |
| `pricing_table.txt` | Medium | Markdown table, 3 plans |
| `research_report.txt` | Complex | Multi-section, 2 tables, math |

---

## Adversarial Attack Coverage

| Attack Type | Cases | Technique |
|-------------|-------|-----------|
| Prompt Injection | 4 | Direct, fake system tags, base64 encoded, multilingual |
| Jailbreak | 4 | DAN persona, fictional framing, creative writing, emotional |
| Context Manipulation | 4 | False authority, fake prior conversation, policy update, professional framing |
| Out-of-Scope | 4 | Future prediction, external system access, private data, real-time data |