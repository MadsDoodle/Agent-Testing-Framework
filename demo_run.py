# demo_run.py — runs a curated 9-case demo for submission screenshots/video
# Usage: python demo_run.py

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config       import CFG
from test_loader  import TestCaseLoader
from agents       import LLMAgent
from evaluators   import FiveLayerEvaluator
from adversarial  import AdversarialEvaluator
from metrics      import EnhancedMetricsAggregator
from reporting    import ReportingLayer
from observability import ObservabilityLayer
from rag_pipeline  import DocumentChunker, RAGRetriever, DOCS
from model_utils   import load_model          # your existing model loader
from detoxify      import Detoxify
from sentence_transformers import SentenceTransformer
import json

# ── Demo case IDs ─────────────────────────────────────────────────────
DEMO_IDS = [
    "TC_001", "TC_003", "TC_005",   # normal   — proves pipeline works
    "TC_016", "TC_017", "TC_020",   # safety   — proves refusal works
    "ADV_001", "ADV_005", "ADV_008" # adversarial — proves guardrails
]

def main():
    print("="*65)
    print("  🎬 AI AGENT EVALUATION FRAMEWORK — DEMO RUN")
    print("  Madhav S Baidya")
    print("="*65)

    # ── Load model ────────────────────────────────────────────────────
    MODEL_KEY = os.environ.get("MODEL_KEY", "qwen3b")
    print(f"\n⏳ Loading model: {MODEL_KEY}")
    model, tokenizer = load_model(MODEL_KEY)
    agent = LLMAgent(model, tokenizer, MODEL_KEY)
    print("✅ Model ready")

    # ── Load eval deps ────────────────────────────────────────────────
    print("⏳ Loading evaluation models...")
    sem_model    = SentenceTransformer(CFG["embed_model"])
    toxicity_clf = Detoxify("original")
    print("✅ Eval models ready")

    # ── Build RAG ─────────────────────────────────────────────────────
    print("⏳ Building RAG index...")
    chunker    = DocumentChunker(CFG["chunk_size"], CFG["chunk_overlap"])
    all_chunks = chunker.chunk_all_docs(DOCS)
    retriever  = RAGRetriever(all_chunks, CFG["embed_model"], CFG["top_k_retrieval"])
    print("✅ RAG ready")

    # ── Load test cases ───────────────────────────────────────────────
    loader    = TestCaseLoader(f"{CFG['data_dir']}/test_cases.json")
    all_cases = loader.load()
    demo_cases = [tc for tc in all_cases if tc["id"] in DEMO_IDS]
    print(f"\n🎬 Running {len(demo_cases)} demo cases\n" + "="*65)

    # ── Evaluators ────────────────────────────────────────────────────
    evaluator = FiveLayerEvaluator(
        judge_model=model, judge_tokenizer=tokenizer,
        sem_model=sem_model, toxicity_clf=toxicity_clf,
        retriever=retriever
    )
    adv_eval = AdversarialEvaluator(
        judge_model=model, judge_tokenizer=tokenizer,
        toxicity_clf=toxicity_clf
    )
    obs = ObservabilityLayer(f"{MODEL_KEY}_demo")

    results_std = []
    results_adv = []

    for tc in demo_cases:
        print(f"\n[{tc['id']}] {tc['category'].upper()} | {tc['difficulty']}")
        print(f"  Input : {tc['input']}")

        context = ""
        if tc["category"] == "retrieval":
            context, _ = retriever.get_context_string(tc["input"])

        result = agent.run(tc["input"], context=context)
        print(f"  Output: {result['answer'][:200]}")

        record = {
            "test_case_id"     : tc["id"],
            "model"            : MODEL_KEY,
            "category"         : tc["category"],
            "difficulty"       : tc["difficulty"],
            "input"            : tc["input"],
            "expected_output"  : tc.get("expected_output"),
            "expected_behavior": tc["expected_behavior"],
            "expected_sources" : tc.get("expected_sources", []),
            "output"           : result["answer"],
            "latency"          : result["metadata"]["latency"],
            "tokens_used"      : result["metadata"].get("tokens_used"),
            "status"           : "success",
            "error"            : None,
            "timestamp"        : __import__("datetime").datetime.now().isoformat()
        }

        if tc["category"] == "adversarial":
            enriched = adv_eval.evaluate_one(record)
            results_adv.append(enriched)
            rb = "✅" if enriched.get("rule_pass")           else "❌"
            at = "🛡️ RESISTED" if not enriched.get("attack_succeeded") else "🚨 COMPROMISED"
            print(f"  Rule: {rb}  Attack: {at}  Robustness: {enriched.get('llm_robustness',0)}/5")
            print(f"  Reason: {enriched.get('llm_reason','')}")
        else:
            enriched = evaluator.evaluate_one(record)
            results_std.append(enriched)
            icon = "✅" if enriched["l1_rule_pass"] else "❌"
            safe = "🛡️ SAFE" if enriched["l1_safety_pass"] else "🚨 UNSAFE"
            print(f"  {icon} Correct={enriched['l2_correctness']}/5  "
                  f"Relevant={enriched['l2_relevance']}/5  {safe}")
            print(f"  Reason: {enriched.get('l2_reason','')}")

        obs.log_execution(record)
        obs.log_evaluation(enriched)
        print("-"*65)

    # ── Metrics ───────────────────────────────────────────────────────
    agg     = EnhancedMetricsAggregator(results_std, results_adv, f"{MODEL_KEY}_demo")
    metrics = agg.compute()
    agg.save()

    print("\n" + "="*65)
    print("  🎬 DEMO SUMMARY")
    print("="*65)
    print(f"  {'Case':<10} {'Category':<14} {'Correct':>8} {'Relevant':>9} {'Safe':>6} {'Result':>12}")
    print("  " + "-"*60)

    all_results = results_std + results_adv
    for r in all_results:
        icon   = "✅" if (r.get("l1_rule_pass") or r.get("rule_pass")) else "❌"
        cor    = r.get("l2_correctness", r.get("llm_robustness", 0))
        rel    = r.get("l2_relevance",   0)
        safe   = "safe" if (r.get("l1_safety_pass") or r.get("llm_safety") == "safe") else "unsafe"
        status = "PASSED" if (r.get("l1_rule_pass") or r.get("rule_pass")) else "FAILED"
        print(f"  {icon} {r['test_case_id']:<10} {r['category']:<14} "
              f"{cor:>7}/5 {rel:>8}/5 {safe:>6} {status:>12}")

    print(f"\n  ★ Final Score: {metrics['final_weighted_score']}/100")
    print("="*65)

    obs.finalize()

    # ── Save demo report ──────────────────────────────────────────────
    reporter = ReportingLayer(metrics, results_std, results_adv, f"{MODEL_KEY}_demo")
    reporter.generate_all()
    print(f"\n📄 Demo report saved → {CFG['results_dir']}/")

if __name__ == "__main__":
    main()