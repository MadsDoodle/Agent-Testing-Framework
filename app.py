# app.py — Gradio interface entry point
# Run with: python app.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config          import CFG
from test_loader     import TestCaseLoader
from agents          import HTTPAgent
from evaluators      import FiveLayerEvaluator
from adversarial     import AdversarialEvaluator
from metrics         import EnhancedMetricsAggregator
from reporting       import ReportingLayer
from rag_pipeline    import RAGRetriever
import json, threading, queue as q_module
from datetime import datetime
import gradio as gr

# ── Pre-load at startup ───────────────────────────────────────────────
tc_loader     = TestCaseLoader(f"{CFG['data_dir']}/test_cases.json")
UI_TEST_CASES = tc_loader.load()

# These are injected at runtime from the notebook after model loading
sem_model    = None
toxicity_clf = None
retriever    = None

def set_runtime_deps(sem, tox, ret):
    """Call this from notebook after loading models."""
    global sem_model, toxicity_clf, retriever
    sem_model    = sem
    toxicity_clf = tox
    retriever    = ret


# ════════════════════════════════════════════════════════════════════
# FILTER HELPERS
# ════════════════════════════════════════════════════════════════════

def get_filtered_cases(categories, difficulties, tags_str):
    tags     = [t.strip().lower() for t in tags_str.split(",") if t.strip()] if tags_str else []
    filtered = []
    for tc in UI_TEST_CASES:
        if tc["category"]   not in categories:  continue
        if tc["difficulty"] not in difficulties: continue
        if tags:
            tc_tags = [t.lower() for t in tc.get("tags", [])]
            if not any(tag in tc_tags for tag in tags):
                continue
        filtered.append(tc)
    return filtered


def update_case_count(categories, difficulties, tags_str):
    filtered   = get_filtered_cases(categories, difficulties, tags_str)
    cat_counts = {}
    for tc in filtered:
        cat_counts[tc["category"]] = cat_counts.get(tc["category"], 0) + 1
    breakdown = "  |  ".join(f"{c}: {n}" for c, n in sorted(cat_counts.items()))
    return f"{len(filtered)} / {len(UI_TEST_CASES)} cases selected   →   {breakdown}"


# ════════════════════════════════════════════════════════════════════
# CORE PIPELINE — background thread
# ════════════════════════════════════════════════════════════════════

def run_full_pipeline(url, api_format, api_key, model_name,
                      filtered_cases, log_queue):

    log_lines = []

    def log(msg):
        log_lines.append(msg)
        log_queue.put(("LOG", "\n".join(log_lines)))

    def log_section(title):
        log_lines.append(f"\n---\n## {title}\n")
        log_queue.put(("LOG", "\n".join(log_lines)))

    def log_case_header(idx, total, tc_id, category, difficulty):
        log_lines.append(
            f"\n---\n### [{idx}/{total}] `{tc_id}` &nbsp;|&nbsp; "
            f"**{category.upper()}** &nbsp;|&nbsp; `{difficulty}`"
        )
        log_queue.put(("LOG", "\n".join(log_lines)))

    def log_result_block(record, enriched):
        lines = []
        lines.append(f"\n**📥 Input:** {record['input']}\n")
        lines.append(f"**📤 Output:**\n\n{enriched['output']}\n")
        lines.append(
            f"**⏱️ Latency:** `{record['latency']}s` &nbsp; "
            f"**Status:** {'✅ success' if record['status'] == 'success' else '❌ failure'}\n"
        )
        l1_safe   = "✅" if enriched.get("l1_safety_pass")       else "❌"
        l1_rule   = "✅" if enriched.get("l1_rule_pass")         else "❌"
        l1_attack = "🚨 BREACHED" if enriched.get("l1_attack_succeeded") else "🛡️ OK"
        l2_cor    = enriched.get("l2_correctness", 0)
        l2_rel    = enriched.get("l2_relevance",   0)
        l2_saf    = enriched.get("l2_safety",  "unknown")
        reason    = enriched.get("l2_reason",  "") or "*no reason returned*"
        sem       = enriched.get("l3_semantic_score")
        tox       = enriched.get("l4_toxicity_score")
        tox_pass  = "✅" if enriched.get("l4_toxicity_pass") else "⚠️ FLAG"

        lines.append("| Layer | Result |")
        lines.append("|-------|--------|")
        lines.append(f"| **L1 Rules**    | Safety {l1_safe} &nbsp; Rule {l1_rule} &nbsp; Attack {l1_attack} |")
        lines.append(f"| **L2 Judge**    | Correct **{l2_cor}/5** &nbsp; Relevant **{l2_rel}/5** &nbsp; Safety `{l2_saf}` |")
        lines.append(f"| **L2 Reason**   | {reason} |")
        lines.append(f"| **L3 Semantic** | {f'`{sem:.3f}`' if sem is not None else 'N/A — no ground truth'} |")
        lines.append(f"| **L4 Toxicity** | {f'`{tox:.4f}`' if tox is not None else 'N/A'} &nbsp; {tox_pass} |")

        if enriched.get("l5_grounding_available"):
            grd  = enriched.get("l5_grounding_score")
            sh   = enriched.get("l5_source_hit")
            docs = enriched.get("l5_retrieved_docs", [])
            lines.append(
                f"| **L5 RAG** | Grounding `{grd}` &nbsp; "
                f"SourceHit {'✅' if sh else '❌'} &nbsp; Docs: `{docs}` |"
            )
        if enriched.get("l2_parse_error"):
            lines.append(f"\n> ⚠️ **Judge parse error** — raw: `{enriched.get('l2_raw','')[:120]}`\n")

        log_lines.extend(lines)
        log_queue.put(("LOG", "\n".join(log_lines)))

    def log_adv_result_block(record, enriched):
        lines = []
        lines.append(f"\n**📥 Input:** {record['input']}\n")
        lines.append(f"**📤 Output:**\n\n{enriched['output']}\n")
        lines.append(
            f"**⏱️ Latency:** `{record['latency']}s` &nbsp; "
            f"**Status:** {'✅ success' if record['status'] == 'success' else '❌ failure'}\n"
        )
        rb_icon  = "✅" if enriched.get("rule_pass")                  else "❌"
        at_icon  = "🛡️ RESISTED" if not enriched.get("attack_succeeded") else "🚨 COMPROMISED"
        rob      = enriched.get("llm_robustness", 0)
        saf      = enriched.get("llm_safety",     "?")
        reason   = enriched.get("llm_reason",     "") or "*no reason returned*"
        tox      = enriched.get("adv_toxicity")
        tox_pass = "✅" if enriched.get("adv_toxicity_pass") else "⚠️ FLAG"

        lines.append("| Layer | Result |")
        lines.append("|-------|--------|")
        lines.append(f"| **Rule Check** | {rb_icon} &nbsp; Attack: **{at_icon}** |")
        lines.append(f"| **Robustness** | **{rob}/5** &nbsp; Safety `{saf}` |")
        lines.append(f"| **Reason**     | {reason} |")
        if tox is not None:
            lines.append(f"| **Toxicity**   | `{tox:.4f}` &nbsp; {tox_pass} |")
        if enriched.get("llm_parse_error"):
            lines.append(f"\n> ⚠️ **Judge parse error** — raw: `{enriched.get('llm_raw','')[:120]}`\n")

        log_lines.extend(lines)
        log_queue.put(("LOG", "\n".join(log_lines)))

    # ── Setup ─────────────────────────────────────────────────────────
    model_key   = model_name.strip().replace(" ", "_") or "custom_model"
    results_std = []
    results_adv = []

    agent = HTTPAgent(
        url=url.strip(), api_format=api_format,
        api_key=api_key.strip(), model_name=model_name.strip()
    )
    judge_agent = HTTPAgent(
        url=url.strip(), api_format=api_format,
        api_key=api_key.strip(), model_name=model_name.strip()
    )
    evaluator = FiveLayerEvaluator(
        judge_agent=judge_agent, sem_model=sem_model,
        toxicity_clf=toxicity_clf, retriever=retriever
    )

    std_cases = [tc for tc in filtered_cases if tc["category"] != "adversarial"]
    adv_cases = [tc for tc in filtered_cases if tc["category"] == "adversarial"]

    log_section("🚀 Evaluation Started")
    log(f"- **Total cases:** {len(filtered_cases)}")
    log(f"- **Standard:** {len(std_cases)} &nbsp;&nbsp; **Adversarial:** {len(adv_cases)}")
    log(f"- **Categories:** `{sorted(set(tc['category']   for tc in filtered_cases))}`")
    log(f"- **Difficulties:** `{sorted(set(tc['difficulty'] for tc in filtered_cases))}`")

    # ── Standard ──────────────────────────────────────────────────────
    if std_cases:
        log_section(f"📋 Standard Test Cases — {len(std_cases)}")
        for i, tc in enumerate(std_cases):
            log_case_header(i+1, len(std_cases), tc["id"], tc["category"], tc["difficulty"])
            context, retrieved_docs = "", []
            if tc["category"] == "retrieval" and retriever:
                context, retrieved_docs = retriever.get_context_string(tc["input"])
                log(f"> 📚 **Retrieved docs:** `{[r['doc_name'] for r in retrieved_docs]}`")
            try:
                result = agent.run(tc["input"], context=context)
                status, error = "success", None
            except Exception as e:
                result = {"answer": f"[ERROR: {e}]", "metadata": {"latency": 0, "tokens_used": 0}}
                status, error = "failure", str(e)
                log(f"> ❌ **Execution error:** {error}")

            record = {
                "test_case_id"     : tc["id"],       "model"            : model_key,
                "category"         : tc["category"],  "difficulty"       : tc["difficulty"],
                "input"            : tc["input"],      "expected_output"  : tc.get("expected_output"),
                "expected_behavior": tc["expected_behavior"],
                "expected_sources" : tc.get("expected_sources", []),
                "output"           : result["answer"],
                "latency"          : result["metadata"]["latency"],
                "tokens_used"      : result["metadata"].get("tokens_used"),
                "status"           : status,           "error"            : error,
                "timestamp"        : datetime.now().isoformat()
            }
            enriched = evaluator.evaluate_one(record)
            results_std.append(enriched)
            log_result_block(record, enriched)

        std_path = f"{CFG['results_dir']}/{model_key}_evaluated.json"
        with open(std_path, "w") as f:
            json.dump(results_std, f, indent=2)
        log(f"\n> 💾 **Standard results saved** → `{std_path}`")

    # ── Adversarial ───────────────────────────────────────────────────
    if adv_cases:
        log_section(f"🔴 Adversarial Test Cases — {len(adv_cases)}")
        adv_eval = AdversarialEvaluator(judge_agent=judge_agent, toxicity_clf=toxicity_clf)
        for i, tc in enumerate(adv_cases):
            log_case_header(i+1, len(adv_cases), tc["id"], "adversarial", tc.get("difficulty","hard"))
            log(f"> ⚔️ **Attack type:** `{tc.get('attack_type','adversarial').upper()}`")
            try:
                result = agent.run(tc["input"])
                status, error = "success", None
            except Exception as e:
                result = {"answer": f"[ERROR: {e}]", "metadata": {"latency": 0, "tokens_used": 0}}
                status, error = "failure", str(e)

            record = {
                "test_case_id"          : tc["id"],         "model"    : model_key,
                "category"              : "adversarial",
                "attack_type"           : tc.get("attack_type", "unknown"),
                "input"                 : tc["input"],
                "expected_behavior"     : tc["expected_behavior"],
                "expected_response_type": tc.get("expected_response_type", "refusal"),
                "output"                : result["answer"],
                "latency"               : result["metadata"]["latency"],
                "tokens_used"           : result["metadata"].get("tokens_used"),
                "status"                : status,            "error"    : error,
                "timestamp"             : datetime.now().isoformat()
            }
            enriched = adv_eval.evaluate_one(record)
            results_adv.append(enriched)
            log_adv_result_block(record, enriched)

        adv_path = f"{CFG['results_dir']}/{model_key}_adversarial_evaluated.json"
        with open(adv_path, "w") as f:
            json.dump(results_adv, f, indent=2)
        log(f"\n> 💾 **Adversarial results saved** → `{adv_path}`")

    # ── Metrics ───────────────────────────────────────────────────────
    log_section("🧮 Final Metrics")
    agg     = EnhancedMetricsAggregator(results_std, results_adv, model_key)
    metrics = agg.compute()
    agg.save()

    log("| Metric | Score |")
    log("|--------|-------|")
    log(f"| Accuracy      | **{metrics['accuracy_score']}/100** |")
    log(f"| Relevance     | **{metrics['relevance_score']}/100** |")
    log(f"| Safety        | **{metrics['safety_score']}/100** |")
    log(f"| Robustness    | **{metrics['robustness_score']}/100** |")
    log(f"| Toxicity Rate | **{metrics['toxicity_rate']}%** |")
    if metrics.get("semantic_similarity"):
        log(f"| Semantic Sim  | **{metrics['semantic_similarity']}/100** |")
    if metrics.get("rag_grounding_score"):
        log(f"| RAG Grounding | **{metrics['rag_grounding_score']}/100** |")
    log(f"| ⭐ Final Score | **{metrics['final_weighted_score']}/100** |")

    try:
        reporter = ReportingLayer(metrics, results_std, results_adv, model_key)
        reporter.generate_all()
        log(f"\n> 📄 **Reports saved** → `{CFG['results_dir']}/`")
    except Exception as e:
        log(f"\n> ⚠️ Report error: {e}")

    log_section("✅ Evaluation Complete")
    log_queue.put(f"__METRICS__{json.dumps(metrics)}")
    log_queue.put("__DONE__")


# ════════════════════════════════════════════════════════════════════
# GRADIO EVAL
# ════════════════════════════════════════════════════════════════════

def gradio_eval(url, api_format, api_key, model_name,
                filter_category, filter_difficulty, filter_tags):

    if not url.strip():
        yield "❌ Please enter a valid API endpoint URL.", "", ""
        return

    filtered_cases = get_filtered_cases(filter_category, filter_difficulty, filter_tags)
    if not filtered_cases:
        yield "❌ No test cases match your filters.", "", ""
        return

    log_queue     = q_module.Queue()
    final_metrics = {}
    current_md    = "*Starting evaluation...*"

    def worker():
        run_full_pipeline(url, api_format, api_key, model_name,
                          filtered_cases, log_queue)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        try:
            msg = log_queue.get(timeout=90)
            if msg == "__DONE__":
                break
            if isinstance(msg, str) and msg.startswith("__METRICS__"):
                try:
                    final_metrics = json.loads(msg.replace("__METRICS__", ""))
                except Exception:
                    pass
                continue
            if isinstance(msg, tuple) and msg[0] == "LOG":
                current_md = msg[1]
                score_text = ""
                if final_metrics:
                    m = final_metrics
                    score_text = (
                        f"Accuracy   : {m.get('accuracy_score','...')}/100\n"
                        f"Relevance  : {m.get('relevance_score','...')}/100\n"
                        f"Safety     : {m.get('safety_score','...')}/100\n"
                        f"Robustness : {m.get('robustness_score','...')}/100\n"
                        f"Toxicity % : {m.get('toxicity_rate','...')}%\n"
                        f"────────────────────\n"
                        f"★ FINAL    : {m.get('final_weighted_score','...')}/100"
                    )
                yield current_md, score_text, ""
        except Exception:
            break

    m = final_metrics or {}
    score_text = (
        f"Accuracy   : {m.get('accuracy_score',   'N/A')}/100\n"
        f"Relevance  : {m.get('relevance_score',   'N/A')}/100\n"
        f"Safety     : {m.get('safety_score',      'N/A')}/100\n"
        f"Robustness : {m.get('robustness_score',  'N/A')}/100\n"
        f"Toxicity % : {m.get('toxicity_rate',     'N/A')}%\n"
        f"Sem. Sim   : {m.get('semantic_similarity','N/A')}/100\n"
        f"RAG Ground : {m.get('rag_grounding_score','N/A')}/100\n"
        f"────────────────────\n"
        f"★ FINAL    : {m.get('final_weighted_score','N/A')}/100"
    )
    cat_table = ""
    for cat, s in m.get("category_breakdown", {}).items():
        cat_table += (
            f"{cat:<14} | Correct: {s['avg_correctness']}/5 | "
            f"Relevant: {s['avg_relevance']}/5 | "
            f"Safe: {s['safety_pct']}% | Pass: {s['rule_pass_pct']}%\n"
        )
    yield current_md + "\n\n---\n## ✅ Evaluation Complete", score_text, cat_table


# ════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════

with gr.Blocks(title="AI Agent Evaluation Framework", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    <div style="text-align:center; padding:12px 0 6px 0;">
        <h1 style="margin-bottom:4px;">🧪 AI Agent Evaluation Framework</h1>
        <p style="font-size:15px; color:#888; margin:0;">
            SOTA Multi-Layer Evaluation Pipeline &nbsp;|&nbsp;
            Built by <strong>Madhav S Baidya</strong>
        </p>
    </div>
    """)

    gr.Markdown("---")
    gr.Markdown("### ⚙️ Agent Configuration")

    with gr.Row():
        with gr.Column(scale=3):
            url_input = gr.Textbox(
                label="🌐 AI API Endpoint URL",
                placeholder="http://localhost:5050/v1/chat/completions",
                lines=1
            )
        with gr.Column(scale=1):
            format_input = gr.Dropdown(
                choices=["openai", "huggingface", "custom"],
                value="openai", label="API Format"
            )

    with gr.Row():
        key_input   = gr.Textbox(label="🔑 API Key (blank for local)", type="password", lines=1)
        model_input = gr.Textbox(
            label="Model Name / ID", value="qwen3b", lines=1,
            placeholder="qwen3b  |  gpt-3.5-turbo  |  mistralai/Mistral-7B-Instruct-v0.2"
        )

    gr.Markdown("---")
    gr.Markdown("### 🎛️ Test Case Filters")

    with gr.Row():
        with gr.Column(scale=1):
            filter_category = gr.CheckboxGroup(
                choices=["normal","edge","adversarial","safety","retrieval"],
                value=["normal","edge","adversarial","safety","retrieval"],
                label="📂 Categories"
            )
        with gr.Column(scale=1):
            filter_difficulty = gr.CheckboxGroup(
                choices=["easy","medium","hard"],
                value=["easy","medium","hard"],
                label="📊 Difficulty Levels"
            )
        with gr.Column(scale=1):
            filter_tags = gr.Textbox(
                label="🏷️ Filter by Tag (optional, comma-separated)",
                placeholder="factual, rag, jailbreak, prompt_injection",
                lines=2, value=""
            )

    with gr.Row():
        case_count_display = gr.Textbox(
            label="📋 Cases Selected",
            value=f"All {len(UI_TEST_CASES)} cases selected",
            interactive=False, lines=1
        )

    gr.Markdown("---")
    start_btn = gr.Button("🚀 START EVALUATION", variant="primary", size="lg")
    gr.Markdown("---")
    gr.Markdown("### 📊 Results")

    with gr.Row():
        with gr.Column(scale=3):
            log_box = gr.Markdown(
                value="*Waiting for evaluation to start...*",
                label="📋 Live Evaluation Logs",
                height=800
            )
        with gr.Column(scale=1):
            score_box = gr.Textbox(label="📊 Live Scores", lines=14, interactive=False)

    cat_box = gr.Textbox(label="📈 Category Breakdown", lines=8, interactive=False)

    gr.Markdown(
        "<div style='text-align:center;color:#666;font-size:12px;margin-top:10px;'>"
        "AI Agent Evaluation Framework &nbsp;·&nbsp; Madhav S Baidya</div>"
    )

    for component in [filter_category, filter_difficulty, filter_tags]:
        component.change(
            fn=update_case_count,
            inputs=[filter_category, filter_difficulty, filter_tags],
            outputs=[case_count_display]
        )

    start_btn.click(
        fn=gradio_eval,
        inputs=[url_input, format_input, key_input, model_input,
                filter_category, filter_difficulty, filter_tags],
        outputs=[log_box, score_box, cat_box]
    )

demo.queue()

if __name__ == "__main__":
    demo.launch(share=True, debug=False)