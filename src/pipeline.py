"""
run_local_pipeline — orchestrates the full evaluation pipeline for a local HF model.
All Drive references replaced with a configurable local output base directory.
"""
import os

from src.config import CFG
from src.test_loader import TestCaseLoader
from src.execution_engine import ExecutionEngine
from src.evaluators import FiveLayerEvaluator
from src.adversarial import AdversarialRunner, AdversarialEvaluator
from src.metrics import EnhancedMetricsAggregator
from src.reporting import ReportingLayer
from src.observability import ObservabilityLayer
from src.visualization import generate_evaluation_plots
from src.drive_utils import save_run_to_drive, zip_and_archive_to_drive
from src.test_cases_data import ADVERSARIAL_CASES

# ── Output base — set via env var or defaults to local "outputs/" folder ────
DRIVE_BASE = os.environ.get("DRIVE_BASE", "outputs")


def run_local_pipeline(agent, model_key: str,
                       retriever=None,
                       judge_model=None, judge_tokenizer=None,
                       sem_model=None, toxicity_clf=None,
                       archive: bool = True):
    """
    Full pipeline:
    1. Load test cases
    2. Run execution engine
    3. 5-layer evaluation
    4. Adversarial run + evaluation
    5. Metrics aggregation
    6. Reporting (JSON + Markdown)
    7. Observability
    8. Visualization plots
    9. Archive (optional)
    """
    obs = ObservabilityLayer(model_key)

    loader     = TestCaseLoader(f"{CFG['data_dir']}/test_cases.json")
    test_cases = loader.load()

    engine    = ExecutionEngine(agent, test_cases, model_key, retriever=retriever)
    collector = engine.run()
    engine.save()

    for r in collector.get_all():
        obs.log_execution(r)

    evaluator = FiveLayerEvaluator(
        judge_model     = judge_model,
        judge_tokenizer = judge_tokenizer,
        sem_model       = sem_model,
        toxicity_clf    = toxicity_clf,
        retriever       = retriever
    )
    evaluated = evaluator.evaluate_all(collector.get_all())
    evaluator.save(evaluated, model_key)

    for r in evaluated:
        obs.log_evaluation(r)
        if r.get("l2_safety") == "unsafe":
            obs.log_safety_violation(r, "llm_judge_flagged_unsafe")
        if r.get("l4_toxicity_score", 0) > 0.5:
            obs.log_safety_violation(r, f"toxicity={r['l4_toxicity_score']}")

    adv_runner    = AdversarialRunner(agent, ADVERSARIAL_CASES, model_key)
    adv_results   = adv_runner.run()
    adv_runner.save()

    adv_eval      = AdversarialEvaluator(
        judge_model=judge_model, judge_tokenizer=judge_tokenizer,
        toxicity_clf=toxicity_clf
    )
    adv_evaluated = [adv_eval.evaluate_one(r) for r in adv_results]

    for r in adv_evaluated:
        obs.log_execution(r)
        if r.get("attack_succeeded"):
            obs.log_adversarial_breach(r)

    agg     = EnhancedMetricsAggregator(evaluated, adv_evaluated, model_key)
    metrics = agg.compute()
    agg.save()
    agg.print_report()

    reporter = ReportingLayer(metrics, evaluated, adv_evaluated, model_key)
    reporter.generate_all()

    obs.finalize()

    # ── Plots ────────────────────────────────────────────────────────────
    plots_dir  = f"{CFG['results_dir']}/plots/{model_key}"
    plot_paths = generate_evaluation_plots(
        metrics, evaluated, adv_evaluated, model_key, save_dir=plots_dir
    )

    if archive:
        # ── Save results ─────────────────────────────────────────────────
        save_run_to_drive(model_key, CFG["results_dir"], DRIVE_BASE)

        # ── Zip and archive ───────────────────────────────────────────────
        zip_and_archive_to_drive(
            model_key  = model_key,
            drive_base = DRIVE_BASE,
            local_dirs = [CFG["results_dir"], "evaluation/logs",
                          CFG["data_dir"],    CFG["rag_docs_dir"]],
            plot_paths = plot_paths
        )

    print(f"\n✅ Full pipeline complete for model: {model_key}")
    return metrics