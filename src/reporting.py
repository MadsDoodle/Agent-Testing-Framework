import json
from collections import Counter
from datetime import datetime

from src.config import CFG


class ReportingLayer:
    """
    Covers all 4.7.x requirements from the design doc:
    4.7.1 Summary Report
    4.7.2 Failure Analysis
    4.7.3 Timing Analysis
    4.7.4 Output Formats — JSON + Markdown
    """

    def __init__(self, metrics: dict, evaluated: list,
                 adv_evaluated: list, model_key: str):
        self.metrics       = metrics
        self.evaluated     = evaluated
        self.adv_evaluated = adv_evaluated
        self.model_key     = model_key

    # ── 4.7.1 Summary Report ─────────────────────────────────────────
    def summary_report(self) -> dict:
        m     = self.metrics
        total = m["total_standard"]
        passed = sum(1 for r in self.evaluated if r.get("l1_rule_pass"))
        failed = total - passed

        return {
            "model"              : self.model_key,
            "generated_at"       : datetime.now().isoformat(),
            "overall_scores"     : {
                "accuracy"       : m["accuracy_score"],
                "relevance"      : m["relevance_score"],
                "safety"         : m["safety_score"],
                "robustness"     : m["robustness_score"],
                "toxicity_rate"  : m["toxicity_rate"],
                "final_score"    : m["final_weighted_score"]
            },
            "pass_fail_stats"    : {
                "total"  : total,
                "passed" : passed,
                "failed" : failed,
                "pass_pct": round(100*passed/total,2) if total else 0
            },
            "category_breakdown": m.get("category_breakdown",{}),
        }

    # ── 4.7.2 Failure Analysis ────────────────────────────────────────
    def failure_analysis(self) -> dict:
        failures     = [r for r in self.evaluated     if not r.get("l1_rule_pass")]
        adv_failures = [r for r in self.adv_evaluated if not r.get("rule_pass")]
        safety_violations = [
            r for r in self.evaluated
            if r.get("l2_safety") == "unsafe" or r.get("l4_toxicity_score", 0) > 0.5
        ]

        # Most common failure category
        fail_cats = Counter(r["category"] for r in failures)
        fail_types = []

        for r in failures:
            reason = []
            if r.get("l1_has_unsafe"):        reason.append("unsafe_content_in_output")
            if r.get("l1_too_short"):          reason.append("response_too_short")
            if not r.get("l1_safety_pass"):    reason.append("safety_rule_failed")
            if r.get("l2_correctness",5) <= 2: reason.append("low_correctness_score")
            if r.get("l2_relevance",5)   <= 2: reason.append("low_relevance_score")
            fail_types.append({
                "id"      : r["test_case_id"],
                "category": r["category"],
                "input"   : r["input"][:80],
                "output"  : r["output"][:80],
                "reasons" : reason,
                "llm_reason": r.get("l2_reason","")
            })

        # Adversarial failures
        compromised = [
            {
                "id"         : r["test_case_id"],
                "attack_type": r["attack_type"],
                "input"      : r["input"][:80],
                "output"     : r["output"][:80],
                "reason"     : r.get("llm_reason","")
            }
            for r in adv_failures if r.get("attack_succeeded")
        ]

        return {
            "total_failures"        : len(failures),
            "failures_by_category"  : dict(fail_cats),
            "weak_categories"       : [cat for cat,n in fail_cats.items() if n >= 2],
            "safety_violations"     : len(safety_violations),
            "failure_details"       : fail_types,
            "adversarial_compromised": compromised,
            "total_adv_failures"    : len(adv_failures)
        }

    # ── 4.7.3 Timing Analysis ─────────────────────────────────────────
    def timing_analysis(self) -> dict:
        std_lats = [r["latency"] for r in self.evaluated     if r.get("latency",0) > 0]
        adv_lats = [r["latency"] for r in self.adv_evaluated if r.get("latency",0) > 0]

        def stats(lats):
            if not lats: return {}
            s = sorted(lats)
            n = len(s)
            return {
                "mean"  : round(sum(s)/n,4),
                "median": round(s[n//2],4),
                "min"   : round(s[0],4),
                "max"   : round(s[-1],4),
                "p90"   : round(s[int(n*0.9)],4)  # bonus: 90th percentile
            }

        # Slowest cases
        slowest = sorted(
            self.evaluated, key=lambda r: r.get("latency",0), reverse=True
        )[:5]

        return {
            "standard_latency"   : stats(std_lats),
            "adversarial_latency": stats(adv_lats),
            "slowest_cases"      : [
                {"id": r["test_case_id"],
                 "category": r["category"],
                 "latency": r["latency"]} for r in slowest
            ]
        }

    # ── 4.7.4a JSON Export ────────────────────────────────────────────
    def export_json(self) -> str:
        full_report = {
            "summary"        : self.summary_report(),
            "failure_analysis": self.failure_analysis(),
            "timing_analysis": self.timing_analysis(),
            "full_metrics"   : self.metrics
        }
        path = f"{CFG['results_dir']}/{self.model_key}_full_report.json"
        with open(path,"w") as f:
            json.dump(full_report, f, indent=2)
        print(f"💾 JSON report → {path}")
        return path

    # ── 4.7.4b Markdown Export ────────────────────────────────────────
    def export_markdown(self) -> str:
        m    = self.metrics
        summ = self.summary_report()
        fail = self.failure_analysis()
        tim  = self.timing_analysis()
        ps   = summ["pass_fail_stats"]
        ls   = tim["standard_latency"]

        md = f"""# AI Agent Evaluation Report
**Model:** `{self.model_key}`
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Overall Scores

| Metric | Score |
|--------|-------|
| Accuracy | {m['accuracy_score']}/100 |
| Relevance | {m['relevance_score']}/100 |
| Safety | {m['safety_score']}/100 |
| Robustness | {m['robustness_score']}/100 |
| Toxicity Rate | {m['toxicity_rate']}% |
| Semantic Similarity | {m.get('semantic_similarity','N/A')}/100 |
| RAG Grounding | {m.get('rag_grounding_score','N/A')}/100 |
| **★ Final Score** | **{m['final_weighted_score']}/100** |

## Pass / Fail Statistics

- Total Cases : {ps['total']}
- Passed      : {ps['passed']}
- Failed      : {ps['failed']}
- Pass Rate   : {ps['pass_pct']}%

## Category Breakdown

| Category | Correctness | Relevance | Safety% | Rule Pass% |
|----------|-------------|-----------|---------|------------|
"""
        for cat, s in m.get("category_breakdown",{}).items():
            md += (f"| {cat} | {s['avg_correctness']}/5 | "
                   f"{s['avg_relevance']}/5 | "
                   f"{s['safety_pct']}% | "
                   f"{s['rule_pass_pct']}% |\n")

        md += f"""
## Failure Analysis

- Total Failures        : {fail['total_failures']}
- Safety Violations     : {fail['safety_violations']}
- Weak Categories       : {', '.join(fail['weak_categories']) or 'None'}
- Adversarial Breaches  : {len(fail['adversarial_compromised'])}

### Failure Details

"""
        for fd in fail["failure_details"]:
            md += f"**[{fd['id']}]** `{fd['category']}` — {', '.join(fd['reasons'])}\n"
            md += f"> Input: {fd['input']}\n\n"

        if fail["adversarial_compromised"]:
            md += "### Adversarial Compromised Cases\n\n"
            for c in fail["adversarial_compromised"]:
                md += f"**[{c['id']}]** `{c['attack_type']}` — {c['reason']}\n"
                md += f"> Input: {c['input']}\n\n"

        md += f"""
## Timing Analysis

| Metric | Standard Cases | Adversarial Cases |
|--------|---------------|-------------------|
| Mean   | {ls.get('mean','N/A')}s | {tim['adversarial_latency'].get('mean','N/A')}s |
| Median | {ls.get('median','N/A')}s | {tim['adversarial_latency'].get('median','N/A')}s |
| Min    | {ls.get('min','N/A')}s | {tim['adversarial_latency'].get('min','N/A')}s |
| Max    | {ls.get('max','N/A')}s | {tim['adversarial_latency'].get('max','N/A')}s |
| P90    | {ls.get('p90','N/A')}s | {tim['adversarial_latency'].get('p90','N/A')}s |

### 5 Slowest Cases

"""
        for s in tim["slowest_cases"]:
            md += f"- `{s['id']}` ({s['category']}) — {s['latency']}s\n"

        md += "\n---\n*Generated by AI Agent Evaluation Framework*\n"

        path = f"{CFG['results_dir']}/{self.model_key}_report.md"
        with open(path,"w") as f:
            f.write(md)
        print(f"📄 Markdown report → {path}")
        return path

    def generate_all(self):
        print("\n📊 Generating full reports...")
        json_path = self.export_json()
        md_path   = self.export_markdown()
        print(f"✅ Reports ready:\n   {json_path}\n   {md_path}")
        return json_path, md_path