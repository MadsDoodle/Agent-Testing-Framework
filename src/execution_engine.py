import json
import os
from datetime import datetime

from src.test_loader import ResponseCollector


class ExecutionEngine:
    """
    Orchestrates test execution exactly per the design doc pipeline:
    Load input → Send to agent → Capture response → Record metadata
    Decoupled from evaluation ─ just runs and collects.
    """

    def __init__(self, agent, test_cases: list,
                 model_key: str, retriever=None):
        self.agent      = agent
        self.test_cases = test_cases
        self.model_key  = model_key
        self.retriever  = retriever
        self.collector  = ResponseCollector()

    def run(self, log_fn=print) -> ResponseCollector:
        total = len(self.test_cases)
        log_fn(f"\n🚀 Execution Engine ─ {total} test cases\n" + "="*65)

        for i, tc in enumerate(self.test_cases):
            log_fn(f"[{i+1}/{total}] {tc['id']} | "
                   f"{tc['category'].upper()} | {tc['difficulty']}")
            log_fn(f"   Input: {tc['input']}")

            # Inject RAG context for retrieval cases
            context      = ""
            retrieved    = []
            if tc["category"] == "retrieval" and self.retriever:
                context, retrieved = self.retriever.get_context_string(tc["input"])
                log_fn(f"   RAG Context from: {[r['doc_name'] for r in retrieved]}")

            try:
                result = self.agent.run(tc["input"], context=context)
                status = "success"
                error  = None
            except Exception as e:
                result = {
                    "answer"  : "",
                    "metadata": {"latency":0,"tokens_used":0,"retrieved_docs":None}
                }
                status = "failure"
                error  = str(e)
                log_fn(f"   ❌ Error: {error}")

            # ── Build output record (matches design doc schema exactly) ──
            record = {
                "test_case_id"    : tc["id"],
                "model"           : self.model_key,
                "category"        : tc["category"],
                "difficulty"      : tc["difficulty"],
                "input"           : tc["input"],
                "expected_output" : tc.get("expected_output"),
                "expected_behavior": tc["expected_behavior"],
                "expected_sources": tc.get("expected_sources", []),
                "output"          : result["answer"],
                "latency"         : result["metadata"]["latency"],
                "tokens_used"     : result["metadata"].get("tokens_used"),
                "retrieved_docs"  : [r["doc_name"] for r in retrieved],
                "status"          : status,
                "error"           : error,
                "timestamp"       : datetime.now().isoformat()
            }

            self.collector.add(record)

            log_fn(f"   Output  : {result['answer'][:120]}"
                   f"{'...' if len(result['answer'])>120 else ''}")
            log_fn(f"   Latency : {result['metadata']['latency']}s  "
                   f"Status: {'✅' if status=='success' else '❌'}")
            log_fn("-"*65)

        self.collector.summary()
        return self.collector

    def save(self) -> str:
        return self.collector.save(self.model_key)