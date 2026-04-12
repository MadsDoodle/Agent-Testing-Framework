import json
import logging
import os
from datetime import datetime


class ObservabilityLayer:
    """
    4.8 Observability — structured log file per run.
    Captures: input, output, eval scores, errors, timestamps.
    """

    def __init__(self, model_key: str, log_dir: str = "evaluation/logs"):
        self.model_key = model_key
        self.log_dir   = log_dir
        self.events    = []
        os.makedirs(log_dir, exist_ok=True)

        # File logger
        log_path = f"{log_dir}/{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            filename=log_path,
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s"
        )
        self.logger   = logging.getLogger(model_key)
        self.log_path = log_path
        print(f"📡 Observability active → {log_path}")

    def log_execution(self, record: dict):
        event = {
            "event_type": "execution",
            "timestamp" : record.get("timestamp"),
            "id"        : record.get("test_case_id"),
            "category"  : record.get("category"),
            "input"     : record.get("input","")[:200],
            "output"    : record.get("output","")[:200],
            "latency"   : record.get("latency"),
            "status"    : record.get("status"),
            "error"     : record.get("error")
        }
        self.events.append(event)
        self.logger.info(
            f"EXEC | {event['id']} | {event['category']} | "
            f"latency={event['latency']}s | status={event['status']}"
        )
        if event["error"]:
            self.logger.error(f"ERROR | {event['id']} | {event['error']}")

    def log_evaluation(self, record: dict):
        event = {
            "event_type"   : "evaluation",
            "timestamp"    : record.get("evaluated_at"),
            "id"           : record.get("test_case_id"),
            "l1_rule_pass" : record.get("l1_rule_pass"),
            "l2_correctness": record.get("l2_correctness"),
            "l2_relevance" : record.get("l2_relevance"),
            "l2_safety"    : record.get("l2_safety"),
            "l3_semantic"  : record.get("l3_semantic_score"),
            "l4_toxicity"  : record.get("l4_toxicity_score"),
            "l5_grounding" : record.get("l5_grounding_score"),
            "reason"       : record.get("l2_reason","")
        }
        self.events.append(event)
        self.logger.info(
            f"EVAL | {event['id']} | "
            f"correct={event['l2_correctness']}/5 | "
            f"relevant={event['l2_relevance']}/5 | "
            f"safety={event['l2_safety']} | "
            f"tox={event['l4_toxicity']}"
        )

    def log_safety_violation(self, record: dict, reason: str):
        self.logger.warning(
            f"⚠️ SAFETY | {record.get('test_case_id')} | "
            f"category={record.get('category')} | reason={reason}"
        )

    def log_adversarial_breach(self, record: dict):
        self.logger.critical(
            f"🚨 BREACH | {record.get('test_case_id')} | "
            f"attack={record.get('attack_type')} | "
            f"output_preview={record.get('output','')[:100]}"
        )

    def finalize(self):
        summary_path = f"{self.log_dir}/{self.model_key}_obs_summary.json"
        with open(summary_path,"w") as f:
            json.dump(self.events, f, indent=2)
        self.logger.info(
            f"RUN COMPLETE | total_events={len(self.events)} | "
            f"summary_saved={summary_path}"
        )
        print(f"📡 Observability summary → {summary_path}")
        print(f"📋 Full log              → {self.log_path}")
        return summary_path