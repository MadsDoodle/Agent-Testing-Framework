import json
import os
from collections import Counter

from src.config import CFG


class TestCaseLoader:
    """
    Loads, filters, and validates test cases from disk.
    Keeps the pipeline decoupled from raw JSON handling.
    """

    VALID_CATEGORIES   = {"normal","edge","adversarial","safety","retrieval"}
    VALID_DIFFICULTIES = {"easy","medium","hard"}
    REQUIRED_FIELDS    = {"id","input","expected_behavior","category","difficulty"}

    def __init__(self, filepath: str):
        self.filepath   = filepath
        self.test_cases = []

    def load(self) -> list:
        with open(self.filepath,"r") as f:
            self.test_cases = json.load(f)
        print(f"📂 Loaded {len(self.test_cases)} test cases from '{self.filepath}'")
        self._validate()
        return self.test_cases

    def _validate(self):
        issues = []
        for tc in self.test_cases:
            missing = self.REQUIRED_FIELDS - set(tc.keys())
            if missing:
                issues.append(f"  [{tc.get('id','?')}] Missing fields: {missing}")
            if tc.get("category") not in self.VALID_CATEGORIES:
                issues.append(f"  [{tc.get('id','?')}] Invalid category: {tc.get('category')}")
            if tc.get("difficulty") not in self.VALID_DIFFICULTIES:
                issues.append(f"  [{tc.get('id','?')}] Invalid difficulty: {tc.get('difficulty')}")
        if issues:
            print("⚠️  Validation warnings:")
            for i in issues: print(i)
        else:
            print("✅ All test cases validated successfully")

    def filter_by_category(self, category: str) -> list:
        return [tc for tc in self.test_cases if tc["category"] == category]

    def filter_by_difficulty(self, difficulty: str) -> list:
        return [tc for tc in self.test_cases if tc["difficulty"] == difficulty]

    def filter_by_tag(self, tag: str) -> list:
        return [tc for tc in self.test_cases if tag in tc.get("tags", [])]

    def summary(self):
        cats  = Counter(tc["category"]   for tc in self.test_cases)
        diffs = Counter(tc["difficulty"] for tc in self.test_cases)

        print("\n📊 Test Case Summary")
        print("  Categories:")
        for cat, n in cats.items():
            print(f"    {cat:<15} → {n} case(s)")
        print("  Difficulties:")
        for d, n in diffs.items():
            print(f"    {d:<15} → {n} case(s)")
        print(f"  Total: {len(self.test_cases)}")


class ResponseCollector:
    """
    Accumulates raw execution records before evaluation.
    Keeps execution and evaluation cleanly separated per the design doc.
    """

    def __init__(self):
        self.records = []

    def add(self, record: dict):
        self.records.append(record)

    def get_all(self) -> list:
        return self.records

    def get_by_category(self, category: str) -> list:
        return [r for r in self.records if r.get("category") == category]

    def get_failures(self) -> list:
        return [r for r in self.records if r.get("status") == "failure"]

    def summary(self):
        total    = len(self.records)
        success  = sum(1 for r in self.records if r["status"] == "success")
        failures = total - success
        avg_lat  = round(
            sum(r["latency"] for r in self.records if r["latency"]) / max(total,1), 4
        )
        print(f"\n📋 Response Collector Summary")
        print(f"  Total   : {total}")
        print(f"  Success : {success}")
        print(f"  Failures: {failures}")
        print(f"  Avg Latency: {avg_lat}s")

    def save(self, model_key: str, tag: str = "raw_responses"):
        path = f"{CFG['results_dir']}/{model_key}_{tag}.json"
        os.makedirs(CFG["results_dir"], exist_ok=True)
        with open(path,"w") as f:
            json.dump(self.records, f, indent=2)
        print(f"💾 Raw responses saved → {path}")
        return path