import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/colab/final/07_2_existing9000_llama3_direct_cot_v6c_colab.ipynb"


def namespace():
    book = json.loads(NOTEBOOK.read_text())
    scope = dict(np=np, pd=pd, re=re, json=json, hashlib=hashlib, Path=Path,
                 f1_score=f1_score, classification_report=classification_report, confusion_matrix=confusion_matrix,
                 LABELS=["Depression", "Neutral", "Happy"], BASE_SEED=42, MAX_ROWS=None,
                 MAX_REASONING_CHARACTERS=6000, REASONING_HEAD_CHARACTERS=3500,
                 REASONING_TAIL_CHARACTERS=2500, LLAMA3_MAX_NEW_TOKENS=1024, V6_MAX_NEW_TOKENS=1536)
    for index in (5, 7, 8):
        exec("".join(book["cells"][index]["source"]), scope)
    return scope


class Phase2Tests(unittest.TestCase):
    def setUp(self):
        self.n = namespace()

    def test_frozen_plan_and_revision(self):
        evidence = self.n["PROTOCOL_EVIDENCE"]
        self.assertEqual(hashlib.sha256(self.n["CACHED_PLAN_JSON"].encode()).hexdigest(),
                         evidence["cached_plan_original_sha256"])
        book = json.loads(NOTEBOOK.read_text())
        source = "".join(book["cells"][5]["source"])
        self.assertEqual(source.count("revision=MODEL_REVISION"), 2)
        self.assertIn("No silent truncation", source)

    def test_exact_method_calls_and_no_reference_label_in_prompts(self):
        calls = []

        def generate(tokenizer, model, messages, **settings):
            calls.append((json.loads(json.dumps(messages)), settings))
            return {"text": "Final label: Happy", "input_tokens": 15, "generated_tokens": 4,
                    "elapsed_seconds": 0.1, "hit_token_limit": False}

        self.n["chat_generate"] = generate
        row = {"example_id": "TEST", "phase1_label": "Neutral", "phase2_original_text": "post sentinel"}
        a = self.n["run_llama3_direct"](None, None, row)
        b = self.n["run_llama3_cot"](None, None, row)
        c = self.n["run_v6_max_prompt"](None, None, row, self.n["CACHED_PLAN"])
        self.assertEqual(len(calls), 7)
        self.assertEqual([s["max_new_tokens"] for _, s in calls], [1024] * 6 + [1536])
        self.assertTrue(all(not settings["do_sample"] for _, settings in calls))
        self.assertEqual([a["final_label"], b["final_label"], c["final_label"]], ["Happy"] * 3)
        self.assertNotIn("Phase 1 classifier predicted", calls[-1][0][-1]["content"])
        self.assertIn(self.n["CACHED_PLAN"]["reasoning_structure"].strip(), calls[-1][0][-1]["content"])
        self.assertEqual(b["generated_tokens_total"], 20)

    def test_parser_ambiguity_and_fallback(self):
        parser = self.n["parse_final_label"]
        self.assertEqual(parser("Final label: Happy"), "Happy")
        self.assertTrue(pd.isna(parser("Depression or Happy")))
        full = pd.DataFrame({"example_id": ["a", "b", "c"], "target_label": ["Depression", "Happy", "Neutral"],
                             "phase1_label": ["Depression", "Neutral", "Neutral"], "phase1_routed": [True, True, False]})
        results = pd.DataFrame({"example_id": ["b", "a"], "pred": ["Happy", None]})
        merged, stats = self.n["summarize_method"](full, results, "pred", "test")
        self.assertEqual(stats["total_rows"], 3)
        self.assertEqual(stats["parse_failures"], 1)
        self.assertEqual(stats["end_to_end_accuracy"], 1)
        self.assertEqual(stats["net_corrections"], 1)
        self.assertEqual(merged.final_label.tolist(), ["Depression", "Happy", "Neutral"])

    def test_stage_resume_and_mismatched_prompt(self):
        calls = []

        def base(*args, **kwargs):
            calls.append(1)
            return {"text": "Final label: Happy"}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            first = self.n["cached_generation"](base, path, "run")
            first(None, None, [{"role": "user", "content": "one"}], seed=42)
            resumed = self.n["cached_generation"](base, path, "run")
            resumed(None, None, [{"role": "user", "content": "one"}], seed=42)
            self.assertEqual(len(calls), 1)
            wrong = self.n["cached_generation"](base, path, "run")
            with self.assertRaises(ValueError):
                wrong(None, None, [{"role": "user", "content": "changed"}], seed=42)

    def test_manifest_and_complete_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fp = self.n["lock_run"](out, {"input": "one"})
            self.assertEqual(fp, self.n["lock_run"](out, {"input": "one"}))
            with self.assertRaises(ValueError):
                self.n["lock_run"](out, {"input": "two"})
            routed = pd.DataFrame({"example_id": ["x"]})
            with self.assertRaises(ValueError):
                self.n["load_method_records"](out, "direct", routed, fp)
            path = self.n["record_path"](out, "direct", "x")
            self.n["save_json"](path, {"example_id": "x", "method": "direct", "run_fingerprint": fp,
                                      "final_label": np.nan})
            result = self.n["load_method_records"](out, "direct", routed, fp)
            self.assertEqual(len(result), 1)
            self.assertTrue(pd.isna(result.final_label.iloc[0]))

    def test_9000_handoff_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            labels = np.repeat(self.n["LABELS"], 3000)
            frame = pd.DataFrame({"example_id": [f"ROW_{i}" for i in range(9000)], "target_label": labels,
                                  "phase1_label": labels, "phase1_confidence": 0.9, "phase1_routed": False,
                                  "phase2_original_text": ""})
            frame.loc[0, ["phase1_confidence", "phase1_routed", "phase2_original_text"]] = [0.6, True, "Original post."]
            path = folder / "final_phase1_reasoning_input.csv"
            frame.to_csv(path, index=False)
            manifest = {"model_file_sha256": {"model.safetensors": self.n["FINAL_WEIGHT_SHA256"]},
                        "temperature": 1.4801950079829693, "threshold": 0.7}
            handoff = {"input_sha256": self.n["digest_file"](path), "expected_total_rows": 9000,
                       "expected_routed_rows": 1, "phase1_manifest": manifest}
            summary = {"total_rows": 9000, "routed_rows": 1, "errors": 0, "routed_errors": 0,
                       "phase1_accuracy": 1, "phase1_macro_f1": 1}
            for name, value in [("experiment_manifest.json", manifest), ("phase2_handoff.json", handoff),
                                ("phase1_summary.json", summary)]:
                self.n["save_json"](folder / name, value)
            full, routed, _, _ = self.n["read_phase1_handoff"](folder)
            self.assertEqual(len(full), 9000)
            self.assertEqual(routed.example_id.tolist(), ["ROW_0"])
            frame.loc[0, "phase1_label"] = "Happy"
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                self.n["read_phase1_handoff"](folder)


if __name__ == "__main__":
    unittest.main()
