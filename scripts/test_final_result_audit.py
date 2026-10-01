"""Exercise the CPU audit without a GPU or private research data."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import binomtest

from audit_9000_saved_colab import audit_9000_saved
from audit_final_results_20260930 import paired, parse_final, read_unique


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class AuditTests(unittest.TestCase):
    def test_paired_counts(self):
        a = np.array([False] * 30 + [True] * 8970)
        b = np.array([True] * 30 + [False] * 13 + [True] * 8957)
        result = paired(a, b, "test")
        self.assertAlmostEqual(result["exact_p"], binomtest(30, 43).pvalue)
        self.assertAlmostEqual(result["change_pp"], 17 / 9000 * 100)

    def test_parser_does_not_guess(self):
        self.assertEqual(parse_final("Happy or Depression"), "")
        self.assertEqual(parse_final("The label is Happy."), "Happy")
        self.assertEqual(parse_final("Happy? No. Final label: Neutral"), "Neutral")

    def test_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            pd.DataFrame({"example_id": ["a", "a"]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                read_unique(path)

    def test_full_saved_audit_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "additional_9000_20260930"
            p1, p2 = base / "phase1", base / "phase2/llama3_direct_cot_v6c_v1"
            p1.mkdir(parents=True)
            p2.mkdir(parents=True)
            labels = ["Depression", "Neutral", "Happy"]
            full = pd.DataFrame({"example_id": [f"SYNTH_{i:05d}" for i in range(9000)],
                                 "target_label": np.repeat(labels, 3000)})
            full["phase1_label"] = full.target_label
            full.loc[:39, "phase1_label"] = "Happy"
            full["phase1_confidence"] = np.where(full.index < 111, .6, .9)
            full["phase1_routed"] = full.index < 111
            source = p1 / "final_phase1_reasoning_input.csv"
            full.to_csv(source, index=False)
            p1manifest = {"model_file_sha256": {"model.safetensors": "0eeb8b16aef6a5df3045c2924d4701305f0ca08dc85d8a66efc3daff33b253d6"},
                          "temperature": 1.4801950079829693, "threshold": .7}
            write_json(p1 / "experiment_manifest.json", p1manifest)
            handoff = {"input_sha256": digest(source.read_bytes()), "expected_total_rows": 9000,
                       "expected_routed_rows": 111, "phase1_manifest": p1manifest}
            write_json(p1 / "phase2_handoff.json", handoff)
            plan = p2 / "frozen_llama3_v6c_plan.json"
            write_json(plan, {"synthetic_plan": True})
            manifest = {"phase1_handoff": handoff, "model_revision": "53346005fb0ef11d3b6a83b12c895cca40156b6c",
                        "source_protocols": {"cached_plan_original_sha256": digest(plan.read_bytes())}}
            write_json(p2 / "experiment_manifest.json", manifest)
            fingerprint = digest(json.dumps(manifest, sort_keys=True).encode())
            summaries = []
            methods = {"direct": ("direct_answer", "Llama 3 Direct"), "cot": ("cot_final_answer", "Llama 3 CoT"),
                       "v6c": ("sd_final_answer", "Llama 3 SELF-DISCOVER v6c")}
            for method, (answer, name) in methods.items():
                count = 5 if method == "cot" else 1
                # Match run_all_methods: generation records carry no targets.
                raw = full.loc[:110, ["example_id", "phase1_label"]].copy()
                raw["final_label"] = "Depression"
                raw.loc[0, "final_label"] = ""
                raw[answer] = raw.final_label.map(lambda x: f"Final label: {x}" if x else "Cannot classify.")
                raw["input_tokens_total"], raw["generated_tokens_total"] = count * 10, count * 4
                raw.sample(frac=1, random_state=42).to_csv(p2 / f"llama3_{method}_results.csv", index=False)
                final = full.copy()
                final["final_label"] = final.phase1_label
                final.loc[1:110, "final_label"] = "Depression"
                final["phase2_fallback_to_phase1"] = final.index == 0
                final.sample(frac=1, random_state=43).to_csv(p2 / f"llama3_{method}_end_to_end_predictions.csv", index=False)
                for row in raw.itertuples():
                    key = digest(row.example_id.encode())
                    write_json(p2 / "records" / method / f"{key}.json", {
                        "example_id": row.example_id, "run_fingerprint": fingerprint, "final_label": row.final_label or None})
                    for i in range(count):
                        request = {"messages": [], "settings": {"do_sample": False, "max_new_tokens": 1536 if method == "v6c" else 1024}}
                        write_json(p2 / "stages" / method / key / f"stage_{i+1:02d}.json", {
                            "run_fingerprint": fingerprint, "request": request,
                            "request_sha256": digest(json.dumps(request, sort_keys=True).encode()),
                            "generation": {"generated_tokens": 4, "input_tokens": 10}})
                summaries.append({"method": name, "total_rows": 9000, "routed_rows": 111, "parse_failures": 1,
                                  "end_to_end_accuracy": final.final_label.eq(final.target_label).mean(),
                                  "end_to_end_macro_f1": f1_score(final.target_label, final.final_label, average="macro"),
                                  "corrected": 39, "introduced": 0})
            pd.DataFrame(summaries).to_csv(p2 / "llama3_reasoning_method_summary.csv", index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                archive = audit_9000_saved(root)
            self.assertTrue(archive.is_file())
            evidence = json.loads((archive.parent / "audit_evidence.json").read_text())
            self.assertEqual(evidence["generation_calls_checked"], 777)
            # Optional labels in another export cannot silently disagree.
            raw_path = p2 / "llama3_direct_results.csv"
            raw_saved = pd.read_csv(raw_path, keep_default_na=False)
            with_targets = raw_saved.copy()
            with_targets["target_label"] = "Neutral"
            with_targets.to_csv(raw_path, index=False)
            with self.assertRaises(AssertionError), contextlib.redirect_stdout(io.StringIO()):
                audit_9000_saved(root)
            with_targets["target_label"] = "Depression"
            with_targets.to_csv(raw_path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                audit_9000_saved(root)
            raw_saved.to_csv(raw_path, index=False)
            # A changed final prediction must fail, including non-routed examples.
            path = p2 / "llama3_direct_end_to_end_predictions.csv"
            bad = pd.read_csv(path)
            bad.loc[bad.example_id == "SYNTH_08000", "final_label"] = "Neutral"
            bad.to_csv(path, index=False)
            with self.assertRaises(AssertionError), contextlib.redirect_stdout(io.StringIO()):
                audit_9000_saved(root)


if __name__ == "__main__":
    unittest.main()
