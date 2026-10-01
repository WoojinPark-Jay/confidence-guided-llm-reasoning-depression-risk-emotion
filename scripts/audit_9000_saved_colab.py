"""CPU-only audit of saved results. No model loading or generation."""

import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score


def audit_9000_saved(root):
    labels = ["Depression", "Neutral", "Happy"]
    base = root / "additional_9000_20260930"
    p1dir = base / "phase1"
    p2dir = base / "phase2/llama3_direct_cot_v6c_v1"
    output = base / "audit_20260930"
    output.mkdir(parents=True, exist_ok=True)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def read(path):
        d = pd.read_csv(path, keep_default_na=False, dtype={"example_id": str})
        if d.example_id.duplicated().any() or d.example_id.eq("").any():
            raise ValueError(f"Duplicate/empty IDs: {path.name}")
        return d.set_index("example_id").sort_index()

    def truth(value):
        if str(value).lower() in ("true", "1"):
            return True
        if str(value).lower() in ("false", "0"):
            return False
        raise ValueError("Invalid Boolean")

    def parse(text):
        m = re.search(r"Final\s+label\s*\*{0,2}\s*:\s*\*{0,2}\s*(Depression|Neutral|Happy)\b", str(text), re.I)
        if m:
            return m[1].capitalize()
        unique = {x.capitalize() for x in re.findall(r"\b(Depression|Neutral|Happy)\b", str(text), re.I)}
        return next(iter(unique)) if len(unique) == 1 else ""

    def holm(values):
        order = np.argsort(values)
        adjusted = np.empty(len(values))
        adjusted[order] = np.minimum(1, np.maximum.accumulate(np.array(values)[order] * np.arange(len(values), 0, -1)))
        return adjusted

    source = p1dir / "final_phase1_reasoning_input.csv"
    handoff = json.loads((p1dir / "phase2_handoff.json").read_text())
    manifest = json.loads((p2dir / "experiment_manifest.json").read_text())
    assert sha(source) == handoff["input_sha256"]
    assert manifest["phase1_handoff"] == handoff
    phase1_manifest = json.loads((p1dir / "experiment_manifest.json").read_text())
    assert handoff["phase1_manifest"] == phase1_manifest
    assert phase1_manifest["model_file_sha256"]["model.safetensors"] == "0eeb8b16aef6a5df3045c2924d4701305f0ca08dc85d8a66efc3daff33b253d6"
    assert phase1_manifest["temperature"] == 1.4801950079829693
    assert phase1_manifest["threshold"] == 0.7
    assert sha(p2dir / "frozen_llama3_v6c_plan.json") == manifest["source_protocols"]["cached_plan_original_sha256"]
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    assert manifest["model_revision"] == "53346005fb0ef11d3b6a83b12c895cca40156b6c"
    full = read(source)
    assert len(full) == 9000 == handoff["expected_total_rows"]
    assert full.target_label.value_counts().to_dict() == dict.fromkeys(labels, 3000)
    routed = full.phase1_routed.map(truth)
    assert routed.sum() == 111 == handoff["expected_routed_rows"]
    assert np.array_equal(routed, full.phase1_confidence < 0.7)
    vectors = {"Phase 1": full.phase1_label.eq(full.target_label).to_numpy()}
    summaries = []
    methods = {"direct": "direct_answer", "cot": "cot_final_answer", "v6c": "sd_final_answer"}
    saved_summary = pd.read_csv(p2dir / "llama3_reasoning_method_summary.csv").set_index("method")
    names = {"direct": "Llama 3 Direct", "cot": "Llama 3 CoT", "v6c": "Llama 3 SELF-DISCOVER v6c"}
    inventory = []
    for method, answer in methods.items():
        raw_path = p2dir / f"llama3_{method}_results.csv"
        final_path = p2dir / f"llama3_{method}_end_to_end_predictions.csv"
        raw, final = read(raw_path), read(final_path)
        assert set(raw.index) == set(full.index[routed])
        assert final.index.equals(full.index)
        assert raw.final_label.isin(labels + [""]).all()
        # Generation records omit reference labels; Phase 1 is the label source.
        # If a separate export includes labels, they must still match by ID.
        if "target_label" in raw.columns:
            assert raw["target_label"].equals(full.loc[raw.index, "target_label"])
        assert raw.phase1_label.equals(full.loc[raw.index, "phase1_label"])
        assert raw[answer].map(parse).equals(raw.final_label)
        recomputed = full.phase1_label.copy()
        recomputed.loc[raw.index] = raw.final_label.where(raw.final_label.isin(labels), full.loc[raw.index, "phase1_label"])
        assert recomputed.equals(final.final_label)
        assert final.target_label.equals(full.target_label)
        assert final.phase1_label.equals(full.phase1_label)
        expected_fallback = pd.Series(False, index=full.index)
        expected_fallback.loc[raw.index] = raw.final_label.eq("")
        assert final.phase2_fallback_to_phase1.map(truth).equals(expected_fallback)
        for eid, row in raw.iterrows():
            key = hashlib.sha256(eid.encode()).hexdigest()
            record = json.loads((p2dir / "records" / method / f"{key}.json").read_text())
            assert record["example_id"] == eid and record["run_fingerprint"] == fingerprint
            assert (record["final_label"] or "") == row.final_label
            stages = sorted((p2dir / "stages" / method / key).glob("stage_*.json"))
            assert len(stages) == (5 if method == "cot" else 1)
            generated_tokens, input_tokens = 0, 0
            for path in stages:
                stage = json.loads(path.read_text())
                assert stage["run_fingerprint"] == fingerprint
                assert hashlib.sha256(json.dumps(stage["request"], sort_keys=True).encode()).hexdigest() == stage["request_sha256"]
                settings = stage["request"]["settings"]
                assert settings["do_sample"] is False
                assert settings["max_new_tokens"] == (1536 if method == "v6c" else 1024)
                generated_tokens += stage["generation"]["generated_tokens"]
                input_tokens += stage["generation"]["input_tokens"]
            assert generated_tokens == row.generated_tokens_total and input_tokens == row.input_tokens_total
        current = recomputed.eq(full.target_label).to_numpy()
        vectors[method] = current
        stats = {"method": names[method], "total_rows": len(full), "routed_rows": len(raw),
                 "parse_failures": int(raw.final_label.eq("").sum()),
                 "end_to_end_accuracy": float(current.mean()),
                 "end_to_end_macro_f1": f1_score(full.target_label, recomputed, labels=labels, average="macro"),
                 "corrected": int((~vectors["Phase 1"] & current).sum()),
                 "introduced": int((vectors["Phase 1"] & ~current).sum())}
        for name, value in stats.items():
            if name != "method":
                assert np.isclose(saved_summary.loc[names[method], name], value, atol=1e-12, rtol=0), name
        summaries.append(stats)
        inventory.append({"method": method, "raw_sha256": sha(raw_path), "end_to_end_sha256": sha(final_path)})
    rows = []
    for a_name, b_name, family in [("Phase 1", m, "phase1_3") for m in methods] + [(m, "v6c", "sd_2") for m in ("direct", "cot")]:
        a, b = vectors[a_name], vectors[b_name]
        c, i = int((~a & b).sum()), int((a & ~b).sum())
        seed = int.from_bytes(hashlib.sha256((a_name+":"+b_name).encode()).digest()[:4], "big")
        draws = np.random.default_rng(seed).multinomial(len(full), [c/len(full), i/len(full), 1-(c+i)/len(full)], size=50000)
        low, high = np.percentile((draws[:, 0]-draws[:, 1])/len(full)*100, [2.5,97.5])
        rows.append({"reference": a_name, "method": b_name, "family": family, "corrected": c, "introduced": i,
                     "change_pp": (c-i)/len(full)*100, "ci_low_pp": low, "ci_high_pp": high,
                     "exact_p": binomtest(c, c+i, .5).pvalue if c+i else 1.0})
    table = pd.DataFrame(rows)
    for _, indices in table.groupby("family").groups.items():
        table.loc[indices, "holm_p"] = holm(table.loc[indices, "exact_p"].tolist())
    pd.DataFrame(summaries).to_csv(output / "row_verified_summary.csv", index=False)
    table.to_csv(output / "paired_statistics.csv", index=False)
    evidence = {"row_audit_passed": True, "total_rows": len(full), "routed_rows": int(routed.sum()),
                "generation_calls_checked": 777, "source_files": inventory, "statistics": table.to_dict("records")}
    (output / "audit_evidence.json").write_text(json.dumps(evidence, indent=2))

    # Include only the missing final results and final 9000 evidence, never weights or secrets.
    missing_final_dirs = [root / "phase2/llama2_cot_matched_1024_v1",
                          root / "mixed_emotion/phase2/llama2_cot_matched_1024_v1",
                          root / "mixed_emotion/phase2/llama2_direct_final_sd_v1",
                          root / "mixed_emotion/phase2/llama3_reasoning_comparison"]
    archive_path = output / "final_results_audit_package.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for directory in [p1dir, p2dir, *missing_final_dirs]:
            if not directory.exists():
                print("Missing saved directory:", directory)
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix in (".csv", ".json") and "batches" not in path.parts:
                    archive.write(path, str(path.relative_to(root)))
        archive.write(output / "audit_evidence.json", "audit_evidence.json")
        archive.write(output / "paired_statistics.csv", "paired_statistics.csv")
    print("AUDIT PASSED: 9000 IDs, 111 routed posts, all three outputs and 777 generation calls verified.")
    print(table.to_string(index=False))
    print("AUDIT_EVIDENCE_JSON=" + json.dumps(evidence))
    print("Saved verification package:", archive_path)
    return archive_path


if __name__ == "__main__":
    from google.colab import drive, files
    drive.mount("/content/drive")
    audit_package = audit_9000_saved(Path("/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/unified_final"))
    files.download(str(audit_package))
