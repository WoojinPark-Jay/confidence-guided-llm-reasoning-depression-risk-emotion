"""Complete the audit using the downloaded Colab package; no model inference."""

import contextlib
import io
import json

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import audit_final_results_20260930 as audit
from audit_9000_saved_colab import audit_9000_saved


def main():
    package = audit.PRIVATE.parent / "received_package"
    with contextlib.redirect_stdout(io.StringIO()) as log:
        audit.main()
        audit_9000_saved(package)
    (audit.PRIVATE / "complete_validation_log.txt").write_text(log.getvalue())
    evidence = json.loads((package / "additional_9000_20260930/audit_20260930/audit_evidence.json").read_text())
    received = json.loads((package / "audit_evidence.json").read_text())
    assert evidence == received, "Recomputed 9000 audit differs from received evidence"
    (audit.OUT / "additional9000_audit_evidence.json").write_text(json.dumps(evidence, indent=2))
    metrics = pd.read_csv(audit.OUT / "row_verified_metrics.csv").to_dict("records")
    inventory = json.loads((audit.PRIVATE / "source_inventory.json").read_text())
    bases = {}
    for dataset, file in [("Reddit", "phase1_test_predictions.csv"), ("Mixed Emotion", "phase1_mixed_emotion_predictions.csv")]:
        frame = audit.read_unique(audit.DOWNLOADS / "distilbert_phase1_final_outputs (1)" / file)
        frame["target_label"] = frame["target_label" if "target_label" in frame else "label_str"].map(audit.label)
        frame["phase1_label"] = frame.phase1_label.map(audit.label)
        frame["phase1_routed"] = frame.phase1_routed.map(audit.boolean)
        bases[dataset] = frame
    sources = []
    for dataset, prefix in [("Reddit", "phase2"), ("Mixed Emotion", "mixed_emotion/phase2")]:
        folder = package / prefix / "llama2_cot_matched_1024_v1"
        sources.append((dataset, "Llama 2", "CoT", folder, "llama2_cot_matched", "llama2_final_answer", "llama2_final_label"))
    sources.append(("Mixed Emotion", "Llama 2", "Direct", package / "mixed_emotion/phase2/llama2_direct_final_sd_v1",
                    "llama2_direct", "direct_answer", "final_label"))
    for method, answer in [("Direct", "direct_answer"), ("CoT", "cot_final_answer")]:
        sources.append(("Mixed Emotion", "Llama 3", method, package / "mixed_emotion/phase2/llama3_reasoning_comparison",
                        "llama3_" + method.lower(), answer, "final_label"))
    for dataset, model, method, folder, stem, answer, column in sources:
        path, full_path = folder / f"{stem}_results.csv", folder / f"{stem}_end_to_end_predictions.csv"
        final, stat, _ = audit.audit_raw(bases[dataset], path, answer, end_to_end=full_path, final_col=column)
        metrics.append({"dataset": dataset, "model": model, "method": method, **stat})
        inventory.append({"dataset": dataset, "model": model, "method": method, "raw_file": str(path), "full_file": str(full_path), **stat})
        pd.DataFrame({"target_label": bases[dataset].target_label, "phase1_label": bases[dataset].phase1_label,
                      "final_label": final}).to_csv(audit.PRIVATE / (f"{dataset}_{model}_{method}".replace(" ", "_") + "_aligned_predictions.csv"))
    additional = "Additional Reddit 9000"
    base9000 = audit.read_unique(package / "additional_9000_20260930/phase1/final_phase1_reasoning_input.csv")
    base9000["phase1_routed"] = base9000.phase1_routed.map(audit.boolean)
    bases[additional] = base9000
    folder = package / "additional_9000_20260930/phase2/llama3_direct_cot_v6c_v1"
    for method, slug in [("Direct", "direct"), ("CoT", "cot"), ("SELF-DISCOVER", "v6c")]:
        path = folder / f"llama3_{slug}_end_to_end_predictions.csv"
        full = audit.read_unique(path).reindex(base9000.index)
        raw_path = folder / f"llama3_{slug}_results.csv"
        raw = audit.read_unique(raw_path)
        correct, before = full.final_label.eq(full.target_label), full.phase1_label.eq(full.target_label)
        stat = {"n": len(full), "routed_rows": len(raw), "parse_failures": int(raw.final_label.eq("").sum()),
                "accuracy_percent": correct.mean() * 100,
                "macro_f1_percent": f1_score(full.target_label, full.final_label, labels=audit.LABELS, average="macro") * 100,
                "corrected": int((~before & correct).sum()), "introduced": int((before & ~correct).sum()),
                "recovered_labels": 0, "raw_sha256": audit.sha(raw_path), "full_sha256": audit.sha(path)}
        metrics.append({"dataset": additional, "model": "Llama 3", "method": method, **stat})
        inventory.append({"dataset": additional, "model": "Llama 3", "method": method, "raw_file": str(raw_path), "full_file": str(path), **stat})
        full[["target_label", "phase1_label", "final_label"]].to_csv(audit.PRIVATE / (f"{additional}_Llama_3_{method}".replace(" ", "_") + "_aligned_predictions.csv"))
    metrics = pd.DataFrame(metrics).sort_values(["dataset", "model", "method"])
    expected = pd.read_csv(audit.OUT / "phase1_paired_statistics.csv").set_index(["dataset", "model", "method"]).sort_index()
    vectors = {}
    for row in metrics.itertuples():
        key = (row.dataset, row.model, row.method)
        prior = expected.loc[key]
        assert abs(row.accuracy_percent - prior.accuracy_percent) < 1e-10
        assert abs(row.macro_f1_percent - prior.macro_f1_percent) < 0.00006
        assert row.corrected == prior.a_wrong_b_correct and row.introduced == prior.a_correct_b_wrong
        prefix = "_".join(key).replace(" ", "_")
        aligned = audit.read_unique(audit.PRIVATE / f"{prefix}_aligned_predictions.csv")
        assert aligned.index.equals(bases[row.dataset].index)
        vectors[key] = aligned.final_label.eq(aligned.target_label).to_numpy()
        report = classification_report(aligned.target_label, aligned.final_label, labels=audit.LABELS, output_dict=True, zero_division=0)
        (audit.OUT / f"{prefix}_classification_report.json").write_text(json.dumps(report, indent=2))
        pd.DataFrame(confusion_matrix(aligned.target_label, aligned.final_label, labels=audit.LABELS),
                     index=audit.LABELS, columns=audit.LABELS).to_csv(audit.OUT / f"{prefix}_confusion_matrix.csv", index_label="target_label")
        expected.loc[key, "evidence"] = "row_verified"
        expected.loc[key, "macro_f1_percent"] = row.macro_f1_percent
    metrics.to_csv(audit.OUT / "row_verified_metrics.csv", index=False)
    expected.reset_index().to_csv(audit.OUT / "phase1_paired_statistics.csv", index=False)
    (audit.PRIVATE / "source_inventory.json").write_text(json.dumps(inventory, indent=2))
    cross = []
    for dataset, model in sorted({key[:2] for key in vectors}):
        for method in ("Direct", "CoT"):
            sd_key = (dataset, model, "SELF-DISCOVER")
            stat = audit.paired(vectors[(dataset, model, method)], vectors[sd_key], str(sd_key)+method)
            if dataset == additional:
                # Keep the executed Colab bootstrap seed/CI for this comparison.
                source = next(s for s in evidence["statistics"] if s["reference"] == method.lower())
                assert stat["a_wrong_b_correct"] == source["corrected"] and stat["a_correct_b_wrong"] == source["introduced"]
                stat.update({k: source[k] for k in ("ci_low_pp", "ci_high_pp")})
            cross.append({"dataset": dataset, "model": model, "comparison": "SD vs " + method, "status": "row_verified", **stat})
    cross = pd.DataFrame(cross)
    for _, indices in cross.groupby(["dataset", "model"]).groups.items():
        cross.loc[indices, "holm_p_family2"] = audit.holm_adjust(cross.loc[indices, "exact_p"].tolist())
    cross.to_csv(audit.OUT / "sd_vs_baselines_paired_statistics.csv", index=False)
    print("All 15 final conditions row-verified; 9000 audit reproduced exactly.")
    print(cross.to_string(index=False))


if __name__ == "__main__":
    main()
