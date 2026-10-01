"""Audit available final raw predictions; distinguish aggregate-only evidence."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from paired_end_to_end_analysis import exact_mcnemar_p, holm_adjust, paired_bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"
OUT = ROOT / "reports/final_results_audit_20260930"
PRIVATE = ROOT.parent / "final_results_audit_20260930/private"
LABELS = ["Depression", "Neutral", "Happy"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(value):
    text = str(value).removesuffix("_Group").strip()
    if text not in LABELS:
        raise ValueError(f"Invalid label: {value!r}")
    return text


def boolean(value):
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError(f"Invalid Boolean: {value!r}")


def parse_final(text):
    match = re.search(r"Final\s+label\s*\*{0,2}\s*:\s*\*{0,2}\s*(Depression|Neutral|Happy)\b", str(text), re.I)
    if match:
        return match[1].capitalize()
    unique = {s.capitalize() for s in re.findall(r"\b(Depression|Neutral|Happy)\b", str(text), re.I)}
    return next(iter(unique)) if len(unique) == 1 else ""


def read_unique(path):
    frame = pd.read_csv(path, keep_default_na=False, dtype={"example_id": str})
    if frame.example_id.duplicated().any() or frame.example_id.str.strip().eq("").any():
        raise ValueError(f"Duplicate or empty IDs: {path}")
    return frame.set_index("example_id").sort_index()


def paired(a, b, key):
    corrected = int((~a & b).sum())
    introduced = int((a & ~b).sum())
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
    low, high = paired_bootstrap_ci(a, b, np.random.default_rng(seed))
    return {"n": len(a), "a_wrong_b_correct": corrected, "a_correct_b_wrong": introduced,
            "change_pp": (corrected - introduced) / len(a) * 100,
            "ci_low_pp": low, "ci_high_pp": high, "exact_p": exact_mcnemar_p(introduced, corrected)}


def audit_raw(base, path, answer_col, recover=False, end_to_end=None, plan=None, final_col="final_label"):
    raw = read_unique(path)
    expected = base.index[base.phase1_routed]
    if set(raw.index) != set(expected):
        raise ValueError(f"Missing or extra routed IDs: {path}")
    raw = raw.reindex(expected)
    for col in ("phase1_label", "target_label"):
        if not raw[col].map(label).equals(base.loc[expected, col]):
            raise ValueError(f"{col} mismatch: {path}")
    if not np.allclose(raw.phase1_confidence, base.loc[expected].phase1_confidence, rtol=0, atol=1e-12):
        raise ValueError(f"Confidence mismatch: {path}")
    saved = raw[final_col]
    if not saved.isin(LABELS + [""]).all():
        raise ValueError("Invalid stored labels")
    parsed = raw[answer_col].map(parse_final)
    recovered = saved.eq("") & parsed.isin(LABELS)
    if ((saved != parsed) & ~recovered).any():
        raise ValueError(f"Stored label and parser conflict: {path}")
    if recovered.any() and not recover:
        raise ValueError(f"Unexpected unreviewed parser recovery: {path}")
    if plan is not None:
        plan_data = json.loads(plan.read_text())
        for col, key in [("sd_selected_modules", "selected_modules"), ("sd_adapted_modules", "adapted_modules"),
                         ("sd_reasoning_structure", "reasoning_structure")]:
            if not raw[col].map(str.strip).eq(plan_data[key].strip()).all():
                raise ValueError(f"Cached plan mismatch in {col}: {path}")
    final = base.phase1_label.copy()
    chosen = parsed if recover else saved
    final.loc[expected] = chosen.where(chosen.isin(LABELS), base.loc[expected, "phase1_label"])
    if end_to_end is not None:
        full = read_unique(end_to_end)
        if set(full.index) != set(base.index):
            raise ValueError("Full prediction IDs mismatch")
        full = full.reindex(base.index)
        for col in ("phase1_label", "target_label"):
            if not full[col].map(label).equals(base[col]):
                raise ValueError(f"Full {col} mismatch")
        if not full.final_label.equals(final):
            raise ValueError("Raw-to-full final prediction mismatch")
        expected_fallback = pd.Series(False, index=base.index)
        expected_fallback.loc[expected] = chosen.eq("")
        if not full.phase2_fallback_to_phase1.map(boolean).equals(expected_fallback):
            raise ValueError("Fallback flags mismatch")
    correct = final.eq(base.target_label).to_numpy()
    before = base.phase1_label.eq(base.target_label).to_numpy()
    metrics = {"n": len(base), "routed_rows": len(raw), "parse_failures": int(chosen.eq("").sum()),
               "accuracy_percent": float(correct.mean() * 100),
               "macro_f1_percent": f1_score(base.target_label, final, labels=LABELS, average="macro") * 100,
               "corrected": int((~before & correct).sum()), "introduced": int((before & ~correct).sum()),
               "recovered_labels": int(recovered.sum()), "raw_sha256": sha(path),
               "full_sha256": sha(end_to_end) if end_to_end else "", "plan_sha256": sha(plan) if plan else ""}
    return final, metrics, raw.loc[recovered, [answer_col]].assign(recovered_label=parsed[recovered])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    p1root = DOWNLOADS / "distilbert_phase1_final_outputs (1)"
    bases = {}
    for dataset, filename, n, nr in [("Reddit", "phase1_test_predictions.csv", 12000, 218),
                                    ("Mixed Emotion", "phase1_mixed_emotion_predictions.csv", 300, 86)]:
        base = read_unique(p1root / filename)
        base["target_label"] = base.get("target_label", base.label_str).map(label)
        base["phase1_label"] = base.phase1_label.map(label)
        base["phase1_routed"] = base.phase1_routed.map(boolean)
        assert len(base) == n and base.phase1_routed.sum() == nr
        assert np.array_equal(base.phase1_routed, base.phase1_confidence < 0.7)
        bases[dataset] = base
    sources = []
    for model in ("llama2", "llama3"):
        for slug, dataset in [("reddit", "Reddit"), ("mixed_emotion", "Mixed Emotion")]:
            raw = next(DOWNLOADS.rglob(f"{model}_self_discover_{slug}_sd_v6c_compact_results.csv"))
            full = next(DOWNLOADS.rglob(f"{model}_self_discover_{slug}_v6c_compact_end_to_end_predictions.csv"))
            plan = next(DOWNLOADS.rglob(f"{model}_self_discover_task_level_reasoning_structure_compact.json"))
            sources.append((dataset, model.replace("llama", "Llama "), "SELF-DISCOVER", raw, "sd_final_answer", False, full, plan))
    sources += [
        ("Reddit", "Llama 3", "Direct", DOWNLOADS / "llama3_direct_results.csv", "direct_answer", False, None, None),
        ("Reddit", "Llama 3", "CoT", DOWNLOADS / "llama3_cot_results.csv", "cot_final_answer", True, None, None),
        ("Reddit", "Llama 2", "Direct", DOWNLOADS / "llama2_direct_results-2.csv", "direct_answer", False, None, None),
    ]
    vectors, audited, integrity = {}, [], []
    for dataset, model, method, path, col, recover, full, plan in sources:
        final, metrics, recovery = audit_raw(bases[dataset], path, col, recover, full, plan)
        key = f"{dataset}_{model}_{method}".replace(" ", "_")
        vectors[(dataset, model, method)] = final.eq(bases[dataset].target_label).to_numpy()
        audited.append({"dataset": dataset, "model": model, "method": method, **metrics})
        integrity.append({"dataset": dataset, "model": model, "method": method, "raw_file": str(path),
                          "full_file": str(full) if full else None, **metrics})
        pd.DataFrame({"target_label": bases[dataset].target_label, "phase1_label": bases[dataset].phase1_label,
                      "final_label": final}).to_csv(PRIVATE / f"{key}_aligned_predictions.csv")
        if len(recovery):
            recovery.to_csv(PRIVATE / f"{key}_explicit_label_recovery.csv")
        pd.DataFrame(confusion_matrix(bases[dataset].target_label, final, labels=LABELS),
                     index=LABELS, columns=LABELS).to_csv(OUT / f"{key}_confusion_matrix.csv", index_label="target_label")
        report = classification_report(bases[dataset].target_label, final, labels=LABELS, output_dict=True, zero_division=0)
        (OUT / f"{key}_classification_report.json").write_text(json.dumps(report, indent=2))
    audited = pd.DataFrame(audited)
    audited.to_csv(OUT / "row_verified_metrics.csv", index=False)
    (PRIVATE / "source_inventory.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2))
    aggregate = pd.read_csv(ROOT / "reports/unified_v6c_revision_20260929/evidence/final_results_and_count_statistics.csv")
    for _, row in audited.iterrows():
        expected = aggregate[(aggregate.dataset == row.dataset) & (aggregate.model == row.model) & (aggregate.method == row.method)].iloc[0]
        assert row.corrected == expected.corrected and row.introduced == expected.introduced, row.to_dict()
        assert abs(row.macro_f1_percent - expected.macro_f1_percent) < 0.00006, row.to_dict()
    # Count statistics need only (+1,-1,0) paired correctness counts, not inferred IDs.
    additional = pd.DataFrame([
        ["Additional Reddit 9000", "Llama 3", "Direct", 97.7242, 32, 19, 9000, 8782],
        ["Additional Reddit 9000", "Llama 3", "CoT", 97.7234, 28, 15, 9000, 8782],
        ["Additional Reddit 9000", "Llama 3", "SELF-DISCOVER", 97.7688, 30, 13, 9000, 8782],
    ], columns=["dataset", "model", "method", "macro_f1_percent", "corrected", "introduced", "N", "baseline_correct"])
    all_rows = pd.concat([aggregate, additional], ignore_index=True)
    counts = []
    for _, row in all_rows.iterrows():
        n, c, i = int(row.N), int(row.corrected), int(row.introduced)
        a = np.zeros(n, dtype=bool); b = np.zeros(n, dtype=bool)
        b[:c] = True; a[c:c+i] = True
        stats = paired(a, b, f"{row.dataset}:{row.model}:{row.method}")
        verified = (row.dataset, row.model, row.method) in vectors
        counts.append({"dataset": row.dataset, "model": row.model, "method": row.method,
                       "evidence": "row_verified" if verified else "reported_counts_only",
                       "accuracy_percent": (row.baseline_correct + c - i) / n * 100,
                       "macro_f1_percent": row.macro_f1_percent, **stats,
                       "holm_family": "additional_3" if n == 9000 else "main_12"})
    counts = pd.DataFrame(counts)
    for _, indices in counts.groupby("holm_family").groups.items():
        counts.loc[indices, "holm_p"] = holm_adjust(counts.loc[indices, "exact_p"].tolist())
    counts.to_csv(OUT / "phase1_paired_statistics.csv", index=False)
    cross = []
    for dataset in bases:
        for model in ("Llama 2", "Llama 3"):
            for method in ("Direct", "CoT"):
                sd_key, comparator = (dataset, model, "SELF-DISCOVER"), (dataset, model, method)
                if sd_key not in vectors or comparator not in vectors:
                    cross.append({"dataset": dataset, "model": model, "comparison": "SD vs " + method,
                                  "status": "needs_comparator_rows"})
                    continue
                cross.append({"dataset": dataset, "model": model, "comparison": "SD vs " + method,
                              "status": "row_verified", **paired(vectors[comparator], vectors[sd_key], str(sd_key)+method)})
    for method in ("Direct", "CoT"):
        cross.append({"dataset": "Additional Reddit 9000", "model": "Llama 3", "comparison": "SD vs " + method,
                      "status": "needs_9000_rows"})
    cross = pd.DataFrame(cross)
    for _, indices in cross.groupby(["dataset", "model"]).groups.items():
        if cross.loc[indices, "status"].eq("row_verified").all():
            cross.loc[indices, "holm_p_family2"] = holm_adjust(cross.loc[indices, "exact_p"].tolist())
    cross.to_csv(OUT / "sd_vs_baselines_paired_statistics.csv", index=False)
    print(audited[["dataset", "model", "method", "accuracy_percent", "macro_f1_percent", "parse_failures", "corrected", "introduced"]].to_string(index=False))
    print("\nSD vs comparators\n", cross.to_string(index=False))
    print("\nAdditional9000 vs Phase1\n", counts[counts.n.eq(9000)].to_string(index=False))


if __name__ == "__main__":
    main()
