"""Handoff validation, resumable generation, and full-denominator evaluation.

The notebook supplies the unchanged protocol functions before this runtime.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = {"direct": "Llama 3 Direct", "cot": "Llama 3 CoT", "v6c": "Llama 3 SELF-DISCOVER v6c"}
FINAL_WEIGHT_SHA256 = "0eeb8b16aef6a5df3045c2924d4701305f0ca08dc85d8a66efc3daff33b253d6"


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def save_csv(path, frame):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def read_phase1_handoff(folder):
    handoff = json.loads((folder / "phase2_handoff.json").read_text())
    path = folder / "final_phase1_reasoning_input.csv"
    if digest_file(path) != handoff["input_sha256"]:
        raise ValueError("Phase 1 input does not match its handoff hash.")
    manifest = json.loads((folder / "experiment_manifest.json").read_text())
    if handoff["phase1_manifest"] != manifest:
        raise ValueError("Handoff and Phase 1 manifest disagree.")
    if manifest["model_file_sha256"].get("model.safetensors") != FINAL_WEIGHT_SHA256:
        raise ValueError("This export was not produced with the received final seed42 weights.")
    if manifest["temperature"] != 1.4801950079829693 or manifest["threshold"] != 0.7:
        raise ValueError("Phase 1 temperature/threshold changed.")
    frame = pd.read_csv(path, keep_default_na=False, dtype={"example_id": str})
    required = {"example_id", "target_label", "phase1_label", "phase1_confidence", "phase1_routed", "phase2_original_text"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {required - set(frame.columns)}")
    if len(frame) != 9000 or len(frame) != handoff["expected_total_rows"]:
        raise ValueError("Expected exactly 9000 posts.")
    if frame.example_id.duplicated().any() or frame.example_id.str.strip().eq("").any():
        raise ValueError("Duplicate/empty example IDs.")
    if not frame.phase1_label.isin(LABELS).all() or not frame.target_label.isin(LABELS).all():
        raise ValueError("Invalid labels.")
    if frame.target_label.value_counts().to_dict() != dict.fromkeys(LABELS, 3000):
        raise ValueError("Expected 3000 reference examples per class.")
    frame["phase1_routed"] = frame.phase1_routed.map(normalize_bool)
    confidences = pd.to_numeric(frame.phase1_confidence, errors="raise")
    if not confidences.between(0, 1).all() or not np.array_equal(frame.phase1_routed, confidences < 0.7):
        raise ValueError("Routing flags do not match calibrated confidence < 0.70.")
    if int(frame.phase1_routed.sum()) != handoff["expected_routed_rows"]:
        raise ValueError("Routed count does not match the handoff.")
    saved = json.loads((folder / "phase1_summary.json").read_text())
    correct = frame.phase1_label.eq(frame.target_label)
    routed_error_count = int((~correct & frame.phase1_routed).sum())
    if (saved["total_rows"] != len(frame) or saved["routed_rows"] != int(frame.phase1_routed.sum())
            or saved["errors"] != int((~correct).sum()) or saved["routed_errors"] != routed_error_count
            or not np.isclose(saved["phase1_accuracy"], correct.mean(), atol=1e-12, rtol=0)
            or not np.isclose(saved["phase1_macro_f1"], f1_score(frame.target_label, frame.phase1_label,
                                                              labels=LABELS, average="macro"), atol=1e-12, rtol=0)):
        raise ValueError("Saved Phase 1 summary does not match its predictions.")
    routed = frame.loc[frame.phase1_routed].copy()
    if routed.phase2_original_text.str.strip().eq("").any():
        raise ValueError("A routed post has no original text.")
    previous = routed.get("phase2_input_was_truncated", pd.Series(False, index=routed.index)).map(normalize_bool)
    limited = routed.phase2_original_text.map(limit_reasoning_text)
    routed["phase2_original_text"] = limited.map(lambda pair: pair[0])
    routed["phase2_input_was_truncated"] = previous | limited.map(lambda pair: pair[1])
    return frame, routed.reset_index(drop=True), handoff, saved


def record_path(output, method, example_id):
    key = hashlib.sha256(str(example_id).encode()).hexdigest()
    return output / "records" / method / f"{key}.json"


def lock_run(output, manifest):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "experiment_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Run settings/input/runtime changed. Preserve this run and use a new output directory.")
    else:
        if list(output.glob("records/*/*.json")) or list(output.glob("stages/*/*/*.json")) or list(output.glob("*_results.csv")):
            raise ValueError("Results without a matching manifest; refusing to mix runs.")
        save_json(path, manifest)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def cached_generation(base_generate, cache_dir, fingerprint):
    counter = 0

    def generate(tokenizer, model, messages, **kwargs):
        nonlocal counter
        counter += 1
        path = cache_dir / f"stage_{counter:02d}.json"
        request = {"messages": messages, "settings": kwargs}
        request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["run_fingerprint"] != fingerprint or saved["request_sha256"] != request_hash:
                raise ValueError("Cached generation does not match this prompt/run.")
            return saved["generation"]
        generation = base_generate(tokenizer, model, messages, **kwargs)
        save_json(path, {"run_fingerprint": fingerprint, "request_sha256": request_hash,
                         "request": request, "generation": generation})
        return generation

    return generate


def load_method_records(output, method, routed, fingerprint, require_complete=True):
    expected = set(routed.example_id)
    rows = []
    for path in sorted((output / "records" / method).glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("run_fingerprint") != fingerprint or record.get("method") != method:
            raise ValueError("Stale or mixed result record.")
        if record["example_id"] not in expected or path != record_path(output, method, record["example_id"]):
            raise ValueError("Unexpected result ID or filename.")
        if record.get("final_label") not in LABELS + [None]:
            raise ValueError("Invalid parsed label in result record.")
        rows.append(record)
    frame = pd.DataFrame(rows)
    observed = set(frame.example_id) if len(frame) else set()
    if len(rows) != len(observed):
        raise ValueError("Duplicate result IDs.")
    if require_complete and observed != expected:
        raise ValueError(f"{method}: {len(observed)}/{len(expected)} complete; finish generation first.")
    if len(frame):
        frame = frame.set_index("example_id").reindex([x for x in routed.example_id if x in observed]).reset_index()
    return frame


def write_method_results(full, routed, results, output, method):
    prediction_col = f"llama3_{method}_final_label"
    merged, summary = summarize_method(full, results.rename(columns={"final_label": prediction_col}),
                                       prediction_col, METHODS[method])
    save_csv(output / f"llama3_{method}_results.csv", results)
    save_csv(output / f"llama3_{method}_end_to_end_predictions.csv", merged)
    save_json(output / f"llama3_{method}_summary.json", summary)
    save_json(output / f"llama3_{method}_classification_report.json",
              classification_report(merged.target_label, merged.final_label, labels=LABELS,
                                    output_dict=True, zero_division=0))
    matrix = confusion_matrix(merged.target_label, merged.final_label, labels=LABELS)
    save_csv(output / f"llama3_{method}_confusion_matrix.csv",
             pd.DataFrame(matrix, columns=LABELS).assign(target_label=LABELS))
    return summary


def run_all_methods(full, routed, handoff, output):
    global chat_generate
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before running Llama 3.")
    if hashlib.sha256(CACHED_PLAN_JSON.encode()).hexdigest() != PROTOCOL_EVIDENCE["cached_plan_original_sha256"]:
        raise ValueError("The frozen v6c plan changed.")
    manifest = {
        "protocol": "existing9000_final_llama3_direct_cot_v6c_v1",
        "phase1_handoff": handoff, "source_protocols": PROTOCOL_EVIDENCE,
        "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
        "methods": list(METHODS), "do_sample": False, "base_seed": BASE_SEED,
        "direct_cot_max_new_tokens_per_call": LLAMA3_MAX_NEW_TOKENS,
        "v6c_max_new_tokens": V6_MAX_NEW_TOKENS, "quantization": "nf4_double_quant_fp16",
        "max_reasoning_characters": MAX_REASONING_CHARACTERS,
        "reasoning_head_characters": REASONING_HEAD_CHARACTERS, "reasoning_tail_characters": REASONING_TAIL_CHARACTERS,
        "routed_id_sha256": hashlib.sha256("\n".join(sorted(routed.example_id)).encode()).hexdigest(),
        "total_rows": len(full), "routed_rows": len(routed), "new_plan_discovery": False,
        "parser": "existing_final_label_regex_then_unique_label_else_phase1_fallback",
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "packages": {name: importlib.metadata.version(name) for name in (
            "transformers", "accelerate", "bitsandbytes", "numpy", "pandas", "scikit-learn")},
    }
    fingerprint = lock_run(output, manifest)
    plan_path = output / "frozen_llama3_v6c_plan.json"
    if plan_path.exists() and digest_file(plan_path) != PROTOCOL_EVIDENCE["cached_plan_original_sha256"]:
        raise ValueError("Saved frozen plan differs from the received original.")
    if not plan_path.exists():
        plan_path.write_text(CACHED_PLAN_JSON)
    done_by_method = {}
    for method in METHODS:
        saved = load_method_records(output, method, routed, fingerprint, require_complete=False)
        done_by_method[method] = set(saved.example_id) if len(saved) else set()
        print(f"{METHODS[method]}: {len(saved)} completed, {len(routed) - len(saved)} pending")
    if all(len(done) == len(routed) for done in done_by_method.values()):
        print("All generations are saved. No model loaded; proceed to the comparison table.")
        return
    base_generate = chat_generate
    tokenizer, model = load_chat_model(MODEL_NAME)
    if getattr(model.config, "_commit_hash", MODEL_REVISION) not in (None, MODEL_REVISION):
        raise ValueError("Resolved model revision differs from the frozen revision.")
    save_json(output / "loaded_model_environment.json", {
        "revision": getattr(model.config, "_commit_hash", MODEL_REVISION),
        "context_limit": int(model.config.max_position_embeddings),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
    })
    runners = {"direct": run_llama3_direct, "cot": run_llama3_cot, "v6c": run_v6_max_prompt}
    try:
        for method, runner in runners.items():
            for _, row in tqdm(routed.iterrows(), total=len(routed), desc=METHODS[method]):
                if row.example_id in done_by_method[method]:
                    continue
                path = record_path(output, method, row.example_id)
                chat_generate = cached_generation(base_generate, output / "stages" / method / path.stem, fingerprint)
                # Evaluation labels never enter the runner or the generation prompts.
                inference_row = {key: row[key] for key in ("example_id", "phase1_label", "phase2_original_text")}
                if method == "v6c":
                    generated = runner(tokenizer, model, inference_row, task_structure=CACHED_PLAN)
                else:
                    generated = runner(tokenizer, model, inference_row)
                save_json(path, {**generated, "example_id": row.example_id, "method": method,
                                 "run_fingerprint": fingerprint, "phase1_label": row.phase1_label,
                                 "phase2_input_was_truncated": bool(row.phase2_input_was_truncated),
                                 "generation_calls": 5 if method == "cot" else 1})
            results = load_method_records(output, method, routed, fingerprint)
            summary = write_method_results(full, routed, results, output, method)
            print(METHODS[method], "complete:", summary)
    finally:
        chat_generate = base_generate
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def export_all_results(full, routed, output):
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    current_hash = digest_file(PHASE1_DIR / "final_phase1_reasoning_input.csv")
    if current_hash != manifest["phase1_handoff"]["input_sha256"]:
        raise ValueError("Phase 1 input changed after generation.")
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    summaries = []
    costs = []
    for method in METHODS:
        results = load_method_records(output, method, routed, fingerprint)
        summaries.append(write_method_results(full, routed, results, output, method))
        costs.append({"method": METHODS[method], "routed_rows": len(routed),
                      "generation_calls": int(results.generation_calls.sum()),
                      **{key: float(results[key].sum()) for key in (
                          "input_tokens_total", "generated_tokens_total", "generation_seconds_total")},
                      "new_plan_discovery_calls": 0})
    summary = pd.DataFrame(summaries)
    save_csv(output / "llama3_reasoning_method_summary.csv", summary)
    save_csv(output / "llama3_generation_cost_summary.csv", pd.DataFrame(costs))
    return summary
