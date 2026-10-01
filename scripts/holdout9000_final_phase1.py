"""Inference-only evaluation of the existing 9,000 posts with the final model."""

import hashlib
import importlib.metadata
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

LABELS = ["Depression", "Neutral", "Happy"]
TEMPERATURE = 1.4801950079829693
THRESHOLD = 0.70
MAX_LENGTH = 256
BATCH_SIZE = 64
OLD_MODEL_SHA256 = "8b123d526cfd32c6ad7c03d58bc629aa6978df9d750fbcac2f07b9f4702e489b"
FINAL_MODEL_FILES = {
    "config.json": "b0d6bfc8cb80089a4df54ddaf13fb1b726ca09f5f28536f793d56afdab708858",
    "model.safetensors": "0eeb8b16aef6a5df3045c2924d4701305f0ca08dc85d8a66efc3daff33b253d6",
    "tokenizer.json": "8b79639ec74b46604e730f505186eaafb1006d2fd00f2c4930d168bb7f894680",
    "tokenizer_config.json": "e1c2a61a99bda00f6c55303a210b30e2f92dcf8b555e215812e2eb583e177ffd",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path, frame):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def lock_manifest(output, manifest):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "experiment_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Input/model/runtime settings changed. Preserve this folder and use a new output directory.")
    else:
        if list(output.glob("*.csv")) or list(output.glob("batches/*.json")):
            raise ValueError("Existing results without a matching manifest; refusing to mix runs.")
        atomic_json(path, manifest)


def read_data(source, expected_hashes):
    """Read only data files, never load or execute the old bundle's model/code."""
    frames = {}
    for name, expected in expected_hashes.items():
        if source.is_file():
            with zipfile.ZipFile(source) as archive:
                candidates = [n for n in archive.namelist() if n.endswith("/data/" + name)]
                if len(candidates) != 1:
                    raise ValueError(f"Archive must contain exactly one data/{name}")
                payload = archive.read(candidates[0])
        else:
            payload = (source / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"9000-post source hash mismatch: {name}")
        frames[name] = pd.read_csv(io.BytesIO(payload), keep_default_na=False, dtype={"example_id": str})
    inputs = frames["evaluation_inputs.csv"]
    for name, frame in frames.items():
        if len(frame) != 9000 or frame.example_id.eq("").any() or frame.example_id.duplicated().any():
            raise ValueError(f"Expected 9000 distinct nonempty IDs: {name}")
        if set(frame.example_id) != set(inputs.example_id):
            raise ValueError(f"ID mismatch: {name}")
    if inputs.text.str.strip().eq("").any():
        raise ValueError("Empty Phase 1 text")
    labels = frames["evaluation_labels.csv"]
    labels["label_str"] = labels.label_str.replace({label + "_Group": label for label in LABELS})
    expected_ids = labels.label_str.map({label: i for i, label in enumerate(LABELS)})
    if expected_ids.isna().any() or not np.array_equal(expected_ids.to_numpy(), labels.label.to_numpy()):
        raise ValueError("Reference label strings and numeric IDs disagree")
    if labels.label_str.value_counts().to_dict() != dict.fromkeys(LABELS, 3000):
        raise ValueError("Expected 3000 posts per class")
    raw = frames["phase2_inputs.csv"].rename(columns={
        "was_truncated": "phase2_input_was_truncated",
        "full_character_count": "phase2_full_character_count",
    })
    if raw.phase2_original_text.str.strip().eq("").any():
        raise ValueError("Empty Phase 2 original text")
    # The original-text CSV is in a different order: joins must always use IDs.
    return inputs[["example_id", "text"]], labels, raw


def resolve_paths(drive_root, source_override=None, model_override=None):
    project = drive_root / "confidence_guided_llm_reasoning"
    model = Path(model_override) if model_override else project / "distilbert_classification_fine_tuned/seed_42_model"
    if not (model / "config.json").is_file():
        raise FileNotFoundError(
            f"Final seed42 model not found: {model}\n"
            "Set MODEL_DIR to the existing final seed42 folder. Do not use the old 9000-package checkpoint. "
            "No training or model download is performed by this notebook."
        )
    if source_override:
        source = Path(source_override)
    else:
        source = drive_root / "CGSLR_New_Holdout_9000_Run_Package.zip"
        if not source.exists():
            candidates = list(drive_root.rglob("CGSLR_New_Holdout_9000_Run_Package.zip"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    "Could not uniquely locate the old 9000-post ZIP. Set DATA_SOURCE to its Drive path "
                    "or to its extracted same_source_evaluation/data folder."
                )
            source = candidates[0]
    if not source.exists():
        raise FileNotFoundError(source)
    return source, model


def model_files(model_dir):
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        weights = sorted(model_dir.glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"No model weights in {model_dir}")
    metadata = [p for p in model_dir.iterdir() if p.is_file() and
                (p.suffix == ".json" or p.name in {"vocab.txt", "tokenizer.model"})]
    hashes = {p.name: sha256(p) for p in sorted(set(weights + metadata))}
    if OLD_MODEL_SHA256 in hashes.values():
        raise ValueError("This is the old operational checkpoint, not the final seed42 model.")
    if hashes != FINAL_MODEL_FILES:
        raise ValueError("Model/tokenizer files differ from the received final seed42 package.")
    return hashes


def check_label_order(names):
    normalized = [{label + "_Group": label for label in LABELS}.get(name, name) for name in names]
    if normalized != LABELS:
        raise ValueError(f"Unexpected checkpoint label order: {names}")
    return normalized


def prediction_rows(batch, logits):
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (len(batch), 3) or not np.isfinite(logits).all():
        raise ValueError("Invalid classifier logits")
    calibrated = softmax(logits / TEMPERATURE, axis=1)
    raw = softmax(logits, axis=1)
    rows = []
    for identifier, z, p, p_raw in zip(batch.example_id, logits, calibrated, raw):
        idx = int(p.argmax())
        rows.append({
            "example_id": identifier,
            "phase1_label": LABELS[idx],
            "phase1_confidence": float(p[idx]),
            "phase1_raw_confidence": float(p_raw.max()),
            "phase1_routed": bool(p[idx] < THRESHOLD),
            "temperature": TEMPERATURE,
            "routing_threshold": THRESHOLD,
            **{f"phase1_logit_{label.lower()}": float(z[i]) for i, label in enumerate(LABELS)},
            **{f"phase1_probability_{label.lower()}": float(p[i]) for i, label in enumerate(LABELS)},
        })
    return rows


def verify_batch(saved, expected_ids, fingerprint):
    if saved.get("run_fingerprint") != fingerprint:
        raise ValueError("Cached batch belongs to a different run")
    rows = saved["predictions"]
    if [r["example_id"] for r in rows] != list(expected_ids):
        raise ValueError("Cached batch ID/order mismatch")
    for row in rows:
        c = row["phase1_confidence"]
        if (row["phase1_label"] not in LABELS or not np.isfinite(c) or not 0 <= c <= 1
                or not isinstance(row["phase1_routed"], bool)
                or row["phase1_routed"] != (c < THRESHOLD)):
            raise ValueError("Invalid cached prediction")


def export_results(output, inputs, labels, raw, predictions, manifest):
    if predictions.example_id.tolist() != inputs.example_id.tolist():
        raise ValueError("Prediction IDs/order differ from the 9000 inputs")
    final = inputs.merge(predictions, on="example_id", validate="one_to_one")
    final = final.merge(labels[["example_id", "label_str"]].rename(columns={"label_str": "target_label"}),
                        on="example_id", validate="one_to_one")
    final = final.merge(raw, on="example_id", validate="one_to_one")
    routed = final.phase1_routed.astype(bool)
    errors = final.phase1_label.ne(final.target_label)
    summary = {
        "dataset": "existing_same_source_9000_final_pipeline",
        "total_rows": len(final), "routed_rows": int(routed.sum()),
        "phase1_accuracy": float(accuracy_score(final.target_label, final.phase1_label)),
        "phase1_macro_f1": float(f1_score(final.target_label, final.phase1_label, labels=LABELS, average="macro")),
        "errors": int(errors.sum()), "routed_errors": int((routed & errors).sum()),
        "coverage": float((~routed).mean()), "routing_rate": float(routed.mean()),
        "accepted_error_rate": float(errors[~routed].mean()) if (~routed).any() else None,
        "temperature": TEMPERATURE, "routing_threshold": THRESHOLD,
    }
    # The full file preserves the denominator; Phase 2 consumes only routed rows.
    final.loc[~routed, "phase2_original_text"] = ""
    atomic_csv(output / "phase1_predictions.csv", final.drop(columns="phase2_original_text"))
    atomic_csv(output / "final_phase1_reasoning_input.csv", final)
    atomic_csv(output / "routed_phase2_inputs.csv", final.loc[routed].reset_index(drop=True))
    atomic_json(output / "phase1_summary.json", summary)
    atomic_csv(output / "phase1_summary.csv", pd.DataFrame([summary]))
    report = classification_report(final.target_label, final.phase1_label, labels=LABELS, output_dict=True, zero_division=0)
    atomic_json(output / "classification_report.json", report)
    matrix = confusion_matrix(final.target_label, final.phase1_label, labels=LABELS)
    atomic_csv(output / "confusion_matrix.csv", pd.DataFrame(matrix, columns=LABELS).assign(target_label=LABELS))
    atomic_json(output / "phase2_handoff.json", {
        "input_path": str(output / "final_phase1_reasoning_input.csv"),
        "input_sha256": sha256(output / "final_phase1_reasoning_input.csv"),
        "expected_total_rows": len(final), "expected_routed_rows": int(routed.sum()),
        "planned_methods": ["Llama 3 Direct", "Llama 3 CoT", "Llama 3 SELF-DISCOVER v6c"],
        "phase2_executed": False, "phase1_manifest": manifest,
    })
    return summary


def run_phase1(source, model_dir, output, expected_hashes):
    import torch
    import transformers
    from tqdm.auto import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab before running Phase 1.")
    inputs, labels, raw = read_data(source, expected_hashes)
    hashes = model_files(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).float().cuda().eval()
    # The source seed42 notebook uses 0=Depression, 1=Neutral, 2=Happy.
    names = [model.config.id2label.get(i) for i in range(3)]
    if model.config.num_labels != 3:
        raise ValueError(f"Unexpected checkpoint label order: {names}")
    check_label_order(names)
    if model.config.model_type != "distilbert":
        raise ValueError("Expected a DistilBERT sequence classifier")
    manifest = {
        "protocol": "final_seed42_existing9000_phase1_v1",
        "source_data_sha256": expected_hashes, "model_file_sha256": hashes,
        "model_dir": str(model_dir), "label_order": LABELS,
        "checkpoint_id2label": names,
        "temperature": TEMPERATURE, "threshold": THRESHOLD,
        "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE, "precision": "fp32",
        "training_performed": False, "calibration_fitted": False, "threshold_searched": False,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__, "scipy": importlib.metadata.version("scipy"),
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
    }
    lock_manifest(output, manifest)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    batches = output / "batches"
    batches.mkdir(exist_ok=True)
    all_rows = []
    completed = sum((batches / f"batch_{s:05d}.json").exists() for s in range(0, len(inputs), BATCH_SIZE))
    print(f"Saved batches: {completed}; total batches: {(len(inputs) + BATCH_SIZE - 1) // BATCH_SIZE}")
    for start in tqdm(range(0, len(inputs), BATCH_SIZE), desc="Final DistilBERT / 9000"):
        batch = inputs.iloc[start:start + BATCH_SIZE]
        path = batches / f"batch_{start:05d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
        else:
            encoded = tokenizer(batch.text.tolist(), padding=True, truncation=True,
                                max_length=MAX_LENGTH, return_tensors="pt", return_token_type_ids=False)
            tokens = int(encoded["attention_mask"].sum())
            encoded = {k: v.cuda() for k, v in encoded.items()}
            torch.cuda.synchronize()
            began = time.perf_counter()
            with torch.inference_mode():
                logits = model(**encoded).logits.float().cpu().numpy()
            torch.cuda.synchronize()
            saved = {"run_fingerprint": fingerprint, "predictions": prediction_rows(batch, logits),
                     "forward_seconds": time.perf_counter() - began, "input_tokens": tokens}
            atomic_json(path, saved)
        verify_batch(saved, batch.example_id, fingerprint)
        all_rows.extend(saved["predictions"])
    summary = export_results(output, inputs, labels, raw, pd.DataFrame(all_rows), manifest)
    del model, tokenizer
    torch.cuda.empty_cache()
    print("Phase 1 complete. Llama generation has NOT started.")
    print("Saved:", output)
    return pd.DataFrame([summary])
