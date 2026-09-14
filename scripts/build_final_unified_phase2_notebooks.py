#!/usr/bin/env python3
"""Build the two Colab notebooks for the final unified Phase 2 experiment."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "colab" / "final"
LEGACY_FINAL_NOTEBOOK = OUT_DIR / "04_5_reddit_test_routed_phase2_original_text_primary_final_colab.ipynb"


def source(text: str) -> list[str]:
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


def notebook(name: str, cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def legacy_code_cell(marker: str) -> dict:
    """Copy an established prompt cell verbatim into the new standalone notebook."""
    payload = json.loads(LEGACY_FINAL_NOTEBOOK.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cell_source = "".join(cell.get("source", []))
        if marker in cell_source:
            return code(cell_source)
    raise ValueError(f"Could not find legacy prompt cell containing: {marker}")


SETUP = code(
    r"""
    # Run once in a fresh Colab GPU runtime. Restart the runtime after installation.
    %pip install -q -U pandas tqdm scikit-learn sentencepiece protobuf accelerate transformers "bitsandbytes>=0.46.1"

    import importlib.metadata as importlib_metadata
    print("bitsandbytes:", importlib_metadata.version("bitsandbytes"))
    print("transformers:", importlib_metadata.version("transformers"))
    print("accelerate:", importlib_metadata.version("accelerate"))
    print("\nRestart the runtime once, then run from the imports cell.")
    """
)


IMPORTS = code(
    r"""
    import gc
    import hashlib
    import json
    import os
    import re
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from IPython.display import display
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    LABELS = ["Depression", "Neutral", "Happy"]
    LABEL_SET = set(LABELS)
    """
)


CONFIG = code(
    r"""
    # Both Phase 2 notebooks must point to this same final Phase 1 export.
    PHASE1_INPUT_PATH = Path(
        "/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/"
        "unified_final/phase1/final_phase1_reasoning_input.csv"
    )
    DRIVE_OUTPUT_ROOT = Path(
        "/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/"
        "unified_final/phase2"
    )

    # Frozen final Phase 1 contract. Any accidental input change fails before inference.
    EXPECTED_TOTAL_ROWS = 12_000
    EXPECTED_ROUTED_ROWS = 218

    MAX_ROWS = None  # None for the final run; use a small integer only for a smoke test.
    MAX_REASONING_CHARACTERS = 6000
    REASONING_HEAD_CHARACTERS = 3500
    REASONING_TAIL_CHARACTERS = 2500
    LOAD_IN_4BIT = True
    RESUME_FROM_EXISTING = True
    BASE_SEED = 42

    # Canonical required columns in PHASE1_INPUT_PATH:
    # example_id, target_label, phase1_label, phase1_confidence,
    # phase1_routed, phase2_original_text
    # title + selftext may replace phase2_original_text; aliases below are normalized.
    """
)


DRIVE_AND_INPUT = code(
    r"""
    try:
        from google.colab import drive, userdata
        drive.mount("/content/drive")
    except Exception as exc:
        raise RuntimeError("Google Drive must be mounted before the final run.") from exc

    # Optional for the public NousResearch checkpoints used below. A token can
    # still be supplied through Colab Secrets to reduce Hugging Face rate limits.
    try:
        hf_token = userdata.get("HF_TOKEN")
    except Exception:
        hf_token = None
        print("HF_TOKEN is not set; continuing with public Hugging Face access.")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    DRIVE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not PHASE1_INPUT_PATH.exists():
        raise FileNotFoundError(f"Final Phase 1 export not found: {PHASE1_INPUT_PATH}")


    def normalize_bool(value):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        raise ValueError(f"Unrecognized Boolean value: {value!r}")


    def normalize_label(value):
        text = str(value).strip().lower()
        for label in LABELS:
            if text == label.lower():
                return label
        return np.nan


    def minimally_sanitize_original_text(title, selftext):
        title = "" if pd.isna(title) else str(title).strip()
        body = "" if pd.isna(selftext) else str(selftext).strip()
        if body.lower() in {"[deleted]", "[removed]", "nan", "none"}:
            body = ""
        combined = title if not body else f"{title}\n\n{body}"
        combined = re.sub(r"https?://\S+|www\.\S+", " [URL] ", combined, flags=re.I)
        combined = re.sub(r"(?<!\w)(?:/u/|u/)[A-Za-z0-9_-]+", "[USER]", combined)
        combined = combined.replace("\r\n", "\n").replace("\r", "\n")
        combined = re.sub(r"[ \t]+", " ", combined)
        return re.sub(r"\n{3,}", "\n\n", combined).strip()


    def limit_reasoning_text(value):
        value = "" if pd.isna(value) else str(value)
        if len(value) <= MAX_REASONING_CHARACTERS:
            return value, False
        marker = "\n\n[Middle omitted to fit model context; beginning and ending preserved.]\n\n"
        available = MAX_REASONING_CHARACTERS - len(marker)
        head_characters = min(REASONING_HEAD_CHARACTERS, available)
        tail_characters = min(REASONING_TAIL_CHARACTERS, available - head_characters)
        return (
            value[:head_characters]
            + marker
            + value[-tail_characters:],
            True,
        )


    def normalize_phase1_export(path):
        frame = pd.read_csv(path)
        aliases = {
            "target_label": ["target_label", "label_str", "true_label", "reference_label"],
            "phase1_label": ["phase1_label", "prediction", "predicted_label"],
            "phase1_confidence": ["phase1_confidence", "calibrated_confidence", "confidence"],
            "phase1_routed": ["phase1_routed", "routed", "route_indicator"],
            "phase2_original_text": ["phase2_original_text", "original_text", "reasoning_text"],
        }
        for canonical, candidates in aliases.items():
            if canonical in frame.columns:
                continue
            found = next((candidate for candidate in candidates if candidate in frame.columns), None)
            if found:
                frame = frame.rename(columns={found: canonical})

        if "phase2_original_text" not in frame.columns:
            if {"title", "selftext"}.issubset(frame.columns):
                frame["phase2_original_text"] = [
                    minimally_sanitize_original_text(title, body)
                    for title, body in zip(frame["title"], frame["selftext"])
                ]
            else:
                raise ValueError(
                    "Phase 1 export needs phase2_original_text, or both title and selftext."
                )

        required = {
            "example_id", "target_label", "phase1_label", "phase1_confidence",
            "phase1_routed", "phase2_original_text",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Final Phase 1 export is missing columns: {sorted(missing)}")

        frame["example_id"] = frame["example_id"].astype(str)
        if frame["example_id"].duplicated().any():
            duplicates = frame.loc[frame["example_id"].duplicated(), "example_id"].head().tolist()
            raise ValueError(f"Duplicate example_id values found: {duplicates}")

        frame["target_label"] = frame["target_label"].map(normalize_label)
        frame["phase1_label"] = frame["phase1_label"].map(normalize_label)
        frame["phase1_routed"] = frame["phase1_routed"].map(normalize_bool)
        frame["phase1_confidence"] = pd.to_numeric(frame["phase1_confidence"], errors="raise")
        if frame[["target_label", "phase1_label"]].isna().any().any():
            raise ValueError("target_label and phase1_label must use Depression, Neutral, or Happy.")
        if not frame["phase1_confidence"].between(0, 1).all():
            raise ValueError("phase1_confidence must be between 0 and 1.")

        if "phase2_input_was_truncated" in frame.columns:
            previously_truncated = (
                frame["phase2_input_was_truncated"].fillna(False).map(normalize_bool)
            )
        else:
            previously_truncated = pd.Series(False, index=frame.index)
        limited = frame["phase2_original_text"].map(limit_reasoning_text)
        frame["phase2_original_text"] = limited.map(lambda pair: pair[0])
        frame["phase2_input_was_truncated"] = (
            previously_truncated | limited.map(lambda pair: pair[1])
        )

        routed = frame[frame["phase1_routed"]].copy()
        if routed.empty:
            raise ValueError("No routed rows were found in the final Phase 1 export.")
        if routed["phase2_original_text"].str.strip().eq("").any():
            raise ValueError("At least one routed row has empty Phase 2 original text.")
        if EXPECTED_TOTAL_ROWS is not None and len(frame) != EXPECTED_TOTAL_ROWS:
            raise ValueError(f"Expected {EXPECTED_TOTAL_ROWS} total rows, found {len(frame)}.")
        if EXPECTED_ROUTED_ROWS is not None and len(routed) != EXPECTED_ROUTED_ROWS:
            raise ValueError(f"Expected {EXPECTED_ROUTED_ROWS} routed rows, found {len(routed)}.")

        if MAX_ROWS is not None:
            routed = routed.head(min(MAX_ROWS, len(routed))).copy()
        return frame.reset_index(drop=True), routed.reset_index(drop=True)


    phase1_df, routed_df = normalize_phase1_export(PHASE1_INPUT_PATH)
    routed_id_hash = hashlib.sha256(
        "\n".join(sorted(routed_df["example_id"])).encode("utf-8")
    ).hexdigest()
    print("Phase 1 rows:", len(phase1_df))
    print("Routed rows selected for this run:", len(routed_df))
    print("Routed example_id SHA-256:", routed_id_hash)
    display(routed_df[["example_id", "target_label", "phase1_label", "phase1_confidence"]].head())
    """
)


MODEL_HELPERS = code(
    r"""
    def stable_seed(example_id, method, stage):
        key = f"{BASE_SEED}:{example_id}:{method}:{stage}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


    def set_generation_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def load_chat_model(model_name):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        quantization_config = None
        if LOAD_IN_4BIT:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
        model.eval()
        return tokenizer, model


    def format_messages_without_chat_template(messages):
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        dialogue = [m for m in messages if m.get("role") != "system"]
        system_text = "\n".join(system_parts).strip()
        prompt, pending_user, first_turn = "", None, True
        for message in dialogue:
            role = message.get("role")
            content = str(message.get("content", "")).strip()
            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user is not None:
                if first_turn and system_text:
                    prompt += f"<s>[INST] <<SYS>>\n{system_text}\n<</SYS>>\n\n{pending_user} [/INST] {content} </s>"
                else:
                    prompt += f"<s>[INST] {pending_user} [/INST] {content} </s>"
                pending_user, first_turn = None, False
        if pending_user is not None:
            if first_turn and system_text:
                prompt += f"<s>[INST] <<SYS>>\n{system_text}\n<</SYS>>\n\n{pending_user} [/INST]"
            else:
                prompt += f"<s>[INST] {pending_user} [/INST]"
        return prompt


    def build_chat_inputs(tokenizer, model, messages):
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = format_messages_without_chat_template(messages)
        encoded = tokenizer(prompt, return_tensors="pt")
        return {key: value.to(model.device) for key, value in encoded.items()}


    def chat_generate(
        tokenizer, model, messages, *, max_new_tokens, seed,
        do_sample=False, temperature=None, top_p=None,
    ):
        set_generation_seed(seed)
        model_inputs = build_chat_inputs(tokenizer, model, messages)
        input_ids = model_inputs["input_ids"]
        terminators = [tokenizer.eos_token_id]
        if "<|eot_id|>" in tokenizer.get_vocab():
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in terminators:
                terminators.append(eot_id)
        kwargs = dict(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
        )
        if do_sample and temperature is not None:
            kwargs["temperature"] = temperature
        if do_sample and top_p is not None:
            kwargs["top_p"] = top_p
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = outputs[0][input_ids.shape[-1]:]
        return {
            "text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
            "input_tokens": int(input_ids.shape[-1]),
            "generated_tokens": int(generated.shape[-1]),
            "elapsed_seconds": float(elapsed),
            "hit_token_limit": bool(generated.shape[-1] >= max_new_tokens),
        }


    def parse_final_label(output):
        text = str(output)
        match = re.search(
            r"Final\s+label\s*\*{0,2}\s*:\s*\*{0,2}\s*"
            r"(Depression|Neutral|Happy)\b",
            text,
            flags=re.I,
        )
        if match:
            return normalize_label(match.group(1))

        # Some models return a clear terminal label without the requested prefix.
        # Recover it only when the final-stage answer contains one unique allowed
        # label; refusals and outputs mentioning multiple labels remain unparsed.
        labels = {
            normalize_label(label)
            for label in re.findall(r"\b(Depression|Neutral|Happy)\b", text, flags=re.I)
        }
        return next(iter(labels)) if len(labels) == 1 else np.nan


    def load_results(path):
        if path.exists() and path.stat().st_size > 0:
            frame = pd.read_csv(path)
            return frame.drop_duplicates(subset=["example_id"], keep="last")
        return pd.DataFrame()


    def save_result(path, row):
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = load_results(path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        updated = updated.drop_duplicates(subset=["example_id"], keep="last")
        temporary = path.with_suffix(".tmp.csv")
        updated.to_csv(temporary, index=False)
        os.replace(temporary, path)


    def clear_model(tokenizer, model):
        del tokenizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def summarize_method(full_phase1, routed_results, prediction_col, method):
        processed_ids = set(routed_results["example_id"].astype(str))
        evaluation_base = full_phase1
        scope = "full_end_to_end"
        if MAX_ROWS is not None:
            evaluation_base = full_phase1[full_phase1["example_id"].isin(processed_ids)].copy()
            scope = "smoke_routed_subset"
        merged = evaluation_base.merge(
            routed_results[["example_id", prediction_col]],
            on="example_id", how="left", validate="one_to_one",
        )
        merged["final_label"] = merged["phase1_label"]
        routed_mask = merged["phase1_routed"]
        parsed_routed_mask = routed_mask & merged[prediction_col].isin(LABELS)
        merged.loc[parsed_routed_mask, "final_label"] = merged.loc[
            parsed_routed_mask, prediction_col
        ]
        merged["phase2_parse_success"] = ~routed_mask | parsed_routed_mask
        merged["phase2_fallback_to_phase1"] = routed_mask & ~parsed_routed_mask
        merged["phase1_correct"] = merged["phase1_label"].eq(merged["target_label"])
        merged["final_correct"] = merged["final_label"].eq(merged["target_label"])
        corrected = int((routed_mask & ~merged["phase1_correct"] & merged["final_correct"]).sum())
        introduced = int((routed_mask & merged["phase1_correct"] & ~merged["final_correct"]).sum())
        parsed = routed_results[prediction_col].isin(LABELS)
        summary = {
            "method": method,
            "scope": scope,
            "total_rows": len(merged),
            "routed_rows": int(routed_mask.sum()),
            "parsed_routed_rows": int(parsed.sum()),
            "parse_failures": int((~parsed).sum()),
            "phase1_accuracy": float(merged["phase1_correct"].mean()),
            "end_to_end_accuracy": float(merged["final_correct"].mean()),
            "end_to_end_macro_f1": float(
                f1_score(merged["target_label"], merged["final_label"], labels=LABELS, average="macro")
            ),
            "corrected": corrected,
            "introduced": introduced,
            "net_corrections": corrected - introduced,
        }
        return merged, summary
    """
)


LLAMA2_PROMPT = code(
    r'''
    LLAMA2_MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
    LLAMA2_OUTPUT_DIR = DRIVE_OUTPUT_ROOT / "llama2_cot_matched_llama3"
    LLAMA2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LLAMA2_RESULTS_PATH = LLAMA2_OUTPUT_DIR / "llama2_cot_matched_results.csv"
    LLAMA2_PROMPT_VERSION = "final-llama2-cot-v3-matched-llama3"
    MAX_NEW_TOKENS_COT = 512

    LLAMA2_CLASSIFICATION_POLICY = """Classification Policy:
    - Depression: unresolved sadness, hopelessness, emotional distress, emotional exhaustion, withdrawal, self-devaluation, or a clearly negative overall trajectory is dominant.
    - Neutral: the text is mainly factual, routine, balanced, informational, or emotionally mild, without a dominant positive or distress-related state.
    - Happy: happiness, relief, gratitude, accomplishment, fulfillment, or a clearly positive resolution is dominant.

    Assess the dominant emotional meaning of the full text. For mixed or shifting emotions, consider the overall trajectory and final takeaway. Do not decide from isolated words, and do not make clinical diagnoses or treatment recommendations."""

    LLAMA2_COT_REQUESTS = [
        "You are an expert annotator for research-oriented, non-clinical emotion classification. Assist in analyzing emotions in text data. Do not make a clinical diagnosis, infer a medical condition, or provide treatment advice.",
        "I will first provide a piece of text. Independently assess its emotional content before comparing it with a Phase 1 AI-generated label. The only permitted labels are Depression, Neutral, and Happy.",
        "Independently analyze the text. State the dominant emotion and textual evidence before considering the Phase 1 label.\n\n{policy}\n\nUse only Depression, Neutral, or Happy for your provisional label.",
        """The Phase 1 classifier predicted: {phase1_label}

    Compare that prediction with your independent assessment. Confirm it only when it is supported by the dominant emotional meaning of the full text. Otherwise, explain the correction using textual evidence only.""",
        "Provide the final Phase 2 decision. Use only one exact label: Depression, Neutral, or Happy. Do not use synonyms or additional labels. End the response with exactly one line: Final label: [label]",
    ]


    def run_llama2_cot_one(tokenizer, model, row):
        example_id = row["example_id"]
        messages = [
            {"role": "system", "content": LLAMA2_COT_REQUESTS[0]},
            {"role": "user", "content": LLAMA2_COT_REQUESTS[1]},
        ]
        stages = []

        ack = chat_generate(
            tokenizer, model, messages, max_new_tokens=MAX_NEW_TOKENS_COT,
            seed=stable_seed(example_id, "llama2_cot", "ack"),
            do_sample=False,
        )
        stages.append(ack)
        messages.append({"role": "assistant", "content": ack["text"]})
        messages.append({"role": "user", "content": f"Text:\n{row['phase2_original_text']}"})

        text_response = chat_generate(
            tokenizer, model, messages, max_new_tokens=MAX_NEW_TOKENS_COT,
            seed=stable_seed(example_id, "llama2_cot", "text_response"),
            do_sample=False,
        )
        stages.append(text_response)
        messages.append({"role": "assistant", "content": text_response["text"]})
        messages.append({
            "role": "user",
            "content": LLAMA2_COT_REQUESTS[2].format(
                policy=LLAMA2_CLASSIFICATION_POLICY
            ),
        })

        independent = chat_generate(
            tokenizer, model, messages, max_new_tokens=MAX_NEW_TOKENS_COT,
            seed=stable_seed(example_id, "llama2_cot", "independent"),
            do_sample=False,
        )
        stages.append(independent)
        messages.append({"role": "assistant", "content": independent["text"]})
        messages.append({"role": "user", "content": LLAMA2_COT_REQUESTS[3].format(phase1_label=row["phase1_label"])})

        comparison = chat_generate(
            tokenizer, model, messages, max_new_tokens=MAX_NEW_TOKENS_COT,
            seed=stable_seed(example_id, "llama2_cot", "comparison"),
            do_sample=False,
        )
        stages.append(comparison)
        messages.append({"role": "assistant", "content": comparison["text"]})
        messages.append({"role": "user", "content": LLAMA2_COT_REQUESTS[4]})

        final = chat_generate(
            tokenizer, model, messages, max_new_tokens=MAX_NEW_TOKENS_COT,
            seed=stable_seed(example_id, "llama2_cot", "final"),
            do_sample=False,
        )
        stages.append(final)
        return {
            "llama2_ack": ack["text"],
            "llama2_text_response": text_response["text"],
            "llama2_independent": independent["text"],
            "llama2_comparison": comparison["text"],
            "llama2_final_answer": final["text"],
            "llama2_final_label": parse_final_label(final["text"]),
            "input_tokens_total": sum(item["input_tokens"] for item in stages),
            "generated_tokens_total": sum(item["generated_tokens"] for item in stages),
            "generation_seconds_total": sum(item["elapsed_seconds"] for item in stages),
            "any_stage_hit_token_limit": any(item["hit_token_limit"] for item in stages),
        }
    ''')


LLAMA2_RUN = code(
    r"""
    results = load_results(LLAMA2_RESULTS_PATH)
    if not RESUME_FROM_EXISTING and LLAMA2_RESULTS_PATH.exists():
        LLAMA2_RESULTS_PATH.unlink()
        results = pd.DataFrame()
    completed = set(results.get("example_id", pd.Series(dtype=str)).astype(str))
    pending = routed_df[~routed_df["example_id"].isin(completed)].copy()
    print(f"Llama 2 CoT: {len(completed)} completed, {len(pending)} pending")

    if not pending.empty:
        tokenizer, model = load_chat_model(LLAMA2_MODEL_NAME)
        model_revision = getattr(model.config, "_commit_hash", None)
        for _, row in tqdm(pending.iterrows(), total=len(pending), desc="Llama 2 CoT"):
            generated = run_llama2_cot_one(tokenizer, model, row)
            result = {
                "example_id": row["example_id"],
                "target_label": row["target_label"],
                "phase1_label": row["phase1_label"],
                "phase1_confidence": row["phase1_confidence"],
                "phase2_original_text": row["phase2_original_text"],
                "phase2_input_was_truncated": row["phase2_input_was_truncated"],
                "model_name": LLAMA2_MODEL_NAME,
                "model_revision": model_revision,
                "prompt_version": LLAMA2_PROMPT_VERSION,
                "routed_id_hash": routed_id_hash,
                **generated,
            }
            save_result(LLAMA2_RESULTS_PATH, result)
        clear_model(tokenizer, model)

    results = load_results(LLAMA2_RESULTS_PATH)
    if len(results) != len(routed_df) or set(results["example_id"].astype(str)) != set(routed_df["example_id"]):
        raise ValueError("Llama 2 result IDs do not exactly match the routed input IDs.")
    end_to_end, summary = summarize_method(
        phase1_df, results, "llama2_final_label", "Llama 2 CoT matched"
    )
    end_to_end.to_csv(LLAMA2_OUTPUT_DIR / "llama2_cot_matched_end_to_end_predictions.csv", index=False)
    pd.DataFrame([summary]).to_csv(LLAMA2_OUTPUT_DIR / "llama2_cot_matched_summary.csv", index=False)
    display(pd.DataFrame([summary]))
    display(results[["example_id", "target_label", "phase1_label", "llama2_final_label"]].head())
    print("Saved:", LLAMA2_OUTPUT_DIR)
    """
)


LLAMA3_PROMPTS = code(
    r'''
    LLAMA3_MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"
    LLAMA3_OUTPUT_DIR = DRIVE_OUTPUT_ROOT / "llama3_reasoning_comparison"
    LLAMA3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_METHODS = ["direct", "cot", "self_discover"]
    LLAMA3_MAX_NEW_TOKENS = 1024

    CLASSIFICATION_POLICY = """Classification Policy:
    - Depression: unresolved sadness, hopelessness, emotional distress, emotional exhaustion, withdrawal, self-devaluation, or a clearly negative overall trajectory is dominant.
    - Neutral: the text is mainly factual, routine, balanced, informational, or emotionally mild, without a dominant positive or distress-related state.
    - Happy: happiness, relief, gratitude, accomplishment, fulfillment, or a clearly positive resolution is dominant.

    Assess the dominant emotional meaning of the full text. For mixed or shifting emotions, consider the overall trajectory and final takeaway. Do not decide from isolated words, and do not make clinical diagnoses or treatment recommendations."""

    DIRECT_PROMPT_VERSION = "final-llama3-direct-v1-original-text"
    DIRECT_TEMPLATE = """You are an expert annotator for research-oriented, non-clinical emotion classification.

    {policy}

    Text:
    {text}

    The Phase 1 classifier predicted: {phase1_label}

    Select the final label that best represents the text. Return no rationale and end with exactly one line in this form:
    Final label: Depression
    or Final label: Neutral
    or Final label: Happy"""

    COT_PROMPT_VERSION = "final-llama3-cot-v2-original-text"
    COT_REQUESTS = [
        "You are an expert annotator for research-oriented, non-clinical emotion classification. Assist in analyzing emotions in text data. Do not make a clinical diagnosis, infer a medical condition, or provide treatment advice.",
        "I will first provide a piece of text. Independently assess its emotional content before comparing it with a Phase 1 AI-generated label. The only permitted labels are Depression, Neutral, and Happy.",
        "Independently analyze the text. State the dominant emotion and textual evidence before considering the Phase 1 label.\n\n{policy}\n\nUse only Depression, Neutral, or Happy for your provisional label.",
        "The Phase 1 classifier predicted: {phase1_label}\n\nCompare that prediction with your independent assessment. Confirm it only when it is supported by the dominant emotional meaning of the full text. Otherwise, explain the correction using textual evidence only.",
        "Provide the final Phase 2 decision. Use only one exact label: Depression, Neutral, or Happy. Do not use synonyms or additional labels. End the response with exactly one line: Final label: [label]",
    ]

    SELF_DISCOVER_PROMPT_VERSION = "final-llama3-self-discover-v1-original-text"
    ''')


LLAMA3_METHODS = code(
    r'''
    def run_llama3_direct(tokenizer, model, row):
        prompt = DIRECT_TEMPLATE.format(
            policy=CLASSIFICATION_POLICY,
            text=row["phase2_original_text"],
            phase1_label=row["phase1_label"],
        )
        result = chat_generate(
            tokenizer, model, [{"role": "user", "content": prompt}],
            max_new_tokens=LLAMA3_MAX_NEW_TOKENS,
            seed=stable_seed(row["example_id"], "llama3_direct", "final"),
            do_sample=False,
        )
        return {
            "direct_answer": result["text"],
            "final_label": parse_final_label(result["text"]),
            "input_tokens_total": result["input_tokens"],
            "generated_tokens_total": result["generated_tokens"],
            "generation_seconds_total": result["elapsed_seconds"],
            "any_stage_hit_token_limit": result["hit_token_limit"],
        }


    def run_llama3_cot(tokenizer, model, row):
        example_id = row["example_id"]
        messages = [
            {"role": "system", "content": COT_REQUESTS[0]},
            {"role": "user", "content": COT_REQUESTS[1]},
        ]
        stages = []
        ack = chat_generate(
            tokenizer, model, messages, max_new_tokens=LLAMA3_MAX_NEW_TOKENS,
            seed=stable_seed(example_id, "llama3_cot", "ack"), do_sample=False,
        )
        stages.append(ack)
        messages += [
            {"role": "assistant", "content": ack["text"]},
            {"role": "user", "content": f"Text:\n{row['phase2_original_text']}"},
        ]
        text_response = chat_generate(
            tokenizer, model, messages, max_new_tokens=LLAMA3_MAX_NEW_TOKENS,
            seed=stable_seed(example_id, "llama3_cot", "text_response"), do_sample=False,
        )
        stages.append(text_response)
        messages += [
            {"role": "assistant", "content": text_response["text"]},
            {"role": "user", "content": COT_REQUESTS[2].format(policy=CLASSIFICATION_POLICY)},
        ]
        independent = chat_generate(
            tokenizer, model, messages, max_new_tokens=LLAMA3_MAX_NEW_TOKENS,
            seed=stable_seed(example_id, "llama3_cot", "independent"), do_sample=False,
        )
        stages.append(independent)
        messages += [
            {"role": "assistant", "content": independent["text"]},
            {"role": "user", "content": COT_REQUESTS[3].format(phase1_label=row["phase1_label"])},
        ]
        comparison = chat_generate(
            tokenizer, model, messages, max_new_tokens=LLAMA3_MAX_NEW_TOKENS,
            seed=stable_seed(example_id, "llama3_cot", "comparison"), do_sample=False,
        )
        stages.append(comparison)
        messages += [
            {"role": "assistant", "content": comparison["text"]},
            {"role": "user", "content": COT_REQUESTS[4]},
        ]
        final = chat_generate(
            tokenizer, model, messages, max_new_tokens=LLAMA3_MAX_NEW_TOKENS,
            seed=stable_seed(example_id, "llama3_cot", "final"), do_sample=False,
        )
        stages.append(final)
        return {
            "cot_ack": ack["text"],
            "cot_text_response": text_response["text"],
            "cot_independent": independent["text"],
            "cot_comparison": comparison["text"],
            "cot_final_answer": final["text"],
            "final_label": parse_final_label(final["text"]),
            "input_tokens_total": sum(item["input_tokens"] for item in stages),
            "generated_tokens_total": sum(item["generated_tokens"] for item in stages),
            "generation_seconds_total": sum(item["elapsed_seconds"] for item in stages),
            "any_stage_hit_token_limit": any(item["hit_token_limit"] for item in stages),
        }


    def run_llama3_self_discover(tokenizer, model, row):
        example_id = row["example_id"]
        task = build_self_discover_task(row["phase2_original_text"], row["phase1_label"])
        select = select_prompt.replace("{Task}", task).replace("{resonining_modules}", reasoning_modules)
        selected = chat_generate(
            tokenizer, model, [{"role": "user", "content": select}],
            max_new_tokens=LLAMA3_MAX_NEW_TOKENS,
            seed=stable_seed(example_id, "llama3_self_discover", "select"), do_sample=False,
        )
        adapt = adapt_prompt.replace("{Task}", task).replace("{selected_modules}", selected["text"])
        adapted = chat_generate(
            tokenizer, model, [{"role": "user", "content": adapt}],
            max_new_tokens=LLAMA3_MAX_NEW_TOKENS,
            seed=stable_seed(example_id, "llama3_self_discover", "adapt"), do_sample=False,
        )
        implement = implement_prompt.replace("{Task}", task).replace("{adapted_modules}", adapted["text"])
        structure = chat_generate(
            tokenizer, model, [{"role": "user", "content": implement}],
            max_new_tokens=LLAMA3_MAX_NEW_TOKENS,
            seed=stable_seed(example_id, "llama3_self_discover", "implement"), do_sample=False,
        )
        final_prompt = (
            f"Using the following reasoning structure:\n{structure['text']}\n\n"
            f"Solve this task, providing your answer:\n{task}\n\n"
            "Note1: Write the question number before your answer.\n"
            "Note2: Do not write anything besides your answer.\n"
            "Note3: End with exactly one final label in the format Final label: Depression, "
            "Final label: Neutral, or Final label: Happy."
        )
        messages = [
            {"role": "system", "content": "You are an expert annotator for research-oriented, non-clinical emotion classification. Do not provide clinical diagnosis, medical inference, treatment advice, or professional mental health advice."},
            {"role": "user", "content": final_prompt},
        ]
        final = chat_generate(
            tokenizer, model, messages,
            max_new_tokens=LLAMA3_MAX_NEW_TOKENS,
            seed=stable_seed(example_id, "llama3_self_discover", "final"), do_sample=False,
        )
        stages = [selected, adapted, structure, final]
        return {
            "sd_selected_modules": selected["text"],
            "sd_adapted_modules": adapted["text"],
            "sd_reasoning_structure": structure["text"],
            "sd_final_answer": final["text"],
            "final_label": parse_final_label(final["text"]),
            "input_tokens_total": sum(item["input_tokens"] for item in stages),
            "generated_tokens_total": sum(item["generated_tokens"] for item in stages),
            "generation_seconds_total": sum(item["elapsed_seconds"] for item in stages),
            "any_stage_hit_token_limit": any(item["hit_token_limit"] for item in stages),
        }


    METHOD_CONFIG = {
        "direct": {
            "runner": run_llama3_direct,
            "prompt_version": DIRECT_PROMPT_VERSION,
            "path": LLAMA3_OUTPUT_DIR / "llama3_direct_results.csv",
        },
        "cot": {
            "runner": run_llama3_cot,
            "prompt_version": COT_PROMPT_VERSION,
            "path": LLAMA3_OUTPUT_DIR / "llama3_cot_results.csv",
        },
        "self_discover": {
            "runner": run_llama3_self_discover,
            "prompt_version": SELF_DISCOVER_PROMPT_VERSION,
            "path": LLAMA3_OUTPUT_DIR / "llama3_self_discover_results.csv",
        },
    }
    ''')


LLAMA3_RUN = code(
    r"""
    tokenizer, model = load_chat_model(LLAMA3_MODEL_NAME)

    for method in RUN_METHODS:
        config = METHOD_CONFIG[method]
        if not RESUME_FROM_EXISTING and config["path"].exists():
            config["path"].unlink()
        existing = load_results(config["path"])
        completed = set(existing.get("example_id", pd.Series(dtype=str)).astype(str))
        pending = routed_df[~routed_df["example_id"].isin(completed)].copy()
        print(f"{method}: {len(completed)} completed, {len(pending)} pending")
        for _, row in tqdm(pending.iterrows(), total=len(pending), desc=f"Llama 3 {method}"):
            generated = config["runner"](tokenizer, model, row)
            result = {
                "example_id": row["example_id"],
                "target_label": row["target_label"],
                "phase1_label": row["phase1_label"],
                "phase1_confidence": row["phase1_confidence"],
                "phase2_original_text": row["phase2_original_text"],
                "phase2_input_was_truncated": row["phase2_input_was_truncated"],
                "method": method,
                "model_name": LLAMA3_MODEL_NAME,
                "model_revision": getattr(model.config, "_commit_hash", None),
                "prompt_version": config["prompt_version"],
                "decoding": "greedy",
                "routed_id_hash": routed_id_hash,
                **generated,
            }
            save_result(config["path"], result)

    clear_model(tokenizer, model)
    print("All requested Llama 3 methods finished or resumed.")
    """
)


LLAMA3_QA = code(
    r"""
    summaries = []
    result_frames = {}
    expected_ids = set(routed_df["example_id"])

    for method in RUN_METHODS:
        config = METHOD_CONFIG[method]
        results = load_results(config["path"])
        observed_ids = set(results["example_id"].astype(str)) if not results.empty else set()
        if len(results) != len(routed_df) or observed_ids != expected_ids:
            missing = sorted(expected_ids - observed_ids)[:10]
            extra = sorted(observed_ids - expected_ids)[:10]
            raise ValueError(f"{method} IDs mismatch. Missing={missing}, extra={extra}")
        if results["routed_id_hash"].nunique() != 1 or results["routed_id_hash"].iloc[0] != routed_id_hash:
            raise ValueError(f"{method} was not generated from the current routed ID set.")

        prediction_col = f"llama3_{method}_final_label"
        results = results.rename(columns={"final_label": prediction_col})
        end_to_end, summary = summarize_method(phase1_df, results, prediction_col, f"Llama 3 {method}")
        end_to_end.to_csv(LLAMA3_OUTPUT_DIR / f"llama3_{method}_end_to_end_predictions.csv", index=False)
        summaries.append(summary)
        result_frames[method] = results

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(LLAMA3_OUTPUT_DIR / "llama3_reasoning_method_summary.csv", index=False)

    comparison = routed_df[["example_id", "target_label", "phase1_label", "phase1_confidence"]].copy()
    for method, results in result_frames.items():
        prediction_col = f"llama3_{method}_final_label"
        comparison = comparison.merge(
            results[["example_id", prediction_col]],
            on="example_id", how="left", validate="one_to_one",
        )
    comparison.to_csv(LLAMA3_OUTPUT_DIR / "llama3_reasoning_method_predictions_wide.csv", index=False)

    display(summary_df)
    display(comparison.head())
    print("Saved:", LLAMA3_OUTPUT_DIR)
    """
)


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    llama2_name = "05_1_final_unified_phase2_llama2_cot_colab.ipynb"
    llama2_matched_name = (
        "05_1_final_unified_phase2_llama2_cot_matched_colab_20260914.ipynb"
    )
    llama3_name = "05_2_final_unified_phase2_llama3_reasoning_methods_colab.ipynb"
    llama3_fresh_name = (
        "05_2_final_unified_phase2_llama3_reasoning_methods_colab_20260914.ipynb"
    )

    llama2_cells = [
        markdown(
            """
            # Final Unified Phase 2: Matched Llama 2 CoT

            This notebook consumes the final DistilBERT Phase 1 export and runs Llama 2 CoT on the routed cases only. Its classification policy, five-stage CoT structure, greedy decoding, parser, and failure fallback match the Llama 3 CoT comparison. The model-specific chat template is retained, and a separate output directory prevents reuse of the earlier sampled Llama 2 run.

            The frozen final input contract is 12,000 test rows with 218 routed cases. The notebook checks both counts before loading the LLM so an accidental input substitution cannot trigger a costly run.
            """
        ),
        SETUP, IMPORTS,
        markdown("## Configuration and final Phase 1 input contract"),
        CONFIG, DRIVE_AND_INPUT,
        markdown("## Model, generation, persistence, and evaluation helpers"),
        MODEL_HELPERS,
        markdown("## Llama 3-matched Llama 2 CoT prompt and runner"),
        LLAMA2_PROMPT,
        markdown("## Run or resume Llama 2 CoT"),
        LLAMA2_RUN,
    ]

    llama3_cells = [
        markdown(
            """
            # Final Unified Phase 2: Llama 3 Reasoning-Method Comparison

            This notebook consumes the same final DistilBERT Phase 1 export as the Llama 2 notebook and evaluates three Llama 3 paths on exactly the same routed example IDs:

            1. direct constrained classification,
            2. chain-of-thought re-evaluation,
            3. SELF-DISCOVER re-evaluation.

            The model, routed inputs, original-text policy, label set, parser, and greedy decoding are fixed across the three Llama 3 paths. The established SELF-DISCOVER prompt cells are copied verbatim from Final 04.5; the direct and CoT paths are the added comparators. Every result is saved row by row and can resume safely. Unambiguous terminal labels are parsed uniformly; a true parse failure is recorded and retains the Phase 1 label in the end-to-end output.
            """
        ),
        SETUP, IMPORTS,
        markdown("## Configuration and final Phase 1 input contract"),
        CONFIG, DRIVE_AND_INPUT,
        markdown("## Model, generation, persistence, and evaluation helpers"),
        MODEL_HELPERS,
        markdown("## Llama 3 direct, CoT, and SELF-DISCOVER prompts"),
        LLAMA3_PROMPTS,
        legacy_code_cell('reasoning_modules = """'),
        legacy_code_cell('SELF_DISCOVER_TASK_TEMPLATE = """'),
        markdown("## Method runners"),
        LLAMA3_METHODS,
        markdown("## Run or resume all three methods"),
        LLAMA3_RUN,
        markdown("## Cross-method integrity checks and final summaries"),
        LLAMA3_QA,
    ]

    outputs = {
        llama2_name: notebook(llama2_name, llama2_cells),
        llama2_matched_name: notebook(llama2_matched_name, llama2_cells),
        llama3_name: notebook(llama3_name, llama3_cells),
        llama3_fresh_name: notebook(llama3_fresh_name, llama3_cells),
    }
    for name, payload in outputs.items():
        path = OUT_DIR / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    build()
