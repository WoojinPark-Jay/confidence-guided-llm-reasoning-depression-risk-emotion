"""Build the Llama 3 cleaned-input run from the final Reddit notebook."""

import json

from build_final_unified_phase2_notebooks import OUT_DIR, code


INPUT_CHECK = '''
    # Verify the saved Phase 1 cleaned text before replacing the Phase 2 input.
    required_cleaned = {"example_id", "text", "target_label", "phase1_label",
                        "phase1_confidence", "phase1_routed", "temperature",
                        "routing_threshold"}
    missing_cleaned = required_cleaned - set(frame.columns)
    if missing_cleaned:
        raise ValueError(f"Final input is missing: {sorted(missing_cleaned)}")
    if frame["text"].isna().any():
        raise ValueError("Missing Phase 1 cleaned text.")
    cleaned_pairs = sorted(
        [str(example_id), str(text)]
        for example_id, text in zip(frame["example_id"], frame["text"])
    )
    cleaned_hash = hashlib.sha256(json.dumps(
        cleaned_pairs, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if cleaned_hash != "f7c1aa34fe80dfb00bb6ba48b480b50c7daf92e7e4fc4d985be7b6f08dfaa995":
        raise ValueError("Cleaned text does not match the verified final Phase 1 export.")
    if not np.allclose(frame["temperature"], 1.4801950079829693, rtol=0, atol=1e-12):
        raise ValueError("Unexpected final temperature.")
    if not np.allclose(frame["routing_threshold"], 0.70, rtol=0, atol=1e-12):
        raise ValueError("Unexpected routing threshold.")
    if not frame["phase1_routed"].map(normalize_bool).eq(
        frame["phase1_confidence"] < 0.70
    ).all():
        raise ValueError("Routing flags disagree with the fixed threshold.")
    if int(frame["phase1_label"].eq(frame["target_label"]).sum()) != 11630:
        raise ValueError("Unexpected final Phase 1 accuracy.")
    frame["phase2_original_text"] = frame["text"].astype(str)
    frame["phase2_input_was_truncated"] = False
    frame["phase2_input_policy_version"] = "reddit-phase1-cleaned-head-tail-6000-v1"
    frame["phase2_input_source"] = "saved_phase1_text_column"
'''


def build():
    source = OUT_DIR / "05_2_final_unified_phase2_llama3_reasoning_methods_colab_20260914.ipynb"
    name = "07_reddit_cleaned_llama3_reasoning_methods_colab_20260915.ipynb"
    nb = json.loads(source.read_text(encoding="utf-8"))
    nb["metadata"]["colab"]["name"] = name
    nb["cells"][0]["source"] = [
        "# Reddit Cleaned Input: Llama 3 Direct / CoT / SELF-DISCOVER\n",
        "\nUses the saved final Phase 1 text column on the same 218 routed cases.\n",
        "The existing Drive input contains the cleaned text; no upload is needed.\n",
        "All 12,000 predictions contribute to end-to-end metrics.\n",
        "Prompts, greedy decoding, 1,024-token generation limit, and parser are inherited from the final Reddit run.\n",
        "The existing 6,000-character head/tail policy is retained and logged.\n",
        "Results resume within the separate unified_final/reddit_cleaned/phase2 folder.\n",
    ]
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        if "# Both Phase 2 notebooks" in text:
            text = text.replace('"unified_final/phase2"', '"unified_final/reddit_cleaned/phase2"')
        elif "def normalize_phase1_export(path):" in text:
            marker = "    frame = pd.read_csv(path)\n"
            if text.count(marker) != 1:
                raise ValueError("Input loader changed; inspect before building.")
            text = text.replace(marker, marker + INPUT_CHECK, 1)
            text += '\nprint("Input: saved Phase 1 cleaned text")\n'
            text += 'print("Routed inputs shortened by the 6000-character policy:", int(routed_df["phase2_input_was_truncated"].sum()))\n'
        elif '"decoding": "greedy"' in text:
            text = text.replace(
                '"decoding": "greedy",',
                '"decoding": "greedy",\n'
                '            "phase2_input_policy_version": row["phase2_input_policy_version"],\n'
                '            "phase2_input_source": row["phase2_input_source"],',
            )
        nb["cells"][i] = code(text)
    path = OUT_DIR / name
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    build()
