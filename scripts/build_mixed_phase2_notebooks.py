"""Package verified Mixed Emotion predictions into standalone Phase 2 notebooks."""

import argparse
import base64
import csv
import hashlib
import json
import zlib
from pathlib import Path

from build_final_unified_phase2_notebooks import (
    CONFIG, DRIVE_AND_INPUT, OUT_DIR, code,
)


def build(input_path):
    with input_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len({r["example_id"] for r in rows}) == 300
    packaged = []
    for row in rows:
        routed = row["phase1_routed"].lower() == "true"
        assert routed == (float(row["phase1_confidence"]) < 0.7)
        assert abs(float(row["temperature"]) - 1.4801950079829693) < 1e-10
        assert float(row["routing_threshold"]) == 0.7
        packaged.append({
            "example_id": row["example_id"],
            "target_label": row["target_label"],
            "phase1_label": row["phase1_label"],
            "phase1_confidence": float(row["phase1_confidence"]),
            "phase1_routed": routed,
            "phase2_original_text": row["text"],
        })
    assert sum(r["phase1_routed"] for r in packaged) == 86
    assert sum(r["target_label"] == r["phase1_label"] for r in packaged) == 251
    payload = json.dumps(packaged, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(zlib.compress(payload)).decode("ascii")
    config = "".join(CONFIG["source"])
    config = config.replace(
        "unified_final/phase1/final_phase1_reasoning_input.csv",
        "unified_final/mixed_emotion/phase1/final_mixed_phase1_reasoning_input.csv",
    ).replace('"unified_final/phase2"', '"unified_final/mixed_emotion/phase2"')
    config = config.replace("EXPECTED_TOTAL_ROWS = 12_000", "EXPECTED_TOTAL_ROWS = 300")
    config = config.replace("EXPECTED_ROUTED_ROWS = 218", "EXPECTED_ROUTED_ROWS = 86")
    prepare = f'''
import base64
import zlib

embedded_payload = zlib.decompress(base64.b64decode({encoded!r}))
assert hashlib.sha256(embedded_payload).hexdigest() == {digest!r}
embedded_frame = pd.DataFrame(json.loads(embedded_payload))
PHASE1_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
if PHASE1_INPUT_PATH.exists():
    existing_frame = pd.read_csv(PHASE1_INPUT_PATH)
    pd.testing.assert_frame_equal(
        existing_frame[embedded_frame.columns].reset_index(drop=True),
        embedded_frame, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
    )
else:
    embedded_frame.to_csv(PHASE1_INPUT_PATH, index=False)
print("Verified embedded Mixed Emotion input: 300 rows, 86 routed.")
'''
    input_source = "".join(DRIVE_AND_INPUT["source"])
    marker = "if not PHASE1_INPUT_PATH.exists():"
    input_source = input_source.replace(marker, prepare + "\n" + marker, 1)
    pairs = [
        ("05_1_final_unified_phase2_llama2_cot_matched_colab_20260914.ipynb",
         "06_1_mixed_emotion_llama2_cot_matched_colab.ipynb", "Llama 2 matched CoT"),
        ("05_2_final_unified_phase2_llama3_reasoning_methods_colab_20260914.ipynb",
         "06_2_mixed_emotion_llama3_reasoning_methods_colab.ipynb", "Llama 3 Direct / CoT / SELF-DISCOVER"),
    ]
    for source_name, name, title in pairs:
        nb = json.loads((OUT_DIR / source_name).read_text())
        nb["metadata"]["colab"]["name"] = name
        nb["cells"][0]["source"] = [
            f"# Mixed Emotion: {title}\n",
            "\nVerified final DistilBERT predictions are embedded; no input upload is required.\n",
            "Mount the same Google Drive account. Evaluate 86 routed cases and report all 300 examples.\n",
            "Phase 1 accuracy: 251/300; temperature: 1.4801950079829693; threshold: 0.70.\n",
            "Generation settings and prompts are inherited from the corresponding final Reddit notebook.\n",
            "Outputs are saved under unified_final/mixed_emotion/phase2 and support row-level resume.\n",
        ]
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            text = "".join(cell["source"])
            if "# Both Phase 2 notebooks" in text:
                nb["cells"][i] = code(config)
            elif 'drive.mount("/content/drive")' in text:
                nb["cells"][i] = code(input_source)
        dest = OUT_DIR / name
        dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
        print(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    build(parser.parse_args().input)
