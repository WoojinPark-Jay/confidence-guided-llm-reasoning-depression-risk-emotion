"""Verify unchanged prompts and exact reuse of the Mixed Emotion payload."""
import ast
import base64
import hashlib
import json
import tempfile
import zlib

from build_final_unified_phase2_notebooks import OUT_DIR
from build_mixed_sd_countercheck_notebook import find_cell
from test_reddit_sd_round2 import load, capture


def main():
    name = "10_mixed_emotion_llama3_sd_countercheck_colab.ipynb"
    nb = json.loads((OUT_DIR / name).read_text())
    original = json.loads((OUT_DIR / "06_2_mixed_emotion_llama3_reasoning_methods_colab.ipynb").read_text())
    input_text = "".join(find_cell(nb, 'drive.mount("/content/drive")')["source"])
    assert input_text == "".join(find_cell(original, 'drive.mount("/content/drive")')["source"])
    tree = ast.parse(input_text)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "embedded_payload" for t in n.targets))
    encoded = ast.literal_eval(assignment.value.args[0].args[0])
    payload = zlib.decompress(base64.b64decode(encoded))
    assert hashlib.sha256(payload).hexdigest() in input_text
    rows = json.loads(payload)
    assert len(rows) == len({r["example_id"] for r in rows}) == 300
    assert sum(r["phase1_routed"] for r in rows) == 86
    assert sum(r["phase1_label"] == r["target_label"] for r in rows) == 251
    assert all(r["phase2_original_text"].strip() for r in rows)
    config = "".join(find_cell(nb, "# Both Phase 2 notebooks")["source"])
    assert "EXPECTED_TOTAL_ROWS = 300" in config and "EXPECTED_ROUTED_ROWS = 86" in config
    assert '"unified_final/mixed_emotion/phase2"' in config
    with tempfile.TemporaryDirectory() as directory:
        reddit = load("09_reddit_llama3_sd_independent_round2_colab.ipynb", directory)
        mixed = load(name, directory)
        assert mixed["RUN_METHODS"] == ["sd_independent_countercheck"]
        assert capture(reddit, "sd_independent_countercheck") == capture(mixed, "sd_independent_countercheck")
        assert reddit["LLAMA3_OUTPUT_DIR"] != mixed["LLAMA3_OUTPUT_DIR"]
    print("PASS: Python cells compile; prompts identical across all four stages;")
    print("verified existing 300-row/86-routed payload, 251 baseline correct, isolated output.")


if __name__ == "__main__":
    main()
