"""Check Reddit input wiring and unchanged Mixed Emotion model/prompt code."""
import json

from build_final_unified_phase2_notebooks import OUT_DIR
from build_mixed_sd_countercheck_notebook import find_cell
from test_mixed_llama2_direct_sd import main as test_methods


def main():
    nb = json.loads((OUT_DIR / "12_reddit_llama2_direct_final_sd_colab.ipynb").read_text())
    mixed = json.loads((OUT_DIR / "11_mixed_emotion_llama2_direct_final_sd_colab.ipynb").read_text())
    reddit = json.loads((OUT_DIR / "05_2_final_unified_phase2_llama3_reasoning_methods_colab_20260914.ipynb").read_text())
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] == "code" and not any(l.startswith("%") for l in text.splitlines()):
            compile(text, "cell", "exec")
    for marker in ['LLAMA2_MODEL_NAME =', 'INDEPENDENT_TASK =', 'def chat_generate(', 'summaries = []']:
        assert find_cell(nb, marker)["source"] == find_cell(mixed, marker)["source"]
    for marker in ['# Both Phase 2 notebooks', 'drive.mount("/content/drive")']:
        assert find_cell(nb, marker)["source"] == find_cell(reddit, marker)["source"]
    config = "".join(find_cell(nb, '# Both Phase 2 notebooks')["source"])
    assert 'EXPECTED_TOTAL_ROWS = 12_000' in config
    assert 'EXPECTED_ROUTED_ROWS = 218' in config
    assert '"unified_final/phase2"' in config and 'mixed_emotion' not in config
    test_methods()
    print("PASS: Reddit 12,000/218 contract, original-text loader, separate output root,")
    print("and byte-identical model, prompts, runners, parser and summary code.")


if __name__ == "__main__":
    main()
