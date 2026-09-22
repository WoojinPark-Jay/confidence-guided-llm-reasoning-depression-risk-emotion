"""CPU-only equivalence tests for Llama 2 Mixed Emotion additions."""
import ast
import json
import tempfile
from pathlib import Path

from build_final_unified_phase2_notebooks import OUT_DIR, LLAMA3_METHODS
from build_mixed_sd_countercheck_notebook import find_cell
from test_reddit_sd_round2 import load, capture


def main():
    nb = json.loads((OUT_DIR / "11_mixed_emotion_llama2_direct_final_sd_colab.ipynb").read_text())
    old_nb = json.loads((OUT_DIR / "10_mixed_emotion_llama3_sd_countercheck_colab.ipynb").read_text())
    assert find_cell(nb, 'drive.mount("/content/drive")')["source"] == find_cell(old_nb, 'drive.mount("/content/drive")')["source"]
    with tempfile.TemporaryDirectory() as directory:
        ns = {"DRIVE_OUTPUT_ROOT": Path(directory), "json": json}
        for cell in nb["cells"]:
            text = "".join(cell["source"])
            if cell["cell_type"] != "code":
                continue
            if not any(line.startswith("%") for line in text.splitlines()):
                compile(text, "cell", "exec")
            if any(marker in text for marker in ['LLAMA2_MODEL_NAME =', 'reasoning_modules = """',
                    'SELF_DISCOVER_TASK_TEMPLATE = """', 'INDEPENDENT_TASK =']):
                exec(text, ns)
        old = load("10_mixed_emotion_llama3_sd_countercheck_colab.ipynb", directory)
        assert capture(ns, "sd_independent_countercheck") == capture(old, "sd_independent_countercheck")
        calls = []

        def generate(tokenizer, model, messages, **kwargs):
            calls.append((messages, kwargs))
            return {"text": "Final label: Happy", "input_tokens": 8,
                    "generated_tokens": 4, "elapsed_seconds": 1, "hit_token_limit": False}

        row = {"example_id": "test", "phase2_original_text": "Test original passage.", "phase1_label": "Neutral"}
        ns["chat_generate"] = generate
        result = ns["run_llama2_direct"](None, None, row)
        actual = list(calls)
        source = "".join(LLAMA3_METHODS["source"])
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "run_llama3_direct")
        exec(ast.get_source_segment(source, node), old)
        old.update(chat_generate=generate, stable_seed=lambda *args: 42, parse_final_label=lambda text: "Happy")
        calls.clear()
        old["run_llama3_direct"](None, None, row)
        assert calls == actual and len(calls) == 1
        assert "Phase 1 classifier predicted: Neutral" in calls[0][0][0]["content"]
        assert len(json.loads(result["stage_logs_json"])) == 1
        assert ns["RUN_METHODS"] == ["direct", "sd_independent_countercheck"]
        assert ns["LLAMA2_MODEL_NAME"] == "NousResearch/Llama-2-7b-chat-hf"
        assert ns["LLAMA2_MODEL_REVISION"] == "351844e75ed0bcbbe3f10671b3c808d2b83894ee"
        assert ns["LLAMA2_MAX_NEW_TOKENS"] == 1024
        assert ns["LLAMA2_OUTPUT_DIR"].name == "llama2_direct_final_sd_v1"
    print("PASS: syntax, identical Mixed input, Llama 2 revision and output isolation,")
    print("unchanged Direct/SD prompts, one/four generation stages, phase1-label policy and logs.")


if __name__ == "__main__":
    main()
