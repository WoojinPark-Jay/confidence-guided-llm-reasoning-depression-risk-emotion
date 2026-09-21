"""Test exact prompt differences against independent SD without a GPU."""
import json
import tempfile
from pathlib import Path

from build_final_unified_phase2_notebooks import OUT_DIR


def load(name, directory):
    nb = json.loads((OUT_DIR / name).read_text())
    ns = {"DRIVE_OUTPUT_ROOT": Path(directory), "json": json}
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] != "code":
            continue
        if not any(line.startswith("%") for line in text.splitlines()):
            compile(text, name, "exec")
        if any(marker in text for marker in [
            "LLAMA3_MODEL_NAME =", 'reasoning_modules = """',
            'SELF_DISCOVER_TASK_TEMPLATE = """', "INDEPENDENT_TASK =",
        ]):
            exec(text, ns)
    return ns


def capture(ns, method):
    calls = []

    def generate(tokenizer, model, messages, **kwargs):
        calls.append((messages, kwargs))
        return {"text": "Final label: Happy", "input_tokens": 8,
                "generated_tokens": 4, "elapsed_seconds": 1, "hit_token_limit": False}

    ns.update(chat_generate=generate, stable_seed=lambda *args: 42,
              parse_final_label=lambda text: "Happy")
    row = {"example_id": "test", "phase1_label": "SECRET_PHASE1",
           "phase2_original_text": "Test original passage."}
    result = ns["run_sd_hypothesis"](None, None, row, method)
    assert len(calls) == 4
    assert result["generated_tokens_total"] == 16
    assert len(json.loads(result["stage_logs_json"])) == 4
    assert "SECRET_PHASE1" not in json.dumps(calls)
    assert all(kwargs["max_new_tokens"] == 1024 and not kwargs["do_sample"] for _, kwargs in calls)
    return calls


def main():
    with tempfile.TemporaryDirectory() as directory:
        old = load("08_reddit_llama3_self_discover_hypotheses_colab.ipynb", directory)
        new = load("09_reddit_llama3_sd_independent_round2_colab.ipynb", directory)
        baseline = capture(old, "sd_independent")
        check = capture(new, "sd_independent_countercheck")
        neutral = capture(new, "sd_independent_neutral_plan")
        assert baseline[:3] == check[:3], "Countercheck must change only final prompt."
        assert baseline[3] != check[3]
        assert "Before committing" in check[3][0][-1]["content"]
        assert baseline[3] == neutral[3], "Neutral plan must preserve final template."
        for i in range(3):
            assert baseline[i] != neutral[i]
            assert "This stage designs a reasoning procedure" in neutral[i][0][-1]["content"]
        assert len(new["RUN_METHODS"]) == 2
        assert old["LLAMA3_OUTPUT_DIR"] != new["LLAMA3_OUTPUT_DIR"]
        print("PASS: syntax, two variants, stage isolation, no Phase 1 label,")
        print("four calls per case, token totals, logs and separate output paths.")


if __name__ == "__main__":
    main()
