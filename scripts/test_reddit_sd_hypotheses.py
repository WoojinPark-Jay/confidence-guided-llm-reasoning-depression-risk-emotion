"""CPU-only prompt isolation and four-stage runner tests."""
import json
import tempfile
from pathlib import Path

from build_final_unified_phase2_notebooks import OUT_DIR


def main():
    payload = json.loads((OUT_DIR / "08_reddit_llama3_self_discover_hypotheses_colab.ipynb").read_text())
    with tempfile.TemporaryDirectory() as directory:
        ns = {"DRIVE_OUTPUT_ROOT": Path(directory), "json": json}
        for cell in payload["cells"]:
            text = "".join(cell["source"])
            if cell["cell_type"] != "code":
                continue
            if not any(line.startswith("%") for line in text.splitlines()):
                compile(text, "notebook_cell", "exec")
            if any(marker in text for marker in [
                'LLAMA3_MODEL_NAME =', 'reasoning_modules = """',
                'SELF_DISCOVER_TASK_TEMPLATE = """', 'INDEPENDENT_TASK =',
            ]):
                exec(text, ns)
        row = {"example_id": "test", "phase2_original_text": "A unique test passage.",
               "phase1_label": "SECRET_PHASE1"}
        calls = []

        def generate(tokenizer, model, messages, **kwargs):
            calls.append((messages, kwargs))
            return {"text": "Final label: Happy", "input_tokens": 10,
                    "generated_tokens": 5, "elapsed_seconds": 1.0, "hit_token_limit": False}

        ns.update(chat_generate=generate, stable_seed=lambda *args: 42,
                  parse_final_label=lambda answer: "Happy")
        for method in ns["METHOD_CONFIG"]:
            calls.clear()
            result = ns["run_sd_hypothesis"](None, None, row, method)
            assert len(calls) == 4
            assert result["input_tokens_total"] == 40
            assert result["generated_tokens_total"] == 20
            assert len(json.loads(result["stage_logs_json"])) == 4
            assert all(k["max_new_tokens"] == 1024 and not k["do_sample"] for _, k in calls)
            prompts = "\n".join(m["content"] for messages, _ in calls for m in messages)
            assert "A unique test passage." in prompts
            if method == "sd_independent":
                assert "SECRET_PHASE1" not in prompts
            else:
                assert "SECRET_PHASE1" in prompts
            if method == "sd_evidence":
                assert "Whole-text evidence handling:" in prompts
                assert "classify based on that final takeaway rather than" not in prompts
            if method == "sd_compact":
                assert "at most three" in prompts
        helpers = "".join(payload["cells"][5]["source"])
        assert helpers.count("revision=LLAMA3_MODEL_REVISION") == 2
        assert "target_label" not in ns["run_sd_hypothesis"].__code__.co_consts
        print("PASS: cell syntax, four variants, four stages, hidden Phase 1 label,")
        print("evidence-rule replacement, token/time totals, prompt logs and pinned model revision.")


if __name__ == "__main__":
    main()
