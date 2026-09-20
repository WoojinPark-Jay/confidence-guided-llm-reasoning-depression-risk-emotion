"""Build isolated, resumable SELF-DISCOVER hypothesis runs without altering finals."""

import json
from build_final_unified_phase2_notebooks import (
    OUT_DIR, SETUP, IMPORTS, CONFIG, DRIVE_AND_INPUT, MODEL_HELPERS,
    LLAMA3_PROMPTS, LLAMA3_QA, legacy_code_cell, code, markdown, notebook,
)


VARIANTS = r'''
LLAMA3_MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
LLAMA3_OUTPUT_DIR = DRIVE_OUTPUT_ROOT / "llama3_sd_hypotheses_v1"
LLAMA3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_METHODS = ["sd_independent", "sd_evidence", "sd_compact"]
# Optionally append sd_baseline to rerun the unchanged prompt with the same parser.

INDEPENDENT_TASK = """You are an expert annotator for research-oriented, non-clinical emotion classification.
Classify the dominant emotion of the supplied text independently. No other classifier's label is supplied.
Do not make a clinical diagnosis, infer a medical condition, or provide treatment advice.
Text:
{text}

1. Identify the dominant emotion, overall sentiment, final emotional trajectory and takeaway.
2. Assess the textual evidence for your independent classification.
3. Select exactly one label: Depression, Neutral, Happy.
{guidelines}
End with exactly one line: Final label: [label].
"""

EVIDENCE_GUIDANCE = """Whole-text evidence handling:
Identify the author's own expressed emotional state, distinguishing it from quotations,
other people's feelings, a factual topic, or a question. Check negation, irony and metaphor
in their surrounding context rather than treating positive or negative words literally.
Weigh the whole text. A final wish, plan, or attempt to cope is not automatically achieved
relief or positive resolution. Conversely, a mention of difficulty does not automatically
outweigh clearly expressed satisfaction. Use the existing class definitions, not clinical
inference. Compare the strongest evidence for your leading label and its alternative,
then choose the better-supported one. Mixed cues alone do not require Neutral.
"""

def hypothesis_task(row, variant):
    task = build_self_discover_task(row["phase2_original_text"], row["phase1_label"])
    if variant == "sd_independent":
        guidelines = SELF_DISCOVER_TASK_TEMPLATE.split("Classification Guidelines:", 1)[1]
        guidelines = guidelines.split("Output Constraint:", 1)[0]
        task = INDEPENDENT_TASK.format(
            text=row["phase2_original_text"], guidelines="Classification Guidelines:" + guidelines
        )
    elif variant == "sd_evidence":
        start = task.index("Mixed and Shifting Emotion Handling:")
        end = task.index("Output Constraint:", start)
        task = task[:start] + EVIDENCE_GUIDANCE + "\n" + task[end:]
        task = task.replace(
            "overall tone and final emotional trajectory", "overall tone supported by the full text"
        ).replace(
            "overall sentiment, final emotional trajectory, and final takeaway",
            "overall sentiment and contextual evidence across the whole text",
        )
    return task

def run_sd_hypothesis(tokenizer, model, row, variant):
    task = hypothesis_task(row, variant)
    logs = []

    def generate(stage, prompt, final=False):
        messages = [{"role": "user", "content": prompt}]
        if final:
            messages.insert(0, {"role": "system", "content":
                "You are an expert annotator for research-oriented, non-clinical emotion classification. Do not provide clinical diagnosis, medical inference, treatment advice, or professional mental health advice."})
        result = chat_generate(
            tokenizer, model, messages, max_new_tokens=LLAMA3_MAX_NEW_TOKENS,
            seed=stable_seed(row["example_id"], variant, stage), do_sample=False,
        )
        logs.append({"stage": stage, "messages": messages, **result})
        return result["text"]

    select = select_prompt.replace("{Task}", task).replace("{resonining_modules}", reasoning_modules)
    if variant == "sd_compact":
        select += "\nSelect at most three relevant modules. Give one short reason per module; no classification yet."
    selected = generate("select", select)
    adapt = adapt_prompt.replace("{Task}", task).replace("{selected_modules}", selected)
    if variant == "sd_compact":
        adapt += "\nUse at most three short, nonredundant evidence-checking instructions; no classification yet."
    adapted = generate("adapt", adapt)
    implement = implement_prompt.replace("{Task}", task).replace("{adapted_modules}", adapted)
    if variant == "sd_compact":
        implement += "\nReturn a compact JSON plan with at most three steps. Do not solve or preselect a label."
    structure = generate("implement", implement)
    prompt = (
        f"Using the following reasoning structure:\n{structure}\n\n"
        f"Solve this task, providing your answer:\n{task}\n\n"
        "Note1: Write the question number before your answer.\n"
        "Note2: Do not write anything besides your answer.\n"
        "Note3: End with exactly one final label in the format Final label: Depression, "
        "Final label: Neutral, or Final label: Happy."
    )
    if variant == "sd_compact":
        prompt += "\nKeep the evidence explanation within 120 words, without repeating the plan."
    answer = generate("final", prompt, final=True)
    return {
        "sd_selected_modules": selected, "sd_adapted_modules": adapted,
        "sd_reasoning_structure": structure, "sd_final_answer": answer,
        "final_label": parse_final_label(answer),
        "stage_logs_json": json.dumps(logs, ensure_ascii=False),
        "input_tokens_total": sum(s["input_tokens"] for s in logs),
        "generated_tokens_total": sum(s["generated_tokens"] for s in logs),
        "generation_seconds_total": sum(s["elapsed_seconds"] for s in logs),
        "any_stage_hit_token_limit": any(s["hit_token_limit"] for s in logs),
    }

METHOD_CONFIG = {name: {"path": LLAMA3_OUTPUT_DIR / f"llama3_{name}_results.csv",
                       "prompt_version": name + "-v1"}
                 for name in ["sd_independent", "sd_evidence", "sd_compact", "sd_baseline"]}
'''

RUN = r'''
import importlib.metadata as package_metadata
import platform

if not RUN_METHODS or len(set(RUN_METHODS)) != len(RUN_METHODS):
    raise ValueError("Choose unique hypothesis names.")
if not set(RUN_METHODS).issubset(METHOD_CONFIG):
    raise ValueError("Unknown hypothesis.")
if not RESUME_FROM_EXISTING:
    raise ValueError("Use a new output directory instead of deleting prior experiments.")
if not routed_df["phase1_routed"].all():
    raise ValueError("Only routed examples may be generated.")

manifest = {
    "experiment": "exploratory-reddit-sd-v1", "model": LLAMA3_MODEL_NAME,
    "revision": LLAMA3_MODEL_REVISION, "max_new_tokens": LLAMA3_MAX_NEW_TOKENS,
    "decoding": "greedy", "base_seed": BASE_SEED, "load_in_4bit": LOAD_IN_4BIT,
    "max_rows": MAX_ROWS, "methods": RUN_METHODS, "routed_id_hash": routed_id_hash,
    "input_sha256": hashlib.sha256(PHASE1_INPUT_PATH.read_bytes()).hexdigest(),
    "implementation_sha256": IMPLEMENTATION_SHA256,
    "environment": {"python": platform.python_version(),
                    "packages": {p: package_metadata.version(p) for p in
                                 ["torch", "transformers", "accelerate", "bitsandbytes"]},
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"},
    "effective_tasks_sha256": hashlib.sha256(json.dumps([
        [method, str(row["example_id"]), hypothesis_task(row, method)]
        for method in RUN_METHODS for _, row in routed_df.iterrows()
    ], ensure_ascii=False).encode()).hexdigest(),
}
manifest_path = LLAMA3_OUTPUT_DIR / "experiment_manifest.json"
if manifest_path.exists():
    if json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Run settings/input changed. Use a new output directory; do not mix results.")
else:
    if any(LLAMA3_OUTPUT_DIR.glob("*_results.csv")):
        raise ValueError("Existing results without a manifest; use a new directory.")
    manifest_path.write_text(json.dumps(manifest, indent=2))

for method in RUN_METHODS:
    existing = load_results(METHOD_CONFIG[method]["path"])
    if not existing.empty:
        if not set(existing["example_id"].astype(str)).issubset(set(routed_df["example_id"])):
            raise ValueError("Unexpected saved IDs.")
        if not existing["prompt_version"].eq(METHOD_CONFIG[method]["prompt_version"]).all():
            raise ValueError("Saved prompt version mismatch.")

tokenizer, model = load_chat_model(LLAMA3_MODEL_NAME)
try:
    for method in RUN_METHODS:
        config = METHOD_CONFIG[method]
        existing = load_results(config["path"])
        completed = set(existing.get("example_id", pd.Series(dtype=str)).astype(str))
        pending = routed_df[~routed_df["example_id"].isin(completed)]
        print(f"{method}: {len(completed)} completed, {len(pending)} pending")
        for _, row in tqdm(pending.iterrows(), total=len(pending), desc=method):
            # Ground truth is deliberately unavailable to the generation runner.
            inference_row = {key: row[key] for key in ["example_id", "phase2_original_text", "phase1_label"]}
            generated = run_sd_hypothesis(tokenizer, model, inference_row, method)
            save_result(config["path"], {
                "example_id": row["example_id"], "target_label": row["target_label"],
                "phase1_label": row["phase1_label"], "phase1_confidence": row["phase1_confidence"],
                "phase2_input_was_truncated": row["phase2_input_was_truncated"],
                "method": method, "prompt_version": config["prompt_version"],
                "model_revision": LLAMA3_MODEL_REVISION, "routed_id_hash": routed_id_hash,
                **generated,
            })
finally:
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
'''


def build():
    import hashlib
    name = "08_reddit_llama3_self_discover_hypotheses_colab.ipynb"
    config = "".join(CONFIG["source"]).replace(
        '"unified_final/phase2"', '"unified_final/reddit_sd_exploration/phase2"'
    )
    helpers = "".join(MODEL_HELPERS["source"]).replace(
        'model_name, trust_remote_code=True, token=os.environ.get("HF_TOKEN")',
        'model_name, revision=LLAMA3_MODEL_REVISION, trust_remote_code=True, token=os.environ.get("HF_TOKEN")',
    )
    helpers = helpers.replace('        torch_dtype=torch.float16,',
                              '        revision=LLAMA3_MODEL_REVISION,\n        torch_dtype=torch.float16,')
    modules = legacy_code_cell('reasoning_modules = """')
    task = legacy_code_cell('SELF_DISCOVER_TASK_TEMPLATE = """')
    digest = hashlib.sha256((helpers + VARIANTS + RUN + config +
        "".join(modules["source"]) + "".join(task["source"])).encode()).hexdigest()
    cells = [markdown("""# Reddit SELF-DISCOVER: three hypothesis variants

기존 Drive 입력의 12,000건 중 동일한 218건을 원문으로 재평가합니다.
세 버전을 순차 실행하며 기존 실험 파일은 변경하지 않습니다.
이 Reddit 결과를 이용한 프롬프트 개선은 탐색 실험이며 독립 최종 평가가 아닙니다.

- sd_independent: 모든 단계에서 Phase 1 라벨을 숨깁니다.
- sd_evidence: 마지막 문장 우선 규칙을 전체 문맥 근거 규칙으로 바꿉니다.
- sd_compact: 기존 지침을 유지하되 모듈과 계획의 길이를 제한합니다.

GPU 런타임에서 설치 셀을 실행하고 런타임을 재시작한 뒤 imports부터 실행하세요.
기존 Google Drive 계정으로 연결하면 파일 업로드는 필요 없습니다.
기본 실행량: 218건 x 3버전 x 4단계 = 2,616회 생성.
중단 후 같은 설정으로 재실행하면 완료된 사례부터 이어갑니다.
MAX_ROWS smoke test는 별도 출력 폴더에서 실행하세요.
"""), SETUP, IMPORTS, code(config), DRIVE_AND_INPUT, code(helpers),
        LLAMA3_PROMPTS, modules, task, code(VARIANTS),
        code(f'IMPLEMENTATION_SHA256 = "{digest}"\n' + RUN), LLAMA3_QA]
    payload = notebook(name, cells)
    for i, cell in enumerate(payload["cells"]):
        cell["id"] = f"sd-hypothesis-{i:02d}"
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    build()
