"""Build separate CoT-1024 reruns while preserving CoT prompts and old outputs."""
import hashlib
import json

from build_final_unified_phase2_notebooks import OUT_DIR, code, markdown
from build_mixed_sd_countercheck_notebook import find_cell


PREFLIGHT = '''
import importlib.metadata as package_metadata
environment = {p: package_metadata.version(p) for p in
               ["torch", "transformers", "accelerate", "bitsandbytes"]}
contract = {"input_sha256": hashlib.sha256(PHASE1_INPUT_PATH.read_bytes()).hexdigest(),
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "model_revision": LLAMA2_MODEL_REVISION, "max_new_tokens": MAX_NEW_TOKENS_COT,
            "routed_id_hash": routed_id_hash, "max_rows": MAX_ROWS,
            "load_in_4bit": LOAD_IN_4BIT, "base_seed": BASE_SEED,
            "context_policy": "preserve-input-cap-generation-to-remaining-v1"}
manifest_path = LLAMA2_OUTPUT_DIR / "experiment_manifest.json"
if not RESUME_FROM_EXISTING:
    raise ValueError("Use a new output folder; existing results must not be deleted.")
if manifest_path.exists():
    if json.loads(manifest_path.read_text()) != contract:
        raise ValueError("Input or experiment settings changed. Existing results preserved.")
else:
    if LLAMA2_RESULTS_PATH.exists():
        raise ValueError("Results without manifest; choose a new output folder.")
    manifest_path.write_text(json.dumps(contract, indent=2))
env_path = LLAMA2_OUTPUT_DIR / "runtime_environments.json"
history = json.loads(env_path.read_text()) if env_path.exists() else []
if environment not in history:
    history.append(environment)
    env_path.write_text(json.dumps(history, indent=2))
'''


def build():
    context_nb = json.loads((OUT_DIR / '12_reddit_llama2_direct_final_sd_colab.ipynb').read_text())
    helpers = find_cell(context_nb, 'def chat_generate(')
    pairs = [
        ('05_1_final_unified_phase2_llama2_cot_matched_colab_20260914.ipynb',
         '13_reddit_llama2_cot_1024_colab.ipynb', 'Reddit', 218),
        ('06_1_mixed_emotion_llama2_cot_matched_colab.ipynb',
         '14_mixed_emotion_llama2_cot_1024_colab.ipynb', 'Mixed Emotion', 86),
    ]
    for source, name, dataset, count in pairs:
        nb = json.loads((OUT_DIR / source).read_text())
        nb['metadata']['colab']['name'] = name
        run_index = None
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue
            text = ''.join(cell['source'])
            if 'def chat_generate(' in text:
                text = ''.join(helpers['source'])
            elif 'def run_llama2_cot_one(' in text:
                text = text.replace('MAX_NEW_TOKENS_COT = 512', 'MAX_NEW_TOKENS_COT = 1024')
                text = text.replace('"llama2_cot_matched_llama3"', '"llama2_cot_matched_1024_v1"')
                text = 'LLAMA2_MODEL_REVISION = "351844e75ed0bcbbe3f10671b3c808d2b83894ee"\n' + text
                text = text.replace('    example_id = row["example_id"]', '''    example_id = row["example_id"]
    stage_logs = []
    stage_names = ["ack", "text_response", "independent", "comparison", "final"]
    def logged_generate(tokenizer, model, messages, **kwargs):
        saved_messages = [dict(message) for message in messages]
        result = chat_generate(tokenizer, model, messages, **kwargs)
        stage_logs.append({"stage": stage_names[len(stage_logs)], "messages": saved_messages, **result})
        return result''')
                for variable in ('ack', 'text_response', 'independent', 'comparison', 'final'):
                    text = text.replace(variable + ' = chat_generate(', variable + ' = logged_generate(')
                text = text.replace('"llama2_final_label": parse_final_label(final["text"]),',
                    '"llama2_final_label": (np.nan if any(s.get("context_capacity_exceeded", False) for s in stages) '
                    'else parse_final_label(final["text"])),\n'
                    '        "stage_logs_json": json.dumps(stage_logs, ensure_ascii=False),\n'
                    '        "any_stage_context_budget_adjusted": any(s.get("context_budget_adjusted", False) for s in stages),\n'
                    '        "any_stage_context_capacity_exceeded": any(s.get("context_capacity_exceeded", False) for s in stages),')
            elif 'results = load_results(LLAMA2_RESULTS_PATH)' in text:
                run_index = i
                text = text.replace('generated = run_llama2_cot_one(tokenizer, model, row)',
                    'generated = run_llama2_cot_one(tokenizer, model, {k: row[k] for k in '
                    '["example_id", "phase1_label", "phase2_original_text"]})')
                text = text.replace('"prompt_version": LLAMA2_PROMPT_VERSION,',
                    '"prompt_version": LLAMA2_PROMPT_VERSION,\n'
                    '            "requested_max_new_tokens": MAX_NEW_TOKENS_COT,\n'
                    '            "runtime_environment_json": json.dumps(environment, sort_keys=True),')
                text = text.replace('"Llama 2 CoT matched"', '"Llama 2 CoT matched 1024"')
                text = PREFLIGHT + text
            nb['cells'][i] = code(text)
        if run_index is None:
            raise ValueError('Missing run cell.')
        nb['cells'][0] = markdown(f'''# {dataset}: Llama 2 CoT (maximum 1,024 tokens per stage)

기존 CoT 프롬프트와 5단계 대화는 그대로 두고 생성 상한을 512에서 1,024로 바꿉니다.
기존 입력 {count}건을 재평가하며 최종 지표는 전체 평가 데이터 기준입니다.
기존 Drive 계정으로 연결하세요. 입력은 기존 노트북과 같고 추가 업로드는 필요 없습니다.
설치 셀 실행 후 런타임을 재시작하고 imports부터 실행하세요.

- 모델 revision은 최종 Llama 2 Direct/SD와 동일하게 고정합니다.
- greedy decoding, 4-bit NF4, 기존 파싱 및 Phase 1 fallback 유지.
- 원문을 추가로 자르지 않고, 문맥 여유가 부족하면 생성 상한을 줄여 기록합니다.
- 입력 자체가 한도를 넘으면 해당 사례는 실패로 기록하고 Phase 1 예측을 유지합니다.
- 기존 512 결과는 보존하고 phase2/llama2_cot_matched_1024_v1/에 별도 저장합니다.
- 같은 설정으로 재개하면 저장된 사례는 건너뜁니다. 모든 단계의 응답과 토큰 수를 저장합니다.
''')
        digest = hashlib.sha256(json.dumps(nb['cells'], sort_keys=True).encode()).hexdigest()
        nb['cells'][run_index] = code(f'IMPLEMENTATION_SHA256 = "{digest}"\n' + ''.join(nb['cells'][run_index]['source']))
        for i, cell in enumerate(nb['cells']):
            cell['id'] = f'cot-1024-{i:02d}'
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
        path = OUT_DIR / name
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
        print(path)


if __name__ == '__main__':
    build()
