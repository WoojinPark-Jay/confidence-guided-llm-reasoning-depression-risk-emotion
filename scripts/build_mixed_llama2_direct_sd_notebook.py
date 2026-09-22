"""Add Llama 2 Direct and final independent SD without changing existing runs."""
import ast
import hashlib
import json

from build_final_unified_phase2_notebooks import OUT_DIR, LLAMA3_METHODS, code, markdown


def build():
    nb = json.loads((OUT_DIR / "10_mixed_emotion_llama3_sd_countercheck_colab.ipynb").read_text())
    name = "11_mixed_emotion_llama2_direct_final_sd_colab.ipynb"
    nb["metadata"]["colab"]["name"] = name
    methods = "".join(LLAMA3_METHODS["source"])
    tree = ast.parse(methods)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_llama3_direct")
    direct = ast.get_source_segment(methods, node)
    direct = direct.replace('"direct_answer": result["text"],',
        '"direct_answer": result["text"],\n'
        '        "stage_logs_json": json.dumps([{"stage": "final", "messages": '
        '[{"role": "user", "content": prompt}], **result}], ensure_ascii=False),')
    run_index = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        if "INDEPENDENT_TASK =" in text:
            text = text.replace("53346005fb0ef11d3b6a83b12c895cca40156b6c",
                                "351844e75ed0bcbbe3f10671b3c808d2b83894ee")
            text = text.replace("llama3_sd_independent_countercheck_v1", "llama2_direct_final_sd_v1")
            text = text.replace('RUN_METHODS = ["sd_independent_countercheck"]',
                                'RUN_METHODS = ["direct", "sd_independent_countercheck"]')
            text += "\n\n" + direct + "\n"
            text += 'METHOD_CONFIG["direct"]["prompt_version"] = DIRECT_PROMPT_VERSION\n'
        if 'LLAMA3_MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"' in text:
            text = text.replace("NousResearch/Meta-Llama-3-8B-Instruct", "NousResearch/Llama-2-7b-chat-hf")
        if "IMPLEMENTATION_SHA256 =" in text:
            run_index = i
            text = text.split("\n", 1)[1]
            text = text.replace("mixed-sd-independent-countercheck-v1", "mixed-llama2-direct-final-sd-v1")
            text = text.replace(
                '            generated = run_sd_hypothesis(tokenizer, model, inference_row, method)',
                '            if method == "direct":\n'
                '                generated = run_llama3_direct(tokenizer, model, inference_row)\n'
                '            else:\n'
                '                generated = run_sd_hypothesis(tokenizer, model, inference_row, method)')
        if "def chat_generate(" in text:
            text = text.replace('    input_ids = model_inputs["input_ids"]',
                '    input_ids = model_inputs["input_ids"]\n'
                '    context_limit = getattr(model.config, "max_position_embeddings", None)\n'
                '    if context_limit and input_ids.shape[-1] + max_new_tokens > context_limit:\n'
                '        raise ValueError("Prompt plus generation budget exceeds model context; "\n'
                '                         "stopping without silently truncating the comparison input.")')
        text = text.replace("LLAMA3", "LLAMA2").replace("llama3", "llama2").replace("Llama 3", "Llama 2")
        nb["cells"][i] = code(text)
    if run_index is None:
        raise ValueError("Missing execution cell.")
    nb["cells"][0] = markdown("""# Mixed Emotion: Llama 2 Direct + final SELF-DISCOVER

기존 Mixed Emotion 300건 중 같은 86건을 두 방법으로 순차 재평가합니다.
입력이 포함되어 있어 별도 업로드가 필요 없습니다. 기존 Drive 파일은 일치 여부를 확인합니다.

- Direct: 기존 Llama 3 Direct와 동일한 프롬프트. Phase 1 라벨을 제공합니다.
- 최종 SD: 독립 판단 + 반대 근거 점검. 기존 Llama 3 최종 SD와 동일한 프롬프트이며 Phase 1 라벨을 숨깁니다.
- 정답 라벨은 두 방법 모두 생성 함수에 전달하지 않습니다.
- NousResearch/Llama-2-7b-chat-hf, 고정 revision, 4-bit NF4, greedy, 단계별 최대 1,024토큰.
- 모델별 chat template은 유지합니다. 기존 Llama 2 CoT의 상한은 512토큰이므로 해당 차이는 남아 있습니다.

설치 셀 실행 → 런타임 재시작 → imports부터 순서대로 실행하세요.
Direct 86회 + SD 344회 = 총 430회 생성합니다. 모델은 한 번 로드합니다.
결과는 전체 300건 기준 정확도/F1, 수정/신규 오류/순수정과 실패 건수로 출력합니다.
최종 라벨 미추출 시 Phase 1 예측을 유지하고, 단계별 원시 응답을 저장합니다.

별도 결과 위치: unified_final/mixed_emotion/phase2/llama2_direct_final_sd_v1/
기존 CoT 및 Llama 3 결과는 변경하지 않습니다. 동일 설정으로 중단 후 재개 가능합니다.
""")
    run = "".join(nb["cells"][run_index]["source"])
    digest = hashlib.sha256((json.dumps(nb["cells"][:run_index], sort_keys=True) + run).encode()).hexdigest()
    nb["cells"][run_index] = code(f'IMPLEMENTATION_SHA256 = "{digest}"\n' + run)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"mixed-llama2-direct-sd-{i:02d}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    path = OUT_DIR / name
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    build()
