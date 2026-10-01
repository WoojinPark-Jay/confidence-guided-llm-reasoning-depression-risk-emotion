"""Reuse final Direct/CoT source and the collaborator's v6c runner and cached plan."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/colab/final/07_2_existing9000_llama3_direct_cot_v6c_colab.ipynb"


def notebook_source(path):
    book = json.loads(path.read_text())
    return "\n\n".join("".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code")


def selected(source, names):
    tree = ast.parse("\n".join(line for line in source.splitlines() if not line.startswith("%")))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            found[node.name] = ast.get_source_segment("\n".join(line for line in source.splitlines()
                                                               if not line.startswith("%")), node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = ast.get_source_segment("\n".join(line for line in source.splitlines()
                                                                       if not line.startswith("%")), node)
    return "\n\n".join(found[name] for name in names) + "\n"


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(sd_notebook, plan_path):
    direct_path = ROOT / "notebooks/colab/final/05_2_final_unified_phase2_llama3_reasoning_methods_colab.ipynb"
    baseline = notebook_source(direct_path)
    sd = notebook_source(sd_notebook)
    assert selected(baseline, ["CLASSIFICATION_POLICY"]) == selected(sd, ["CLASSIFICATION_POLICY"])
    source = selected(baseline, [
        "normalize_bool", "normalize_label", "limit_reasoning_text", "stable_seed", "set_generation_seed",
        "load_chat_model", "format_messages_without_chat_template", "build_chat_inputs", "chat_generate",
        "parse_final_label", "summarize_method", "CLASSIFICATION_POLICY", "DIRECT_PROMPT_VERSION",
        "DIRECT_TEMPLATE", "COT_PROMPT_VERSION", "COT_REQUESTS", "run_llama3_direct", "run_llama3_cot",
    ])
    source += selected(sd, ["ANNOTATOR_SYSTEM_PROMPT", "V6_TEMPLATE", "build_v6_prompt",
                            "stage_totals", "run_v6_max_prompt"])
    # Pin the known repository revision; preserve the original prompts and generation behavior.
    source = source.replace('model_name, trust_remote_code=True, token=os.environ.get("HF_TOKEN")',
                            'model_name, revision=MODEL_REVISION, trust_remote_code=True, token=os.environ.get("HF_TOKEN")')
    source = source.replace('        model_name,\n        torch_dtype=',
                            '        model_name,\n        revision=MODEL_REVISION,\n        torch_dtype=')
    source = source.replace('    input_ids = model_inputs["input_ids"]\n',
                            '    input_ids = model_inputs["input_ids"]\n'
                            '    context = int(model.config.max_position_embeddings)\n'
                            '    if input_ids.shape[-1] + max_new_tokens > context:\n'
                            '        raise ValueError(f"Context exceeded: {input_ids.shape[-1]} + {max_new_tokens} > {context}. No silent truncation.")\n')
    plan_text = plan_path.read_text()
    plan = json.loads(plan_text)
    assert plan["model_name"] == "NousResearch/Meta-Llama-3-8B-Instruct"
    assert plan["implement_prompt"] == "compact_v1_max_3_steps"
    evidence = {"direct_cot_notebook_sha256": hashlib.sha256(direct_path.read_bytes()).hexdigest(),
                "sd_notebook_sha256": hashlib.sha256(sd_notebook.read_bytes()).hexdigest(),
                "cached_plan_original_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "protocol_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "runtime_source_sha256": hashlib.sha256((ROOT / "scripts/holdout9000_phase2_runtime.py").read_bytes()).hexdigest()}
    constants = '''LABELS = ["Depression", "Neutral", "Happy"]
BASE_SEED = 42
MAX_ROWS = None
MAX_REASONING_CHARACTERS = 6000
REASONING_HEAD_CHARACTERS = 3500
REASONING_TAIL_CHARACTERS = 2500
LOAD_IN_4BIT = True
LLAMA3_MAX_NEW_TOKENS = 1024
V6_MAX_NEW_TOKENS = 1536
MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
'''
    imports = '''import gc, hashlib, importlib.metadata, json, os, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from google.colab import drive, userdata
drive.mount("/content/drive")
try:
    token = userdata.get("HF_TOKEN")
except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
    token = None
if token:
    os.environ["HF_TOKEN"] = token
BASE = Path("/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/unified_final/additional_9000_20260930")
PHASE1_DIR = BASE / "phase1"
OUTPUT_DIR = BASE / "phase2/llama3_direct_cot_v6c_v1"
'''
    cells = [
        cell("markdown", """# 2단계: 기존 9,000건 / Llama 3 Direct · CoT · 최종 SELF-DISCOVER v6c

**새 GPU 런타임에서 위부터 모두 실행. Drive 연결은 jay.seizethemoment@gmail.com.**
완료된 1단계 출력과 handoff를 자동으로 읽는다. 파일 업로드와 경로 수정은 필요 없다.

- 최종 DistilBERT가 라우팅한 동일 사례만 세 방식으로 재평가한다. 전체 성능의 분모는 9,000건이다.
- Direct/CoT는 기존 최종 프롬프트, v6c는 동료에게 받은 최종 공통 계획과 프롬프트를 그대로 사용한다. 공통 계획을 새로 생성하지 않는다.
- 원문 입력, 4-bit NF4/fp16, greedy decoding. Direct/CoT 각 호출 상한 1,024, v6c 1,536토큰으로 기존 설정을 유지한다.
- 파싱 실패는 Phase 1 라벨을 유지한다. 자동 재생성이나 정답을 보고 라벨을 고치는 단계는 없다.
- 생성 호출별·사례별 결과를 저장한다. 중단되면 같은 GPU 종류와 환경에서 모두 실행하면 완료된 호출과 사례를 건너뛴다.
- 이 노트북은 분류 모델 학습·추론·보정·임계값 탐색을 다시 하지 않는다.
"""),
        cell("markdown", "## 1. 설치\n새 런타임에서도 반드시 실행한다. HF_TOKEN은 선택 사항이며, 없다고 Drive 오류로 처리하지 않는다."),
        cell("code", '%pip install -q "transformers==5.17.0" "bitsandbytes>=0.46.1" accelerate sentencepiece protobuf pandas numpy scipy scikit-learn tqdm\n'),
        cell("markdown", "## 2. Drive 및 고정 프로토콜"),
        cell("code", imports + constants),
        cell("code", source),
        cell("markdown", "## 3. 받은 최종 공통 계획\n아래 문자열은 제공된 JSON 원본이다. 변경하거나 재생성하지 않는다."),
        cell("code", "CACHED_PLAN_JSON = " + repr(plan_text) + "\nCACHED_PLAN = json.loads(CACHED_PLAN_JSON)\nPROTOCOL_EVIDENCE = " + repr(evidence) + "\n"),
        cell("code", (ROOT / "scripts/holdout9000_phase2_runtime.py").read_text()),
        cell("markdown", "## 4. 1단계 결과 검증\n9,000건과 handoff 해시, 모델 출처, 라우팅 규칙을 확인한다. 오류가 있으면 LLM을 불러오기 전에 멈춘다."),
        cell("code", "phase1_df, routed_df, handoff, phase1_summary = read_phase1_handoff(PHASE1_DIR)\ndisplay(pd.DataFrame([phase1_summary]))\nprint('세 방식 각각 평가할 사례:', len(routed_df))\n"),
        cell("markdown", "## 5. 세 방식 실행 또는 이어서 실행\n모델은 한 번만 불러온다. 이 셀은 수 시간이 걸릴 수 있다. 완료된 방법은 다시 실행하지 않는다."),
        cell("code", "run_all_methods(phase1_df, routed_df, handoff, OUTPUT_DIR)\n"),
        cell("markdown", "## 6. 최종 비교표\n세 방식이 모두 완료됐는지 확인한 뒤 전체 9,000건 기준 결과표를 만든다. 원시 생성 기록도 같은 출력 폴더에 보존된다."),
        cell("code", "summary = export_all_results(phase1_df, routed_df, OUTPUT_DIR)\ndisplay(summary)\nprint('Saved:', OUTPUT_DIR)\n"),
    ]
    for index, item in enumerate(cells):
        item["id"] = f"additional9000-phase2-{index:02d}"
        if item["cell_type"] == "code":
            ast.parse("\n".join(line for line in "".join(item["source"]).splitlines() if not line.startswith("%")))
    audit_heading = cell("markdown", "## 7. 저장 결과 검증 (CPU 전용)\n완료 후에는 아래 셀만 실행한다. 모델 재실행이나 GPU가 필요 없다. 저장 파일 검증과 대응 통계를 계산하고 관련 파일 ZIP을 다운로드한다.")
    audit_heading["id"] = "audit-saved-results-heading"
    audit_cell = cell("code", (ROOT / "scripts/audit_9000_saved_colab.py").read_text())
    audit_cell["id"] = "audit-saved-results-20260930"
    cells.extend([audit_heading, audit_cell])
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"name": OUT.name, "provenance": []}}}
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd-notebook", type=Path, required=True)
    parser.add_argument("--cached-plan", type=Path, required=True)
    args = parser.parse_args()
    build(args.sd_notebook, args.cached_plan)
