"""Reuse round-two countercheck verbatim with the verified Mixed Emotion input."""

import hashlib
import json

from build_final_unified_phase2_notebooks import OUT_DIR, code, markdown


def find_cell(nb, marker):
    matches = [c for c in nb["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    if len(matches) != 1:
        raise ValueError(f"Expected one source cell for {marker!r}.")
    return matches[0]


def build():
    nb = json.loads((OUT_DIR / "09_reddit_llama3_sd_independent_round2_colab.ipynb").read_text())
    mixed = json.loads((OUT_DIR / "06_2_mixed_emotion_llama3_reasoning_methods_colab.ipynb").read_text())
    name = "10_mixed_emotion_llama3_sd_countercheck_colab.ipynb"
    nb["metadata"]["colab"]["name"] = name
    config = find_cell(mixed, "# Both Phase 2 notebooks")
    inputs = find_cell(mixed, 'drive.mount("/content/drive")')
    run_index = None
    for i, cell in enumerate(nb["cells"]):
        text = "".join(cell["source"])
        if cell["cell_type"] != "code":
            continue
        if "# Both Phase 2 notebooks" in text:
            nb["cells"][i] = code("".join(config["source"]))
        elif 'drive.mount("/content/drive")' in text:
            nb["cells"][i] = code("".join(inputs["source"]))
        elif "INDEPENDENT_TASK =" in text:
            text = text.replace(
                'RUN_METHODS = ["sd_independent_countercheck", "sd_independent_neutral_plan"]',
                'RUN_METHODS = ["sd_independent_countercheck"]',
            ).replace("llama3_sd_independent_round2_v1", "llama3_sd_independent_countercheck_v1")
            nb["cells"][i] = code(text)
        elif "IMPLEMENTATION_SHA256 =" in text:
            run_index = i
    if run_index is None:
        raise ValueError("Missing execution cell.")
    nb["cells"][0] = markdown("""# Mixed Emotion: independent SELF-DISCOVER + countercheck

Reddit round 2의 sd_independent_countercheck 프롬프트를 변경 없이 실행합니다.
다른 방법은 다시 실행하지 않습니다.

- 기존 최종 DistilBERT의 Mixed Emotion 300건(정답 251건) 중 동일한 86건을 재평가합니다.
- 검증된 입력이 포함되어 있어 별도 업로드가 필요 없습니다. 기존 Drive 파일이 있으면 일치 여부를 먼저 확인합니다.
- 동일한 모델 revision, greedy decoding, 4단계, 단계별 최대 1,024토큰을 유지합니다.
- 생성 프롬프트에는 Phase 1 라벨과 정답을 전달하지 않습니다.
- 출력: unified_final/mixed_emotion/phase2/llama3_sd_independent_countercheck_v1/
- 기존 Mixed Emotion 결과는 변경하지 않습니다. 같은 설정으로 중단 후 재개할 수 있습니다.

설치 셀 실행 → 런타임 재시작 → imports부터 순서대로 실행하세요.
86건 x 4단계 = 344회 생성합니다. 마지막 표는 전체 300건 기준 정확도/F1,
수정/신규 오류/순수정과 파싱 실패를 보여줍니다.
파싱 실패는 Phase 1 예측을 유지하며, 원시 응답도 함께 저장합니다.
""")
    run = "".join(nb["cells"][run_index]["source"]).split("\n", 1)[1]
    run = run.replace("exploratory-reddit-sd-independent-round2-v1", "mixed-sd-independent-countercheck-v1")
    digest = hashlib.sha256((json.dumps(nb["cells"][:run_index], sort_keys=True) + run).encode()).hexdigest()
    nb["cells"][run_index] = code(f'IMPLEMENTATION_SHA256 = "{digest}"\n' + run)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"mixed-countercheck-{i:02d}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    path = OUT_DIR / name
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    build()
