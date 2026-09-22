"""Run the Mixed Emotion Llama 2 methods unchanged on the final Reddit input."""
import hashlib
import json

from build_final_unified_phase2_notebooks import OUT_DIR, code, markdown
from build_mixed_sd_countercheck_notebook import find_cell


def build():
    nb = json.loads((OUT_DIR / "11_mixed_emotion_llama2_direct_final_sd_colab.ipynb").read_text())
    reddit = json.loads((OUT_DIR / "05_2_final_unified_phase2_llama3_reasoning_methods_colab_20260914.ipynb").read_text())
    name = "12_reddit_llama2_direct_final_sd_colab.ipynb"
    nb["metadata"]["colab"]["name"] = name
    run_index = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        if "# Both Phase 2 notebooks" in text:
            nb["cells"][i] = code("".join(find_cell(reddit, "# Both Phase 2 notebooks")["source"]))
        elif 'drive.mount("/content/drive")' in text:
            nb["cells"][i] = code("".join(find_cell(reddit, 'drive.mount("/content/drive")')["source"]))
        elif "IMPLEMENTATION_SHA256 =" in text:
            run_index = i
    if run_index is None:
        raise ValueError("Missing execution cell.")
    nb["cells"][0] = markdown("""# Reddit: Llama 2 Direct + final SELF-DISCOVER

기존 최종 Reddit 12,000건 중 같은 218건을 원문으로 재평가합니다.
Mixed Emotion에서 실행한 Llama 2 Direct와 최종 SD의 프롬프트 및 생성 설정을 그대로 사용합니다.
기존 CoT, Llama 3, Mixed Emotion 결과는 변경하지 않습니다.

- 입력: unified_final/phase1/final_phase1_reasoning_input.csv (기존 Drive 파일)
- Direct: Phase 1 예측을 제공하는 기존 Direct 프롬프트.
- 최종 SD: Phase 1 예측을 숨기는 독립 판단 + 반대 근거 점검, 4단계.
- Llama 2 고정 revision, 4-bit NF4, greedy, 단계별 최대 1,024토큰.
- 기존 원문 6,000자 head/tail 정책 유지. 모델 context를 초과하면 몰래 자르지 않고 중단합니다.
- 결과: unified_final/phase2/llama2_direct_final_sd_v1/

설치 셀 실행 → 런타임 재시작 → imports부터 순서대로 실행하세요.
기존 Drive 계정에 연결하면 추가 파일 업로드는 필요 없습니다.
Direct 218회 + SD 872회 = 총 1,090회 생성합니다.
중단 후 동일 설정으로 재개할 수 있으며, 완료 사례를 다시 생성하지 않습니다.
마지막 표는 전체 12,000건 기준 정확도/F1, 수정/신규 오류/순수정과 파싱 실패입니다.
파싱 실패 시 Phase 1 예측을 유지하며 단계별 원시 응답을 저장합니다.
기존 Llama 2 CoT는 최대 512토큰이므로 해당 설정 차이는 남아 있습니다.
""")
    run = "".join(nb["cells"][run_index]["source"]).split("\n", 1)[1]
    run = run.replace("mixed-llama2-direct-final-sd-v1", "reddit-llama2-direct-final-sd-v1")
    digest = hashlib.sha256((json.dumps(nb["cells"][:run_index], sort_keys=True) + run).encode()).hexdigest()
    nb["cells"][run_index] = code(f'IMPLEMENTATION_SHA256 = "{digest}"\n' + run)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"reddit-llama2-direct-sd-{i:02d}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    path = OUT_DIR / name
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    build()
