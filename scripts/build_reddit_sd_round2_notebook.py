"""Build two isolated extensions of the round-one independent SD prompt."""

import hashlib
import json

from build_final_unified_phase2_notebooks import OUT_DIR, code, markdown
from build_reddit_sd_hypotheses_notebook import VARIANTS, RUN


CHECK = """Before committing to a final label:
1. Form a provisional classification from the full text.
2. Identify the strongest text-supported alternative label.
3. Find the strongest evidence against your provisional interpretation.
   Check whether it changes the dominant emotional meaning or only qualifies it.
4. Keep or revise your provisional label based on the stronger evidence.
Do not invent an alternative interpretation merely to create disagreement.
Do not default to Neutral because two labels are plausible.
Do not infer feelings that the author has not expressed or supported through context.
End with exactly one line: Final label: Depression, Final label: Neutral, or Final label: Happy.
"""

NEUTRAL = """This stage designs a reasoning procedure; it does not classify the post.
Do not choose, recommend, or imply a provisional emotion label.
Do not assume which emotional interpretation will turn out to be correct.
Describe what evidence should be examined and how competing interpretations
should be distinguished.
Keep the procedure applicable regardless of whether the final answer is
Depression, Neutral, or Happy.
Leave all label decisions to the final execution stage.
"""


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Source changed; expected one occurrence: {old!r}")
    return text.replace(old, new, 1)


def variant_source():
    text = VARIANTS.replace("llama3_sd_hypotheses_v1", "llama3_sd_independent_round2_v1")
    text = replace_once(text, 'RUN_METHODS = ["sd_independent", "sd_evidence", "sd_compact"]',
                        'RUN_METHODS = ["sd_independent_countercheck", "sd_independent_neutral_plan"]')
    text = text.replace('# Optionally append sd_baseline to rerun the unchanged prompt with the same parser.',
                        '# Two independent-prompt variants only; prior outputs are never reused.')
    text = replace_once(text, 'if variant == "sd_independent":', 'if variant in RUN_METHODS:')
    text = replace_once(text,
        'for name in ["sd_independent", "sd_evidence", "sd_compact", "sd_baseline"]',
        'for name in RUN_METHODS')
    text = replace_once(text, '    answer = generate("final", prompt, final=True)',
        '    if variant == "sd_independent_countercheck":\n'
        '        prompt += "\\n\\n" + COUNTERCHECK\n'
        '    answer = generate("final", prompt, final=True)')
    text = replace_once(text, '        messages = [{"role": "user", "content": prompt}]',
        '        if variant == "sd_independent_neutral_plan" and not final:\n'
        '            prompt += "\\n\\n" + NEUTRAL_PLAN\n'
        '            if stage == "implement":\n'
        '                prompt += "\\nReturn a reasoning plan with questions or checks to perform, " \\\n'
        '                          "not a completed analysis or a plan containing a predetermined answer."\n'
        '        messages = [{"role": "user", "content": prompt}]')
    return 'COUNTERCHECK = ' + repr(CHECK) + '\nNEUTRAL_PLAN = ' + repr(NEUTRAL) + '\n' + text


def build():
    source = OUT_DIR / "08_reddit_llama3_self_discover_hypotheses_colab.ipynb"
    nb = json.loads(source.read_text())
    name = "09_reddit_llama3_sd_independent_round2_colab.ipynb"
    nb["metadata"]["colab"]["name"] = name
    nb["cells"][0] = markdown("""# Reddit independent SELF-DISCOVER: round 2

두 가지 신규 버전만 실행합니다. 기존 결과와 프롬프트는 변경하지 않습니다.

1. sd_independent_countercheck: 마지막 판단 단계에 반대 근거 점검 추가.
2. sd_independent_neutral_plan: 구조 생성 3단계에서 라벨을 미리 정하지 않도록 지침 추가.

두 버전 모두 Phase 1 예측을 숨깁니다. 기존 원문 218건, 모델 revision,
4단계 구조, greedy decoding, 단계별 최대 1,024토큰, 파싱/fallback을 유지합니다.
기존 독립 판단형 결과는 비교 기준이며 이 노트북에서 다시 실행하지 않습니다.
이는 이미 확인한 Reddit 데이터에서의 탐색 실험입니다.

기존 계정으로 Drive를 연결하면 추가 파일 업로드 없이 실행됩니다.
설치 셀 실행 후 런타임 재시작, imports부터 순서대로 실행하세요.
기본 실행량은 218 x 2 x 4 = 1,744회 생성입니다.
두 버전의 전체 12,000건 기준 요약표와 사례별 비교 CSV를 자동 저장합니다.
출력 폴더: reddit_sd_exploration/phase2/llama3_sd_independent_round2_v1/
중단 후 동일 설정으로 재개할 수 있습니다. smoke test는 별도 출력 폴더를 쓰세요.
""")
    run_index = None
    for i, cell in enumerate(nb["cells"]):
        text = "".join(cell["source"])
        if cell["cell_type"] == "code" and "INDEPENDENT_TASK =" in text:
            nb["cells"][i] = code(variant_source())
        elif cell["cell_type"] == "code" and "IMPLEMENTATION_SHA256 =" in text:
            run_index = i
    if run_index is None:
        raise ValueError("Run cell not found.")
    run = RUN.replace("exploratory-reddit-sd-v1", "exploratory-reddit-sd-independent-round2-v1")
    digest = hashlib.sha256((json.dumps(nb["cells"][:run_index], sort_keys=True) + run).encode()).hexdigest()
    nb["cells"][run_index] = code(f'IMPLEMENTATION_SHA256 = "{digest}"\n' + run)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"sd-round2-{i:02d}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    path = OUT_DIR / name
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    build()
