"""Build a self-contained Colab notebook without embedding private data/weights."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "independent_evaluation_20260911/private/run_package/same_source_evaluation/data"
OUT = ROOT / "notebooks/colab/final/07_1_final_distilbert_existing9000_colab.ipynb"


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(asset_config):
    asset = json.loads(Path(asset_config).read_text())
    if not asset.get("parts") or not all(part.get("file_id") for part in asset["parts"]):
        raise ValueError("Upload the private asset parts first and record their verified Drive file IDs.")
    names = ["evaluation_inputs.csv", "evaluation_labels.csv", "phase2_inputs.csv", "selection_manifest.csv"]
    hashes = {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in names}
    cells = [
        cell("markdown", """# 1단계: 최종 DistilBERT로 기존 9,000건 재평가

GPU 런타임을 선택하고 위에서부터 실행합니다. Drive 연결 시 기존 실험 파일이 있는 `jay.seizethemoment@gmail.com` 계정을 선택합니다.

- 최종 seed42 저장 모델로 **추론만** 합니다. 학습, 온도 재보정, 임계값 탐색은 하지 않습니다.
- T = 1.4801950079829693, 임계값 = 0.70, 최대 길이 = 256, 평가 배치 = 64, fp32.
- 받은 최종 모델과 기존 9,000건 데이터를 비공개 Drive 파일에서 자동으로 가져옵니다. 수동 업로드나 경로 수정은 필요하지 않습니다.
- 모델·토크나이저·데이터의 SHA256을 확인합니다. 예전 모델은 사용하지 않습니다.
- 새로운 출력 폴더를 사용합니다. 기존 12,000건, Mixed Emotion, 예전 9,000건 결과는 변경하지 않습니다.
- 다음 단계는 이번에 라우팅된 사례에 대한 Llama 3 Direct / CoT / 최종 v6c입니다. 이 노트북은 LLM을 실행하지 않습니다.

이미 평가 이력이 있는 동일한 9,000건의 추가 평가입니다. 새로 확보한 미사용 평가셋으로 표기하지 않습니다.
"""),
        cell("markdown", "## 1. 패키지 설치\n새 런타임에서도 이 셀부터 실행합니다. 이 분류 단계에는 HF_TOKEN과 bitsandbytes가 필요하지 않습니다."),
        cell("code", '%pip install -q "transformers==5.17.0" pandas numpy scipy scikit-learn tqdm safetensors\n'),
        cell("markdown", "## 2. Drive 연결 및 파일 자동 준비\nDrive 연결과 추가 인증에서 `jay.seizethemoment@gmail.com`을 선택합니다. 처음 한 번만 약 250MB를 내려받고, 다음부터 저장된 파일을 검증해 재사용합니다. 기존 실험 폴더는 변경하지 않습니다."),
        cell("code", '''from google.colab import drive
from pathlib import Path
drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive")
OUTPUT_DIR = DRIVE_ROOT / "confidence_guided_llm_reasoning/outputs_final/unified_final/additional_9000_20260930/phase1"
'''),
        cell("code", (ROOT / "scripts/holdout9000_assets.py").read_text() +
             "\nASSET_CONFIG = " + json.dumps(asset, indent=4) + '''
DATA_SOURCE, MODEL_DIR = prepare_assets(
    DRIVE_ROOT, ASSET_CONFIG["parts"], ASSET_CONFIG["archive_sha256"], ASSET_CONFIG["expected_files"]
)
'''),
        cell("markdown", "## 3. 추론 함수\n입력 해시·ID·클래스 수를 검증하고, 모델 해시와 실행 설정을 저장합니다. 원문은 행 순서가 아닌 example_id로 결합합니다."),
        cell("code", (ROOT / "scripts/holdout9000_final_phase1.py").read_text()),
        cell("markdown", "## 4. 입력 확인\n9,000건, 클래스별 3,000건인지 확인합니다. 이 셀에서는 모델 추론을 실행하지 않습니다."),
        cell("code", "EXPECTED_DATA_HASHES = " + json.dumps(hashes, indent=4) + '''
DATA_PATH, FINAL_MODEL_PATH = resolve_paths(DRIVE_ROOT, DATA_SOURCE, MODEL_DIR)
checked_inputs, checked_labels, checked_raw = read_data(DATA_PATH, EXPECTED_DATA_HASHES)
verified_model_hashes = model_files(FINAL_MODEL_PATH)
checkpoint_config = json.loads((FINAL_MODEL_PATH / "config.json").read_text())
check_label_order([checkpoint_config["id2label"][str(i)] for i in range(3)])
print("Data:", DATA_PATH)
print("Model:", FINAL_MODEL_PATH)
print("Output:", OUTPUT_DIR)
print("Rows:", len(checked_inputs))
display(checked_labels.label_str.value_counts().rename("rows").to_frame())
del checked_inputs, checked_labels, checked_raw
'''),
        cell("markdown", "## 5. 최종 모델 분류 실행\n중단 후에는 같은 GPU 종류·환경·경로에서 위 셀부터 실행하면 저장된 배치를 건너뜁니다. 설정이 달라지면 결과를 섞지 않고 멈춥니다."),
        cell("code", "summary = run_phase1(DATA_PATH, FINAL_MODEL_PATH, OUTPUT_DIR, EXPECTED_DATA_HASHES)\ndisplay(summary)\n"),
        cell("markdown", """## 6. 다음 단계에 넘길 파일

- `final_phase1_reasoning_input.csv`: 전체 9,000건 예측·정답 및 라우팅된 사례의 원문. 다음 Llama 3 코드의 입력.
- `routed_phase2_inputs.csv`: 새로 라우팅된 사례만 별도로 저장.
- `phase1_summary.csv`: 정확도, Macro F1, 라우팅 건수.
- `experiment_manifest.json`: 모델 해시 및 설정 기록.
- `phase2_handoff.json`: 다음 단계 입력 경로·해시·건수. 정답은 평가에만 사용하며 프롬프트 입력으로 사용하지 않습니다.

분류 결과표를 공유하면 새 라우팅 수를 기준으로 다음 노트북을 연결합니다. 이전 라우팅 137건이나 본 실험 218건을 강제하지 않습니다.
"""),
        cell("code", '''print("Next-stage input:", OUTPUT_DIR / "final_phase1_reasoning_input.csv")
print("Handoff:", OUTPUT_DIR / "phase2_handoff.json")
print("No Llama model was run in this notebook.")
'''),
    ]
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": OUT.name, "provenance": []},
    }, "cells": cells}
    for i, item in enumerate(cells):
        item["id"] = f"holdout-phase1-{i:02d}"
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-config", required=True, type=Path)
    build(parser.parse_args().asset_config)
