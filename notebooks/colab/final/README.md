# 최종 실험 노트북 안내

2026-09-26 기준. 아래는 최신 비교 결과에 사용한 실행 코드다. 기존 노트북은 실험 이력 보존을 위해 삭제하지 않았다.

## 최신 결과에 대응하는 파일

| 데이터 | 모델 / 방법 | 노트북 |
|---|---|---|
| Reddit | Llama 2 Direct + 최종 SD | [12_reddit_llama2_direct_final_sd_colab.ipynb](12_reddit_llama2_direct_final_sd_colab.ipynb) |
| Reddit | Llama 2 CoT | [13_reddit_llama2_cot_1024_colab.ipynb](13_reddit_llama2_cot_1024_colab.ipynb) |
| Reddit | Llama 3 Direct + CoT | [05_2_final_unified_phase2_llama3_reasoning_methods_colab_20260914.ipynb](05_2_final_unified_phase2_llama3_reasoning_methods_colab_20260914.ipynb) |
| Reddit | Llama 3 최종 SD | [09_reddit_llama3_sd_independent_round2_colab.ipynb](09_reddit_llama3_sd_independent_round2_colab.ipynb) |
| Mixed Emotion | Llama 2 Direct + 최종 SD | [11_mixed_emotion_llama2_direct_final_sd_colab.ipynb](11_mixed_emotion_llama2_direct_final_sd_colab.ipynb) |
| Mixed Emotion | Llama 2 CoT | [14_mixed_emotion_llama2_cot_1024_colab.ipynb](14_mixed_emotion_llama2_cot_1024_colab.ipynb) |
| Mixed Emotion | Llama 3 Direct + CoT | [06_2_mixed_emotion_llama3_reasoning_methods_colab.ipynb](06_2_mixed_emotion_llama3_reasoning_methods_colab.ipynb) |
| Mixed Emotion | Llama 3 최종 SD | [10_mixed_emotion_llama3_sd_countercheck_colab.ipynb](10_mixed_emotion_llama3_sd_countercheck_colab.ipynb) |

- 최종 SD는 `sd_independent_countercheck`(독립 판단 + 반대 근거 점검)다.
- **05_2와 06_2에 포함된 기존 `self_discover`는 최종 SD가 아니다.** 해당 파일은 최신 표의 Direct·CoT 결과에 대응한다.
- **09는 두 가설을 함께 실행하는 원래 실험 파일**이다. 최신 표에는 `sd_independent_countercheck` 결과만 사용하며, `sd_independent_neutral_plan`은 개발 이력이다.
- 05_1·06_1의 Llama 2 CoT 512토큰 결과는 13·14의 재실행 결과로 대체했다.
- 07(cleaned 입력), 08(SD 후보 탐색)은 최종 비교표에 사용하지 않는다.

## 실행 및 재현

1. Colab GPU 런타임에서 설치 셀을 실행하고, 런타임 재시작 후 imports부터 순서대로 실행한다. `.ipynb`에는 Colab 전용 Drive 연결 코드가 있어 일반 로컬 Jupyter에서는 경로·인증을 별도로 조정해야 한다.
2. Reddit 입력은 [final_phase1_reasoning_input.csv](../../../data/phase2_inputs/final_phase1_reasoning_input.csv)다. 12,000건 중 218건이 routed이며, 해당 사례의 `phase2_original_text`가 LLM 입력이다. 노트북 기본 경로는 Drive의 `confidence_guided_llm_reasoning/outputs_final/unified_final/phase1/final_phase1_reasoning_input.csv`다. 동료 계정에서는 이 파일을 해당 경로에 두거나 입력 경로를 맞춰야 한다.
3. Mixed Emotion 입력은 기존 검증된 300건/86건 routing 결과가 노트북에 포함돼 있어 추가 업로드가 필요 없다.
4. 기존 결과 폴더와 manifest를 삭제하지 않는다. 재개 기능이 있는 노트북은 완료된 ID를 제외하고 실행한다. 설정 변경 시에는 다른 출력 폴더를 사용한다.
5. 최신 코드의 단계별 생성 상한은 최대 1,024토큰이다. Reddit Llama 2 재개 수정과 CoT 재실행에는 남은 문맥 공간에 맞춰 상한을 줄이고 기록하는 정책이 포함돼 있다. 단계 수가 다르므로 총 생성 예산까지 같다는 뜻은 아니다.

## 결과표와 원시 실행 출력의 구분

[최신 결과 요약](../../../reports/phase2_results_20260918/FINAL_RESULTS_UPDATE_20260925_KO.md)은 원시 응답의 명시적 라벨을 사후 확인해 복구한 집계를 포함한다. 따라서 원래 노트북의 자동 요약과 일부 숫자가 다를 수 있다. 이 안내는 사후 파싱 검토가 모든 노트북의 자동 파서에 이미 통합됐다는 뜻이 아니다. 최종 예측·통계 패키지 정리는 후속 작업이다.

## 생성 및 검증 코드

[scripts 폴더](../../../scripts)에서 `build_*notebook*.py` 생성 스크립트와 `test_*.py` 검증 스크립트를 확인할 수 있다. 최신 추가 파일은 CoT 1,024토큰 재실행용 `build_llama2_cot_1024_notebooks.py`와 `test_llama2_cot_1024.py`다. 로컬 검증은 구문·프롬프트·입력 연결·모의 생성·재개 정책 테스트이며 GPU 추론을 대신하지 않는다.
