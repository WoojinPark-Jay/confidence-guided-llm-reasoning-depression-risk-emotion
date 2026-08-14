# Reddit Phase 2 원문 입력 최종 재실험

작성일: 2026-08-14  
최종 노트북: `notebooks/colab/final/04_5_reddit_test_routed_phase2_original_text_primary_final_colab.ipynb`

## 1. 실험 목적

기존 Reddit Phase 2 실험은 DistilBERT에 사용한 aggressively cleaned text를 Llama reasoning에도 그대로 전달했다. 이 과정에서 부정어, 문장부호, 문장 순서, 시간적 감정 변화가 손실될 수 있었다. 실제 감사 과정에서 원문 제목의 `not`이 제거되어 의미가 반대로 전달되는 사례가 확인되었다.

이번 재실험은 모델이나 routing 정책을 바꾸는 prompt ablation이 아니다. Phase 1은 기존과 동일하게 cleaned text를 사용하고, Phase 2만 원문 `title + selftext`를 최소한으로 비식별 처리해 사용하도록 입력 파이프라인을 바로잡는다.

## 2. 고정되는 조건

| 항목 | 최종 고정값 |
|---|---|
| Phase 1 모델 | 기존 final DistilBERT |
| temperature | Final 01에서 저장된 값 그대로 사용 |
| routing threshold | primary policy `tau=0.70` |
| target risk budget | `alpha=0.05` |
| held-out test | 클래스당 4,000건, 총 12,000건 |
| routed sample | 171건 |
| Llama 2 | 최종 CoT v2 prompt |
| Llama 3 | established SELF-DISCOVER prompt |
| 비교 단위 | 동일한 171개 routed ID |

기존 Final 01 산출물에는 fractional split rounding으로 Happy가 4,001건 포함되어 총 12,001행이 저장되어 있다. 최종 노트북은 예측 결과를 기준으로 행을 선택하지 않는다. 기존 파일 순서에서 각 target class의 첫 4,000건만 유지하여 문서화된 balanced 12,000건으로 자동 정규화한다. 제외되는 1행은 별도 CSV로 기록되며 routed 171건은 변하지 않는다.

## 3. 원문 연결과 비식별 처리

노트북은 Final 01의 `phase1_test_predictions.csv`와 원본 Reddit source의 `title_with_selftext_cleaned` 및 proxy label을 normalized exact key로 연결한다.

최종 추론을 시작하기 전에 다음 조건을 모두 검사한다.

1. held-out test가 정확히 12,000건인지 확인한다.
2. routing threshold가 0.70인지 확인한다.
3. routed row가 정확히 171건인지 확인한다.
4. routed 171건 모두 원문과 연결되는지 확인한다.
5. 동일 cleaned key가 서로 다른 원문으로 연결되는 모호한 사례가 없는지 확인한다.

원문에는 다음 최소 처리만 적용한다.

- URL을 `[URL]`로 치환
- 직접적인 Reddit username 패턴을 `[USER]`로 치환
- 불필요한 연속 공백 정리
- 삭제 또는 제거된 body 처리

부정어, 구두점, 문장 순서, 감정 전환, 결론 부분은 유지한다. 지나치게 긴 글만 시작 3,500자와 끝 2,500자를 보존하며, 해당 여부를 결과 컬럼에 기록한다.

## 4. 실행 방법

1. Colab에서 Final 04.5를 연다.
2. GPU runtime을 선택한다.
3. 첫 setup 셀을 실행한다.
4. 안내대로 runtime을 한 번 재시작한다.
5. import 셀부터 마지막 셀까지 순서대로 실행한다.

Llama 2와 Llama 3는 같은 171건을 순서대로 처리한다. 각 row가 끝날 때마다 Google Drive CSV에 즉시 append하므로 runtime이 종료되어도 완료된 ID는 남는다. 다시 실행하면 기존 CSV의 `example_id`를 읽고 미완료 행부터 재개한다.

## 5. 저장 위치와 핵심 산출물

Google Drive 저장 경로:

```text
/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/
reddit_test_phase2_reasoning_original_text_primary_tau070_final/
```

주요 파일:

| 파일 | 내용 |
|---|---|
| `reddit_test_split_rounding_excluded_row.csv` | 12,001행을 balanced 12,000행으로 정규화하면서 제외된 1행 감사 기록 |
| `reddit_test_primary_routed_original_text_mapping_audit.csv` | 171건 원문 연결 상태와 중복 후보 수 |
| `reddit_test_primary_tau070_original_text_input_rows.csv` | 최종 Phase 2 입력 171건 |
| `reddit_test_llama2_cot_original_text_tau070_results.csv` | Llama 2 row-level reasoning 및 final label |
| `reddit_test_llama3_self_discover_original_text_tau070_results.csv` | Llama 3 SELF-DISCOVER 단계와 final label |
| `reddit_test_phase2_original_text_primary_tau070_combined_outputs.csv` | 두 모델 출력 결합본 |
| `reddit_test_end_to_end_metrics_summary.csv` | Phase 1 및 두 end-to-end 시스템 성능 |
| `reddit_test_phase2_correction_analysis.csv` | corrected, introduced, net correction 결과 |
| `reddit_test_paper_ready_tables.xlsx` | 논문용 표 모음 |
| `reddit_test_phase2_original_text_primary_tau070_outputs.zip` | 전체 산출물 다운로드용 압축 파일 |

## 6. 결과 확정 후 논문 반영

결과가 완료되면 cleaned-text Reddit Phase 2 수치는 최종 결과로 사용하지 않는다. 원문 입력 결과를 기준으로 다음 항목을 갱신한다.

- Abstract와 contribution의 Reddit end-to-end 결과
- 본문의 Reddit routing 및 correction 표
- Llama 2와 Llama 3 corrected/introduced/net correction 수치
- confusion matrix와 error analysis
- Appendix의 실제 routed-case reasoning 예시
- Methods의 Phase 2 input policy와 비식별 처리 설명
- 재현성 문서와 결과 provenance

Mixed Emotion 결과는 원래부터 자연어 원문을 사용했으므로 그대로 유지한다. 이번 재실험은 Reddit과 Mixed Emotion의 Phase 2 입력 조건을 일관되게 맞추는 최종 파이프라인 수정이다.
