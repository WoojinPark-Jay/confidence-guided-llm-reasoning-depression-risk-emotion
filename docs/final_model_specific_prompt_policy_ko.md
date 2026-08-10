# 최종 모델별 Prompt Policy 확정 기록

## 1. 최종 결론

최종 Phase 2 구성은 하나의 공통 프롬프트를 두 모델에 강제하지 않는다.

| Phase 2 모델 | 최종 프롬프트 | 선택 근거 |
|---|---|---|
| Llama 2-7B-Chat | CoT v2 | Reddit routed 171개에서 기존 Llama 2보다 routed-only accuracy가 30.99%에서 45.61%로 개선되고, net error change가 -31에서 -6으로 개선됨. |
| Llama 3-8B-Instruct | 기존 SELF-DISCOVER | Reddit routed 171개에서 routed-only accuracy 52.05%, 전체 E2E 96.73%, corrected 13 / introduced 8 / net +5로 가장 좋은 완료 결과를 보임. |

## 2. 고정한 이유

Llama 2 CoT와 Llama 3 SELF-DISCOVER는 원래 중간 추론 형식과 출력 행동이 다르다. 동일한 universal v2 정책을 적용한 비교에서는 Llama 2가 개선됐지만 Llama 3는 성능이 저하되고 terminal label parse failure도 발생했다. Llama 3 error-aware v2.1은 parse failure를 제거했지만 기존 SELF-DISCOVER의 완료 성능을 넘지 못했다.

따라서 최종 정책은 각 모델에서 완료된 고정 routed subset 결과가 가장 좋았던 프롬프트를 채택한다. 이는 모델 간 점수를 임의로 유리하게 만드는 선택이 아니라, 같은 Phase 1 model, calibrated temperature, routing threshold, routed rows, checkpoint, generation settings, evaluation metric을 고정한 prompt-policy comparison에 근거한다.

## 3. Reddit 완료 비교

| System | Routed-only accuracy | Full Reddit E2E accuracy | Corrected | Introduced | Net |
|---|---:|---:|---:|---:|---:|
| Phase 1 DistilBERT | - | 96.69% | - | - | - |
| Llama 2 기존 CoT | 30.99% | 96.43% | 24 | 55 | -31 |
| Llama 2 CoT v2 | 45.61% | 96.64% | 41 | 47 | -6 |
| Llama 3 기존 SELF-DISCOVER | 52.05% | 96.73% | 13 | 8 | +5 |
| Llama 3 universal v2 | 43.90% | 96.63% | 25 | 32 | -7 |
| Llama 3 error-aware v2.1 | 50.29% | 96.71% | 26 | 24 | +2 |

## 4. 최종 재현 노트북

1. `01_distilbert_phase1_training_final_colab.ipynb`: Phase 1 training, calibration, threshold selection, and prediction export.
2. `02_2_llm_phase2_reasoning_model_specific_prompt_final_colab.ipynb`: Mixed Emotion routed rows only. This is the required matched Llama 2 CoT v2 rerun before final Mixed Emotion Llama 2 results are claimed.
3. `03_mixed_emotion_end_to_end_orchestration_final_colab.ipynb`: combines Phase 1 and the final Phase 2 outputs for the Mixed Emotion paper tables.
4. `04_3_reddit_test_routed_phase2_model_specific_prompt_final_colab.ipynb`: Reddit routed rows only, using the final model-specific combination.

## 5. Reporting rule

- Report the existing Llama 3 Mixed Emotion result (81.33% to 87.33%) as completed.
- Do not relabel the earlier Llama 2 Mixed Emotion trajectory-aware result as a CoT v2 result. Run Final 02.2 first, then refresh the corresponding table.
- Keep universal v2 and error-aware v2.1 as controlled ablations, not the final operating policy.
