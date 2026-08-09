# Phase 2 Universal Prompt Policy v2 변경 이력 및 비교 실험 안내

## 1. 목적

기존 Phase 2 prompt는 감정이 섞이거나 시간에 따라 변화하는 Mixed Emotion 사례를 잘 처리하도록 `final emotional trajectory`를 강하게 강조했다. Mixed Emotion v2.4에서는 Llama 3 SELF-DISCOVER가 Phase 1 accuracy 81.33%에서 end-to-end 87.33%로 개선되는 결과를 보였다.

반면 Reddit held-out test는 일반적인 커뮤니티 게시글이 중심이며, 모든 문장에 명확한 시간적 감정 전환이 존재하지 않는다. 따라서 감정 전환이 실제로 드러난 경우에만 trajectory rule을 적용하고, 그 외에는 전체 텍스트의 dominant emotional meaning을 우선하는 공통 prompt policy를 별도 버전으로 검증한다.

이 문서는 기존 노트북을 덮어쓰지 않는 prompt comparison experiment의 변경 이력이다.

## 2. 비교 대상

| 구분 | 기존 baseline | 개선 비교 버전 |
|---|---|---|
| Mixed Emotion Phase 2 | `02_llm_phase2_reasoning_final_colab.ipynb` | `02_1_llm_phase2_reasoning_universal_prompt_final_colab.ipynb` |
| Reddit routed Phase 2 | `04_reddit_test_routed_phase2_end_to_end_final_colab.ipynb` | `04_1_reddit_test_routed_phase2_universal_prompt_final_colab.ipynb` |
| Phase 1 입력 | 동일한 Final 01 output | 동일한 Final 01 output |
| Llama models | NousResearch Llama 2 7B Chat, Meta-Llama-3-8B-Instruct | 동일 |
| routing 대상 | `phase1_routed=True` row만 사용 | 동일 |
| 변경 범위 | 기존 trajectory-aware policy | Phase 2 prompt policy와 output constraint만 변경 |
| 기존 결과 보존 | 기존 output directory 유지 | 별도 output directory 사용 |

## 3. 핵심 변경 사항

| 번호 | 기존 방식 | Universal Prompt Policy v2 | 변경 이유 |
|---:|---|---|---|
| 1 | 감정이 섞인 문장에서 final trajectory를 우선하는 규칙이 강함 | 먼저 전체 텍스트의 dominant emotional meaning을 판단하고, **명확한** mixed cue 또는 temporal shift가 있을 때만 trajectory를 추가 고려 | Reddit의 일반 문장에 trajectory를 과도하게 추론하지 않기 위함 |
| 2 | Llama 2가 Phase 1 label을 본 뒤 독립 판단을 시작 | Llama 2는 텍스트만으로 독립 판단을 먼저 기록하고, 이후에 Phase 1 label과 비교 | Phase 1 prediction에 대한 anchoring을 줄이고 correction 근거를 명확히 하기 위함 |
| 3 | exact output rule은 있었지만 일부 결과에서 `Sad` 같은 외부 표현이 발생 | Depression, Neutral, Happy만 허용하며 Sad, Positive, Mixed, Anxiety, Other를 명시적으로 금지 | parser failure와 label-space 불일치를 줄이기 위함 |
| 4 | Mixed cue가 있으면 Neutral이 아닌 다른 label을 택하도록 하는 경향 | mixed cue만으로 Neutral을 배제하지는 않되, Neutral은 factual/routine/mild 상태이고 dominant positive/distress가 없을 때만 선택 | Neutral을 단순 회피 label로 쓰지 않으면서, 실제로 중립적인 Reddit 문장도 처리하기 위함 |

## 4. 공통 판단 정책

새 policy는 Llama 2 CoT와 Llama 3 SELF-DISCOVER에 공통으로 들어간다.

1. 텍스트 전체의 dominant emotional meaning을 먼저 판단한다.
2. 단어 하나, 짧은 positive/negative cue, 감정 단어의 단순 평균으로 label을 정하지 않는다.
3. 여러 감정 단서 또는 명확한 시간적 감정 변화가 텍스트에 실제로 있을 때만 overall trajectory와 final takeaway를 추가 고려한다.
4. 전환이 뚜렷하지 않으면, 존재하지 않는 trajectory를 만들지 않고 전체 지배 정서로 판단한다.
5. Depression, Neutral, Happy 이외의 label은 출력하지 않는다.

## 5. Llama 2 CoT 변경 전후

| 출력 단계 | 기존 | 변경 후 |
|---|---|---|
| `LLaMA2_1` | Phase 1 label을 포함한 입력을 본 뒤 dominant emotion 분석 | 텍스트만 본 independent analysis와 provisional label |
| `LLaMA2_2` | Phase 1 label과 비교 | 독립 판단이 끝난 뒤 Phase 1 label과 비교 및 evidence-based correction |
| `LLaMA2_3` | final label을 요구 | percentage breakdown 없이 exact 3-class final label만 요구 |

변경 후 마지막 응답 규칙은 다음과 같다.

```text
Use only one exact label: Depression, Neutral, or Happy.
Do not use synonyms or additional labels such as Sad, Positive, Mixed,
Anxiety, or Other. Do not provide a percentage breakdown.
End the response with exactly one line: Final label: [label]
```

## 6. Llama 3 SELF-DISCOVER 변경 전후

| 항목 | 기존 | 변경 후 |
|---|---|---|
| task framing | overall tone과 final emotional trajectory를 함께 강조 | dominant emotional meaning을 먼저 판단하고 trajectory는 명확한 shift가 있을 때만 적용 |
| independent analysis | Phase 1 label이 입력 context에 함께 존재 | prompt에서 Phase 1 label을 고려하기 전에 독립 판단을 먼저 수행하도록 명시 |
| fixed structure | final trajectory 우선 | cue 확인 → shift 존재 확인 → 필요 시 trajectory 적용 → Phase 1 비교 → exact label 선택 |
| label constraint | `Final label:` 형식 요구 | 동일 형식 유지 + 외부 label/synonym 명시적 금지 |

SELF-DISCOVER의 `SELECT → ADAPT → IMPLEMENT → Answer` column 구조와 모델, generation 설정, routed input row는 바꾸지 않는다. 따라서 v2 결과는 prompt policy의 영향으로 비교할 수 있다.

## 7. 결과 저장 위치

기존 실험 파일은 건드리지 않는다. v2는 아래의 별도 directory로 저장된다.

| 노트북 | Google Drive output directory |
|---|---|
| `02_1` Mixed Emotion | `/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/phase2_llm_reasoning_universal_prompt_v2/` |
| `04_1` Reddit routed test | `/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/reddit_test_phase2_reasoning_universal_prompt_v2/` |

각 결과 row에는 기존처럼 Llama 2의 `LLaMA2_1`, `LLaMA2_2`, `LLaMA2_3`, `LLaMA2_final_label`과 Llama 3의 `LLaMA3_SELECT`, `LLaMA3_ADAPT`, `LLaMA3_IMPLEMENT`, `LLaMA3_Answer`, `LLaMA3_final_label`이 저장된다.

## 8. 실행 및 비교 순서

1. Final 01의 Phase 1 prediction CSV를 그대로 사용한다.
2. `02_1`을 실행해 Mixed Emotion routed row에 대해 v2를 실행한다.
3. `04_1`을 실행해 Reddit held-out routed row에 대해 v2를 실행한다.
4. 기존 baseline과 v2에서 다음 수치를 비교한다.
   - routed-only accuracy
   - full end-to-end accuracy와 macro F1
   - corrected routed errors
   - introduced routed errors
   - net error reduction
   - final-label parse failure 수
5. 어느 prompt가 더 좋더라도, 최종 논문에는 prompt version, model version, dataset version, routing threshold를 함께 기록한다.

## 9. 해석 원칙

- 이 실험은 **prompt policy ablation**이다. Phase 1 model, threshold, routed set, LLM model을 바꾸지 않고 prompt policy만 비교한다.
- Mixed Emotion 결과만 보고 prompt를 선택하면 stress-test tuning으로 보일 수 있다. 따라서 Reddit routed test와 Mixed Emotion에서 같은 v2 policy를 함께 평가한다.
- 최종 논문에는 선택된 policy의 성능뿐 아니라, baseline 대비 어떤 오류가 수정되었고 어떤 오류가 새로 생겼는지를 함께 보고한다.
- 모든 해석은 proxy emotion classification과 supplementary stress-test 범위에 한정하며 clinical diagnosis claim으로 확장하지 않는다.
