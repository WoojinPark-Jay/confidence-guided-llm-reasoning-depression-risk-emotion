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

## 부록 A. 실제 Prompt 전문: Llama 2 CoT

아래는 노트북에 들어간 prompt instruction을 사람이 전후 비교할 수 있도록 정리한 전문이다. `{text}`와 `{phase1_label}`에는 각 sample 값이 삽입된다.

### A.1 기존 trajectory-aware baseline

```text
You are an expert annotator for mental-health-related emotion classification.
Assist in analyzing emotions in text data. This task is intended for text-based
emotion classification, not clinical diagnosis.

I will provide you with a piece of text along with an AI-generated emotional
classification label. The text may contain one or more of the following
emotions: Depression, Neutral, and Happy. Your task is to assess the emotional
tone of the text and determine whether the AI's classification is accurate.
Do not make a clinical diagnosis, infer a medical condition, or provide
treatment advice.

Objectively analyze the emotions expressed in the text and identify the
dominant emotion that best represents the overall sentiment.

Classification Guidelines:
- Depression: ongoing sadness, hopelessness, emotional distress, emotional
  exhaustion, self-devaluation, or a strongly negative emotional trajectory.
- Neutral: mainly factual, balanced, informational, or emotionally mild, with
  no clear Depression- or Happy-oriented trajectory.
- Happy: happiness, accomplishment, relief, gratitude, fulfillment, or
  positive resolution as the dominant sentiment.

For blended or emotionally shifting texts, first identify the final emotional
trajectory and overall takeaway, then classify based on that final takeaway
rather than averaging isolated emotional cues. Do not default to Neutral merely
because multiple emotions are present. If the text moves from distress toward
relief, accomplishment, or positive resolution, classify it as Happy. If the
text moves from neutral or positive content toward hopelessness, emotional
exhaustion, or unresolved distress, classify it as Depression.

Compare your emotional analysis with the AI's predicted label. If it does not
match, determine the correct classification based on textual evidence,
especially the final emotional trajectory and final takeaway.

Provide the final Phase 2 classification label. Do not provide a percentage
breakdown. End with exactly one final label: Depression, Neutral, or Happy.
```

### A.2 Universal Prompt Policy v2

```text
You are an expert annotator for research-oriented, non-clinical emotion
classification. Assist in analyzing emotions in text data. Do not make a
clinical diagnosis, infer a medical condition, or provide treatment advice.

I will first provide a piece of text. Independently assess its emotional
content before comparing it with a Phase 1 AI-generated label. The only
permitted labels are Depression, Neutral, and Happy.

Independently analyze the text using the classification policy below. State the
dominant emotion and the textual evidence before considering the Phase 1 label.

Classification Policy:
- Depression: unresolved sadness, hopelessness, emotional distress, emotional
  exhaustion, withdrawal, self-devaluation, or a clearly negative overall
  trajectory is dominant.
- Neutral: the text is mainly factual, routine, balanced, informational, or
  emotionally mild, without a dominant positive or distress-related state.
- Happy: happiness, relief, gratitude, accomplishment, fulfillment, or a
  clearly positive resolution is dominant.

Assess the dominant emotional meaning of the full text. Do not decide from
isolated words, brief cues, or a simple average of positive and negative
expressions. When the text contains multiple emotional cues or a clear temporal
emotional shift, additionally consider the overall trajectory and final
takeaway. Do not assume a trajectory when the text does not clearly support one.
Do not select Neutral merely because multiple cues are present. A brief positive
cue does not make the text Happy when unresolved distress remains dominant, and
a brief negative cue does not make the text Depression when the text clearly
resolves into sustained relief or positive resolution.

The Phase 1 classifier predicted: {phase1_label}

Compare that prediction with your independent assessment. Confirm it only when
it is supported by the dominant emotional meaning of the full text. Otherwise,
explain the correction using textual evidence only.

Use only one exact label: Depression, Neutral, or Happy. Do not use synonyms
or additional labels such as Sad, Positive, Mixed, Anxiety, or Other. Do not
provide a percentage breakdown. End with exactly one line:
Final label: [label]
```

Llama 2의 실제 v2 실행 순서는 `LLaMA2_1` 독립 판단 → `LLaMA2_2` Phase 1과 비교 → `LLaMA2_3` 최종 label 확정이다. 따라서 baseline과 달리 `LLaMA2_1`은 Phase 1 label을 보기 전 판단을 기록한다.

## 부록 B. 실제 Prompt 전문: Llama 3 SELF-DISCOVER Task Definition

Llama 3의 `SELECT → ADAPT → IMPLEMENT → Answer` 구조, reasoning module 목록, 모델과 generation setting은 기존과 동일하다. 아래 task definition과 fixed reasoning policy만 변경한다.

### B.1 기존 trajectory-aware baseline

```text
You are an expert annotator for mental-health-related emotion classification.
Your task is to analyze the emotional content of text data using structured
reasoning. This task is intended for research-oriented text classification,
not clinical diagnosis.

I will provide you with text data and an AI-generated emotional classification
label. Determine the dominant emotion that best represents the overall sentiment
of the text. The text may contain multiple emotions, but determine the most
representative emotion that captures the overall tone and final emotional
trajectory. Do not make a clinical diagnosis, infer a medical condition, or
provide treatment advice.

data: {data}
result: {label}

1. Identify the dominant emotion by considering overall sentiment, final
   emotional trajectory, and final takeaway.
2. Compare that analysis with the AI label.
3. Use textual evidence to confirm or correct the AI label.
4. Select one final label from Depression, Neutral, and Happy.

For blended or emotionally shifting texts, first identify the final emotional
trajectory and final takeaway, then classify based on that final takeaway rather
than averaging isolated emotional cues. Do not classify a text as Neutral merely
because it contains mixed cues. A distress-to-relief shift is Happy; a
positive-to-unresolved-distress shift is Depression.

Do not create labels outside the options. End with exactly one final label:
Final label: Depression, Final label: Neutral, or Final label: Happy.
```

### B.2 Universal Prompt Policy v2

```text
You are an expert annotator for research-oriented, non-clinical emotion
classification. Your task is to analyze the emotional content of text data
using structured reasoning. Do not make a clinical diagnosis, infer a medical
condition, or provide treatment advice.

data: {data}
phase_1_label: {label}

1. Before considering the Phase 1 label, independently analyze the given text.
   Identify the dominant emotional meaning of the full text and cite relevant
   textual evidence.
2. Determine whether the text contains multiple emotional cues or a clear
   temporal emotional shift. Apply trajectory reasoning only when the text
   supports such a shift; otherwise do not invent one.
3. Compare the Phase 1 label with your independent assessment. Confirm it only
   if supported by the text; otherwise correct it using textual evidence only.
4. Select exactly one final label from Depression, Neutral, and Happy.

Classification Policy:
- Depression: unresolved sadness, hopelessness, emotional distress, emotional
  exhaustion, withdrawal, self-devaluation, or a clearly negative overall
  trajectory is dominant.
- Neutral: factual, routine, balanced, informational, or emotionally mild,
  without a dominant positive or distress-related state.
- Happy: happiness, relief, gratitude, accomplishment, fulfillment, or a
  clearly positive resolution is dominant.

Mixed and Shifting Emotion Rule:
- Assess dominant emotional meaning of the full text; do not classify from
  isolated phrases, brief cues, or a simple average of emotional words.
- Use overall trajectory and final takeaway only when multiple cues or a clear
  temporal shift is present.
- Do not select Neutral merely because multiple cues are present.
- Do not select Happy from a brief positive cue when unresolved distress
  remains dominant.
- Do not select Depression from a brief negative cue when the text clearly
  resolves into sustained relief or positive resolution.

Use only Depression, Neutral, or Happy. Do not output Sad, Positive, Mixed,
Anxiety, Other, or any label outside the three-class space. Base the
justification on textual evidence rather than clinical assumptions. End with:
Final label: [label]
```

## 부록 C. 문장별 핵심 변경 이유

| v2 문장 또는 규칙 | 변경 이유 | 비교할 결과 |
|---|---|---|
| `clear temporal emotional shift` | 실제 전환이 있는 경우에만 trajectory rule 적용 | Reddit 일반 문장에서 과도한 trajectory inference가 줄어드는지 |
| `Do not assume a trajectory` | 모델이 text에 없는 시간 흐름을 만들지 못하게 제한 | Neutral/factual sample의 불필요한 correction 감소 여부 |
| `before considering the Phase 1 label` | Phase 1 label anchoring을 약화 | Llama 2 독립 판단과 최종 판단의 차이, correction quality |
| `Do not use synonyms ... Sad` | 실제 baseline의 parser 불일치 대응 | parse failure와 manual repair 필요 여부 |
| `brief positive/negative cue` rule | 단기 cue를 dominant emotion으로 오인하는 오류 완화 | Depression-Happy 간 corrected/introduced error 변화 |
