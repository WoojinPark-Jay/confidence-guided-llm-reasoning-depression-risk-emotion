# 다음 실험 및 논문 보강 계획

이 문서는 confidence-guided two-phase LLM reasoning 연구를 다음 단계로 진행하기 위해 필요한 실험, 산출물, 논문 반영 항목을 정리한 것이다. 목적은 동료 연구자와 현재 상태를 공유하고, 어떤 순서로 실험과 원고 보강을 진행해야 하는지 명확히 하는 데 있다.

## 1. 현재 연구의 핵심 방향

본 연구의 핵심 아이디어는 다음과 같다.

1. Phase 1에서 DistilBERT, Mistral 7B, Llama 2 7B와 같은 분류 모델이 Reddit 텍스트를 Depression, Neutral, Happy 세 class로 분류한다.
2. 각 예측에 대해 confidence score를 계산한다.
3. confidence가 충분히 높은 샘플은 Phase 1 결과를 그대로 사용한다.
4. confidence가 낮은 샘플만 Phase 2 LLM reasoning 단계로 보낸다.
5. Phase 2에서는 Llama 2 Chain-of-Thought 또는 Llama 3 SELF-DISCOVER 방식으로 재평가하고, 최종 label과 rationale을 생성한다.

따라서 본 연구의 강점은 단순히 LLM을 사용하는 것이 아니라, LLM reasoning을 필요한 샘플에만 선택적으로 적용한다는 점이다. 이 주장을 설득력 있게 만들기 위해서는 confidence threshold, routed subset, Phase 2 correction, cost reduction을 수치로 보여주는 것이 중요하다.

## 2. 가장 먼저 해결해야 할 문제

현재 원고에서 가장 먼저 닫아야 할 부분은 다음 세 가지다.

| 우선순위 | 해결해야 할 문제 | 이유 |
|---|---|---|
| 1 | Mixed Emotion Dataset 300건 기준 결과 업데이트 | 현재 dataset 설명은 300건 기준으로 확장되었지만, 기존 결과는 90건 기준이므로 불일치가 발생할 수 있음 |
| 2 | Risk-coverage threshold sweep 결과 추가 | confidence-guided routing이 논문의 핵심 방법론이므로 threshold 선택 근거가 필요함 |
| 3 | Selective end-to-end evaluation 수행 | Phase 1 결과와 Phase 2 reasoning 결과를 합친 최종 framework 성능을 보여줘야 함 |

이 세 가지가 해결되어야 논문 결과 섹션이 안정적으로 완성될 수 있다.

## 3. 권장 진행 순서

### Step 1. Phase 1 모델 결과 확정

Primary Reddit dataset을 기준으로 Phase 1 모델 성능을 확정한다. 가능하면 class별 약 40,000건 수준의 balanced dataset을 사용한다.

해야 할 일:

1. DistilBERT, Mistral 7B, Llama 2 7B의 최종 학습 또는 기존 학습 결과 확인
2. validation set과 test set에 대해 prediction 수행
3. 각 샘플별 true label, predicted label, confidence score, correctness 저장
4. model별 accuracy, macro precision, macro recall, macro F1, class-level metrics 정리

필요 산출물:

| 파일 | 내용 |
|---|---|
| `phase1_validation_predictions.csv` | validation set의 true label, predicted label, confidence, correctness |
| `phase1_test_predictions.csv` | test set의 true label, predicted label, confidence, correctness |
| `phase1_model_metrics.csv` | Phase 1 모델별 전체 성능 요약 |

### Step 2. Risk-Coverage Threshold Sweep 수행

Phase 1 validation 결과를 이용해 confidence threshold 후보별 성능을 계산한다. 이 단계는 confidence-guided routing의 핵심 근거가 된다.

해야 할 일:

1. 여러 threshold 후보를 설정한다. 예: 0.70, 0.75, 0.80, 0.85, 0.90
2. threshold 이상 샘플은 accepted set으로 분류한다.
3. threshold 미만 샘플은 routed set으로 분류한다.
4. threshold별 coverage, routing rate, accepted accuracy, selective risk를 계산한다.
5. validation 결과를 기준으로 최종 threshold를 선택한다.

필요 산출물:

| 파일 | 내용 |
|---|---|
| `threshold_sweep_results.csv` | threshold별 coverage, routing rate, selective risk, captured errors |
| `selected_threshold.json` | 최종 선택된 threshold와 선택 기준 |

논문에 들어갈 표:

| 항목 | 설명 |
|---|---|
| Threshold | confidence 기준값 |
| Coverage | Phase 1에서 그대로 accept되는 비율 |
| Routing rate | Phase 2로 보내지는 비율 |
| Accepted accuracy | accept된 샘플 중 Phase 1 정확도 |
| Selective risk | accept된 샘플 중 error rate |
| Captured Phase 1 errors | routed subset 안에 포함된 Phase 1 error 수 |

### Step 3. Primary Test Set에서 Selective End-to-End 평가

전체 test set에 대해 Phase 1 prediction을 먼저 수행하고, selected threshold 기준으로 low-confidence 샘플만 Phase 2로 보낸다.

중요한 점:

- 전체 test sample을 전부 LLM reasoning에 넣을 필요는 없다.
- 본 연구의 end-to-end 평가는 selective routing 기준으로 수행한다.
- 즉, high-confidence sample은 Phase 1 결과를 유지하고, low-confidence sample만 Phase 2 결과로 교체한다.

해야 할 일:

1. test set 전체에 Phase 1 prediction 적용
2. selected threshold 기준으로 accepted/routed split 생성
3. routed samples에 대해서만 Phase 2 reasoning 실행
4. accepted Phase 1 결과와 Phase 2 결과를 합쳐 final prediction 생성
5. 전체 test set 기준 final accuracy, macro F1, class-level metrics 계산

필요 산출물:

| 파일 | 내용 |
|---|---|
| `routed_test_samples.csv` | Phase 2로 보내진 test sample 목록 |
| `phase2_reasoning_outputs.csv` | LLM reasoning 결과, rationale, final label |
| `end_to_end_results.csv` | 전체 test set의 final prediction 결과 |
| `end_to_end_metrics.csv` | 최종 framework 성능 요약 |

### Step 4. Phase 2 Correction Analysis 수행

Phase 2 reasoning이 실제로 도움이 되었는지 정리한다. 단순히 최종 accuracy만 보고하면 부족하다. routed subset 안에서 어떤 변화가 있었는지를 보여줘야 한다.

분석 항목:

| 항목 | 의미 |
|---|---|
| Corrected | Phase 1에서 틀렸지만 Phase 2에서 맞춘 샘플 |
| Unchanged correct | Phase 1도 맞고 Phase 2도 맞은 샘플 |
| Unchanged wrong | Phase 1도 틀리고 Phase 2도 틀린 샘플 |
| Worsened | Phase 1은 맞았지만 Phase 2에서 틀리게 바뀐 샘플 |

필요 산출물:

| 파일 | 내용 |
|---|---|
| `phase2_correction_summary.csv` | corrected, unchanged, worsened count |
| `phase2_error_cases.csv` | Phase 2가 실패한 사례 목록 |
| `phase2_representative_cases.csv` | 논문 appendix에 넣을 대표 사례 |

### Step 5. Mixed Emotion Dataset v2.2 300건 평가

Mixed Emotion Dataset은 training이나 threshold selection에 사용하지 않고, supplementary stress-test로만 사용한다.

해야 할 일:

1. 300건 전체에 Phase 1 prediction 적용
2. selected threshold 기준으로 routed sample 선택
3. routed sample에 Phase 2 reasoning 적용
4. final prediction 생성
5. 300건 기준 accuracy, confusion matrix, corrected/worsened count 계산

필요 산출물:

| 파일 | 내용 |
|---|---|
| `mixed_emotion_v2_2_phase1_predictions.csv` | 300건 Phase 1 결과 |
| `mixed_emotion_v2_2_phase2_outputs.csv` | routed sample Phase 2 결과 |
| `mixed_emotion_v2_2_results.csv` | 최종 성능 및 correction summary |
| `mixed_emotion_v2_2_confusion_matrix.csv` | final prediction confusion matrix |

논문 반영 포인트:

- 기존 90건 결과는 최종 원고에서 제거하거나, pilot result로 명확히 분리한다.
- 최종 결과 섹션은 300건 v2.2 기준으로 업데이트한다.
- synthetic dataset이므로 clinical validity claim은 하지 않는다.

### Step 6. Confidence Calibration 분석

Confidence score를 routing 기준으로 사용하는 연구이므로, confidence가 얼마나 신뢰 가능한지 보여주는 보조 분석이 필요하다.

권장 분석:

| 분석 | 목적 |
|---|---|
| Confidence histogram | confidence 분포 확인 |
| Reliability diagram | confidence와 실제 accuracy의 일치 정도 확인 |
| Expected Calibration Error (ECE) | calibration 정도를 수치화 |
| High-confidence error analysis | confidence가 높았지만 틀린 사례 분석 |

필요 산출물:

| 파일 | 내용 |
|---|---|
| `confidence_bins.csv` | confidence bin별 sample count와 accuracy |
| `calibration_results.csv` | ECE 등 calibration metric |
| `high_confidence_errors.csv` | high-confidence error 사례 |

### Step 7. Efficiency / Cost 분석

Selective routing의 장점은 LLM reasoning을 모든 샘플에 적용하지 않는다는 점이다. 따라서 LLM call reduction을 수치로 보여주는 것이 중요하다.

계산할 항목:

| 항목 | 설명 |
|---|---|
| Routed sample count | Phase 2로 보내진 샘플 수 |
| Routing rate | 전체 중 Phase 2로 보내진 비율 |
| LLM call reduction | all-routed baseline 대비 줄어든 LLM 호출 비율 |
| Estimated cost reduction | 가능하면 token/cost 기준 절감 추정 |
| Estimated time reduction | 가능하면 inference time 기준 절감 추정 |

예시 해석:

- routing rate가 20%라면, all-routed LLM baseline 대비 약 80%의 LLM calls를 줄인 것으로 설명할 수 있다.

## 4. 논문에 최종적으로 들어가야 할 표

| 표 | 내용 |
|---|---|
| Table 1 | Reddit dataset class 및 subreddit source distribution |
| Table 2 | Sentiment-aware filtering 전후 count |
| Table 3 | Phase 1 model performance |
| Table 4 | Risk-coverage threshold sweep |
| Table 5 | Selective end-to-end performance |
| Table 6 | Phase 2 correction analysis |
| Table 7 | Mixed Emotion Dataset v2.2 stress-test results |
| Appendix Table A1 | Reddit dataset variable description |
| Appendix Table A2 | Chain-of-Thought prompting protocol |
| Appendix Table A3 | SELF-DISCOVER prompting protocol |
| Appendix Table A4 | Illustrative case-level outputs |
| Appendix Table A5 | Mixed Emotion Dataset generation protocol |

## 5. 논문에 최종적으로 들어가면 좋은 그림

| 그림 | 내용 |
|---|---|
| Figure 1 | Two-phase framework architecture |
| Figure 2 | Confidence score distribution |
| Figure 3 | Risk-coverage curve |
| Figure 4 | Primary test set confusion matrix |
| Figure 5 | Mixed Emotion Dataset confusion matrix 또는 correction flow |

## 6. 최종 원고에서 주의해야 할 표현

본 연구는 clinical diagnosis 모델이 아니라 proxy emotion classification framework이다. 따라서 원고 전체에서 다음 표현을 일관되게 유지해야 한다.

권장 표현:

- depression-risk-related emotion classification
- proxy emotion labels
- research-oriented analysis
- non-clinical social media text
- supplementary stress-test dataset

피해야 할 표현:

- depression diagnosis
- clinical screening system
- detecting clinically depressed individuals
- diagnostic decision-support system

## 7. 바로 다음 회의에서 결정해야 할 사항

다음 회의에서는 아래 항목을 먼저 확정하는 것이 좋다.

1. Phase 1 large-run에 사용할 최종 데이터 크기
2. 최종 Phase 1 기준 모델을 DistilBERT 중심으로 둘지, 세 모델 모두 동일하게 가져갈지
3. threshold sweep에 사용할 candidate threshold 범위
4. Phase 2 reasoning을 Llama 2 CoT와 Llama 3 SELF-DISCOVER 모두 실행할지 여부
5. Mixed Emotion 300건 평가를 어떤 모델 조합으로 수행할지
6. calibration 분석 범위
7. cost/efficiency 분석을 실제 시간/토큰 기준으로 할지, LLM call reduction 기준으로 단순화할지

## 8. 가장 가까운 실행 목표

가장 먼저 만들어야 할 파일은 다음 네 가지다.

1. `phase1_validation_predictions.csv`
2. `phase1_test_predictions.csv`
3. `threshold_sweep_results.csv`
4. `routed_test_samples.csv`

이 네 가지가 생성되면 이후 Phase 2 reasoning, end-to-end evaluation, calibration, mixed emotion evaluation을 순서대로 진행할 수 있다.
