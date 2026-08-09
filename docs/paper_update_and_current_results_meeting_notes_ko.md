# 논문 업데이트 및 현재 실험 결과 공유

이 문서는 다음 미팅에서 현재 원고 업데이트, 확보된 실험 결과, 아직 최종 수치로 확정하지 않은 항목을 함께 검토하기 위한 공유 자료다. 본 연구는 임상 진단 모델이 아니라 Reddit 커뮤니티 맥락으로 구성된 **3-class proxy emotion classification** 과제에서, confidence-guided routing과 Phase 2 LLM reasoning의 효과를 검증하는 것을 목표로 한다.

## 1. 이번에 반영한 방법론 업데이트

### 1.1 데이터 및 평가 분리

- Reddit primary dataset은 class별 40,000개를 사용하도록 구성했다.
- 학습, validation, calibration, held-out test를 분리했다.
- validation은 학습 중 모델 선택과 hyperparameter sweep에 사용한다.
- calibration split은 temperature scaling과 routing threshold 선택에만 사용한다.
- held-out test는 threshold와 temperature를 고정한 뒤 최종 평가에만 사용한다.
- 따라서 test set으로 threshold를 맞추는 leakage를 피한다.

### 1.2 Confidence calibration과 routing 정책

- Phase 1 DistilBERT의 raw softmax confidence 대신 temperature-scaled maximum softmax probability (MSP)를 주 confidence score로 사용한다.
- calibration split에서 negative log-likelihood (NLL)를 최소화해 temperature $T^*$를 선택한다.
- 현재 실행에서는 $T^*=1.7706$으로 추정되었다.
- routing threshold 후보는 $0.70, 0.71, \ldots, 1.00$이다.
- 각 후보에서 accepted set의 one-sided Clopper--Pearson selective-risk upper bound가 5% 이하인지를 확인한다.
- 위험 조건을 통과한 후보 중 coverage가 가장 큰 threshold를 선택한다. 즉, 정해진 위험 기준을 넘지 않는 범위에서 LLM으로 보내는 비율을 최소화하는 정책이다.
- 현재 선택된 운영 threshold는 $\tau^*=0.70$이다. 이 값은 임의로 결과를 보고 고른 값이 아니라, 사전에 정한 후보 범위 안에서 calibration data의 risk--coverage 조건으로 선택된 값이다.

### 1.3 Phase 2 reasoning protocol

- Phase 1 confidence가 threshold보다 낮은 sample만 Phase 2로 routing한다.
- Llama 2는 Chain-of-Thought 기반 재평가를 수행한다.
- Llama 3는 SELF-DISCOVER select/adapt/implement 절차로 재평가를 수행한다.
- 두 prompt 모두 clinical diagnosis, medical condition inference, treatment advice를 요구하지 않도록 제한했다.
- mixed or shifting emotion에서는 단어 하나의 polarity 평균이 아니라, 텍스트의 전체 메시지와 최종 emotional trajectory를 기준으로 3-class label을 선택하도록 명시했다.
- Phase 2 최종 label은 Depression, Neutral, Happy 중 하나로 canonicalize하여 end-to-end 평가에 사용한다.

## 2. 현재 확보된 결과

### 2.1 Reddit held-out test: Phase 1 DistilBERT

아래 Reddit 수치는 현재 실행의 provisional result다. 분할 반올림 문제로 test row가 12,001개가 되었으므로, 논문 최종 표에는 **class별 정확히 4,000개, 총 12,000개**가 되도록 재실행한 수치로 교체한다.

| 항목 | 현재 결과 | 해석 |
|---|---:|---|
| Test accuracy | 96.69% | primary proxy-label task에서의 Phase 1 분류 성능 |
| Macro F1 | 96.69% | 세 class를 균형 있게 본 성능 |
| Temperature $T^*$ | 1.7706 | calibration split에서 NLL 최소화로 추정 |
| Raw NLL -> calibrated NLL | 0.1411 -> 0.1041 | confidence probability의 적합도 개선 |
| Raw ECE -> calibrated ECE | 0.0244 -> 0.0083 | confidence와 실제 정답률의 정렬 개선 |
| Raw Brier -> calibrated Brier | 0.0559 -> 0.0514 | 전체 확률 분포 품질 개선 |
| Selected threshold | 0.70 | calibration risk--coverage 규칙으로 선택 |
| Routed test samples | 171 / 12,001 (1.42%) | Phase 2가 처리할 Reddit held-out candidate |
| Routed Phase 1 errors | 87 | routing된 sample 안의 Phase 1 오류 |
| Error capture | 21.91% | 전체 Phase 1 오류 중 routing이 포착한 비율 |

현재 threshold가 비교적 적은 sample만 routing하는 이유는 primary Reddit dataset에서 calibrated confidence가 전반적으로 높기 때문이다. 반대로 emotional trajectory가 의도적으로 섞인 Mixed Emotion stress-test에서는 같은 threshold로도 더 큰 비율이 routing된다.

### 2.2 Mixed Emotion v2.4 stress-test: end-to-end 결과

Mixed Emotion v2.4는 Depression, Happy, Neutral 각 100개로 구성된 300개 supplementary stress-test다. 학습, threshold 선택, hyperparameter tuning에는 사용하지 않았다. 따라서 primary benchmark나 clinical validity 증거가 아니라, mixed or shifting emotional cues에서 routing과 reasoning의 행동을 보는 보조 평가로 해석한다.

| System | Total | Routed | Accuracy | Change vs. Phase 1 | Corrected routed errors | Introduced routed errors | 현재 상태 |
|---|---:|---:|---:|---:|---:|---:|---|
| Phase 1 only (DistilBERT) | 300 | 44 | 81.33% | - | 0 | 0 | 완료 |
| DistilBERT + Llama 2 CoT | 300 | 44 | 82.33% | +1.00 pp | 13 | 10 | parser canonicalization 재검증 필요 |
| DistilBERT + Llama 3 SELF-DISCOVER | 300 | 44 | 87.33% | +6.00 pp | 18 | 0 | 완료 |

Mixed Emotion Phase 1 routed subset은 44/300 (14.67%)이며, 이 subset 안의 Phase 1 accuracy는 52.27%였다. 즉 routing은 전체 sample을 무차별적으로 LLM에 보내는 방식이 아니라, Phase 1이 상대적으로 취약한 부분을 선별했다. Llama 3는 그 routed subset에서 18개의 오류를 수정하고 새 오류는 만들지 않아 전체 accuracy를 81.33%에서 87.33%로 높였다.

## 3. 원고에 반영한 내용

### 본문

- subreddit source group, class-level sample count, sentiment-aware filtering과 balancing 과정을 명시했다.
- label을 clinical diagnosis가 아닌 proxy emotion label로 한정하는 문장을 Dataset, Discussion, Limitation에 반영했다.
- temperature scaling, NLL 기반 $T^*$ 선택, calibrated MSP, accepted/routed set, coverage, routing rate, selective risk를 수식으로 기술했다.
- Clopper--Pearson risk upper bound와 constrained threshold-selection rule을 추가했다.
- Phase 1 / Phase 2 inference 과정을 algorithm 형태로 정리했다.
- calibration quality table에는 NLL, Brier, ECE, adaptive ECE를 함께 보고하도록 구성했다.
- Reddit primary test와 Mixed Emotion stress-test의 역할을 분리해 서술했다.

### Appendix

- Appendix B: Llama 2 CoT prompt protocol과 structured output rule.
- Appendix C: Llama 3 SELF-DISCOVER protocol과 final-label extraction rule.
- Appendix D: Phase 1이 Happy로 예측했지만 Phase 2가 Depression으로 수정한 routed example을 Llama 2와 Llama 3 각각 제시했다. 저장 column, output excerpt, pipeline role, 해석 방식을 같이 제공한다.
- References 뒤에 Appendix가 오도록 구성하고, appendix table의 빈 공간과 크기를 정리했다.

### References

- 이전 reference audit에서 확인된 citation--claim mismatch, DOI 오류, 최신 LLM/social-media 문헌 부족 문제를 정리했다.
- 실제 수정 내역은 `reference_update_completion_report_ko.md`에 문장 단위로 기록되어 있다.

## 4. 아직 논문 최종값으로 확정하지 않은 항목

1. **Reddit exact 12,000-row rerun**: 현재 12,001-row provisional 수치를 정확히 4,000/class test split 결과로 교체한다.
2. **Reddit routed Phase 2 end-to-end**: 171개 routed Reddit held-out sample만 Llama 2/Llama 3에 보내서 primary test 전체 기준 Phase 1 only 대비 two-phase 성능을 계산한다.
3. **Llama 2 parser canonicalization**: Llama 2가 `Sad`처럼 predefined label space 밖의 표현을 낸 사례가 있어 final-label canonicalization을 재검증한다. 현재 +1.00 pp는 working result로만 취급한다.
4. **Mistral 7B / Llama 2 7B Phase 1 classifier comparison**: full-scale classifier 실험이 완료되면 Table I--IV의 Pending을 실제 값으로 교체한다.
5. **High-confidence accepted-error audit**: threshold 이상인데도 틀린 sample을 별도로 정량/정성 분석한다. 이는 routing이 놓친 위험 사례와 proxy-label 한계를 투명하게 보여주기 위한 보강 실험이다.
6. **최종 원고 수치 갱신**: 위 실험이 완료된 뒤 Results, table, figure, Appendix representative case의 수치를 final artifact 기준으로 동기화한다.

## 5. 다음 미팅에서 확인할 결정 사항

- 현재 confidence policy ($\tau$ candidate range 0.70--1.00, upper-bound risk target 5%)를 본 실험의 운영 정책으로 유지할지, 그리고 threshold sensitivity 결과를 본문/appendix 중 어디에 둘지.
- Llama 3 SELF-DISCOVER를 Phase 2 주 결과로 보고하고, Llama 2 CoT는 비교 또는 ablation으로 둘지.
- Mixed Emotion v2.4를 supplementary controlled stress-test로만 보고하는 현재 서술 범위를 유지할지.
- Reddit routed Phase 2 결과와 high-confidence accepted-error audit을 완료한 뒤 최종 저널/학회 제출용 결과 표를 확정할지.
- 제목은 clinical risk/diagnosis 과장을 피하는 방향을 유지할지. 현재 제목의 `Depression-Risk-Related` 표현은 `Depression-Related Emotion Classification`으로 단순화하는 안을 검토할 수 있다.

## 6. 관련 문서 및 실행 링크

- [최종 End-to-End Colab Workflow](final_end_to_end_workflow_ko.md)
- [High-Confidence Accepted-Error Analysis 계획](high_confidence_accepted_error_analysis_plan_ko.md)
- [Trajectory-aware Phase 2 prompt 실험 계획](phase2_trajectory_prompt_experiment_plan_ko.md)
- [Reference 업데이트 반영 보고서](reference_update_completion_report_ko.md)
- [Final 01: DistilBERT Phase 1 training](https://colab.research.google.com/github/WoojinPark-Jay/confidence-guided-llm-reasoning-depression-risk-emotion/blob/feature/phase2-mixed-emotion-reasoning-colab/notebooks/colab/final/01_distilbert_phase1_training_final_colab.ipynb)
- [Final 02: Mixed Emotion routed Phase 2 reasoning](https://colab.research.google.com/github/WoojinPark-Jay/confidence-guided-llm-reasoning-depression-risk-emotion/blob/feature/phase2-mixed-emotion-reasoning-colab/notebooks/colab/final/02_llm_phase2_reasoning_final_colab.ipynb)
- [Final 03: Mixed Emotion end-to-end orchestration](https://colab.research.google.com/github/WoojinPark-Jay/confidence-guided-llm-reasoning-depression-risk-emotion/blob/feature/phase2-mixed-emotion-reasoning-colab/notebooks/colab/final/03_mixed_emotion_end_to_end_orchestration_final_colab.ipynb)
- [Final 04: Reddit routed Phase 2 end-to-end](https://colab.research.google.com/github/WoojinPark-Jay/confidence-guided-llm-reasoning-depression-risk-emotion/blob/feature/phase2-mixed-emotion-reasoning-colab/notebooks/colab/final/04_reddit_test_routed_phase2_end_to_end_final_colab.ipynb)

## 7. 해석 원칙

- 모든 수치는 proxy-label task 및 supplementary synthetic stress-test 맥락에서 해석한다.
- 높은 accuracy 또는 LLM correction은 depression diagnosis, clinical screening efficacy, treatment recommendation을 의미하지 않는다.
- 최종 제출 원고에는 실행 코드가 만든 CSV/JSON/XLSX/PNG 산출물의 확정 수치만 반영하고, provisional 또는 Pending 표시는 남기지 않는다.
