# 최종 연구 진행 현황 및 남은 작업

작성일: 2026-08-15  
대상 연구: Confidence-Guided Selective LLM Re-Evaluation for Depression-Risk-Related Emotion Classification

## 1. 현재 상태 요약

본 연구의 핵심 end-to-end 구조와 주요 실험은 완료되었다. DistilBERT Phase 1 학습, confidence calibration, calibration-only routing threshold 선택, Reddit 및 Mixed Emotion routing, Llama 2 CoT와 Llama 3 SELF-DISCOVER Phase 2, full-set end-to-end 재조합, paired 통계 분석, high-confidence accepted-error audit까지 완료하였다.

현재 남은 가장 중요한 실험은 동일한 분할과 평가 프로토콜을 적용한 Mistral 7B 및 Llama 2 7B의 supervised Phase 1 classifier 비교다. 이 두 결과와 최종 도식을 추가하면 원고의 주요 `Pending` 부분을 해소할 수 있다.

## 2. 확정된 실험 구조

### 2.1 Phase 1

- Reddit primary dataset에서 Depression, Neutral, Happy를 클래스별 40,000건씩 사용한다.
- train / validation / calibration / held-out test를 70 / 10 / 10 / 10으로 분리한다.
- 최종 논문은 held-out test를 클래스별 4,000건, 총 12,000건으로 고정한다.
- validation split은 모델 및 hyperparameter 선택에 사용한다.
- calibration split은 temperature scaling과 routing threshold 선택에만 사용한다.
- held-out test 정답과 Phase 2 결과는 threshold 선택에 사용하지 않는다.

### 2.2 Confidence calibration과 routing

- Primary confidence score는 temperature-scaled maximum softmax probability(MSP)다.
- 최적 temperature는 calibration negative log-likelihood를 최소화해 선택한다.
- 최종 temperature는 `T*=1.7706`이다.
- 사전 지정한 threshold 후보는 `0.70, 0.71, ..., 1.00`이다.
- 위험예산은 `alpha=0.05`이며, calibration accepted-set의 보수적 오류 상한이 5% 이하인 후보 중 coverage가 가장 높은 threshold를 선택한다.
- 최종 primary threshold는 `tau*=0.70`이다.
- 다른 alpha 또는 threshold의 결과는 민감도 분석 기록이며, held-out LLM 결과를 보고 primary policy를 다시 선택하지 않는다.

### 2.3 Phase 2와 최종 label

- confidence가 `tau*` 이상인 accepted sample은 DistilBERT label을 유지한다.
- confidence가 `tau*` 미만인 routed sample만 Phase 2로 보낸다.
- Mixed Emotion은 저장된 원문 text를 사용한다.
- Reddit Phase 2는 minimally sanitized original `title + selftext`를 사용한다. URL과 직접 사용자명 패턴만 마스킹하고 부정어, 구두점, 문장 순서, 감정 변화는 보존한다.
- 최종 독자용 방법명은 `Llama 2 CoT`와 `Llama 3 SELF-DISCOVER`다. `v2`, `v2.1`은 내부 prompt-development 이력에서만 사용한다.

## 3. 완료 및 확정된 결과

### 3.1 DistilBERT Phase 1 및 calibration

| 항목 | 확정값 |
|---|---:|
| Reddit held-out examples | 12,000 |
| Phase 1 accuracy | 96.69% |
| Temperature | 1.7706 |
| Raw MSP NLL | 0.1411 |
| Temperature-scaled MSP NLL | 0.1041 |
| Raw MSP Brier score | 0.0559 |
| Temperature-scaled MSP Brier score | 0.0514 |
| Raw MSP ECE | 0.0244 |
| Temperature-scaled MSP ECE | 0.0083 |
| Raw MSP Adaptive ECE | 0.0244 |
| Temperature-scaled MSP Adaptive ECE | 0.0138 |
| Primary threshold | 0.70 |
| Reddit routed examples | 171 / 12,000 (1.42%) |
| Reddit routed Phase 1 accuracy | 49.12% |
| Reddit routed Phase 1 errors | 87 |
| Phase 1 error capture | 21.91% |
| Accepted examples | 11,829 |
| Accepted errors / selective risk | 310 / 2.62% |

### 3.2 Reddit original-text end-to-end

| System | Accuracy | Change | Corrected | Introduced | Net |
|---|---:|---:|---:|---:|---:|
| DistilBERT Phase 1 | 96.69% | - | - | - | 0 |
| DistilBERT + Llama 2 CoT | 96.67% | -0.03 pp | 47 | 50 | -3 |
| DistilBERT + Llama 3 SELF-DISCOVER | 96.94% | +0.25 pp | 42 | 12 | +30 |

- Reddit Llama 2 변화는 Phase 1과 통계적으로 구분되지 않는다.
- Reddit Llama 3 paired bootstrap 95% CI는 `[0.13, 0.38]` pp다.
- Reddit Llama 3 exact McNemar p-value는 약 `0.000052`, Holm-adjusted p-value는 약 `0.000156`이다.
- 결론은 “모든 reasoner가 항상 향상시킨다”가 아니라, 동일 routed subset에서도 reasoner에 따라 correction quality가 달라진다는 것이다.

### 3.3 Mixed Emotion end-to-end

| System | Accuracy | Change | Corrected | Introduced | Net |
|---|---:|---:|---:|---:|---:|
| DistilBERT Phase 1 | 81.33% | - | - | - | 0 |
| DistilBERT + Llama 2 CoT | 85.33% | +4.00 pp | 18 | 6 | +12 |
| DistilBERT + Llama 3 SELF-DISCOVER | 87.33% | +6.00 pp | 18 | 0 | +18 |

- 전체 300건 중 44건(14.67%)이 routed되었다.
- Mixed Emotion은 실제 모집단 prevalence를 나타내는 dataset이 아니라 mixed cue와 trajectory shift를 평가하기 위한 controlled stress test다.
- scenario별 결과와 실제 reasoning output 예시는 원고 Appendix에 반영되어 있다.

### 3.4 완료된 추가 분석

- Phase 1 대비 end-to-end paired bootstrap 95% CI
- Exact McNemar test
- 네 개 사전 지정 비교에 대한 Holm correction
- Risk-coverage curve와 threshold sweep
- Raw/scaled calibration reliability 분석
- Confidence score ablation(MSP, margin, negative entropy)
- Threshold bootstrap stability와 class-conditional selective metrics
- Accepted high-confidence error 310건 전수 연결
- 대표 original-post accepted error 6건 정성 분석
- Reddit routed original-text 171건 연결 감사: 171/171 exact match, conflicting original 0건

## 4. 완료된 코드 및 연구 산출물

### 4.1 최종 실행 노트북

1. `notebooks/colab/final/01_distilbert_phase1_training_final_colab.ipynb`
   - DistilBERT 학습, W&B sweep, best checkpoint 저장, calibration, threshold, advanced confidence 분석, Reddit/Mixed Phase 1 export
2. `notebooks/colab/final/02_2_llm_phase2_reasoning_model_specific_prompt_final_colab.ipynb`
   - Mixed Emotion routed sample의 최종 Llama 2 CoT 및 Llama 3 SELF-DISCOVER 결과
3. `notebooks/colab/final/03_mixed_emotion_end_to_end_orchestration_final_colab.ipynb`
   - Mixed Emotion full-set end-to-end 재조합과 논문용 표·그림 생성
4. `notebooks/colab/final/04_5_reddit_test_routed_phase2_original_text_primary_final_colab.ipynb`
   - Reddit primary `tau=0.70` routed 171건의 original-text Phase 2 및 12,000건 end-to-end 결과

### 4.2 재현 및 감사 코드

- `scripts/paired_end_to_end_analysis.py`
- `scripts/high_confidence_accepted_error_audit.py`
- `reports/statistics/paired_end_to_end_statistics.csv`
- `reports/statistics/paired_end_to_end_statistics.json`
- `reports/high_confidence_accepted_error_audit/`

### 4.3 논문 패키지 상태

- IEEE two-column 본문과 one-column Appendix로 구성된 25-page draft를 컴파일했다.
- 본문에는 세 개의 method-specific algorithm이 포함되어 있다.
- 참고문헌은 1--46번의 연속 번호이며 모든 항목이 본문에서 사용된다.
- Appendix에는 prompt protocol, 39-module SELF-DISCOVER pool, prompt-development diagnostics, 실제 Llama 2/Llama 3 output 예시, scenario 결과, Reddit correction, high-confidence accepted-error 사례, Mixed Emotion generation protocol이 포함되어 있다.
- Raw MSP Adaptive ECE 누락을 보완했으며, reader-facing prompt 버전 표기를 정리했다.
- PDF 25페이지를 렌더링하여 표 겹침, 잘림, 미정의 참조, horizontal overflow를 점검했다.

## 5. 다시 실행할 필요가 없는 항목

- DistilBERT primary Phase 1 학습 및 primary `alpha=0.05`, `tau=0.70` 정책
- Mixed Emotion 300건의 최종 Phase 1 및 Phase 2 실행
- Reddit 171건의 original-text Llama 2/Llama 3 Phase 2 실행
- Alpha/threshold를 end-to-end 결과에 맞춰 사후 재선택하는 실험
- 이미 기각된 universal/shared prompt를 최종 prompt로 다시 실행하는 실험
- 단순히 정확도를 높이기 위한 추가 dataset 문장 수정

재현성 확인이 필요한 경우에는 위 실험을 동일 설정으로 다시 실행할 수 있지만, 현재 확정 수치를 바꾸기 위한 반복 실험으로 사용하지 않는다.

## 6. 제출 전 필수 남은 작업

### P0-1. Mistral 7B Phase 1 classifier 완료

- DistilBERT와 동일한 120,000건, 동일 split, 동일 label space를 사용한다.
- supervised classifier로 학습하고 Accuracy, macro Precision, macro Recall, macro F1을 저장한다.
- calibration split에서 temperature, NLL, Brier, ECE, Adaptive ECE, threshold, coverage, routing rate를 계산한다.
- 결과가 완료되기 전까지 DistilBERT가 세 classifier 중 최고라고 단정하지 않는다.

### P0-2. Llama 2 7B Phase 1 classifier 완료

- Mistral과 동일한 matched protocol을 사용한다.
- Phase 2 Llama 2 reasoner 결과와 혼동하지 않도록 `Phase 1 classifier`라고 명시한다.
- 동일한 표와 calibration 항목을 모두 산출한다.

### P0-3. 두 classifier 결과의 논문 반영

결과가 나오면 다음을 갱신한다.

- Main Table I의 `Pending` Accuracy, macro Precision, macro Recall, macro F1
- 필요 시 model-specific calibration/threshold 비교표
- Abstract와 Contributions의 classifier 비교 문장
- Results, Discussion, Limitations의 비교 해석
- Conclusion의 “completed DistilBERT only” 제한 문장
- Supplementary result CSV/JSON과 run manifest

### P0-4. 최종 도식 삽입

- 전체 two-phase architecture
- Phase 1 split, calibration, routing 흐름
- accepted/routed 분기와 Phase 2 final-label replacement
- 가능하면 reliability/risk-coverage 또는 routing distribution 핵심 그림
- 그림의 threshold, routed count, sample 수가 본문 표와 일치하는지 확인한다.

### P0-5. 저자·소속·제출 메타데이터 확정

- 저자 순서, affiliation, corresponding author
- acknowledgments와 funding
- conflict of interest
- data/code availability
- target journal에 맞는 keywords와 cover letter

## 7. 강하게 권장하지만 결과표 완성을 막지는 않는 작업

### P1-1. 독립적인 prompt 확인

현재 prompt-development 비교 일부는 동일 routed evaluation subset을 반복적으로 확인한 탐색적 기록이다. 가능하면 prompt 수정에 사용하지 않은 별도 routed subset 또는 outer fold에서 frozen prompt의 방향성을 한 번 확인한다. 수행하지 못하면 현재 Limitations의 exploratory-development 문장을 유지한다.

### P1-2. 전문가 또는 복수 연구자 검토

- Mixed Emotion과 representative Reddit cases의 일부를 최소 두 명이 독립 검토한다.
- label agreement와 disagreement 사유를 기록한다.
- clinician review가 없으면 generated rationale을 임상 설명 또는 faithful explanation으로 주장하지 않는다.

### P1-3. 실행시간과 효율 기록

- GPU 종류, routed rows, 모델별 실행시간, 생성 호출 수를 기록한다.
- 전체 입력을 LLM에 보내지 않고 Reddit 1.42%, Mixed Emotion 14.67%만 re-evaluate했다는 계산 효율을 정리한다.
- 실제 측정하지 않은 비용은 추정값으로 단정하지 않는다.

## 8. 최종 작업 순서

1. Mistral 7B Phase 1 classifier 실행 및 산출물 저장
2. Llama 2 7B Phase 1 classifier 실행 및 산출물 저장
3. 두 classifier의 calibration 및 threshold 분석 완료
4. Table I과 관련 본문·제한점 갱신
5. 최종 architecture/data-flow 도식 삽입
6. 저자·소속·availability·윤리·funding 메타데이터 확정
7. 원고의 숫자와 supplementary artifact 재대조
8. Overleaf clean compile 후 PDF 전 페이지 시각 검수
9. 제출용 ZIP, PDF, source, supplementary archive 동결
10. IEEE Access submission checklist 최종 서명

## 9. 최종 완료 기준

다음 조건을 모두 만족하면 제출본을 동결한다.

- Table I의 classifier `Pending`이 실제 결과 또는 명시적으로 제외된 비교 설계로 정리되어 있다.
- 본문과 Appendix의 모든 숫자가 prediction-level artifact에서 재현된다.
- Primary routing policy는 `alpha=0.05`, `tau=0.70`으로 일관된다.
- Reddit 결과는 original-text Phase 2 결과만 최종 수치로 사용한다.
- Internal prompt version이 reader-facing method name으로 노출되지 않는다.
- `Depression`, `Neutral`, `Happy` label과 proxy-emotion task 정의가 전 문서에서 일치한다.
- 임상 진단·치료·자동화된 의료 의사결정 주장을 하지 않는다.
- 표·그림·수식이 잘리지 않고 reference/cross-reference 오류가 없다.
- 최종 source, PDF, result artifact, environment/run manifest를 보관한다.

## 10. 다음 미팅에서 결정할 항목

1. Mistral 7B와 Llama 2 7B Phase 1 실행 담당 및 GPU 일정
2. 최종 논문에 넣을 architecture figure 구성
3. 독립 prompt 확인 또는 복수 연구자 annotation을 이번 제출에 포함할지 여부
4. IEEE Access 제출 일정과 저자·소속·corresponding author 확정
5. 코드 및 Mixed Emotion dataset의 공개 범위와 release 시점

이 문서는 최종 현황의 기준 문서다. 실험이 추가되면 완료된 수치와 남은 작업을 이 문서에서 먼저 갱신한 뒤 원고와 다른 설명 문서를 동기화한다.
