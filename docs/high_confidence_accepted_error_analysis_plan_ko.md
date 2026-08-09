# High-Confidence Accepted-Error Analysis 계획

## 1. 목적

Confidence-guided routing은 낮은 confidence 예측만 Phase 2로 보내는 구조다. 따라서 routing threshold 이상으로 accept된 예측 중에도 오류가 남아 있는지, 남아 있다면 어떤 유형인지 별도로 확인한다. 이 분석의 목적은 모델의 한계를 숨기지 않고, threshold 기반 선택적 예측이 실제로 어떤 오류를 줄이고 어떤 오류를 남기는지 투명하게 제시하는 것이다.

이 분석은 현재 Mixed Emotion end-to-end 결과를 다시 만드는 실험이 아니다. Reddit held-out test의 Phase 1 prediction과 calibration/routing 결과를 이용하는 보조 신뢰성 분석이다.

## 2. 입력과 대상

기본 입력 파일은 Final 01에서 저장되는 다음 파일이다.

- `outputs_final/phase1_distilbert/phase1_test_predictions.csv`
- `outputs_final/phase1_distilbert/advanced_confidence_threshold_analysis/high_confidence_accepted_errors.csv`
- `outputs_final/phase1_distilbert/distilbert_phase1_summary.json`

분석 대상은 다음 조건을 모두 만족하는 Reddit held-out test 샘플이다.

1. calibrated MSP가 최종 routing threshold 이상이다.
2. `phase1_accepted=True`이다.
3. `phase1_label`과 reference proxy label이 다르다.

즉, Phase 2로 보내지지 않았지만 Phase 1이 높은 confidence로 잘못 예측한 사례를 분석한다.

## 3. 핵심 질문

1. accepted sample 중 오류가 몇 건이며, accepted set 대비 비율은 얼마인가?
2. 각 proxy class에서 어떤 오류가 많이 남는가?
3. Depression reference를 Happy 또는 Neutral로 예측한 false negative가 얼마나 되는가?
4. 이 오류는 confidence가 threshold 바로 위에 몰려 있는가, 아니면 매우 높은 confidence에서도 발생하는가?
5. routed error와 accepted error는 텍스트 길이, subreddit source, predicted class, confidence 분포에서 어떤 차이가 있는가?

## 4. 분석 절차

### Step 1. 최종 routing policy 고정

최종 temperature와 threshold는 calibration split에서만 선택한 값을 사용한다. 현재 운영 정책은 temperature-scaled MSP와 후보 범위 0.70--1.00을 사용한다. 이 단계에서 test 결과를 보고 threshold를 다시 조정하지 않는다.

### Step 2. accepted error 추출

`phase1_test_predictions.csv`에서 accepted 예측만 남긴 뒤, 실제 label과 predicted label이 다른 행을 추출한다. 각 행에 아래 정보를 보존한다.

| 항목 | 내용 |
|---|---|
| reference label | Reddit proxy label |
| Phase 1 label | DistilBERT final prediction |
| calibrated confidence | temperature-scaled MSP |
| threshold margin | confidence - routing threshold |
| raw confidence | temperature scaling 전 MSP |
| text | audit를 위한 원문 |
| source/subreddit | 데이터 출처가 남아 있는 경우 |
| text length | 길이별 오류 비교용 |

### Step 3. 정량 요약

다음 표를 생성한다.

| 지표 | 정의 |
|---|---|
| accepted count | Phase 1 결과를 그대로 사용한 샘플 수 |
| accepted errors | accepted sample 중 오분류 수 |
| accepted selective risk | accepted errors / accepted count |
| high-confidence Depression false negatives | reference가 Depression인데 Happy 또는 Neutral로 예측된 accepted 오류 수 |
| error-capture rate | 전체 Phase 1 오류 중 routed set에 포함된 비율 |
| confidence bands | 0.70--0.80, 0.80--0.90, 0.90 이상별 오류율 |

### Step 4. 사례 수준 audit

각 오류 유형에서 과도하게 민감한 개인 정보나 위기 표현이 없는 대표 사례를 최대 2--3개씩 선정한다. 사례 표에는 원문을 필요한 범위에서 축약하고, reference proxy label, Phase 1 label, calibrated confidence, threshold, 그리고 오류의 가능한 언어적 원인을 기록한다.

가능한 원인 범주는 다음과 같다.

- subreddit-derived proxy label과 본문 감정의 불일치
- 긍정 또는 부정 단어 하나에 대한 과도한 반응
- 긴 글에서 후반 정서 변화 반영 실패
- 문맥 또는 풍자/인용 표현의 모호성
- 중립적 정보 문장과 감정 표현의 혼합

사례 해석은 임상적 진단이 아니라 proxy-label 분류 오류 분석으로 제한한다.

## 5. 산출물

| 파일 | 내용 |
|---|---|
| `accepted_error_summary.csv` | accepted count, error count, selective risk, Depression false-negative count |
| `accepted_error_by_class.csv` | reference/prediction class별 accepted error matrix |
| `accepted_error_confidence_bands.csv` | confidence 구간별 sample 수와 오류율 |
| `high_confidence_accepted_error_cases.csv` | 사례 audit용 원문, confidence, label, threshold margin |
| `accepted_vs_routed_error_comparison.csv` | accepted error와 routed error의 분포 비교 |
| `high_confidence_error_examples.md` | 논문 Appendix 후보 사례와 해석 |

## 6. 논문 반영 방식

본문에는 다음 두 가지를 간결하게 반영한다.

1. routing effectiveness 표에 accepted errors와 accepted Depression false-negative risk를 포함한다.
2. Limitations에서 threshold 이상 오류가 완전히 제거되지 않으며, 고신뢰 오류 audit가 proxy-label/언어적 한계를 점검하는 보조 분석임을 명시한다.

대표 사례와 상세 표는 Appendix 또는 supplementary material에 둔다. 이 분석은 two-phase framework의 성능 주장을 부풀리기 위한 것이 아니라, Phase 2로 넘어가지 않은 오류를 공개하고 향후 개선 지점을 명확하게 하기 위한 것이다.

## 7. 완료 기준

다음 조건을 만족하면 분석을 완료로 본다.

1. 정확한 12,000-row Reddit held-out test 결과를 사용한다.
2. threshold와 temperature가 calibration split에서 고정된 상태임을 provenance 파일로 확인한다.
3. accepted error 수와 class별 분포를 재현 가능한 CSV로 저장한다.
4. Depression false negative를 포함한 대표 사례를 비임상적 proxy-label 관점에서 검토한다.
5. 본문 표, Limitations, Appendix 후보 문장을 결과에 맞춰 업데이트한다.
