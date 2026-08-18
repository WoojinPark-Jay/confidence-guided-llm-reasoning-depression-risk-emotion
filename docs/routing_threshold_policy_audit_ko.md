# Routing Threshold Policy Audit

## 1. 목적

이 문서는 DistilBERT Phase 1의 confidence-guided routing threshold가 어떤 규칙으로 선택되었는지, 그리고 target selective risk `alpha`와 사전 지정 candidate grid가 각각 어떤 역할을 하는지 명확히 기록한다.

이 연구의 routing은 임상 의사결정이 아니라 Reddit 기반 3-class proxy emotion classification의 계산적 재평가 정책이다. 따라서 `alpha`는 임상 안전 기준이 아니라, Phase 1 accepted prediction의 오류 위험을 제한하기 위해 연구자가 명시한 운영 risk budget이다.

## 2. 현재 고정 운영 정책

| 항목 | 값 | 역할 |
|---|---:|---|
| Confidence score | Temperature-scaled MSP | Phase 1 confidence score |
| Temperature | `T*=1.7706` | calibration NLL 최소화로 추정 |
| Candidate grid | `0.70, 0.71, ..., 1.00` | held-out test 전에 사전 지정 |
| Target selective risk | `alpha=0.05` | accepted-set upper-risk feasibility condition |
| One-sided confidence level | `1-delta=0.95` | Clopper--Pearson upper bound |
| Acceptance rule | `confidence >= tau` | Phase 1 label 유지 |
| Routing rule | `confidence < tau` | Phase 2 LLM reasoning으로 전달 |

`0.70`이라는 candidate-grid 하한은 3-class softmax에서 약한 과반 confidence인 0.50 부근을 바로 accept하지 않기 위한 보수적 운영정책으로, held-out test와 Phase 2 결과를 보기 전에 정했다.

## 3. 선택 규칙

각 candidate threshold `tau`에서 calibration split의 accepted set을 만들고 다음 값을 계산한다.

```text
accepted(tau) = confidence >= tau
coverage(tau) = accepted count / total count
selective risk(tau) = accepted errors / accepted count
upper risk(tau) = one-sided Clopper--Pearson upper bound
```

최종 선택 규칙은 다음과 같다.

```text
1. upper risk(tau) <= alpha 를 만족하는 candidate만 feasible로 남긴다.
2. feasible candidate 중 coverage가 가장 높은 tau를 선택한다.
```

즉, `alpha`는 threshold를 직접 최적화하는 점수가 아니라 **안전성 통과/탈락 조건**이다. 통과한 후보 중에서는 LLM으로 보내는 비율을 최소화하도록, 즉 Phase 1 coverage가 가장 큰 후보를 고른다.

## 4. 저장된 calibration audit 결과

현재 완료된 DistilBERT run의 calibration-side 결과는 다음과 같다.

| Threshold | Accepted count | Routed count | Coverage | Selective risk | 95% one-sided upper risk |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 11,989 | 11 | 99.91% | 3.09% | 3.37% |
| 0.60 | 11,922 | 78 | 99.35% | 2.82% | 3.08% |
| 0.70 | 11,830 | 170 | 98.58% | 2.57% | 2.82% |
| 0.75 | 11,785 | 215 | 98.21% | 2.46% | 2.71% |
| 0.80 | 11,712 | 288 | 97.60% | 2.19% | 2.42% |

현재 pre-specified grid는 0.70--1.00이므로, `tau=0.70`은 `alpha=0.05`를 만족하는 **가장 낮은 candidate**이며 feasible 후보 중 coverage가 가장 높다. 따라서 현재 결과의 올바른 해석은 다음과 같다.

> `tau=0.70`은 `alpha=0.05`만으로 자동 발굴된 전역 최적 threshold가 아니다. 사전 지정한 보수적 candidate range 0.70--1.00 안에서 calibration upper-risk 조건을 충족하고 가장 높은 coverage를 가진 운영 threshold다.

이 값은 held-out Reddit test 성능, Mixed Emotion 결과, Llama 2/3 correction 결과를 보고 선택하거나 조정한 값이 아니다.

## 5. Alpha sensitivity를 해석하는 방법

후보 범위를 0.50--1.00으로 확장해 audit할 경우의 의미는 아래와 같다.

| Target risk `alpha` | 현재 저장된 값으로 확인되는 결과 | 해석 |
|---:|---|---|
| 10% | 0.50이 통과 | risk budget이 느슨하면 Phase 1을 거의 모두 accept |
| 5% | 0.50이 통과 | 5%만으로는 0.70을 강제하지 않음 |
| 3% | 0.60은 탈락, 0.70은 통과 | 정확한 선택값은 0.61--0.70 사이이며, 더 엄격한 risk budget일수록 routing 증가 |

따라서 `alpha=3%, 5%, 10%` sensitivity는 최종 threshold를 test 결과가 가장 좋게 나오는 값으로 바꾸기 위한 작업이 아니다. 이는 risk budget 변화에 따라 routing 규모, Phase 1 accepted risk, Phase 2 correction과 introduced error가 어떻게 변하는지 확인하는 보조 견고성 분석이다.

## 6. Held-out Reddit test에서의 routing 난이도

아래 값은 threshold 선택에 사용되지 않은 held-out test의 사후 진단이다.

| Threshold | Routed samples | Routed Phase 1 errors | Routed Phase 1 error rate |
|---:|---:|---:|---:|
| 0.50 | 14 | 11 | 78.6% |
| 0.60 | 87 | 52 | 59.8% |
| 0.70 | 171 | 87 | 50.9% |
| 0.75 | 233 | 117 | 50.2% |
| 0.80 | 311 | 138 | 44.4% |

`tau=0.70`에서 171개 중 87개가 Phase 1 오류였다는 것은 routing이 전체 test error를 완벽히 포착했다는 뜻은 아니다. 다만 Phase 2가 검토하는 집단이 전체 test 평균보다 훨씬 오류가 농축된 어려운 집단임을 보여준다.

## 7. 논문 서술 원칙

- 주 operating policy는 사전 지정 grid `0.70--1.00`, `alpha=0.05`로 보고한다.
- `alpha=0.05`를 데이터가 자동으로 정한 값이나 임상 안전 기준으로 표현하지 않는다.
- `tau=0.70`을 test 또는 Phase 2 end-to-end 성능으로 최적화했다고 표현하지 않는다.
- `alpha` 또는 threshold sensitivity 결과는 보조 분석으로 제시하고, sensitivity에서 가장 높은 LLM accuracy가 나온 operating point를 주 결과로 교체하지 않는다.
- Phase 2의 효과는 전체 accuracy뿐 아니라 routed error rate, corrected errors, introduced errors, net corrections를 함께 보고한다.

## 8. 재현 산출물

Final 01 실행 후 다음 파일에서 값을 확인할 수 있다.

```text
outputs_final/phase1_distilbert/phase1_selected_threshold.csv
outputs_final/phase1_distilbert/phase1_threshold_calibration_table.csv
outputs_final/phase1_distilbert/phase1_threshold_grid_lower_bound_sensitivity.csv
outputs_final/phase1_distilbert/phase1_test_predictions.csv
outputs_final/phase1_distilbert/advanced_confidence_threshold_analysis/threshold_provenance.json
```

Calibration split은 12,000 rows로 고정되었다. 과거 held-out artifact의 fractional split 반올림으로 남은 12,001 rows는 label별 4,000 rows의 deterministic balanced view로 정규화했으며, 최종 논문과 통계 분석은 정확히 12,000 held-out rows를 사용한다.
