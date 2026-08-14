# High-Confidence Accepted-Error Audit 완료 결과

## 1. 분석 대상

최종 DistilBERT Phase 1의 Reddit held-out prediction에서 논문 규약과 동일하게 class별 4,000건, 총 12,000건을 사용하였다. Temperature-scaled MSP threshold `0.70` 이상으로 Phase 1에서 accept되었지만 reference proxy label과 prediction이 다른 사례를 분석하였다.

이 문서에서 reference는 임상 진단이 아니라 subreddit-derived proxy label이다. 따라서 불일치를 모두 모델의 명백한 의미 해석 실패로 단정하지 않고, `model error`, `proxy label-content mismatch`, `linguistic ambiguity` 가능성을 함께 검토한다.

## 2. 핵심 결과

| 지표 | 결과 |
|---|---:|
| Held-out sample | 12,000 |
| Accepted | 11,829 |
| Routed | 171 |
| Total Phase 1 errors | 397 |
| Accepted errors | 310 |
| Accepted selective risk | 2.62% |
| Routed errors | 87 |
| Error-capture rate | 21.91% |
| Routing precision | 50.88% |
| Accepted Depression sample | 3,939 |
| Accepted Depression false negatives | 125 |
| Accepted Depression FN risk | 3.17% |
| Confidence >= 0.98 accepted errors | 34 |
| Confidence >= 0.98 Depression false negatives | 16 |

`Accepted Depression FN risk`는 accepted된 reference-Depression 3,939건 가운데 Happy 또는 Neutral로 예측된 125건의 비율이다. 따라서 `125 / 3,939 = 3.17%`이다.

## 3. Class별 accepted error

| Reference proxy label | Accepted | Errors | Accepted risk |
|---|---:|---:|---:|
| Depression | 3,939 | 125 | 3.17% |
| Happy | 3,912 | 144 | 3.68% |
| Neutral | 3,978 | 41 | 1.03% |

주요 전이는 Depression -> Happy 115건, Depression -> Neutral 10건, Happy -> Depression 112건, Happy -> Neutral 32건, Neutral -> Depression 17건, Neutral -> Happy 24건이었다.

## 4. Confidence 구간별 accepted error

| Calibrated confidence | Accepted | Errors | Error rate within band |
|---|---:|---:|---:|
| 0.70-0.80 | 140 | 51 | 36.43% |
| 0.80-0.90 | 343 | 89 | 25.95% |
| 0.90-0.95 | 571 | 76 | 13.31% |
| 0.95-0.98 | 1,437 | 60 | 4.18% |
| 0.98-1.00 | 9,338 | 34 | 0.36% |

Confidence가 높을수록 오류율은 빠르게 감소한다. 그러나 `0.98` 이상에서도 오류 34건이 남아 있으므로 temperature scaling과 routing threshold가 high-confidence error를 완전히 제거하지는 않는다.

## 5. 원문 매칭 및 대표 사례 6건

Accepted error 310건의 stored model input을 retained Reddit `title`과 `selftext`에 normalized exact matching으로 연결했으며, 310건 모두 원문에 매칭되었다. 아래 사례는 원문 자체가 stored proxy label을 비교적 명확하게 지지하고 서로 다른 failure mechanism을 보여주는 경우로, class별 2건씩 선정하였다. GitHub 문서에는 민감한 원문 전체를 게시하지 않고 publication-readable 요약만 제공한다.

| ID | Reference -> Prediction | Confidence | 오류 유형 | 핵심 해석 |
|---|---|---:|---|---|
| RED_041439 | Depression -> Happy | 0.9827 | acute-distress cue 누락 | 제목에 현재의 suicidal ideation이 명시되어 있으나, 직전 2주가 괜찮았다는 짧은 문장이 Happy prediction을 지배했다. |
| RED_000879 | Depression -> Happy | 0.9801 | severe negative-state cue 누락 | coercion, alcohol dependence, imprisonment, family separation, custody loss가 서술되지만 Happy로 예측되었다. |
| RED_026909 | Happy -> Neutral | 0.9954 | 기술적 문맥이 affective cue를 가림 | programming 설명이 대부분을 차지해 마지막의 accomplishment cue인 `kinda proud`가 충분히 반영되지 않았다. |
| RED_103680 | Happy -> Neutral | 0.9888 | 제품 세부정보가 excitement를 가림 | 비용과 hardware 세부정보가 명시적인 excitement와 새 프로젝트에 대한 긍정적 평가를 가렸다. |
| RED_010855 | Neutral -> Depression | 0.9817 | mental-health topic-term shortcut | 글은 online diagnosis 가능성에 관한 정보 질문이지만 disorder와 diagnosis가 반복되어 Depression으로 과잉 반응했다. |
| RED_000438 | Neutral -> Happy | 0.9855 | factual question의 affect 오판 | 차에서 들리는 고주파음의 원인과 안전성을 묻는 사실적 질문에 positive affect가 없지만 Happy로 예측되었다. |

## 6. 논문 해석

이 결과는 routing이 실패했다는 뜻이 아니다. Routed subset의 Phase 1 accuracy는 49.12%로 full test보다 훨씬 낮고, 171건만 보내면서 전체 오류 397건 중 87건을 포착했다. 즉 low-confidence routing은 오류가 밀집된 집단을 찾았다.

동시에 accepted set에도 310건의 오류가 남았으며 일부는 confidence 0.98 이상이었다. 그 원인은 하나가 아니다.

1. 현재의 acute distress나 심각한 negative state를 짧은 국소 문장보다 낮게 평가한 사례
2. 기술적·제품 중심 문맥이 명시적 pride 또는 excitement를 가린 사례
3. 정신건강 topic term이나 일반 질문 형식에 감정 label을 잘못 부여한 사례

따라서 본 분석은 threshold를 test 결과에 맞춰 다시 선택하거나 reference label을 사후 변경하기 위한 것이 아니다. 최종 threshold `0.70`은 그대로 유지하며, high-confidence error가 남는 이유와 proxy-label task의 한계를 투명하게 보고하는 residual-risk audit이다. 원문 전체 310건은 민감한 social-media text의 불필요한 재배포를 피하기 위해 공개 GitHub artifact에 포함하지 않는다.

## 7. 재현

```bash
python3 scripts/high_confidence_accepted_error_audit.py \
  --phase1-predictions /path/to/phase1_test_predictions.csv
```

기본 산출물 위치는 `reports/high_confidence_accepted_error_audit/`이다.
