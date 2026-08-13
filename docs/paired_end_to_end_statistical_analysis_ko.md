# Phase 1 대비 Phase 2 paired 통계 분석

## 1. 분석 목적

동일한 입력에서 Phase 1의 정오 여부와 Phase 2를 결합한 end-to-end 정오 여부가 실제로 달라졌는지 분석하였다. 서로 다른 표본에서 계산한 정확도를 단순 비교한 것이 아니라, **같은 sample이 오답에서 정답으로 바뀌었는지 또는 정답에서 오답으로 바뀌었는지**를 비교한다.

이 paired 설계가 필요한 이유는 Phase 1과 end-to-end system이 정확히 같은 test sample을 평가하기 때문이다. 따라서 전체 정확도 차이뿐 아니라 다음 두 변화가 핵심이다.

- `corrected`: Phase 1 오답이 Phase 2 적용 후 정답으로 바뀐 수
- `introduced`: Phase 1 정답이 Phase 2 적용 후 오답으로 바뀐 수
- `net corrections = corrected - introduced`

## 2. 분석 방법

### Exact McNemar test

`corrected`와 `introduced`의 비대칭을 검정한다. Phase 2가 아무 효과도 없다면 두 방향의 변화가 비슷하게 나타날 것으로 기대한다. Exact McNemar test는 변화가 발생한 discordant pair만 사용하므로, 같은 sample을 전후 비교하는 본 실험에 적합하다.

### Paired bootstrap 95% confidence interval

동일 sample의 정오 차이를 한 쌍으로 유지한 채 50,000회 재표집하여 accuracy change의 95% confidence interval을 계산한다.

- interval 전체가 0보다 크면 관측된 개선 방향이 비교적 안정적이다.
- interval이 0을 포함하면 개선과 저하를 명확히 구분하기 어렵다.
- interval 전체가 0보다 작으면 성능 저하 방향이 비교적 안정적이다.

### Holm multiple-comparison correction

다음 네 개의 사전 지정 비교를 함께 보고하므로 family-wise error를 통제한다.

1. Reddit held-out / Llama 2 CoT v2
2. Reddit held-out / Llama 3 SELF-DISCOVER
3. Mixed Emotion / Llama 2 CoT v2
4. Mixed Emotion / Llama 3 SELF-DISCOVER

Holm 보정은 여러 검정을 동시에 수행할 때 우연히 작은 p-value가 발생할 가능성을 줄인다. 난수 시드는 `20260813`로 고정하였다.

## 3. 확정 결과

| Dataset | Reasoner | Phase 1 | End-to-end | Change | Corrected | Introduced | Paired 95% CI | Exact p | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reddit held-out | Llama 2 CoT v2 | 96.69% | 96.60% | -0.09 pp | 38 | 49 | [-0.24, 0.06] pp | 0.284 | 0.526 |
| Reddit held-out | Llama 3 SELF-DISCOVER | 96.69% | 96.74% | +0.05 pp | 13 | 7 | [-0.02, 0.12] pp | 0.263 | 0.526 |
| Mixed Emotion | Llama 2 CoT v2 | 81.33% | 85.33% | +4.00 pp | 18 | 6 | [1.00, 7.33] pp | 0.0227 | 0.0680 |
| Mixed Emotion | Llama 3 SELF-DISCOVER | 81.33% | 87.33% | +6.00 pp | 18 | 0 | [3.33, 8.67] pp | <0.0001 | <0.0001 |

## 4. 결과 해석

1. Reddit에서는 Llama 2와 Llama 3의 변화량이 매우 작고 두 paired 95% confidence interval이 0을 포함한다. 따라서 일반 Reddit held-out sample에서 큰 accuracy improvement가 확인되었다고 주장하지 않는다.
2. Mixed Emotion에서는 두 reasoner 모두 양의 변화량을 보였고 paired 95% confidence interval이 0보다 크다.
3. 네 비교를 Holm 보정하면 Llama 3의 Mixed Emotion 개선만 0.05 기준에서 명확히 유지된다.
4. Llama 2의 Mixed Emotion 결과는 effect size가 `+4.00 pp`로 긍정적이지만 Holm-adjusted p-value가 `0.068`이므로 confirmatory evidence가 아니라 exploratory evidence로 해석한다.
5. 최종 논문 주장은 “Phase 2가 모든 입력에서 accuracy를 크게 높인다”가 아니라, “confidence routing은 어려운 sample을 집중시키며, Phase 2의 실제 효과는 reasoner와 input regime에 따라 달라진다”로 제한한다.

## 5. 논문에서의 보고 원칙

- accuracy change만 제시하지 않고 `corrected`, `introduced`, `net corrections`를 함께 보고한다.
- Reddit 결과는 작은 효과와 불확실성을 그대로 기술한다.
- Mixed Emotion은 controlled stress-test 결과이며 일반 Reddit 분포 전체에 대한 동일한 개선을 의미하지 않는다.
- Llama 3 Mixed Emotion 결과를 가장 강한 paired evidence로 보고한다.
- Llama 2 Mixed Emotion 결과는 긍정적이지만 Holm 보정 후 0.05를 넘으므로 탐색적으로 기술한다.

## 6. 재현 방법

저장소 root에서 아래 명령을 실행한다.

```bash
python3 scripts/paired_end_to_end_analysis.py \
  --reddit-end-to-end /path/to/reddit_test_end_to_end_results.csv \
  --mixed-phase1 /path/to/phase1_mixed_emotion_predictions.csv \
  --mixed-phase2 /path/to/phase2_llm_reasoning_model_specific_final_combined_outputs.csv
```

기본 출력 위치는 `reports/statistics/`이며 다음 파일이 생성된다.

- `paired_end_to_end_statistics.csv`
- `paired_end_to_end_statistics.json`

Reddit 입력 파일에 12,001 rows가 있는 경우 스크립트는 논문의 정확한 12,000-row protocol과 맞추기 위해 class별 4,000 rows를 deterministic하게 선택한다.
