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

1. Reddit held-out / Llama 2 CoT
2. Reddit held-out / Llama 3 SELF-DISCOVER
3. Mixed Emotion / Llama 2 CoT
4. Mixed Emotion / Llama 3 SELF-DISCOVER

Holm 보정은 여러 검정을 동시에 수행할 때 우연히 작은 p-value가 발생할 가능성을 줄인다. 난수 시드는 `20260813`로 고정하였다.

## 3. 확정 결과

| Dataset | Reasoner | Phase 1 | End-to-end | Change | Corrected | Introduced | Paired 95% CI | Exact p | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reddit held-out | Llama 2 CoT | 96.69% | 96.67% | -0.03 pp | 47 | 50 | [-0.18, 0.13] pp | 0.8392 | 0.8392 |
| Reddit held-out | Llama 3 SELF-DISCOVER | 96.69% | 96.94% | +0.25 pp | 42 | 12 | [0.13, 0.38] pp | 0.000052 | 0.000156 |
| Mixed Emotion | Llama 2 CoT | 81.33% | 85.33% | +4.00 pp | 18 | 6 | [1.00, 7.33] pp | 0.0227 | 0.0453 |
| Mixed Emotion | Llama 3 SELF-DISCOVER | 81.33% | 87.33% | +6.00 pp | 18 | 0 | [3.33, 8.67] pp | 0.0000076 | 0.000031 |

## 4. 결과 해석

1. Reddit에서 Llama 2의 변화량은 작고 paired 95% confidence interval이 0을 포함하므로 Phase 1과 구분되는 향상 또는 저하를 주장하지 않는다.
2. Reddit에서 Llama 3는 `+0.25 pp`이며 paired interval 전체가 0보다 크고 Holm 보정 후에도 유의하다. 다만 절대 변화량은 작으므로 대규모 일반 성능 향상으로 과장하지 않는다.
3. Mixed Emotion에서는 두 reasoner 모두 양의 변화량을 보였고 paired 95% confidence interval이 0보다 크다. 두 결과 모두 Holm 보정 후 0.05 기준을 만족한다.
4. 최종 논문 주장은 “Phase 2가 모든 입력에서 accuracy를 크게 높인다”가 아니라, “confidence routing은 어려운 sample을 집중시키며, 원문을 보존한 Phase 2의 효과는 reasoner와 input regime에 따라 달라진다”로 제한한다.

## 5. 논문에서의 보고 원칙

- accuracy change만 제시하지 않고 `corrected`, `introduced`, `net corrections`를 함께 보고한다.
- Reddit Llama 2의 불확실성과 Llama 3의 작지만 양의 paired effect를 구분해 기술한다.
- Mixed Emotion은 controlled stress-test 결과이며 일반 Reddit 분포 전체에 대한 동일한 개선을 의미하지 않는다.
- Mixed Emotion 두 결과와 Reddit Llama 3 결과가 Holm 보정 후 유지됨을 보고하되, controlled stress test와 일반 Reddit 분포를 구분한다.

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

Reddit 입력 파일에 과거 split 반올림으로 12,001 rows가 남아 있는 경우 스크립트는 논문의 정확한 12,000-row protocol과 맞추기 위해 class별 4,000 rows를 deterministic하게 선택한다. 최종 보고 수치는 minimally sanitized original `title + selftext`를 사용한 Phase 2 결과다.
