# Routing 오류 농축 및 Phase 2 수정 기회 분석

작성일: 2026-08-15

용도: 공동연구자 회의, 최종 논문 결과 해석, Appendix 계산 근거 공유

## 1. 이번 분석의 위치

이번 작업은 새로운 모델 학습, threshold 재선택, prompt 재튜닝이 아니다. 이미 동결된 아래 결과를 이용한 사후 **설명용 분해 분석**이다.

- Reddit held-out 12,000건, `alpha=0.05`, `tau=0.70`, routed 171건
- Mixed Emotion 300건, 동일한 Reddit-selected temperature와 threshold 적용, routed 44건
- 최종 Llama 2 CoT 및 Llama 3 SELF-DISCOVER end-to-end 결과

따라서 primary routing policy와 최종 정확도는 바뀌지 않는다. 이번 분석의 목적은 다음 두 질문을 더 명확하게 답하는 것이다.

1. Routing이 전체 오류를 얼마나 진하게 모았는가?
2. 고정된 routed subset 안에서 각 reasoner가 가능한 수정 기회를 얼마나 활용했는가?

## 2. 핵심 정의

평가 세트 크기를 `N`, 전체 Phase 1 오류 수를 `E`, routed 수를 `n_R`, routed subset 안의 Phase 1 오류 수를 `E_R`이라고 둔다.

### 2.1 Routed-error enrichment

```text
Error enrichment = (E_R / n_R) / (E / N)
```

전체 데이터의 오류율과 비교해 routed subset의 오류율이 몇 배 높은지 나타낸다. 값이 클수록 적은 샘플을 보내면서 오류가 많은 구간을 잘 골랐다는 뜻이다.

### 2.2 Conditional routing oracle

```text
Conditional oracle accuracy = Phase 1 accuracy + E_R / N
```

현재 고정된 router가 보낸 Phase 1 오류를 Phase 2가 전부 고치고, 기존 정답에는 새 오류를 하나도 만들지 않는다고 가정한 조건부 상한이다. 실제로 학습한 comparator가 아니며, threshold 선택에 사용하지 않는다.

### 2.3 Realized correction opportunity

```text
Opportunity realization = (corrected - introduced) / E_R
```

고정된 routed Phase 1 오류 중 reasoner의 순개선이 차지하는 비율이다. 이 값은 router가 제공한 수정 기회를 reasoner가 얼마나 효과적으로 이용했는지 설명한다.

## 3. 확정 계산 결과

| Dataset | Full error rate | Routed error rate | Error enrichment | Routed Phase 1 acc. | Llama 2 routed acc. | Llama 3 routed acc. | Conditional oracle e2e |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reddit | 3.31% | 50.88% | 15.38x | 49.12% | 47.37% | 66.67% | 97.42% |
| Mixed Emotion | 18.67% | 47.73% | 2.56x | 52.27% | 79.55% | 93.18% | 88.33% |

세부 count는 다음과 같다.

| Dataset | N | E | n_R | E_R | Llama 2 net | Llama 3 net |
|---|---:|---:|---:|---:|---:|---:|
| Reddit | 12,000 | 397 | 171 | 87 | -3 | +30 |
| Mixed Emotion | 300 | 56 | 44 | 21 | +12 | +18 |

수정 기회 실현율은 다음과 같다.

- Reddit Llama 2: `-3 / 87 = -3.45%`
- Reddit Llama 3: `30 / 87 = 34.48%`
- Mixed Emotion Llama 2: `12 / 21 = 57.14%`
- Mixed Emotion Llama 3: `18 / 21 = 85.71%`

## 4. 결과 해석

### 4.1 Routing 자체는 효과가 있었는가

그렇다. Reddit에서는 전체 오류율이 3.31%인데 routed subset 오류율은 50.88%였다. Routed subset은 전체보다 약 15.4배 오류가 진했다. 전체 오류율을 그대로 따르는 무작위 171건이라면 평균 약 5.7개의 오류가 예상되지만, confidence routing은 실제 오류 87건을 포함했다.

Mixed Emotion에서도 전체 오류율 18.67% 대비 routed 오류율이 47.73%로 약 2.56배 높았다. Stress test 자체가 어렵기 때문에 Reddit보다 배수는 작지만, 동일한 Reddit-selected policy가 더 어려운 subset을 골랐다는 방향은 일치한다.

### 4.2 Phase 2 reasoner는 routing 기회를 활용했는가

- Reddit Llama 2는 routed-only accuracy가 49.12%에서 47.37%로 낮아져 routing 기회를 활용하지 못했다.
- Reddit Llama 3는 66.67%로 높였고, full-set에서는 +0.25 pp와 +30 net correction을 만들었다.
- Mixed Emotion Llama 2는 routed-only accuracy를 79.55%로 높였다.
- Mixed Emotion Llama 3는 93.18%로 높였고, fixed routed errors 21건 중 net 18건을 회수했다.

따라서 최종 주장은 `routing success`와 `reasoning success`를 구분한다. Router는 오류를 집중시켰지만, 최종 이득은 reasoner에 따라 달랐다.

### 4.3 효율성은 어디까지 주장할 수 있는가

- Reddit은 98.58%의 held-out posts에 LLM을 호출하지 않았다.
- Mixed Emotion은 85.33%의 examples에 LLM을 호출하지 않았다.

이는 all-input re-evaluation 대비 **호출 수 감소**다. 실제 GPU runtime, 전력, 비용 절감으로 표현하려면 별도의 실행시간 측정이 필요하다.

## 5. 논문 반영 위치

### Main Results

- Table III: Reddit `Error enrichment = 15.38x` 열 추가
- Reddit end-to-end Table V: routed-only `49.12% / 47.37% / 66.67%` 열 추가
- Mixed Emotion Table VI: routed-only `52.27% / 79.55% / 93.18%` 열 추가
- Routing policy 결과 문단: Reddit 15.38배 오류 농축 및 random-size expectation 설명
- Mixed Emotion 문단: 2.56배 오류 농축 설명
- Interpretation 문단: 호출 감소와 conditional oracle 해석

### Appendix

- Appendix Table A4g: error enrichment, routed-only accuracy, conditional oracle를 한 표에 정리
- 계산식과 count를 함께 제시해 본문 수치를 재현 가능하게 함

## 6. 이 분석으로 바뀌지 않는 것

- `alpha=0.05`, `tau=0.70`
- Reddit routed 171건
- Mixed Emotion routed 44건
- Reddit/Mixed Phase 1 및 end-to-end accuracy
- corrected, introduced, net correction
- paired bootstrap, McNemar, Holm 결과
- frozen prompt와 original-text input policy

## 7. 남은 작업

### 제출 전 필수

1. Mistral 7B supervised Phase 1 classifier 완료
2. Llama 2 7B supervised Phase 1 classifier 완료
3. Table I 및 관련 본문에 matched classifier 결과 반영
4. Architecture 및 workflow figure 삽입
5. 저자, 소속, corresponding author, funding, conflict, data/code availability 확정

### 권장 보강

1. Frozen prompt를 독립 routed subset 또는 outer fold에서 확인
2. Mixed Emotion 및 대표 Reddit 사례의 복수 연구자 또는 전문가 검토
3. GPU, reasoner별 runtime, 생성 token, 호출 수를 기록해 실제 효율 분석 추가

## 8. 회의에서 사용할 한 문장

> Calibration-selected routing은 Reddit에서 전체의 1.42%만 Phase 2로 보내면서 전체보다 약 15.4배 오류가 진한 subset을 만들었고, Llama 3는 그 고정된 subset의 정확도를 49.12%에서 66.67%로 높였다. Mixed Emotion에서는 routed-only accuracy가 52.27%에서 93.18%로 상승했지만, Llama 2의 Reddit 결과처럼 routing 성공이 모든 reasoner의 성공을 보장하지는 않았다.
