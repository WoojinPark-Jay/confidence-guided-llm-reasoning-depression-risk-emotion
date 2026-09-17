# 최종 Phase 1 기반 재평가 실험 결과 요약

2026-09-18 | 공동연구자 검토용

## 1. 비교 범위

선택된 하이퍼파라미터의 최종 DistilBERT 출력에 동일한 routing 정책을 적용하고, 전달된 사례를 LLM으로 재평가했다. 이전 operational checkpoint 결과와 이번 결과는 섞지 않는다. 아래 정확도는 **routed 사례만이 아니라 전체 평가 데이터 기준**이다.

| 데이터 | 전체 건수 | Phase 1 정확도 | LLM 전달 건수 | 전달 비율 | Phase 2 입력 |
|---|---:|---:|---:|---:|---|
| Reddit | 12,000 | 96.9167% | 218 | 1.82% | 원문 |
| Mixed Emotion | 300 | 83.6667% | 86 | 28.67% | 합성 사례의 원래 텍스트 |

- 최종 DistilBERT: reference seed 42 모델, learning rate 4.0368e-5, batch 64, epochs 설정 3, weight decay 0.001. 테스트 성능으로 최고 seed를 고른다는 의미는 아니다.
- 고정 temperature scaling 값 T = 1.480195, routing threshold = 0.70.
- Reddit은 Phase 1에서 정제 텍스트, Phase 2에서 원문을 사용한다. Mixed Emotion은 두 단계에서 같은 원래 텍스트를 사용하며, Reddit식 정제 실험이 아니다.
- Llama 3는 Direct / CoT / SELF-DISCOVER를 비교했다. Direct는 명시적인 단계별 추론을 요구하지 않는 직접 분류 조건이다.
- Llama 2는 Llama 3 CoT와 지침을 맞춘 CoT 조건이다. 두 모델 모두 greedy decoding이지만 최대 생성 토큰은 Llama 2 512, Llama 3 1,024로 다르므로 완전히 동일한 생성 예산 비교는 아니다.

## 2. 주요 결과: 원래 텍스트 재평가

수정 = Phase 1 오답을 정답으로 변경, 신규 오류 = Phase 1 정답을 오답으로 변경. 순수정은 두 수의 차이이며, 정확도 변화는 Phase 1 대비 percentage point(pp)다.

| 데이터 | 재평가 방법 | 최종 정확도 | Macro F1 | 정확도 변화 | 수정 | 신규 오류 | 순수정 |
|---|---|---:|---:|---:|---:|---:|---:|
| Reddit | Llama 2 CoT matched | 96.9417% | 96.9404% | +0.0250 pp | 49 | 46 | +3 |
| Reddit | Llama 3 Direct | 97.1083% | 97.1056% | +0.1917 pp | 68 | 45 | +23 |
| Reddit | Llama 3 CoT | **97.1500%** | **97.1471%** | **+0.2333 pp** | 63 | 35 | +28 |
| Reddit | Llama 3 SELF-DISCOVER | 97.0417% | 97.0402% | +0.1250 pp | 41 | 26 | +15 |
| Mixed Emotion | Llama 2 CoT matched | 79.0000% | 78.9298% | -4.6667 pp | 19 | 33 | -14 |
| Mixed Emotion | Llama 3 Direct | 89.3333% | 89.3803% | +5.6667 pp | 26 | 9 | +17 |
| Mixed Emotion | Llama 3 CoT | 84.6667% | 84.6760% | +1.0000 pp | 15 | 12 | +3 |
| Mixed Emotion | Llama 3 SELF-DISCOVER | **90.3333%** | **90.5072%** | **+6.6667 pp** | 22 | 2 | +20 |

**집계 주의:** Reddit Llama 3는 저장된 생성 결과에서 최종 라벨을 재파싱하고, 라벨이 없는 출력은 Phase 1 예측을 유지하는 fallback을 적용한 집계다. 정답을 보고 라벨을 선택한 것이 아니다. 최초 Colab 출력의 순수정(+22 / +25 / +14)과 구별해야 한다. 재파싱 후 라벨 미추출은 Direct 1건, CoT 1건, SELF-DISCOVER 0건이다. 실패 사례를 평가에서 제외하지 않는다.

Llama 2 matched와 Mixed Emotion 수치는 완료된 Colab 요약 기준이다. 이 조건들은 보고된 파싱 실패가 모두 0건이다. Reddit Llama 3 외 조건의 생성 원시 로그 전체 대조는 별도 마무리 항목으로 남아 있다. 표의 굵은 값은 관측된 최고값이지, 방법 간 통계적으로 유의한 우위라는 뜻은 아니다.

## 3. 추가 확인: Reddit 정제 텍스트로 재평가

동일한 218건에 대해 Llama 3의 입력을 저장된 Phase 1 정제 텍스트로 바꿨다. **현재 논문에 추가하는 결과가 아니라 내부 검토용 참고 실험**으로 구분한다.

| 방법 | 최종 정확도 | Macro F1 | 정확도 변화 | 수정 | 신규 오류 | 순수정 | 파싱 실패 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 96.8500% | 96.8453% | -0.0667 pp | 50 | 58 | -8 | 0 |
| CoT | 96.8250% | 96.8192% | -0.0917 pp | 45 | 56 | -11 | 1 |
| SELF-DISCOVER | 96.9167% | 96.9152% | 0.0000 pp | 12 | 12 | 0 | 0 |

완료된 Colab 요약 기준이며, CoT 실패 1건의 원문 로그는 추가 확인 대상이다. 이 조건에서는 세 방법 모두 원문 입력보다 정확도가 낮았다. 다만 이것만으로 모든 데이터에서 정제 텍스트 재평가가 불가능하다고 일반화하지 않는다. 저장된 텍스트가 같더라도 Phase 1의 256-token 입력과 LLM 입력의 토큰 범위는 동일하지 않다.

이전에 수행한 Llama 2 sampling CoT는 Reddit 정확도 96.8500%, 수정 54건, 신규 오류 62건, 순수정 -8건이었다. 이는 temperature 0.6 / top_p 0.9 / 최대 256토큰 등의 이전 조건으로, 위의 matched CoT(+3건)와 별개이며 현재 주요 비교는 matched 결과를 사용한다.

## 4. 현재 해석

1. **Llama 3의 세 방법 모두 원래 텍스트 입력에서는 두 데이터셋의 Phase 1 정확도를 개선했다.** 단, 최고 방법은 Reddit에서 CoT, Mixed Emotion에서 SELF-DISCOVER로 다르다.
2. **SELF-DISCOVER는 Reddit에서 세 방법 중 개선 폭이 가장 작지만, 성능을 악화시킨 것은 아니다.** 순수정 +15건이다. Mixed Emotion에서는 신규 오류가 2건으로 적고 순수정 +20건으로 가장 크다.
3. **Direct도 중요한 비교 기준이다.** 두 데이터셋에서 개선되므로, 명시적 추론 절차 자체의 추가 이득은 Direct 대비로 판단해야 한다. LLM 재평가의 이득과 특정 추론 방식의 우위를 구분한다.
4. **Llama 2 CoT는 Reddit에서 소폭 개선, Mixed Emotion에서 악화했다.** 재평가가 언제나 도움이 되는 것은 아니며 모델과 방법의 선택이 중요하다는 결과다. Llama 2의 다른 추론 방법까지 검증한 것은 아니다.
5. 이전 checkpoint 실험과 이번 실험은 전달 사례가 달라졌다(Reddit 171→218건, Mixed Emotion 44→86건). 따라서 이전 SELF-DISCOVER의 순위가 그대로 유지되어야 하는 것은 아니다. 순위 변화의 구체적 원인은 사례별 비교 전에는 단정하지 않는다.

## 5. 논문 반영 전 마무리

- 생성 로그를 전체 예측과 ID로 대조하고, 최종 라벨 파싱·fallback·입력 및 출력 잘림을 동일 기준으로 확인한다. 요약값만으로 확인되지 않는 항목을 마무리한다.
- 확정된 전체 예측으로 paired bootstrap CI, exact McNemar 검정, 명시된 비교군에 대한 Holm 보정을 다시 계산한다. Phase 1 대비 개선과 방법 간 차이를 구분한다.
- 논문은 공통된 선택적 재평가 파이프라인과 데이터셋별 관측 결과를 중심으로 정리한다. 한 방법의 보편적 우위나 데이터 특성이 순위 차이를 일으켰다는 인과적 결론은 현재 결과만으로 주장하지 않는다.

## 결과 위치

Google Drive의 `MyDrive/confidence_guided_llm_reasoning/outputs_final/unified_final/` 아래:

| 조건 | 결과 폴더 |
|---|---|
| Reddit / Llama 2 matched | `phase2/llama2_cot_matched_llama3` |
| Reddit / Llama 3 원문 | `phase2/llama3_reasoning_comparison` |
| Mixed Emotion / Llama 2 matched | `mixed_emotion/phase2/llama2_cot_matched_llama3` |
| Mixed Emotion / Llama 3 | `mixed_emotion/phase2/llama3_reasoning_comparison` |
| Reddit / Llama 3 정제 텍스트 | `reddit_cleaned/phase2/llama3_reasoning_comparison` |

Reddit Llama 3 표는 위 폴더의 최초 요약표가 아니라 재파싱·fallback 후 수치다. 논문용 최종 결과 파일을 만들 때 이 집계 기준도 함께 고정한다.
