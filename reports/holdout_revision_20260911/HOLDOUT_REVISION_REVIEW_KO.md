# 9월 11일 추가 평가 반영: 동료 검토용 변경 기록

## 이번 버전의 역할

오늘 먼저 반영한 Phase 1 분류기 수치 및 A100 효율성 결과를 기반으로, 이후 완료된 신규 9,000건 평가를 추가했다. 앞선 12,000건 Reddit, 300건 Mixed Emotion, 분류기 3시드 비교를 덮어쓰지 않았다. 기존 연구의 핵심 주장은 유지하며, 프롬프트 개발에 사용하지 않았던 같은 출처의 표본에서 고정 파이프라인을 평가했다는 근거를 보강했다.

현재 PDF: `CGSLR_IEEE_Access_Holdout_Updated_2026_09_11.pdf`

현재 Overleaf: `CGSLR_IEEE_Access_Overleaf_Holdout_Updated_2026_09_11.zip`

원고: `Paper_20260911_holdout_updated/main.tex`

## 오늘까지 진행한 단계

1. 동료 연구자의 최신 분류기 결과와 common-A100 효율성 결과를 앞선 버전에 반영했다. 비교 정확도는 DistilBERT 96.88±0.04%, Mistral 95.36±0.08%, Llama 2 95.17±0.07%다. 이는 신규 9,000건 결과가 아니며 이번 단계에서 다시 변경하지 않았다.
2. 원래 사용한 120,000건과 정규화 텍스트가 겹치지 않는 후보에서 클래스당 3,000건을 확보했다. 총 9,000건, 표본 내 정규화 중복 0건이다.
3. 기존 실험과 별도 코드·출력 경로에서 operational checkpoint, T, threshold, prompt 정책을 고정하고 Colab 추론을 실행했다. 기존 실험 재학습이나 결과 교체는 하지 않았다.
4. 9,000건 Phase 1 추론, routed 137건에 대한 두 LLM 구성, 최종 집계가 완료되었다.
5. 두 개별 예측 CSV를 내려받아 정답/ID/라우팅/확률/accepted 유지/성능/통계를 전수 재계산했다. 모든 검사 통과. 실행 package manifest hash도 대조했다.
6. 성능·해석·측정 범위에 맞춰 논문을 수정하고 PDF와 Overleaf 패키지를 만들었다. 전체 diff와 남은 검토 목록을 함께 제공한다.

## 핵심 결과

| 구성 | 정확도 | 교정 | 새 오류 | 순교정 | Holm p |
|---|---:|---:|---:|---:|---:|
| Phase 1 | 96.73% | — | — | — | — |
| + Llama 2 CoT | 96.90% | 41 | 26 | +15 | 0.0864 |
| + Llama 3 SELF-DISCOVER | 97.03% | 36 | 9 | +27 | 0.000131 |

Llama 3 정확도 차이의 paired 95% CI는 +0.16~+0.46 pp다. Llama 2는 -0.01~+0.34 pp로 0을 포함한다. 두 구성의 직접 비교 검정은 아니므로 Llama 3가 Llama 2보다 '유의하게 우수'하다고 쓰지 않았다.

## 수정 위치와 전후

### 1. Abstract

이전 끝부분:
> Because prompt development reused the routed Reddit cases, the end-to-end findings are exploratory. The study evaluates proxy emotion labels, not clinical diagnoses; independent data and expert review are needed to establish broader validity.

변경:
> Because prompt development reused the original routed Reddit cases, we additionally evaluated the frozen pipeline on 9,000 previously unused same-source posts. Routing 137 posts (1.52%) raised accuracy from 96.73% to 97.03% with Llama 3, yielding 27 net corrections and a Holm-adjusted p = 0.000131; the Llama 2 gain was not significant.

의도: 원래 분석의 개발 의존성을 지우지 않으면서 추가 검증 성과를 초록에 포함한다. 임상적 주장과 외부 일반화는 여전히 제한한다.

### 2. Study Contributions

세 번째 기여에 신규 holdout이 prompt-development sample 밖에서 고정 파이프라인을 평가한다는 문장을 추가했다. 별도 기여를 과도하게 늘리거나 새로운 알고리즘을 발명한 것처럼 쓰지 않았다.

### 3. Experimental Setup

신규 소절 `Additional Same-Source Holdout Protocol`을 넣었다. 9,000건 구성, 정규화 중복 제외, seed, 고정 checkpoint/T/threshold/prompt, accepted 예측 유지, 파싱 실패 처리, 단일 평가 실행임을 명시했다.

통계 설명은 원래 50,000회 bootstrap/Holm 4개 비교를 유지하고, 신규 평가의 10,000회/Holm 2개 비교를 구분했다. L4 추가 평가와 A100 분류기 benchmark를 서로 다른 측정으로 구분했다.

### 4. Experimental Results

신규 소절 `Additional Same-Source Holdout Results`와 본문 Table 8을 추가했다. 137건 routing, 오류 포착 67/294, accepted risk 2.56%, 전후 정확도, 교정/도입, 신뢰구간과 Holm p를 보고한다.

새 그림을 추가하지 않은 이유: 기존 그림은 원래 실험의 결과를 나타내므로 그대로 유지한다. 신규 결과는 비교표 한 개로 성능과 통계가 함께 읽혀 중복 그림 없이도 충분하다. 기존 그림에 새 표본 수치를 섞지 않았다.

### 5. Interpretation / Limitations

이전:
> Phase 2 generation was not instrumented ... Phase 2 resource measurements ... are the next steps.

변경:
> The additional L4 run records successful Phase 2 generation calls, tokens, and time, but excludes loading and retried work and does not measure energy or an all-input LLM baseline.

이전:
> The next validation should freeze prompts and input handling ... then evaluate once on independent data.

변경:
> The additional 9,000-post holdout evaluates the frozen pipeline on previously unused same-source text and provides separate evidence of improvement, without retroactively changing the status of the original analyses.

의도: 이미 수행한 일을 미완료라고 쓰지 않되 외부/저자 단위 검증까지 완료했다고 확장하지 않는다. Llama 2는 원래 Reddit 및 신규 holdout 모두 '통계적으로 검출되는 개선 없음'으로 표현하여 새 결과와 충돌하지 않도록 했다.

### 6. Conclusion

고정된 구성으로 신규 9,000건 중 1.52%만 재평가하여 유의한 개선을 얻었다는 결론을 추가했다. 기존 핵심인 '어려운 사례를 고르는 것과 실제로 고치는 것은 별개'는 유지했다.

### 7. Appendix F

원래 appendix 번호를 유지하고 F를 새로 추가했다. A6에는 클래스별 F1와 routed accuracy, A6b에는 생성 호출·입력/출력 토큰·시간을 제시했다. 모델 revision과 실행 환경, row seed, 측정 범위도 적었다.

Neutral F1은 두 구성 모두 약간 낮아졌으므로 '모든 클래스 개선'이라고 쓰지 않았다. 모델 family와 reasoning 방식이 함께 달라 SELF-DISCOVER만의 효과라고 주장하지 않았다.

## 그대로 유지한 것

- 기존 operational checkpoint의 96.69%, 기존 Reddit routed 171건 및 12,000건 결과.
- Mixed Emotion 300건의 원래 결과 및 통계.
- 최신 3시드 분류기 비교와 common-A100 benchmark.
- 기존 결과 표, 그림 파일, 참고문헌.
- 임상 진단이 아닌 proxy-label task라는 연구 범위.

## 검증과 기록의 범위

예측 9,000행 및 통계는 직접 재계산했다. 호출 시간/토큰은 Colab 생성 집계표를 확인해 반영했으며, 원시 생성 trace 전체를 재합산하거나 모든 rationale을 평가한 것은 아니다. 이는 연구 수행 결과와 원시 로그 감사 범위를 구분하기 위한 협업 기록이며, 본문에 'AI가 직접 확인했는지' 같은 표현을 넣지는 않았다.

`HOLDOUT_FULL_DIFF.md`에는 본 단계 main.tex의 전체 전후 diff가 있다. `holdout_qa.json`에는 빌드/보존 검사, 별도 결과 검증 JSON에는 개별 예측 재계산 결과가 있다. 기존 분류기 수정 이력은 앞선 Phase 1 efficiency revision 문서에서 확인한다.

## 내일 먼저 확인할 사항

1. 새 결과를 별도 same-source 평가로 배치한 구조와 초록/결론의 강도에 동의하는가?
2. 새로운 평가 집합으로 재조정하지 않았다는 실험 기록과 구성 고정 설명에 동의하는가?
3. 통계 family 분리와 단일 실행 표기가 분명한가?
4. 시간/토큰을 부록 보조 자료로 두고 금전/GPU 절감률 주장은 하지 않는 방향에 동의하는가?
5. 연구자 정보·공개 범위·추가 검토 우선순위를 결정했는가?

자세한 남은 작업은 `REMAINING_REVIEW_KO.md`에 정리했다. 이번 단계에서 원격 Git push는 수행하지 않았다.
