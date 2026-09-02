# 최종 연구 진행 현황 및 남은 작업

최종 갱신: 2026-09-03

용도: 공동연구자 회의, 교수 검토, IEEE Access 제출 준비

대상 연구: *Confidence-Guided Selective LLM Re-Evaluation for Depression-Risk-Related Emotion Classification in Social Media Text*

> **이 문서가 현재 연구 현황의 단일 기준 문서다.**
>
> 과거 prompt 버전, threshold 민감도 실험, cleaned-text Reddit 실행은 재현 이력으로 보존하지만 최종 보고 수치는 아래 동결 결과를 따른다.

## 1. 현재 상태

- Reddit 120,000건의 Phase 1 비교, calibration, selective routing, Reddit/Mixed Emotion Phase 2, 통계 검정, accepted-error audit를 완료했다.
- DistilBERT, Llama 2 7B, Mistral 7B의 **동일 12,000건 held-out test, 3개 seed 비교**를 완료했다.
- Llama 2와 Mistral은 LoRA/QLoRA 또는 전체 fine-tuning이 아니라 **frozen backbone + bias-free linear probe**로 평가했다.
- 최종 architecture vector figure와 편집 가능한 Draw.io 원본을 확정하고 원고에 반영했다.
- 공식 IEEE Access 템플릿 원고는 결과·표·그림·Appendix·참고문헌 감사까지 동기화했다.
- 남은 필수 작업은 저자/소속 등 투고 메타데이터, 공개 범위 확정, Overleaf pdfLaTeX clean compile 및 최종 source/PDF 동결이다.

## 2. 완료된 실험 단계

1. Reddit 원자료에서 클래스별 40,000건, 총 120,000건 구성
2. 클래스별 train/validation/calibration/test = 70/10/10/10 분리
3. 세 Phase 1 모델의 hyperparameter 탐색과 3-seed 평가
4. DistilBERT 최종 operational checkpoint 저장
5. 별도 calibration split에서 temperature scaling 수행
6. Calibration split만 사용해 risk-constrained routing threshold 선택
7. Reddit held-out 12,000건 Phase 1 및 routing 평가
8. Mixed Emotion 300건을 학습·튜닝에 사용하지 않는 stress test로 평가
9. Routed sample만 Llama 2 CoT와 Llama 3 SELF-DISCOVER로 재평가
10. Reddit routed 171건은 minimally sanitized original `title + selftext`로 재실행
11. Accepted sample은 Phase 1 label을 유지하고 routed sample만 parsed Phase 2 label로 교체
12. Full-set accuracy, macro P/R/F1, corrected/introduced/net corrections, confusion matrix 산출
13. Paired bootstrap, exact McNemar, Holm correction 수행
14. Accepted high-confidence Phase 1 오류 310건 감사 및 대표 원문 사례 분석
15. 논문 본문, Appendix, 그림, 참고문헌, 실행 문서와 재현 artifact 정리

## 3. Phase 1 분류기 비교

### 3.1 공통 평가 조건

| 항목 | 설정 |
|---|---|
| 데이터 | Reddit class-balanced 120,000건, 클래스별 40,000건 |
| 분할 | 70/10/10/10, test 12,000건 |
| 평가 seed | 42, 43, 44 |
| 주요 지표 | Accuracy, macro Precision, macro Recall, macro F1 |
| 모델 선택 | Validation macro F1을 목표로 한 Bayesian sweep |
| 해석 범위 | 동일 데이터·분할 아래의 bounded operational screening comparison |

### 3.2 3-seed held-out 결과

| Phase 1 model | Training regime | Accuracy, mean +/- SD | Macro F1, mean +/- SD | 95% CI, accuracy |
|---|---|---:|---:|---:|
| DistilBERT | Full fine-tuning | **96.70 +/- 0.10%** | **96.70 +/- 0.10%** | [96.45, 96.95] |
| Mistral 7B | Frozen backbone + linear probe | 95.56 +/- 0.11% | 95.56 +/- 0.11% | [95.29, 95.83] |
| Llama 2 7B | Frozen backbone + linear probe | 95.09 +/- 0.06% | 95.09 +/- 0.06% | [94.94, 95.24] |

DistilBERT가 이 사전 명시 프로토콜에서 가장 높은 평균 성능을 보여 operational Phase 1 모델로 채택되었다. 이 결과는 fully optimized 7B 모델 전체에 대한 보편적 우위 주장이 아니다. 7B 비교기는 backbone을 동결하고 4,096 x 3, bias-free 분류 헤드 12,288개 파라미터만 학습한 bounded comparator다.

### 3.3 선택된 학습 설정

| Model | Selected learning rate | Batch | Epochs | Weight decay | Trainable regime |
|---|---:|---:|---:|---:|---|
| DistilBERT operational checkpoint | 8.996e-5 | 32 | 3 | 1e-2 | Full fine-tuning |
| Mistral 7B | 2.292e-4 | 16 | 2 | 1e-2 | Frozen backbone, linear probe |
| Llama 2 7B | 2.824e-4 | 64 | 3 | 1e-4 | Frozen backbone, linear probe |

- Llama 2와 Mistral은 공통으로 learning rate `1e-4--1e-2`, batch `[16, 32, 64]`, epochs `[2, 3]`, weight decay `[1e-2, 1e-3, 1e-4]`에서 Bayesian 4회 탐색했다.
- Mistral은 탐색 후보 중 epoch 2가, Llama 2는 epoch 3이 validation macro F1 기준으로 선택된 것이다.
- 두 7B 모델의 서로 다른 최적 batch/epoch는 비교 결함이 아니라 동일 탐색 규칙을 모델별로 적용한 결과다.
- 모델 선택 근거는 이 프로토콜에서의 정확도와 실제 운영 규모를 함께 고려한 것이다. 7B 모델의 최대 성능 자체를 규명하는 것이 본 연구의 목적은 아니다.

## 4. Calibration과 routing 결과

### 4.1 DistilBERT operational checkpoint

3-seed 평균은 모델 비교용 결과이고, 아래 값은 이후 routing과 Phase 2에 사용한 고정 operational checkpoint의 결과다.

| 항목 | 결과 |
|---|---:|
| Reddit held-out accuracy | 96.69% |
| Macro Precision / Recall / F1 | 96.691% / 96.692% / 96.692% |
| Temperature `T*` | 1.7706 |
| NLL, raw -> scaled | 0.1411 -> 0.1041 |
| Brier, raw -> scaled | 0.0559 -> 0.0514 |
| ECE, raw -> scaled | 0.0244 -> 0.0083 |
| Adaptive ECE, raw -> scaled | 0.0244 -> 0.0138 |

Temperature scaling은 class argmax를 바꾸기보다 confidence의 과도한 확신을 완화했다. 네 calibration 지표가 모두 감소했으므로 raw MSP가 아니라 temperature-scaled MSP를 routing score로 사용했다.

### 4.2 고정 routing policy

| 항목 | 결과 |
|---|---:|
| Risk budget `alpha` | 0.05 |
| Candidate threshold grid | 0.70--1.00, step 0.01 |
| Selected threshold `tau*` | 0.70 |
| Reddit routed | 171 / 12,000 (1.42%) |
| Reddit coverage | 98.58% |
| Routed Phase 1 errors | 87 / 전체 397 errors |
| Error capture | 21.91% |
| Routed Phase 1 accuracy | 49.12% |
| Accepted errors / selective risk | 310 / 2.62% |
| Routed-error enrichment | 15.38x |

Threshold는 test 또는 Phase 2 결과를 보고 고른 값이 아니다. Calibration split에서 one-sided risk upper bound가 `alpha=5%` 이하인 후보 중 coverage가 가장 큰 값을 선택했고, 이후 Reddit test와 Mixed Emotion에 고정했다.

## 5. End-to-end 결과

### 5.1 Reddit original-text evaluation

| System | Full-set accuracy | Change | Corrected | Introduced | Net |
|---|---:|---:|---:|---:|---:|
| DistilBERT Phase 1 | 96.69% | - | - | - | 0 |
| + Llama 2 CoT | 96.67% | -0.03 pp | 47 | 50 | -3 |
| + Llama 3 SELF-DISCOVER | **96.94%** | **+0.25 pp** | 42 | 12 | **+30** |

- Llama 2 변화는 Phase 1과 통계적으로 구분되지 않았다.
- Llama 3 paired bootstrap 95% CI는 `[+0.13, +0.38]` pp다.
- Llama 3 exact McNemar `p ~= 0.000052`, Holm-adjusted `p ~= 0.000156`이다.
- Routed-only accuracy는 Phase 1 49.12%, Llama 2 47.37%, Llama 3 66.67%다.
- Llama 3 fixed-routing oracle ceiling은 97.42%이며, routed correction opportunity의 34.48%를 순개선으로 실현했다.

### 5.2 Mixed Emotion stress test

| System | Full-set accuracy | Change | Corrected | Introduced | Net |
|---|---:|---:|---:|---:|---:|
| DistilBERT Phase 1 | 81.33% | - | - | - | 0 |
| + Llama 2 CoT | 85.33% | +4.00 pp | 18 | 6 | +12 |
| + Llama 3 SELF-DISCOVER | **87.33%** | **+6.00 pp** | 18 | 0 | **+18** |

- 300건 중 44건(14.67%)이 routed되었다.
- Routed-only accuracy는 Phase 1 52.27%, Llama 2 79.55%, Llama 3 93.18%다.
- Llama 2 paired CI는 `[+1.00, +7.33]` pp, Holm-adjusted p-value는 약 0.0453이다.
- Llama 3 paired CI는 `[+3.33, +8.67]` pp, Holm-adjusted p-value는 0.0001 미만이다.
- Llama 3 fixed-routing oracle ceiling은 88.33%이며 correction opportunity의 85.71%를 실현했다.
- Mixed Emotion은 실제 prevalence 추정용 자료가 아니라 mixed cue와 emotional trajectory를 통제한 300건 supplementary stress test다.

## 6. 현재 논문의 핵심 주장

1. Calibrated confidence는 전체 입력에 LLM을 적용하지 않고 Phase 1 오류가 농축된 작은 subset을 선별할 수 있다.
2. Routing의 성공과 Phase 2 correction의 성공은 별개의 문제이며, reasoner 선택과 입력 보존이 최종 결과를 좌우한다.
3. Llama 3 SELF-DISCOVER는 Reddit과 Mixed Emotion 모두에서 introduced error를 제한하며 양의 순개선을 보였다.
4. Llama 2 결과를 함께 보고함으로써 LLM re-evaluation이 자동으로 성능을 높인다는 과장을 피한다.
5. Mixed/trajectory stress test에서 reasoning 이점이 더 크게 관찰되었지만, synthetic test를 실제 임상·유병률 성능으로 일반화하지 않는다.
6. 이 연구는 임상 진단이 아니라 subreddit-derived proxy emotion classification과 selective text re-evaluation 연구다.
7. 효율 근거는 routed fraction과 LLM invocation avoidance이며, wall-clock time, 비용, energy를 실측한 것으로 주장하지 않는다.

## 7. 2026-09-03까지 완료한 원고·재현성 작업

- RQ1을 `RQ1a: operational predictive/calibration quality`와 `RQ1b: bounded architecture comparison`으로 명확히 분리했다.
- Abstract, Introduction, Methods, Results, Discussion, Limitations, Conclusion의 Phase 1 실측 수치를 동기화했다.
- Table I과 Phase 1 비교 Figure의 placeholder를 3-seed 실측값으로 교체했다.
- 세 모델의 비교 목적, 학습 범위, trainable parameter 차이와 operational model 선택 근거를 명시했다.
- Llama 2/Mistral을 exhaustive fine-tuning 결과로 오해하지 않도록 frozen linear-probe 범위를 분명히 했다.
- Calibration, routing, Reddit/Mixed end-to-end, paired statistics와 accepted-error audit 결과를 본문과 Appendix에 정렬했다.
- Corrected error만이 아니라 introduced error와 net correction을 함께 보고했다.
- 최종 architecture figure를 벡터 PDF로 삽입하고 editable Draw.io 원본을 보존했다.
- SELF-DISCOVER 상세 도식은 발명 기여로 오해되지 않도록 Appendix에 배치했다.
- Reader-facing 명칭에서 내부 `v2`, `v2.1` 표기를 제거하고 Llama 2 CoT, Llama 3 SELF-DISCOVER로 통일했다.
- Reference 46건 중 39건을 full text 또는 동등한 source로 감사했다. `[9], [20], [22], [27], [31], [38], [45]`는 초록·서지 기반 상태이며 `[22]`, `[31]`이 추가 전문 확인 우선순위다.
- LaTeX reference, table/figure width, float/page break, caption spacing, Appendix 배치와 전 페이지 시각 QA를 수행했다.
- 교수 검토용 PDF, Overleaf ZIP과 SHA-256 manifest를 생성했다.

## 8. 제출 전 남은 작업

### 8.1 필수

1. 최종 저자 순서, 영문 이름, 소속, 이메일, corresponding author 확정
2. ORCID, 저자 약력, funding/Acknowledgment, conflict of interest, AI-use disclosure 입력
3. Data/code availability 문구와 공개 artifact, license, release tag/DOI 범위 확정
4. 실제 제출 메타데이터 반영 후 모든 placeholder 재검색
5. 새 Overleaf 프로젝트에서 공식 IEEE Access `pdfLaTeX` clean compile
6. 최종 source, PDF, supplementary artifact, run manifest의 내용과 checksum 동결

### 8.2 권장 보강

1. Frozen prompt를 prompt 개발에 사용하지 않은 독립 subset에서 확인
2. Mixed Emotion label/rationale와 대표 Reddit 사례에 대한 복수 연구자 또는 전문가 검토
3. GPU, wall-clock time, generation token, 실제 비용의 별도 효율 benchmark
4. 접근 제한 참고문헌 `[22]`, `[31]`의 전문 확보

권장 항목은 연구를 강화하지만 현재 핵심 파이프라인을 다시 설계해야 하는 필수 조건은 아니다. 수행하지 않을 경우 현재 Limitations의 범위 제한을 유지한다.

## 9. 다음 회의에서 결정할 사항

| 결정 | 필요한 결론 |
|---|---|
| 저자·소속 | 순서, corresponding author, ORCID |
| 제출 공개 문구 | Funding, conflict, AI use, ethics/IRB 문구 |
| Artifact 공개 | 코드, Mixed Emotion, 결과 CSV/JSON, 모델 checkpoint의 공개 범위 |
| 권장 실험 | Independent prompt confirmation과 expert review의 이번 논문 포함 여부 |
| 효율 보고 | Invocation avoidance만 유지할지 runtime/cost benchmark를 추가할지 |
| 제출 일정 | 교수 1차 검토 종료일과 IEEE Access 제출일 |

## 10. 기준 파일

### 실행 노트북

- `notebooks/colab/final/01_distilbert_phase1_training_final_colab.ipynb`
- `notebooks/colab/final/02_2_llm_phase2_reasoning_model_specific_prompt_final_colab.ipynb`
- `notebooks/colab/final/03_mixed_emotion_end_to_end_orchestration_final_colab.ipynb`
- `notebooks/colab/final/04_5_reddit_test_routed_phase2_original_text_primary_final_colab.ipynb`

### 주요 문서와 artifact

- End-to-end 실행: `docs/final_end_to_end_workflow_ko.md`
- Threshold 근거: `docs/routing_threshold_policy_audit_ko.md`
- Paired statistics: `docs/paired_end_to_end_statistical_analysis_ko.md`
- Accepted-error audit: `docs/high_confidence_accepted_error_audit_results_ko.md`
- Prompt policy: `docs/final_model_specific_prompt_policy_ko.md`
- 원고 논리 감사: `docs/manuscript_logic_and_model_description_revision_2026_08_31_ko.md`
- Architecture 원본: `docs/figures/architecture/confidence_guided_two_phase_architecture.drawio`
- Architecture 벡터: `docs/figures/architecture/confidence_guided_two_phase_architecture.pdf`
- 제출 체크리스트: `docs/ieee_access_submission_readiness_checklist_ko.md`

새 실험이나 공동 결정이 생기면 이 문서를 먼저 갱신하고 README, 체크리스트, 원고를 같은 커밋에서 동기화한다.
