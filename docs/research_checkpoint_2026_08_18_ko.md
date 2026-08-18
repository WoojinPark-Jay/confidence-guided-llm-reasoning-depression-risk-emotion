# 연구 진행 체크포인트

최종 갱신: 2026-08-18  
대상 연구: Confidence-Guided Selective LLM Re-Evaluation for Depression-Risk-Related Emotion Classification

> 이 문서는 현재 확정된 결과, 최근 논문 보강 내용, 제출 전 남은 작업을 한 번에 확인하기 위한 공동연구 기준 문서다.

## 1. 현재 상태 한 줄 요약

DistilBERT 기반 Phase 1, calibration 기반 selective routing, Llama 2/Llama 3 Phase 2, Reddit 및 Mixed Emotion end-to-end 평가와 통계 검증까지 완료했다. 현재 가장 큰 미완료 항목은 Mistral 7B와 Llama 2 7B의 matched Phase 1 classifier baseline, 최종 도식, 투고 메타데이터다.

## 2. 완료된 전체 파이프라인

1. Reddit 데이터 클래스별 40,000건, 총 120,000건 구성
2. Train / validation / calibration / test = 70 / 10 / 10 / 10 분리
3. W&B sweep과 best hyperparameter 선택
4. Final DistilBERT 학습과 checkpoint 저장
5. Calibration split에서 temperature scaling 수행
6. Temperature-scaled MSP와 risk constraint로 routing threshold 선택
7. Reddit held-out test 12,000건 Phase 1 평가
8. Mixed Emotion 300건 external stress-test 평가
9. Routed sample만 Llama 2 CoT와 Llama 3 SELF-DISCOVER로 재평가
10. Accepted sample은 Phase 1 label을 유지하고 routed sample만 Phase 2 label로 교체
11. Full-set accuracy, corrected/introduced/net correction, confusion matrix 계산
12. Paired bootstrap, exact McNemar, Holm correction 수행
13. High-confidence accepted error 감사와 실제 원문 대표 사례 분석

## 3. 확정된 핵심 결과

### 3.1 Phase 1와 calibration

| 항목 | 확정 결과 |
|---|---:|
| Reddit held-out test | 12,000건 |
| DistilBERT accuracy | 96.69% |
| Macro Precision / Recall / F1 | 96.691% / 96.692% / 96.692% |
| Optimal temperature | 1.7706 |
| NLL, raw -> scaled | 0.1411 -> 0.1041 |
| Brier, raw -> scaled | 0.0559 -> 0.0514 |
| ECE, raw -> scaled | 0.0244 -> 0.0083 |
| Adaptive ECE, raw -> scaled | 0.0244 -> 0.0138 |
| Risk budget | alpha = 0.05 |
| Primary threshold | tau = 0.70 |

Temperature scaling은 class prediction 자체를 바꾸기보다 confidence의 과도한 확신을 완화했다. NLL, Brier, ECE, Adaptive ECE가 모두 감소했으므로 raw softmax보다 calibrated probability를 routing에 사용하는 근거가 확보됐다.

### 3.2 Reddit routing

| 항목 | 결과 |
|---|---:|
| Routed samples | 171 / 12,000 (1.42%) |
| Coverage | 98.58% |
| Routed Phase 1 errors | 87 |
| 전체 Phase 1 errors | 397 |
| Error capture | 21.91% |
| Routed Phase 1 accuracy | 49.12% |
| Routed error rate | 50.88% |
| Overall error rate | 3.31% |
| Error enrichment | 15.38x |
| Accepted errors / selective risk | 310 / 2.62% |

전체의 1.42%만 LLM으로 보내면서 Phase 1 오류의 21.91%를 포착했다. Routed subset의 오류율이 전체보다 15.38배 높으므로 routing은 무작위 전송이 아니라 오류가 집중된 표본을 선별했다.

### 3.3 Reddit original-text end-to-end

| System | Full-set accuracy | Change | Corrected | Introduced | Net |
|---|---:|---:|---:|---:|---:|
| DistilBERT Phase 1 | 96.69% | - | - | - | 0 |
| + Llama 2 CoT | 96.67% | -0.03 pp | 47 | 50 | -3 |
| + Llama 3 SELF-DISCOVER | 96.94% | +0.25 pp | 42 | 12 | +30 |

- Llama 2의 변화는 통계적으로 Phase 1과 구분되지 않았다.
- Llama 3 paired bootstrap 95% CI는 +0.13~+0.38 pp다.
- Llama 3 exact McNemar p-value는 약 0.000052, Holm-adjusted p-value는 약 0.000156이다.
- Llama 3 routed-only accuracy는 66.67%로 Phase 1의 49.12%보다 높았다.
- 최종 Reddit Phase 2는 cleaned text가 아니라 최소 비식별 처리된 original title + selftext를 사용했다.

### 3.4 Mixed Emotion end-to-end

| System | Full-set accuracy | Change | Corrected | Introduced | Net |
|---|---:|---:|---:|---:|---:|
| DistilBERT Phase 1 | 81.33% | - | - | - | 0 |
| + Llama 2 CoT | 85.33% | +4.00 pp | 18 | 6 | +12 |
| + Llama 3 SELF-DISCOVER | 87.33% | +6.00 pp | 18 | 0 | +18 |

- 300건 중 44건(14.67%)이 routed되었다.
- Routed-only accuracy는 Phase 1 52.27%, Llama 2 79.55%, Llama 3 93.18%다.
- Llama 2 paired CI는 +1.00~+7.33 pp, Holm-adjusted p-value는 약 0.0453이다.
- Llama 3 paired CI는 +3.33~+8.67 pp, Holm-adjusted p-value는 0.0001 미만이다.
- Mixed Emotion은 실제 유병률을 추정하는 데이터가 아니라 mixed cue와 emotional trajectory를 통제한 supplementary stress test다.

## 4. 현재 논문의 핵심 주장

1. Calibration 기반 routing은 전체 입력에 LLM을 적용하지 않고 Phase 1 오류가 농축된 표본을 선별할 수 있다.
2. Selective re-evaluation의 효과는 router만으로 결정되지 않으며 reasoner와 입력 보존 방식에 따라 달라진다.
3. Llama 3 SELF-DISCOVER는 Reddit과 Mixed Emotion에서 introduced error를 통제하면서 순개선을 만들었다.
4. Mixed/trajectory 사례에서는 LLM reasoning의 이점이 더 크게 관찰됐다.
5. 이 연구는 임상 진단 시스템이 아니라 subreddit-derived proxy emotion classification과 비임상적 text re-evaluation 연구다.
6. 현재 효율 근거는 LLM 호출 회피율과 routed fraction이다. Wall-clock time, energy, monetary cost를 측정한 것으로 과장하지 않는다.

## 5. 2026-08-18 원고 보강 체크포인트

- Abstract를 약 230단어로 정리했다.
- Introduction에 기존 연구의 공백을 calibration-only selective prediction과 full LLM re-evaluation 사이의 연결 부족으로 명시했다.
- RQ1은 predictive/calibration quality, RQ2는 routing quality, RQ3는 full-set end-to-end correctness change로 평가 근거를 정렬했다.
- Results에서 RQ1~RQ3의 답을 직접 제시했다.
- Corrected errors만 강조하지 않고 introduced errors와 net corrections를 함께 보고하도록 통일했다.
- Interpretability 과장을 줄이고 traceability와 text-grounded rationale 중심으로 표현을 정교화했다.
- Proxy label, clinician validation 부재, synthetic stress-test의 한계를 중복 없이 통합했다.
- 결론을 “calibration은 correction opportunity를 찾고, 실제 개선은 reasoner와 input preservation에 달려 있다”는 구조로 강화했다.
- Figure와 Table의 serif typography, 여백, 축 이름, caption 간격, confusion-matrix 크기를 IEEE 2단 편집 기준에 맞춰 정비했다.
- Reader-facing 명칭에서 내부 prompt 버전 표기를 제거하고 Llama 2 CoT, Llama 3 SELF-DISCOVER로 통일했다.
- Raw MSP Adaptive ECE 누락값을 보완했다.

최신 검토용 Overleaf 패키지:

`Paper_260620_overleaf_content_polished_review_final.zip`

## 6. 재현성과 감사 상태

- Temperature와 threshold는 test set이 아니라 calibration split에서만 선택했다.
- 선택된 temperature와 threshold는 Reddit held-out test와 Phase 2에 고정 적용했다.
- Reddit routed 171건의 original title + selftext 연결을 전수 확인했다.
- Phase 2 결과는 row 단위 CSV로 저장되며 중단 후 이어서 실행할 수 있다.
- Paired bootstrap 50,000회, exact McNemar, Holm multiple-comparison correction을 완료했다.
- Accepted high-confidence error 310건을 감사했고, confidence >= 0.98 오류 34건과 대표 원문 6건을 분석했다.
- High-confidence accepted error는 Phase 2가 틀린 사례가 아니다. 잘못된 Phase 1 class의 calibrated confidence가 `tau=0.70` 이상이어서 accepted되고 routing되지 않은 사례이며, 따라서 Phase 2 output이 존재하지 않는다.
- 이 감사는 confidence-only routing이 모든 오류를 탐지한다고 주장하기 위한 것이 아니라, calibration 후에도 남는 비라우팅 사각지대를 공개하고 유형화하기 위한 것이다.
- LaTeX environment 64쌍, label 34개, reference 14개를 정적으로 검사했고 누락·중복이 없었다.
- 최신 ZIP 내부 main.tex과 작업 원문의 일치성을 확인했다.
- 로컬에는 LaTeX engine이 없어 최신 ZIP의 실제 PDF compile은 Overleaf에서 다시 확인해야 한다.

## 7. 제출 전 남은 작업

### 필수

1. Mistral 7B supervised Phase 1 classifier matched run
2. Llama 2 7B supervised Phase 1 classifier matched run
3. Table I의 Pending 값을 실제 accuracy, macro Precision, Recall, F1로 교체
4. 두 baseline의 calibration과 model-specific routing 결과 반영 여부 결정
5. Architecture와 end-to-end workflow 도식 삽입
6. Overleaf clean compile과 전체 페이지의 표 폭, float, caption, page break 최종 확인
7. 저자 순서, affiliation, corresponding author, funding, conflict of interest 확정
8. Data/code availability와 최종 release 범위 확정

### 권장

1. Frozen prompt를 독립 routed subset 또는 outer fold에서 한 번 확인
2. Mixed Emotion과 대표 Reddit 사례에 대한 복수 연구자 또는 전문가 검토
3. GPU, 모델별 runtime, 생성 token, wall-clock time과 비용 기록
4. 최종 영문 교정과 IEEE Access 제출 체크리스트 점검

## 8. 이번 회의에서 결정할 항목

| 결정 사항 | 필요한 결정 |
|---|---|
| Mistral 7B Phase 1 | 실행 일정 또는 정당화된 제외 |
| Llama 2 7B Phase 1 | 실행 일정 또는 정당화된 제외 |
| Architecture figure | 포함 요소와 제작 담당 확정 |
| Independent prompt confirmation | 본 연구 포함 또는 후속 연구 |
| Human/expert review | 검토 범위와 reviewer 수 |
| 공개 범위 | 코드, Mixed Emotion, 결과 artifact의 release 시점 |
| 투고 정보 | 저자, 소속, corresponding author, funding |
| 목표 일정 | 교수 1차 검토와 IEEE Access 제출 목표일 |

## 9. 현재 판단

- 교수 1차 검토용으로는 방법론, 확정 수치, 통계, 오류 감사, Appendix까지 충분히 정리된 상태다.
- IEEE Access 제출 전에는 최소한 matched Phase 1 baselines의 Pending 처리와 최종 메타데이터·PDF 레이아웃 검수가 필요하다.
- 결과가 강한 부분과 약한 부분을 모두 보고하고 있으므로 “LLM이 언제나 개선한다”는 과장 대신 selective re-evaluation의 조건부 효과를 방어할 수 있다.
- 새로운 결과가 더 좋아 보인다는 이유로 확정 threshold, prompt, dataset 문장을 사후 교체하지 않는다.

## 10. 기준 문서와 실행 파일

- 연구 현황 기준: `docs/final_research_status_and_remaining_work_ko.md`
- End-to-end 실행 안내: `docs/final_end_to_end_workflow_ko.md`
- Threshold 근거: `docs/routing_threshold_policy_audit_ko.md`
- Paired 통계: `docs/paired_end_to_end_statistical_analysis_ko.md`
- High-confidence 오류 감사: `docs/high_confidence_accepted_error_audit_results_ko.md`
- Final 01: `notebooks/colab/final/01_distilbert_phase1_training_final_colab.ipynb`
- Final 02: `notebooks/colab/final/02_2_llm_phase2_reasoning_model_specific_prompt_final_colab.ipynb`
- Final 03: `notebooks/colab/final/03_mixed_emotion_end_to_end_orchestration_final_colab.ipynb`
- Final 04: `notebooks/colab/final/04_5_reddit_test_routed_phase2_original_text_primary_final_colab.ipynb`

실험이나 공동 결정이 추가되면 이 체크포인트를 먼저 갱신한 뒤 README, workflow guide, 원고를 동기화한다.
