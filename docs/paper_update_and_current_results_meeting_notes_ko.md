# 연구 진행 체크포인트: 완료 내용과 남은 작업

업데이트: 2026-08-23

이 문서는 다음 회의에서 현재 논문과 실험 상태를 빠르게 공유하기 위한 최신 체크포인트다. 연구 과제는 임상 진단이 아니라, Reddit 텍스트의 **3-class proxy emotion classification**에서 불확실한 예측만 LLM으로 선택적 재평가하는 confidence-guided two-phase framework다.

## 1. 현재 결론 요약

- Phase 1은 DistilBERT를 운영 분류기로 사용한다.
- Temperature-scaled MSP를 confidence score로 사용하고, calibration split에서 routing policy를 고정한다.
- 주 운영 정책은 `alpha=5%`, candidate threshold `0.70--1.00`, step `0.01`, 선택 threshold `tau=0.70`이다.
- Reddit held-out test는 논문 기준 class별 4,000건, 총 12,000건으로 보고한다.
- Reddit Phase 2는 최종 primary rerun에서 routed 171건에 원문 수준의 `title + selftext`를 연결해 Llama 2와 Llama 3를 평가했다.
- Mixed Emotion v2.4는 학습이나 threshold 선택에 사용하지 않은 300건 supplementary stress-test다.
- 현재 결과상 Llama 3 SELF-DISCOVER가 두 stress-test 조건에서 가장 일관된 개선을 보이고, Llama 2 CoT는 데이터 조건에 따라 개선 폭이 다르거나 Reddit에서 소폭 하락한다. 이 차이는 숨길 항목이 아니라 선택적 재평가자의 모델 의존성을 보여주는 결과로 해석한다.

## 2. 완료된 작업

### 2.1 Phase 1 학습, calibration, routing

- Reddit primary dataset에서 class별 40,000건을 샘플링하도록 최종 Colab workflow를 구성했다.
- `train / validation / calibration / held-out test`를 분리했다.
- validation은 학습과 hyperparameter/sweep 및 checkpoint 선택에 사용한다.
- calibration split은 temperature scaling과 routing threshold 선택에만 사용한다.
- held-out test는 temperature와 threshold를 고정한 뒤 최종 평가에만 사용한다.
- Temperature-scaled MSP를 주 confidence score로 확정했다.
- calibration 결과:

| 지표 | Raw MSP | Temperature-scaled MSP |
|---|---:|---:|
| Temperature | 1.0000 | 1.7706 |
| NLL | 0.1411 | 0.1041 |
| Brier score | 0.0559 | 0.0514 |
| ECE | 0.0244 | 0.0083 |
| Adaptive ECE | 산출 대상 아님 | 0.0138 |

- 주 threshold 규칙은 calibration data에서 후보를 모두 평가한 뒤, one-sided Clopper--Pearson risk upper bound가 `alpha=0.05` 이하인 후보 중 coverage가 가장 높은 후보를 선택하는 방식이다.
- 따라서 `0.70`은 test accuracy를 보고 사후에 고른 숫자가 아니라, 사전에 고정한 calibration candidate grid에서 선택된 operating policy다.

### 2.2 Reddit held-out test와 원문-input Phase 2

논문에는 정확히 12,000건 기준으로 다음 수치를 사용한다.

| 시스템 | 전체 정확도 | Routed | End-to-end 정확도 | 변화 |
|---|---:|---:|---:|---:|
| DistilBERT Phase 1 only | 96.69% | 171 / 12,000 (1.42%) | 96.69% | - |
| DistilBERT -> Llama 2 CoT | 96.69% | 동일 171건 | 96.67% | -0.03 pp |
| DistilBERT -> Llama 3 SELF-DISCOVER | 96.69% | 동일 171건 | 96.94% | +0.25 pp |

추가 routed-set 분석에서는 Phase 1 오류 포착과 Phase 2 변경을 분리해 보고한다.

- Phase 1 전체 오류: 397건
- Routed sample: 171건
- Routed subset의 Phase 1 오류: 87건
- Error capture: 21.91%
- Llama 2: 47 corrected, 50 introduced, net correction -3
- Llama 3: 42 corrected, 12 introduced, net correction +30

이 결과는 “많이 보내면 항상 좋아진다”는 주장을 하지 않는다. 현재 primary policy는 전체 LLM 호출을 1.42%로 제한하면서, 오류가 상대적으로 농축된 subset을 재평가하는 계산적 절충안이다.

### 2.3 Mixed Emotion v2.4 stress-test

Mixed Emotion v2.4는 Depression, Neutral, Happy 각 100건의 300건 controlled supplementary stress-test다. Phase 1 학습, hyperparameter tuning, calibration, threshold 선택에는 사용하지 않았다.

| 시스템 | 전체 정확도 | Routed | End-to-end 변화 | Corrected | Introduced | Net |
|---|---:|---:|---:|---:|---:|---:|
| DistilBERT Phase 1 only | 81.33% | 44 / 300 (14.67%) | - | - | - | - |
| DistilBERT -> Llama 2 CoT v2 | 85.33% | 동일 44건 | +4.00 pp | 18 | 6 | +12 |
| DistilBERT -> Llama 3 SELF-DISCOVER | 87.33% | 동일 44건 | +6.00 pp | 18 | 0 | +18 |

Mixed Emotion은 실제 Reddit 모집단의 일반화 성능을 증명하는 데이터가 아니라, 감정 공존·감정 전환·최종 trajectory가 있는 어려운 입력에서 routing과 reasoning이 어떻게 작동하는지 확인하는 보조 평가로 해석한다.

### 2.4 통계 및 오류 분석

- 동일 sample의 Phase 1과 Phase 2 결과를 paired unit으로 유지했다.
- paired bootstrap 50,000회로 accuracy change의 95% interval을 계산했다.
- exact McNemar test로 discordant pair의 방향성을 검정했다.
- 여러 비교에 Holm correction을 적용했다.
- Reddit Llama 3와 Mixed Emotion Llama 2/Llama 3의 paired improvement는 논문에 통계 결과와 함께 반영했다.
- accepted high-confidence error audit을 정확한 12,000-row protocol으로 완료했다.
- `tau=0.70` 이상으로 accept된 오류 310건, accepted selective risk 2.62%, accepted reference-Depression 3,939건 중 Depression false negative 125건(3.17%)을 확인했다.
- confidence 0.98 이상에서도 오류 34건이 남는 사례를 확인하고, 대표 사례를 original retained text와 normalized exact matching으로 검증했다.
- 대표 오류는 acute distress false negative, mixed trajectory, proxy-label/content ambiguity, topic-term shortcut 유형으로 정리했다.

### 2.5 Prompt, Appendix, manuscript 보강

- Llama 2 CoT와 Llama 3 SELF-DISCOVER를 같은 최종 label space로 canonicalize했다.
- Mixed/shifted emotion에서는 isolated keyword가 아니라 전체 메시지와 final emotional trajectory를 보도록 prompt를 정리했다.
- Llama 2의 percentage breakdown이 최종 label을 불안정하게 만들 수 있어, 최종 label을 명시적 `Final label:` 계약으로 추출하도록 정리했다.
- Appendix B에는 Llama 2 CoT prompt protocol, Appendix C에는 Llama 3 SELF-DISCOVER protocol과 실제 output-column 구조를 넣었다.
- Appendix D에는 Phase 1 오류가 Phase 2에서 Depression으로 수정된 Llama 2/Llama 3 실제 사례와 reasoning excerpt를 넣었다.
- SELF-DISCOVER workflow 도식은 본문에서 모델 기여처럼 보이지 않도록 Appendix C의 `Appendix Figure C1`로 배치했다.
- references, terminology, table numbering, figure caption, page break와 appendix 간격을 반복 검수했다.
- 문헌·수식·알고리즘·calibration 지표(NLL, Brier, ECE, Adaptive ECE)의 설명을 보강했다.

## 3. 현재 Git/실행 산출물

### 3.1 최종 실행 순서

1. `01_distilbert_phase1_training_final_colab.ipynb`
2. `02_llm_phase2_reasoning_final_colab.ipynb`
3. `03_mixed_emotion_end_to_end_orchestration_final_colab.ipynb`
4. `04_5_reddit_test_routed_phase2_original_text_primary_final_colab.ipynb`

Final 04.5는 `tau=0.70`으로 고정된 Reddit routed 171건을 원문 `title + selftext`에 연결하고, row-level resume 및 Google Drive append 저장을 사용한다.

### 3.2 주요 저장 위치

- Phase 1: `/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/phase1_distilbert/`
- Mixed Emotion Phase 2: `/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/phase2_llm_reasoning/`
- Mixed Emotion orchestration: `/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/end_to_end_orchestration/`
- Reddit original-text Phase 2: `/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/reddit_test_phase2_reasoning_original_text_primary_tau070_final/`
- paired statistics: `reports/statistics/`
- accepted-error audit: `reports/high_confidence_accepted_error_audit/`

각 Phase 2 row는 완료 직후 CSV에 append되며, 재실행 시 이미 저장된 `example_id`를 건너뛴다. 따라서 runtime이 종료되어도 완료된 row가 사라지는 구조가 아니다.

### 3.3 논문 패키지

현재 Overleaf source of truth는 로컬의 다음 패키지다.

`/Users/woojinpark/Documents/헬스케어 논문/Paper_260620_overleaf_original_text_phase2_final_self_discover_figure.zip`

이 패키지에는 최신 본문 수치, 원문-input Reddit 결과, Appendix C1 SELF-DISCOVER vector figure, Appendix D output examples, table/figure layout 수정이 반영되어 있다. Reddit 원문 행 자체는 공개 Git에 올리지 않고, 재현에 필요한 코드·문서·요약 산출물만 저장소에 유지한다.

## 4. 아직 남은 필수 작업

### 4.1 실제 classifier 비교값 확보

- Mistral 7B Phase 1 full-scale classifier run
- Llama 2 7B Phase 1 full-scale classifier run
- 두 모델의 matched split, accuracy, macro precision, macro recall, macro F1, calibration 지표를 실제 실행값으로 확보
- 완료 후 논문의 `Pending/TBD`를 실제 값으로 교체

현재 값이 없는 모델을 임의의 숫자로 채우지 않는다. 실제 결과가 나오기 전까지는 표에서 `Pending` 또는 `TBD`로 명확히 표시해야 한다.

### 4.2 제출 직전 동기화와 시각 검수

- 새 classifier 결과가 들어오면 Table I--III, calibration comparison, model-comparison figure, Results 문단을 동시에 갱신
- Final 04.5의 최신 CSV/통계와 Reddit 표·figure의 값이 일치하는지 재확인
- Overleaf에서 새로 compile하고 표가 잘리지 않는지, Appendix 제목과 figure caption이 다음 페이지로 어색하게 분리되지 않는지 페이지별 확인
- 저자명, 소속, acknowledgements, data/code availability, ethical statement와 최종 reference metadata 입력
- 최종 PDF와 제출 ZIP의 파일 목록 및 컴파일 재현성 확인

## 5. 선택적 후속 작업

아래 항목은 현재 논문의 핵심 결론을 성립시키기 위한 필수 조건은 아니며, 시간과 공동 연구자 합의가 있을 때 추가한다.

- 실제 wall-clock runtime, GPU 사용량, LLM 호출 수와 비용을 분리해 측정
- frozen prompt를 독립 실행 환경에서 재확인
- 복수 연구자 또는 전문가의 소규모 rationale review
- high-confidence accepted-error 유형을 더 큰 표본으로 확장
- 새로운 human-annotated external validation set 추가

## 6. 이번 회의에서 확인할 것

1. Mistral/Llama 2 Phase 1 classifier 비교를 실제로 추가할지 결정
2. 추가하지 않는다면 해당 모델을 “비교 예정”으로 남길지, 본문 범위에서 제외할지 결정
3. 현재 primary routing policy `alpha=5%, tau=0.70`을 최종 운영 정책으로 유지
4. Mixed Emotion은 supplementary stress-test로 유지하고 primary Reddit 일반화 주장과 분리
5. 최신 Overleaf ZIP을 기준으로 최종 저자·윤리·재현성 항목을 채운 뒤 제출용 PDF를 확정

## 7. 해석상 주의

- `alpha=5%`는 임상 안전 기준이 아니라 calibration 단계에서 연구자가 사전 지정한 accepted-risk budget이다.
- threshold는 test 성능이 가장 좋은 값을 사후 선택한 것이 아니다.
- routing precision, error capture, accepted error는 서로 다른 지표이므로 하나의 숫자로 대체하지 않는다.
- LLM Phase 2의 correction은 diagnosis나 treatment recommendation이 아니며, text-grounded research re-evaluation이다.
- 최종 원고에는 실제 실행으로 확인된 수치만 사용하고, 예시·계획·TBD 값은 완료 결과처럼 서술하지 않는다.

## 관련 문서

- [최종 End-to-End Workflow](final_end_to_end_workflow_ko.md)
- [Reddit 원문-input 최종 재실험 기록](reddit_phase2_original_text_primary_final_run_ko.md)
- [Routing threshold policy audit](routing_threshold_policy_audit_ko.md)
- [Paired end-to-end statistical analysis](paired_end_to_end_statistical_analysis_ko.md)
- [High-confidence accepted-error audit 결과](high_confidence_accepted_error_audit_results_ko.md)
- [IEEE Access 제출 준비 체크리스트](ieee_access_submission_readiness_checklist_ko.md)
- [최종 model-specific prompt policy](final_model_specific_prompt_policy_ko.md)

## 실행 링크

- [Final 01: DistilBERT Phase 1](https://colab.research.google.com/github/WoojinPark-Jay/confidence-guided-llm-reasoning-depression-risk-emotion/blob/main/notebooks/colab/final/01_distilbert_phase1_training_final_colab.ipynb)
- [Final 02: Mixed Emotion Phase 2](https://colab.research.google.com/github/WoojinPark-Jay/confidence-guided-llm-reasoning-depression-risk-emotion/blob/main/notebooks/colab/final/02_llm_phase2_reasoning_final_colab.ipynb)
- [Final 03: Mixed Emotion orchestration](https://colab.research.google.com/github/WoojinPark-Jay/confidence-guided-llm-reasoning-depression-risk-emotion/blob/main/notebooks/colab/final/03_mixed_emotion_end_to_end_orchestration_final_colab.ipynb)
- [Final 04.5: Reddit original-text Phase 2](https://colab.research.google.com/github/WoojinPark-Jay/confidence-guided-llm-reasoning-depression-risk-emotion/blob/main/notebooks/colab/final/04_5_reddit_test_routed_phase2_original_text_primary_final_colab.ipynb)
