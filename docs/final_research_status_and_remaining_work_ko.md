# 최종 연구 진행 현황 및 남은 작업

최종 갱신: 2026-08-29

용도: 공동연구자 회의 및 IEEE Access 제출 준비

대상 연구: Confidence-Guided Selective LLM Re-Evaluation for Depression-Risk-Related Emotion Classification

> **이 문서가 현재 연구 현황의 기준 문서다.**
>
> 오늘 회의에서는 먼저 Section 1--3으로 결과와 결론을 확인하고, Section 8의 결정 항목을 논의한다. 세부 구현이나 과거 실험 이력이 필요할 때만 Section 9의 연결 문서를 확인한다.

## 1. 오늘 회의 5분 요약

### 1.1 현재 어디까지 완료되었는가

- Reddit 120,000건 기반 DistilBERT Phase 1 학습과 W&B hyperparameter sweep 완료
- 별도 calibration split을 이용한 temperature scaling 완료
- calibration-only risk constraint를 이용한 primary routing threshold 선택 완료
- Reddit held-out test 12,000건과 Mixed Emotion 300건의 Phase 1 평가 완료
- routed sample에 대한 Llama 2 CoT 및 Llama 3 SELF-DISCOVER Phase 2 완료
- Reddit은 cleaned text가 아니라 original `title + selftext`로 최종 Phase 2 재실험 완료
- Phase 1 only 대비 full-set end-to-end 결과, corrected/introduced/net correction 분석 완료
- Paired bootstrap, exact McNemar, Holm correction 완료
- High-confidence accepted error 310건 감사와 실제 원문 대표 사례 6건 분석 완료
- Routing 오류 농축, routed-only accuracy, conditional correction opportunity 분석 완료
- 최종 confidence-guided two-phase architecture 도식 완성 및 원고 삽입 완료
- 편집 가능한 Draw.io 원본, publication-ready 벡터 PDF, 재현 안내를 GitHub `main`에 보존 완료
- 최종 논문 초안, Appendix, 재현 코드, 결과 CSV/JSON, Overleaf ZIP 정리 완료

### 1.2 현재 핵심 결론

1. DistilBERT는 Reddit held-out test에서 `96.69%` 정확도를 기록했다.
2. Primary routing policy는 `alpha=0.05`, `tau=0.70`이며 Reddit 12,000건 중 171건(1.42%)을 Phase 2로 보냈다.
3. Routed Reddit subset의 오류율은 `50.88%`로 전체 오류율 `3.31%`보다 약 `15.38배` 높았다. Routing이 적은 샘플 안에 오류를 실제로 집중시켰다.
4. Reddit routed-only 정확도는 Phase 1 `49.12%`, Llama 2 `47.37%`, Llama 3 `66.67%`였다. Full-set에서는 Llama 3가 `96.69% -> 96.94%`, net correction `+30`을 기록했다.
5. Mixed Emotion의 routed-only 정확도는 Phase 1 `52.27%`, Llama 2 `79.55%`, Llama 3 `93.18%`였으며 full-set 정확도는 각각 `81.33%`, `85.33%`, `87.33%`였다.
6. 따라서 최종 주장은 “LLM이 항상 성능을 높인다”가 아니다. **Calibration 기반 routing은 어려운 표본을 선별하며, 원문을 보존한 selective re-evaluation의 효과는 reasoner와 input regime에 따라 달라진다**는 것이 핵심 결론이다.

### 1.3 현재 가장 중요한 남은 작업

| 우선순위 | 남은 작업 | 완료 후 달라지는 부분 |
|---|---|---|
| 필수 | Mistral 7B supervised Phase 1 classifier | Table I의 Mistral `Pending` 해소 |
| 필수 | Llama 2 7B supervised Phase 1 classifier | Table I의 Llama 2 classifier `Pending` 해소 |
| 필수 | 두 classifier 실측치의 표·그림·본문 반영 | 모든 `TBD`/planning estimate 제거 |
| 필수 | 최신 원고와 아키텍처를 공식 IEEE Access 패키지에 최종 동기화 | 제출 소스와 교수 검토본의 내용 일치 |
| 필수 | 저자·소속·funding·availability 확정 | 실제 투고 메타데이터 완성 |
| 권장 | Frozen prompt 독립 확인 | Prompt 개발과 최종 평가의 분리 강화 |
| 권장 | 복수 연구자 또는 전문가 사례 검토 | Proxy label과 rationale 신뢰성 보강 |
| 권장 | 실행시간 및 효율 정리 | Selective routing의 계산 효율 근거 보강 |

## 2. 연구 질문과 현재 답변

| 연구 질문 | 현재 근거 | 현재 답변 |
|---|---|---|
| Phase 1 classifier의 confidence를 믿을 수 있는가? | NLL, Brier, ECE, Adaptive ECE가 temperature scaling 후 모두 감소 | Raw softmax보다 calibrated MSP가 routing에 더 적합함 |
| Routing이 실제 어려운 표본을 찾는가? | Reddit routed 오류율 50.88%, 전체 오류율 3.31%, error enrichment 15.38배 | 전체의 1.42%만 보내면서 Phase 1 오류의 21.91%를 집중시킴 |
| Phase 2가 전체 정확도를 높이는가? | Reddit/Mixed full-set end-to-end 결과 | Llama 3는 두 입력 regime에서 개선, Llama 2는 Mixed에서만 개선 |
| Reasoner가 routed correction opportunity를 활용하는가? | Reddit Llama 3 34.48%, Mixed Llama 3 85.71%의 conditional opportunity 실현 | Router가 만든 기회를 활용하는 정도는 reasoner와 dataset에 따라 다름 |
| Mixed/trajectory 사례에 reasoning이 유효한가? | Mixed Emotion Llama 2 +4.00 pp, Llama 3 +6.00 pp | Controlled stress-test에서는 명확한 개선이 관찰됨 |
| 모든 accepted prediction이 안전한가? | Accepted error 310건, selective risk 2.62%, 대표 원문 6건 | 아님. High-confidence 오류가 남으며 한계와 감사 결과를 함께 보고해야 함 |
| 임상 진단 모델로 주장할 수 있는가? | Subreddit-derived proxy labels와 synthetic stress test 사용 | 불가. 연구 범위는 proxy emotion classification과 selective re-evaluation임 |

## 3. 확정 결과

### 3.1 DistilBERT Phase 1 및 calibration

| 항목 | 확정값 |
|---|---:|
| Reddit held-out examples | 12,000 |
| Phase 1 accuracy | 96.69% |
| 최적 temperature | 1.7706 |
| Raw / scaled NLL | 0.1411 / 0.1041 |
| Raw / scaled Brier score | 0.0559 / 0.0514 |
| Raw / scaled ECE | 0.0244 / 0.0083 |
| Raw / scaled Adaptive ECE | 0.0244 / 0.0138 |
| Primary risk budget | `alpha=0.05` |
| Primary threshold | `tau=0.70` |
| Reddit routed | 171 / 12,000 (1.42%) |
| Routed Phase 1 accuracy | 49.12% |
| Routed Phase 1 errors | 87 |
| 전체 Phase 1 error capture | 21.91% |
| Full / routed error rate | 3.31% / 50.88% |
| Routed-error enrichment | 15.38x |
| Accepted examples | 11,829 |
| Accepted errors / selective risk | 310 / 2.62% |

### 3.2 Reddit original-text end-to-end

| System | Accuracy | Change | Corrected | Introduced | Net |
|---|---:|---:|---:|---:|---:|
| DistilBERT Phase 1 | 96.69% | - | - | - | 0 |
| DistilBERT + Llama 2 CoT | 96.67% | -0.03 pp | 47 | 50 | -3 |
| DistilBERT + Llama 3 SELF-DISCOVER | 96.94% | +0.25 pp | 42 | 12 | +30 |

- Llama 2 변화는 Phase 1과 통계적으로 구분되지 않는다.
- Llama 3 paired bootstrap 95% CI는 `[0.13, 0.38]` pp다.
- Llama 3 exact McNemar p-value는 약 `0.000052`, Holm-adjusted p-value는 약 `0.000156`이다.
- Llama 3의 절대 향상 폭은 작지만, 동일 표본 paired comparison에서 양의 효과가 확인되었다.
- Routed-only accuracy는 Phase 1 `49.12%`, Llama 2 `47.37%`, Llama 3 `66.67%`다.
- Fixed routing oracle은 97.42%이며 Llama 3는 routed error correction opportunity의 34.48%를 순개선으로 실현했다.

### 3.3 Mixed Emotion end-to-end

| System | Accuracy | Change | Corrected | Introduced | Net |
|---|---:|---:|---:|---:|---:|
| DistilBERT Phase 1 | 81.33% | - | - | - | 0 |
| DistilBERT + Llama 2 CoT | 85.33% | +4.00 pp | 18 | 6 | +12 |
| DistilBERT + Llama 3 SELF-DISCOVER | 87.33% | +6.00 pp | 18 | 0 | +18 |

- 300건 중 44건(14.67%)이 routed되었다.
- Routed error rate는 47.73%로 전체 error rate 18.67%보다 2.56배 높다.
- Routed-only accuracy는 Phase 1 `52.27%`, Llama 2 `79.55%`, Llama 3 `93.18%`다.
- Fixed routing oracle은 88.33%이며 Llama 3는 correction opportunity의 85.71%를 순개선으로 실현했다.
- Mixed Emotion은 real-world prevalence dataset이 아니라 mixed cue와 trajectory shift를 통제해 평가하는 supplementary stress test다.
- Scenario별 결과와 실제 reasoning output 예시는 Appendix에 반영되어 있다.

## 4. 이번 단계에서 완료한 작업

### 4.1 데이터와 Phase 1

- Reddit primary dataset 클래스별 40,000건 구성
- Train / validation / calibration / test = 70 / 10 / 10 / 10 분리
- W&B sweep과 best hyperparameter 선택
- Final DistilBERT checkpoint 및 hyperparameter 저장
- Reddit held-out test를 클래스별 4,000건, 총 12,000건으로 통일
- Mixed Emotion 300건을 학습에 사용하지 않는 external stress-test로 분리

### 4.2 Confidence calibration과 routing

- Calibration NLL 최소화를 통한 temperature fitting
- Temperature-scaled MSP를 primary confidence로 고정
- NLL, Brier, ECE, Adaptive ECE 전후 비교
- Raw MSP Adaptive ECE 누락값 재계산 및 Final 01 export 보완
- Candidate grid `0.70--1.00`, step `0.01` 고정
- Calibration upper-risk constraint와 coverage-maximization에 따른 `tau=0.70` 선택
- Risk-coverage, score ablation, bootstrap stability, class-conditional analysis 완료

### 4.3 Phase 2와 end-to-end

- Mixed Emotion routed 44건 Llama 2/Llama 3 실행
- Reddit routed 171건을 original `title + selftext`에 exact link
- URL과 직접 사용자명만 최소 마스킹하고 문장 구조와 부정 표현 보존
- Row-level resumable CSV 저장 및 full-set label recomposition
- Corrected, introduced, net correction과 confusion matrix 생성
- Reader-facing 방법명을 `Llama 2 CoT`, `Llama 3 SELF-DISCOVER`로 통일
- 내부 `v2`, `v2.1`은 prompt-development history에서만 유지

### 4.4 통계와 오류 감사

- Paired bootstrap 50,000회
- Exact McNemar test
- 네 비교에 대한 Holm multiple-comparison correction
- Accepted high-confidence error 310건 전수 original-post linkage
- Depression false negative와 대표 failure mode 분석
- 실제 원문과 reference/prediction 구분을 포함한 대표 사례 6건 Appendix 반영

### 4.5 논문과 재현 패키지

- 25-page IEEE-style draft 작성
- Method-specific algorithm 3개 수록
- Main result table과 Appendix 구조 재편
- 실제 stored Llama 2/Llama 3 output 예시 수록
- SELF-DISCOVER 39-module pool과 generated trace의 차이 설명
- 참고문헌 46개 번호·본문 사용 여부 전수 확인
- 내부 prompt version과 최종 방법명 분리
- 모든 25페이지 렌더링 후 표 잘림, 겹침, cross-reference, horizontal overflow 검수
- Paired statistics와 high-confidence audit 재현 스크립트 및 CSV/JSON 저장
- Main Results에 Reddit/Mixed routed-error enrichment와 routed-only accuracy 반영
- Main Table III에 error enrichment, Table V와 VI에 routed-only accuracy 열 추가
- Appendix Table A4g에 conditional oracle과 correction-opportunity 계산식 및 결과 반영
- Reddit 98.58%, Mixed Emotion 85.33%의 all-input 대비 LLM invocation avoidance를 실제 runtime 절감과 구분해 서술

## 5. 동결된 설계와 다시 돌리지 않을 실험

다음 항목은 새로운 오류가 발견되지 않는 한 현재 논문용 최종값으로 동결한다.

- DistilBERT primary Phase 1 checkpoint와 12,000건 held-out protocol
- Temperature `1.7706`
- Primary routing policy `alpha=0.05`, `tau=0.70`
- Mixed Emotion 300건과 routed 44건의 최종 결과
- Reddit original-text routed 171건의 최종 결과
- Llama 2 CoT와 Llama 3 SELF-DISCOVER 최종 prompt
- Reddit/Mixed paired statistical results
- End-to-end 결과를 보고 alpha 또는 threshold를 사후 재선택하는 실험은 수행하지 않음
- 기각된 universal/shared prompt를 최종 정책으로 재실행하지 않음
- 정확도를 높이기 위해 Mixed Emotion 문장을 추가 수정하지 않음

동일 설정의 재현 실행은 가능하지만, 새로운 결과가 더 좋아 보인다는 이유만으로 확정 수치를 교체하지 않는다.

## 6. 제출 전 남은 작업

### 6.1 필수: Mistral 7B Phase 1 classifier

- DistilBERT와 동일한 120,000건, split, label space 사용
- Accuracy, macro Precision, macro Recall, macro F1 저장
- Temperature, NLL, Brier, ECE, Adaptive ECE 저장
- Model-specific threshold, coverage, routing rate 저장
- 완료 전까지 DistilBERT가 세 classifier 중 최고라고 단정하지 않음

### 6.2 필수: Llama 2 7B Phase 1 classifier

- Mistral과 동일한 matched protocol 사용
- Phase 2의 Llama 2 reasoner와 혼동하지 않도록 항상 `Phase 1 classifier`로 표기
- 동일한 classification 및 calibration 산출물 저장

### 6.3 필수: 두 classifier 결과의 원고 반영

결과가 나오면 다음 위치를 함께 갱신한다.

- Main Table I의 Mistral/Llama 2 `Pending`
- 필요 시 model-specific calibration/threshold 비교
- Abstract와 Contributions의 classifier 비교 문장
- Results와 Discussion의 비교 결과
- Limitations의 “completed DistilBERT only” 문장
- Conclusion의 classifier 선택 근거
- Supplementary CSV/JSON과 run manifest

### 6.4 완료: 최종 도식과 재현 원본

- 전체 selective two-phase architecture를 완성하고 원고에 삽입했다.
- Reddit split, calibration, fixed threshold, accepted/routed 경로, Phase 2 re-evaluation, 최종 label selection 및 audit artifact 흐름을 하나의 도식에 연결했다.
- 편집 가능한 Draw.io 원본과 publication-ready 벡터 PDF를 다음 위치에 공개했다.
  - `docs/figures/architecture/confidence_guided_two_phase_architecture.drawio`
  - `docs/figures/architecture/confidence_guided_two_phase_architecture.pdf`
  - `docs/figures/architecture/README.md`
- GitHub `main` 반영 커밋: `5096c47` (`Add final editable architecture figure`)

### 6.5 필수: 공식 IEEE Access 패키지 최종 동기화

- 공식 `ieeeaccess.cls` 기반 패키지에 최신 본문, 표, Appendix, 최종 architecture PDF를 동기화한다.
- 교수 검토용 최신 원고와 공식 제출용 소스 사이에 수치, 그림 번호, caption, reference 차이가 없는지 확인한다.
- Overleaf에서 pdfLaTeX clean compile 후 전 페이지를 다시 검수한다.

### 6.6 필수: 투고 메타데이터

- 저자 순서와 affiliation
- Corresponding author
- Acknowledgments와 funding
- Conflict of interest
- Data/code availability
- Keywords와 cover letter

### 6.7 권장 보강

- Frozen prompt를 독립 routed subset 또는 outer fold에서 한 번 확인
- Mixed Emotion 및 대표 Reddit 사례에 대한 복수 연구자 또는 전문가 검토
- GPU, 모델별 실행시간, 생성 token, 실제 wall-clock time 기록
- 현재 보고된 호출 감소량을 실제 계산시간 또는 비용 측정과 연결

이 권장 작업을 수행하지 못하면 해당 부분을 감추지 않고 현재 Limitations의 탐색적 prompt-development, proxy label, 비임상적 rationale 제한을 유지한다.

## 7. 남은 작업의 실행 순서

1. Mistral 7B Phase 1 classifier 실행
2. Llama 2 7B Phase 1 classifier 실행
3. 두 모델의 calibration 및 routing 결과 생성
4. Main Table I과 관련 본문 갱신
5. 최신 원고와 architecture를 공식 IEEE Access 패키지에 동기화
6. 저자·소속·availability·윤리·funding 확정
7. 원고 숫자와 prediction-level artifact 최종 대조
8. Overleaf clean compile 및 전 페이지 시각 검수
9. Source, PDF, supplementary, environment/run manifest 동결
10. IEEE Access 제출 체크리스트 최종 확인

## 8. 오늘 회의에서 결정할 항목

| 번호 | 결정할 사항 | 선택 또는 결정 기준 |
|---:|---|---|
| 1 | Mistral 7B와 Llama 2 7B Phase 1 실행 일정 | 동일 split과 평가 산출물을 만들 수 있는 GPU 일정 |
| 2 | 두 classifier를 반드시 본문 baseline으로 완료할지 | IEEE Access 제출 시 Table I의 비교 완결성 |
| 3 | 공식 IEEE Access 패키지 최종 동기화 일정 | 최신 원고·아키텍처·Appendix를 제출 패키지에 반영할 일정 |
| 4 | Independent prompt confirmation 포함 여부 | 일정 대비 prompt-selection bias 방어 효과 |
| 5 | 복수 연구자 또는 전문가 검토 범위 | Mixed 300건 전체 또는 층화 표본 검토 |
| 6 | 코드와 Mixed Emotion dataset 공개 범위 | 익명화, 라이선스, release 시점 |
| 7 | 저자·소속·corresponding author | 실제 투고 정보 확정 |
| 8 | IEEE Access 목표 제출일 | 남은 실험과 도식 완료 가능 일정 |

### 회의 종료 시 남겨야 할 결정 기록

- Mistral 7B Phase 1: 실행 / 제외 / 보류
- Llama 2 7B Phase 1: 실행 / 제외 / 보류
- Independent prompt confirmation: 포함 / 후속 연구
- Human/expert review: 범위 확정 / 후속 연구
- 공식 IEEE Access 패키지 동기화 담당과 마감일
- 투고 목표일

연구자별 역할은 이 문서에서 미리 나누지 않는다. 회의에서 필요한 일정과 실행 범위만 확정한다.

## 9. 문서가 많을 때 확인 순서

### 먼저 볼 문서

1. `docs/final_research_status_and_remaining_work_ko.md`

   현재 결과, 결론, 남은 작업, 회의 결정사항을 확인하는 기준 문서
2. `docs/final_end_to_end_workflow_ko.md`

   Final notebook 실행 순서와 Google Drive 산출물 위치
3. `docs/ieee_access_submission_readiness_checklist_ko.md`

   실제 투고 직전 점검표

### 특정 질문이 있을 때만 볼 문서

- `docs/reddit_phase2_original_text_primary_final_run_ko.md`: Reddit 원문 Phase 2 연결과 최종 결과
- `docs/routing_threshold_policy_audit_ko.md`: `alpha=0.05`, `tau=0.70` 선택 근거
- `docs/paired_end_to_end_statistical_analysis_ko.md`: Bootstrap, McNemar, Holm 결과
- `docs/high_confidence_accepted_error_audit_results_ko.md`: Accepted high-confidence error 감사
- `docs/final_model_specific_prompt_policy_ko.md`: 최종 model-specific prompt와 내부 개발 이력
- `docs/routing_concentration_and_correction_opportunity_update_ko.md`: 이번 오류 농축, routed-only, conditional oracle 분석과 원고 반영 위치

### 과거 이력으로만 유지하는 문서와 노트북

- Universal/shared prompt 비교 문서
- `v2`, `v2.1` prompt iteration history
- Cleaned-text Reddit Phase 2 결과
- Alpha/threshold sensitivity notebook
- 기존 `10`, `13`, `14` 계열 개발 노트북

이 항목들은 재현성과 변경 이력을 위해 보존하지만 최종 논문 수치의 근거로 사용하지 않는다.

## 10. 최종 실행 파일

1. `notebooks/colab/final/01_distilbert_phase1_training_final_colab.ipynb`
   - DistilBERT 학습, sweep, calibration, threshold, Reddit/Mixed Phase 1
2. `notebooks/colab/final/02_2_llm_phase2_reasoning_model_specific_prompt_final_colab.ipynb`
   - Mixed Emotion routed sample의 Llama 2/Llama 3 Phase 2
3. `notebooks/colab/final/03_mixed_emotion_end_to_end_orchestration_final_colab.ipynb`
   - Mixed Emotion full-set end-to-end 결과와 논문용 표·그림
4. `notebooks/colab/final/04_5_reddit_test_routed_phase2_original_text_primary_final_colab.ipynb`
   - Reddit `tau=0.70` routed 171건 original-text Phase 2와 12,000건 end-to-end 결과

재현 및 감사:

- `scripts/paired_end_to_end_analysis.py`
- `scripts/high_confidence_accepted_error_audit.py`
- `reports/statistics/`
- `reports/high_confidence_accepted_error_audit/`

## 11. 제출본 동결 기준

다음 조건을 만족하면 최종 제출본을 동결한다.

- Table I의 classifier `Pending`이 실제 결과 또는 명시적으로 정당화된 제외로 정리됨
- 모든 본문·Appendix 숫자가 prediction-level artifact에서 재현됨
- Primary routing policy가 `alpha=0.05`, `tau=0.70`으로 일관됨
- Reddit 최종 수치는 original-text Phase 2 결과만 사용함
- Internal prompt version이 reader-facing 방법명으로 노출되지 않음
- `Depression`, `Neutral`, `Happy` label과 proxy-emotion task 정의가 일치함
- 임상 진단, 치료 권고, 자동 의료 의사결정 주장을 하지 않음
- 표, 그림, 수식, Appendix, References에 잘림·겹침·누락이 없음
- Source, PDF, result artifact, environment/run manifest가 함께 보관됨

실험이나 결정이 추가되면 이 문서를 먼저 갱신한 뒤 README, workflow guide, 원고를 동기화한다.
