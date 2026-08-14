# 논문 내용 보강 및 제출 준비 현황

업데이트 기준일: 2026-08-14

## 1. 문서 목적

이 문서는 현재 논문에서 완료한 방법론, 실험, 결과 해석, Appendix, 재현성 보강 사항을 한 곳에서 확인하기 위한 협업용 기록이다. 논문 원본과 민감한 Reddit 원문 전체는 GitHub에 게시하지 않고, 공개 가능한 방법·집계 결과·작업 상태만 정리한다.

## 2. 현재 상태 요약

- 논문 내용 및 논리 구조: 약 95% 완료
- 완료된 주요 보강 작업: 13개 묶음
- 제출 전 필수 잔여 작업: 3개
- 추가 권장 작업: 2개
- 현재 가장 큰 미완료 항목: Mistral 7B 및 Llama 2 7B의 matched Phase 1 classifier 실험

현재는 방법론을 새로 설계하거나 논문을 다시 쓰는 단계가 아니다. 남은 비교 실험값과 그림을 채운 뒤 제출용 대조 검수를 수행하는 단계이다.

## 3. 완료된 주요 보강 작업 12개

### 3.1 연구 주장과 임상적 경계 정리

- 연구 대상을 임상 진단이 아닌 `Depression`, `Neutral`, `Happy`의 연구용 proxy emotion classification으로 명확히 제한하였다.
- Depression-risk-related 표현이 임상 진단 능력으로 오해되지 않도록 제목, Abstract, Introduction, Limitations, Conclusion의 주장을 조정하였다.
- 결과는 preliminary research signal이며 실제 임상 의사결정에는 clinician-annotated validation이 필요하다는 범위를 명시하였다.

### 3.2 데이터 분할 및 held-out 평가 규약 고정

- Reddit 데이터는 class별 40,000건, 총 120,000건으로 구성하였다.
- Train/validation/calibration/test를 70/10/10/10으로 분리하였다.
- 논문에는 class별 4,000건, 총 12,000건의 held-out test set으로 통일하였다.
- Calibration split과 test split의 역할을 분리하여 test leakage를 방지하였다.

### 3.3 DistilBERT Phase 1 학습 결과 확정

- W&B Bayesian sweep, best hyperparameter 선택, final checkpoint 저장 흐름을 정리하였다.
- 최종 DistilBERT held-out accuracy는 96.69%이다.
- 최종 checkpoint를 calibration, Reddit held-out evaluation, Mixed Emotion inference에 동일하게 적용하였다.
- Mistral 7B와 Llama 2 7B classifier 값은 추정하지 않고 실제 실험 전까지 `Pending`으로 유지하였다.

### 3.4 Confidence calibration 방법론 보강

- Raw MSP와 temperature-scaled MSP를 구분하였다.
- Calibration split에서 NLL을 최소화하는 optimal temperature를 추정하는 과정을 수식으로 정리하였다.
- 최종 DistilBERT temperature는 1.7706이다.
- NLL, Brier score, ECE, Adaptive ECE의 정의와 상호 보완 관계를 설명하였다.
- Temperature scaling 전후의 calibration quality를 표와 reliability analysis로 연결하였다.

### 3.5 Risk-controlled routing 정책 정리

- Accepted set, routed set, coverage, routing rate, selective risk를 수식으로 정의하였다.
- Clopper-Pearson upper confidence bound를 사용해 accepted risk를 보수적으로 평가하였다.
- Calibration set에서 `0.70, 0.71, ..., 1.00` 후보를 평가하였다.
- Primary policy는 `alpha=0.05` risk constraint를 만족하면서 coverage가 가장 높은 threshold를 선택하도록 고정하였다.
- 최종 primary threshold는 `tau=0.70`이다.
- 최종 원고에는 사전 정의한 `alpha=0.05`, calibration-selected `tau=0.70` 정책만 보고하고, Phase 2 결과를 이용한 사후 threshold 재선택은 하지 않았음을 명확히 하였다.

### 3.6 Advanced confidence-threshold 분석 보강

- Risk-coverage curve, threshold sweep, calibration reliability, confidence distribution을 산출하였다.
- MSP, margin, negative entropy confidence score 비교를 정리하였다.
- Per-class selective metrics, bootstrap stability, threshold provenance, high-confidence errors를 재현 산출물로 남겼다.
- Threshold 선택 결과와 sensitivity 결과를 본문 주 결과와 보조 분석으로 구분하였다.

### 3.7 Reddit Phase 2 원문-input end-to-end 결과 확정

- Primary policy에서 12,000건 중 171건(1.42%)이 routed되었다.
- Routed subset에는 Phase 1 오류 87건이 포함되었고 routed-only Phase 1 accuracy는 49.12%였다.
- Routed 171건 전부를 retained original `title + selftext`에 exact matching했고, conflicting original은 0건이었다.
- Phase 2 입력은 URL과 직접 username 패턴만 최소 마스킹하고, 부정어, 문장부호, 문장 순서, 감정 전환을 보존하였다.
- Llama 2 CoT는 47건을 수정하고 50건의 오류를 새로 만들어 net correction -3, end-to-end accuracy 96.67%를 기록하였다.
- Llama 3 SELF-DISCOVER는 42건을 수정하고 12건의 오류를 새로 만들어 net correction +30, end-to-end accuracy 96.94%를 기록하였다.
- Llama 3의 +0.25 percentage-point 변화는 paired bootstrap 95% CI `[0.13, 0.38]`, exact McNemar `p < 0.0001`, Holm-adjusted `p=0.0002`로 양의 paired effect가 확인되었다.
- Routing 성공과 reasoning 성공을 분리하여, low-confidence subset을 잘 찾는 것만으로 최종 정확도 향상이 보장되지는 않는다고 해석하였다.

### 3.8 Mixed Emotion stress-test 결과 확정

- Synthetic Mixed Emotion v2.4는 300건, class별 100건으로 고정하였다.
- 이 데이터는 training, hyperparameter tuning, temperature fitting, threshold selection에 사용하지 않고 supplementary stress test로만 사용하였다.
- Phase 1 baseline accuracy는 81.33%이며 44건(14.67%)이 routed되었다.
- Llama 2 CoT는 18건 수정, 6건 도입 오류로 85.33%(+4.00 pp)를 기록하였다.
- Llama 3 SELF-DISCOVER는 18건 수정, 0건 도입 오류로 87.33%(+6.00 pp)를 기록하였다.
- Synthetic stress-test 결과를 자연 발생 데이터의 prevalence 또는 임상적 일반화 증거로 과장하지 않도록 서술 범위를 제한하였다.

### 3.9 Prompt policy 및 SELF-DISCOVER 설명 보강

- Llama 2 CoT와 Llama 3 SELF-DISCOVER의 prompt/output contract를 Appendix에 정리하였다.
- Llama 2는 independent assessment, Phase 1 comparison, canonical terminal label 구조를 갖는 CoT prompt를 최종 정책으로 고정하였다. 내부 버전 식별자는 논문 표기에서 제거하였다.
- Llama 3는 universal prompt variant가 아니라 기존 SELF-DISCOVER policy를 최종 구성으로 유지하였다.
- SELECT, ADAPT, IMPLEMENT, Answer 필드가 실제 output에서 어떤 역할을 하는지 설명하였다.
- 모듈 번호만 나열하던 예시를 실제 processing 단계와 독자가 이해할 수 있는 기능 설명으로 보강하였다.

### 3.10 실제 Phase 2 correction 사례 추가

- Phase 1이 Happy로 틀렸으나 Phase 2가 Depression으로 수정한 실제 Llama 2 및 Llama 3 사례를 각각 제시하였다.
- Llama 2의 단계별 응답과 Llama 3의 SELECT/ADAPT/IMPLEMENT/Answer 필드를 실제 저장 컬럼과 연결하였다.
- 최종 label뿐 아니라 어떤 textual evidence와 reasoning trace로 수정했는지 확인할 수 있도록 구성하였다.

### 3.11 통계 분석과 결과 해석 보강

- 동일 sample에 대한 paired bootstrap confidence interval과 exact McNemar test를 추가하였다.
- 다중 비교에는 Holm adjustment를 적용하였다.
- Reddit의 작은 변화와 Mixed Emotion의 양의 효과를 같은 강도로 주장하지 않고 데이터 regime별로 분리하여 해석하였다.
- 유의성 결과는 필요한 범위에서만 보고하고, 부정적 결과를 과장하거나 숨기지 않도록 정리하였다.

### 3.12 High-confidence accepted-error 원문 감사 완료

- `tau=0.70`에서 accepted되었지만 틀린 310건을 별도로 분석하였다.
- Accepted selective risk는 2.62%이며, accepted reference-Depression 3,939건 중 false negative는 125건(3.17%)이었다.
- Confidence 0.98 이상에서도 accepted error 34건이 남았다.
- 310건 전부를 retained Reddit `title`과 `selftext`에 normalized exact matching으로 연결하였다.
- 명확한 원문 근거가 있는 대표 사례 6건을 Depression, Happy, Neutral에서 각각 2건씩 선정하였다.
- 대표 failure mode는 acute distress 누락, severe negative-state 누락, technical context가 pride/excitement를 가리는 현상, mental-health topic-term shortcut, factual question에 대한 affect 오판이다.
- 원문 검토는 reference label이나 평가 지표를 사후 변경하는 용도가 아니라 residual-risk transparency를 위한 audit로 한정하였다.
- 민감한 Reddit 원문 전체 310건은 GitHub에 게시하지 않고 로컬 제한 산출물로 관리한다.

### 3.13 최종 원고 및 Overleaf 패키지 검증

- Abstract, Methods, Results, Discussion, Limitations, Conclusion을 최종 원문-input 결과에 맞춰 갱신하였다.
- Reddit, Mixed Emotion, paired statistics, routing policy, calibration 결과를 본문 표에 일관되게 반영하였다.
- Appendix에는 실제 Llama 2/Llama 3 correction output, canonical SELF-DISCOVER module pool, observed Reddit original-text correction, high-confidence accepted-error 원문 사례를 포함하였다.
- 최종 PDF 25페이지를 페이지별 렌더링하여 표 잘림, 텍스트 겹침, horizontal overflow, undefined reference가 없음을 확인하였다.
- 논문 LaTeX/PDF 원본은 GitHub에 게시하지 않고 로컬 Overleaf package로 관리하며, GitHub에는 공개 가능한 방법·집계 결과·재현 문서만 유지한다.

## 4. 현재 남은 제출 전 필수 작업 3개

### [ ] 4.1 Mistral 7B 및 Llama 2 7B Phase 1 classifier matched run

- 동일 데이터 분할, metric, calibration protocol로 full-scale classifier 실험을 완료한다.
- 실제 결과로 본문 비교표의 `Pending`을 교체한다.
- 결과가 준비되지 않을 경우 두 모델을 완료된 성능 비교 대상으로 제시하지 않고 planned comparison 범위에서 제거하거나 명확히 제한한다.

### [ ] 4.2 최종 핵심 그림 삽입

- Two-phase architecture diagram
- Calibration/reliability figure
- Risk-coverage 또는 threshold-selection figure
- 핵심 confusion matrix/end-to-end comparison figure
- 그림 번호, caption, 본문 cross-reference를 고정한다.

### [~] 4.3 IEEE Access 제출 전 최종 대조 검수

- 저자명, 소속, 교신저자 정보를 확정한다.
- 본문, 표, Appendix, CSV의 현재 완료 수치를 대조하였다. 남은 Phase 1 comparison 결과가 추가되면 다시 대조한다.
- 현재 표·수식·알고리즘·참고문헌 번호를 검수하였다. 그림 삽입 후 최종 번호를 다시 고정한다.
- 현재 25페이지 PDF의 overflow, 표 잘림, 겹침, 부자연스러운 page break를 페이지별로 확인하였다.
- 제출 파일에 임시 문구, 내부 메모, 추정값이 없는지 확인한다.

## 5. 추가 권장 작업 2개

### [ ] 5.1 Phase 2 비용 및 처리 효율 보고

- Routed sample 수, 평균 처리시간, GPU 종류, token 사용량, 대략적인 실행비용을 표로 정리한다.
- 모든 sample에 LLM을 적용하는 방식과 selective routing의 계산 효율 차이를 설명한다.

### [ ] 5.2 Human/expert rationale review

- 실제 임상 효능 검증이 아니라 rationale의 textual faithfulness, clarity, harmful overclaim 여부를 검토한다.
- 제출 일정상 수행하지 못할 경우 Limitations와 Future Work에 명시한다.

## 6. 제출 전 판단 기준

필수 3개가 완료되면 현재 원고는 IEEE Access 제출용 최종 패키지 검수 단계로 넘어갈 수 있다. 권장 2개는 논문의 실용성과 설명가능성 방어를 강화하지만, 연구 범위를 명확히 제한한다면 모두가 제출의 절대 조건은 아니다.

## 7. 관련 문서

- [최종 End-to-End Colab Workflow](final_end_to_end_workflow_ko.md)
- [IEEE Access 제출 준비 체크리스트](ieee_access_submission_readiness_checklist_ko.md)
- [논문 업데이트 및 현재 결과 미팅 노트](paper_update_and_current_results_meeting_notes_ko.md)
- [High-Confidence Accepted-Error Audit 완료 결과](high_confidence_accepted_error_audit_results_ko.md)
- [Routing Threshold Policy Audit](routing_threshold_policy_audit_ko.md)
- [최종 Model-Specific Prompt Policy](final_model_specific_prompt_policy_ko.md)
- [Reference 업데이트 완료 보고서](reference_update_completion_report_ko.md)
