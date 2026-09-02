# IEEE Access 최종 제출 준비 체크리스트

최종 갱신: 2026-09-03

대상 원고: *Confidence-Guided Selective LLM Re-Evaluation for Depression-Risk-Related Emotion Classification in Social Media Text*

> 현재 결과와 상세 수치는 `docs/final_research_status_and_remaining_work_ko.md`를 기준으로 한다. 이 문서는 실제 제출 직전의 실행 체크리스트다.

## 1. 현재 판정

- **핵심 실험:** 완료
- **Phase 1 matched comparison:** 완료
- **Phase 2와 end-to-end 평가:** 완료
- **통계·오류 감사:** 완료
- **공식 IEEE Access 템플릿 이식 및 교수 검토용 패키지:** 완료
- **제출 전 필수 미완료:** 저자/소속 등 실제 메타데이터, 공개 범위, Overleaf pdfLaTeX clean compile, 최종 artifact 동결

Phase 1 분류기 실측 placeholder는 제거되었으며, 남은 작업은 새로운 핵심 실험보다 투고 정보와 최종 제출 패키지 확정에 가깝다.

## 2. 완료된 연구 요소

- [x] Reddit 클래스별 40,000건, 총 120,000건 구성
- [x] Train/validation/calibration/test = 70/10/10/10 분리
- [x] DistilBERT W&B sweep, 3-seed 평가와 operational checkpoint 저장
- [x] Llama 2 7B 및 Mistral 7B frozen linear-probe Bayesian sweep과 3-seed 평가
- [x] 동일 12,000건 held-out test에서 세 Phase 1 모델 비교
- [x] Temperature scaling 및 NLL, Brier, ECE, Adaptive ECE 평가
- [x] Calibration-only risk-constrained threshold 선택
- [x] Reddit held-out 12,000건 Phase 1와 routing 평가
- [x] Reddit routed 171건 original `title + selftext` Phase 2 재평가
- [x] Mixed Emotion 300건 Phase 1/Phase 2/end-to-end 평가
- [x] Llama 2 CoT와 Llama 3 SELF-DISCOVER 비교
- [x] Corrected/introduced/net correction, routed-error enrichment, conditional oracle 분석
- [x] Paired bootstrap 95% CI, exact McNemar, Holm correction
- [x] Accepted high-confidence Phase 1 오류의 정량·정성 감사
- [x] Prompt, SELF-DISCOVER module pool, 저장 output 예시, synthetic generation protocol 부록
- [x] 참고문헌 46건 감사 및 원고 인용 수정
- [x] 최종 architecture 도식과 editable Draw.io 원본 보존
- [x] Table/Figure/Appendix 레이아웃과 reader-facing 명칭 정리
- [x] 공식 `ieeeaccess.cls` 기반 Overleaf 패키지와 교수 검토 PDF 생성

## 3. 최종 실측값 동기화 확인

- [x] DistilBERT: accuracy/macro F1 `96.70 +/- 0.10%`
- [x] Mistral 7B linear probe: accuracy/macro F1 `95.56 +/- 0.11%`
- [x] Llama 2 7B linear probe: accuracy/macro F1 `95.09 +/- 0.06%`
- [x] Operational DistilBERT: held-out accuracy `96.69%`, `T*=1.7706`, `tau*=0.70`
- [x] Reddit Llama 3: `96.94%`, `+0.25 pp`, net `+30`
- [x] Mixed Emotion Llama 3: `87.33%`, `+6.00 pp`, net `+18`
- [x] Abstract, Table I, Phase 1 figure, RQ1, Results, Discussion, Limitations, Conclusion 동기화
- [x] `TBD`, `Pending`, planning estimate를 보고 결과 영역에서 제거

## 4. 제출 전 반드시 완료할 작업

### [ ] 4.1 저자와 제출 메타데이터

- 최종 저자 순서, 영문 이름, 소속, 이메일, corresponding author를 확정한다.
- 제출 저자의 ORCID와 영문 biography를 입력한다.
- Funding/Acknowledgment와 conflict-of-interest 문구를 실제 상황대로 입력한다.
- AI-assisted writing/coding 사용 공개 문구를 IEEE 정책과 실제 사용 범위에 맞게 확정한다.
- Ethics/IRB 또는 exemption 관련 표현을 기관 판단과 맞춘다.

### [ ] 4.2 Data/code availability와 공개 범위

- GitHub 공개 범위, Mixed Emotion 데이터, 결과 CSV/JSON, checkpoint 공개 여부를 확정한다.
- License, release tag, archival DOI 또는 장기 보존 경로를 정한다.
- 원고의 availability 문구와 실제 공개 파일이 일치하는지 확인한다.

### [ ] 4.3 공식 제출 패키지 clean compile

- 메타데이터가 반영된 최신 source와 figure를 새 Overleaf 프로젝트에 올린다.
- Compiler를 **pdfLaTeX**로 설정하고 clean compile한다.
- Undefined reference/citation, overfull box, font substitution 경고를 점검한다.
- Abstract, 표, 그림, Appendix, References, biography의 page break와 잘림을 전 페이지 확인한다.
- 최신 source와 최종 PDF의 수치, 그림 번호, caption, reference를 대조한다.

### [ ] 4.4 최종 동결

- Source, PDF, supplementary artifact, environment/run manifest를 같은 릴리스로 묶는다.
- SHA-256 checksum과 생성일을 기록한다.
- 제출 직전 원고에서 `TBD`, `Pending`, placeholder author, 임시 경로를 다시 검색한다.
- 최종 제출본 이후에는 사전 명시된 재현 오류가 아닌 한 threshold, prompt, dataset을 결과에 맞춰 변경하지 않는다.

## 5. 권장하지만 1차 제출의 절대 조건은 아닌 작업

### [ ] 5.1 Frozen-prompt 독립 확인

- 보고 prompt를 더 이상 수정하지 않고 prompt 개발에 사용하지 않은 subset에서 방향성을 확인한다.

### [ ] 5.2 복수 연구자 또는 전문가 검토

- Mixed Emotion label/rationale와 대표 Reddit 사례를 독립 검토한다.
- 수행하지 않을 경우 clinical validity를 주장하지 않고 현재 한계를 유지한다.

### [ ] 5.3 실제 처리시간과 계산 효율

- GPU, wall-clock time, 생성 token, routed call 수와 비용을 기록한다.
- 현재 논문은 invocation avoidance를 보고하며 실제 비용·energy 절감을 실측한 것으로 과장하지 않는다.

### [ ] 5.4 참고문헌 전문 감사 보완

- 46건 중 39건은 전문 또는 동등한 full web source로 검증했다.
- `[9], [20], [22], [27], [31], [38], [45]`는 초록·서지 기반 감사 상태다.
- `[22]`, `[31]` 전문 확보를 우선하되, 현재 서지의 존재가 틀렸다는 의미는 아니다.

## 6. 최종 작업 순서

1. 저자·소속·ORCID·corresponding author·biography 확정
2. Funding, conflict, AI use, ethics, availability 문구 확정
3. 공개 artifact와 release 범위 확정
4. 최신 source/figure/metadata를 공식 IEEE Access Overleaf 프로젝트에 동기화
5. pdfLaTeX clean compile과 전 페이지 시각·논리 검수
6. Source/PDF/supplementary/run manifest/checksum 동결
7. IEEE Access 제출 시스템 메타데이터와 PDF를 최종 대조 후 제출

## 7. 제출 가능성 판정

방법론, Phase 1 비교, routing, Phase 2, paired statistics, accepted-error audit, Appendix와 architecture는 1차 제출용 핵심 내용이 갖춰진 상태다. 4절의 실제 투고 정보와 clean compile을 완료하면 IEEE Access 제출 후보본으로 동결할 수 있다. 권장 항목은 리뷰 방어력을 높이지만 현재 연구 구조를 다시 설계해야 하는 필수 조건은 아니다.

## 8. 공식 지침

- IEEE Access Submission Guidelines: https://ieeeaccess.ieee.org/authors/submission-guidelines/
- IEEE Access Preparing Your Article: https://ieeeaccess.ieee.org/authors/preparing-your-article/
- IEEE Access Post-Acceptance Guide: https://ieeeaccess.ieee.org/authors/post-acceptance-guide/
