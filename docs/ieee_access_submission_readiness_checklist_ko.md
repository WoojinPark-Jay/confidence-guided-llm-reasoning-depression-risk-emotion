# IEEE Access 최종 제출 준비 체크리스트

최종 갱신: 2026-08-29
대상 원고: *Confidence-Guided Selective LLM Re-Evaluation for Depression-Risk-Related Emotion Classification in Social Media Text*

> 이 문서는 분류 실험을 제외한 현재 완성도와 IEEE Access 실제 제출 전 남은 작업을 공동연구자가 한번에 확인하기 위한 기준 문서다.

## 1. 현재 판정

- **연구 실험과 방법론: 약 90--95% 완료**
- **가장 큰 미완료:** Mistral 7B와 Llama 2 7B의 matched Phase 1 classifier 실측 실험
- **제출 파일 필수 미완료:** 저자/소속/교신저자, ORCID, 연구비·Acknowledgment, AI 사용 공개, 저자 약력, 실측치 반영 후 최종 일관성 감사
- **템플릿 상태:** 2026-05-13 공식 IEEE Access LaTeX 템플릿으로 원고 이식 완료

분류 실험과 제출 메타데이터를 완성하고, 수치·표·그림·본문 일관성을 다시 검수하면 IEEE Access 1차 제출 가능 단계다.

## 2. 완료된 핵심 연구 요소

- [x] Reddit 클래스별 40,000건, 총 120,000건 구성
- [x] Train/validation/calibration/test = 70/10/10/10 분리
- [x] W&B sweep, best hyperparameter 선택, DistilBERT 최종 checkpoint 저장
- [x] Temperature scaling 및 NLL, Brier, ECE, adaptive ECE 평가
- [x] Calibration split에서 risk-constrained routing threshold 선택
- [x] Reddit held-out 12,000건 Phase 1 평가
- [x] Reddit original title + selftext 기반 171건 Phase 2 재평가
- [x] Mixed Emotion 300건 Phase 1/Phase 2/end-to-end 평가
- [x] Llama 2 CoT와 Llama 3 SELF-DISCOVER 비교
- [x] Corrected/introduced/net correction, routing concentration, conditional oracle 분석
- [x] Paired bootstrap 95% CI, exact McNemar, Holm correction
- [x] High-confidence accepted-error 정량·정성 감사
- [x] Prompt, SELF-DISCOVER module pool, 저장 output 예시, synthetic generation protocol 부록
- [x] 참고문헌 46건 감사 및 발견된 원고 문제 9건 수정
- [x] 최종 two-phase architecture 도식 완성 및 원고 삽입
- [x] 편집 가능한 Draw.io 원본과 publication-ready 벡터 PDF를 GitHub `main`에 보존
- [x] 기존 IEEEtran 최종본 독립 컴파일·전 페이지 시각 검수
- [x] 2026-05-13 공식 `ieeeaccess.cls` 기반 새 Overleaf 패키지 생성

## 3. 제출 전 반드시 완료할 작업

### [ ] 3.1 Mistral 7B와 Llama 2 7B Phase 1 matched classifier 실험

- DistilBERT와 동일한 Reddit split, label, preprocessing, evaluation protocol을 적용한다.
- 모델별 Accuracy, Macro Precision, Macro Recall, Macro F1을 저장한다.
- 학습 설정, PEFT 방식, seed, checkpoint/revision, GPU와 package 버전을 보존한다.
- 현재 Table 1과 Figure 1의 italic TBD planning estimate를 실측치로 교체한다.
- RQ1, Methods, Results, Discussion, Limitations, Conclusion의 모델 비교 문장을 같이 갱신한다.
- 실측 결과 전에 DistilBERT의 비교 우위를 단정하지 않는다.

### [ ] 3.2 저자와 제출 메타데이터

- 최종 저자 순서, 영문 이름, 소속, 이메일, 교신저자를 확정한다.
- 제출 저자의 공개·완성된 ORCID 프로필을 확인한다.
- 모든 저자의 짧은 영문 약력을 참고문헌 이후에 추가한다.
- 연구비, 이해상충, 데이터/코드 공개 위치, IRB 또는 면제 판단을 실제 상황대로 기재한다.
- AI-generated manuscript text 사용 범위를 IEEE 정책에 맞게 Acknowledgment에 공개한다.

### [x] 3.3 IEEE Access 공식 LaTeX 템플릿 이식

- 공식 지침: https://ieeeaccess.ieee.org/authors/submission-guidelines/
- 사용 릴리스: `ACCESS_latex_template_20260513`
- 새 패키지: `Paper_260620_ieee_access_template_final`
- `ieeeaccess.cls`, spot-color, logo, font asset 포함
- Access 전용 title/byline, abstract, keywords, `\EOD` 구조 반영
- 로컬 QA 환경에서 31페이지 전체 원고 호환성 확인
- Overleaf 컴파일러는 반드시 **pdfLaTeX**로 설정한다.

### [ ] 3.4 실측치 반영 후 최종 일관성 감사

- `TBD`, `Pending`, `planning estimate`, `must be replaced` 표현을 제출본에서 전부 제거한다.
- Abstract, Table 1, Figure 1, RQ1, Discussion, Limitations, Conclusion의 수치와 해석을 대조한다.
- 최신 architecture PDF와 Appendix/본문 수정본을 공식 `ieeeaccess.cls` 패키지에 최종 동기화한다.
- 최종 LaTeX 소스와 PDF 내용이 일치하는지 확인한다.
- 약어 첫 정의, 표·그림 본문 인용, 참고문헌 번호, 잘린 표/그림을 다시 검수한다.
- 최종 패키지를 새 Overleaf 프로젝트에 올리고 pdfLaTeX로 clean compile한다.

## 4. 강하게 권장하지만 1차 제출의 절대 조건은 아닌 작업

### [ ] 4.1 실제 처리시간과 계산 효율

- GPU, 벽시간, routed sample 수, 추론 token/호출 수를 기록한다.
- 현재 결과는 LLM 호출 회피율을 보여주지만 실제 비용·에너지 절감을 실측한 것으로 과장하지 않는다.

### [ ] 4.2 Frozen-prompt 독립 확인

- 보고 prompt를 더 이상 변경하지 않고 고정한다.
- 가능하면 prompt 수정에 사용하지 않은 소규모 별도 subset에서 방향성을 확인한다.

### [ ] 4.3 복수 연구자 또는 전문가 사례 검토

- Synthetic Mixed Emotion label과 rationale의 신뢰성을 보강한다.
- 수행하지 못하는 경우 현재 원고처럼 clinical validity를 주장하지 않고 한계를 명시한다.

### [ ] 4.4 참고문헌 전문 감사 보완

- 참고문헌 46건 중 39건은 전문 또는 동등한 full web source로 검증했다.
- 접근 제한이 있던 `[9], [20], [22], [27], [31], [38], [45]`는 초록·서지 기반 감사 상태다.
- 추가 전문 확보 우선순위는 `[22]`, `[31]`이다. 현재 인용의 존재와 서지가 틀렸다는 뜻은 아니다.

## 5. 최종 작업 순서

1. Mistral 7B와 Llama 2 7B matched Phase 1 실험
2. Table 1, Figure 1, RQ1 및 관련 해석에 실측치 반영
3. 저자/소속/ORCID/교신저자/약력/연구비/Acknowledgment 입력
4. AI 사용, 윤리, 데이터·코드 공개 문구 확정
5. 최신 본문·architecture·Appendix를 공식 IEEE Access 패키지에 동기화
6. 전체 수치·용어·인용·약어·레이아웃 감사
7. IEEE Access 공식 템플릿 Overleaf에서 pdfLaTeX clean compile
8. LaTeX source, PDF, supplementary artifact 최종 동기화 후 제출

## 6. 당장 제출해도 되는지에 대한 최종 판정

현재 프레임워크, 라우팅, Phase 2, 통계, 오류 감사, prompt 부록, reference audit, 최종 architecture 도식은 완성단계다. 다만 실측 분류 비교가 없고 저자 정보가 빈 현재 본은 **교수 검토용 완성본**이지 **즉시 제출본**은 아니다. 3절의 필수 항목을 완료하면 IEEE Access 1차 제출 가능 상태로 판정한다.

## 7. 공식 지침

- IEEE Access Submission Guidelines: https://ieeeaccess.ieee.org/authors/submission-guidelines/
- IEEE Access Preparing Your Article: https://ieeeaccess.ieee.org/authors/preparing-your-article/
- IEEE Access Post-Acceptance Guide: https://ieeeaccess.ieee.org/authors/post-acceptance-guide/
