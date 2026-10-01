# 공동연구자 리뷰용 최종 연구 현황과 제출 전 작업

2026-10-01 기준

## 한 줄 결론

**계획한 연구 실험, 행별 결과 검증, 대응 통계 분석, 주요 원고 반영은 완료했다. 현재 결과를 완성하기 위해 반드시 다시 돌려야 할 모델 실험은 없다. 남은 핵심 작업은 Figure 1 교체, 최신 원고 기준 참고문헌 재검수, 제출 정보 확정, 재현 자료 정리, 최종 PDF 통합 검수다.**

## 현재 상태 한눈에 보기

| 영역 | 상태 | 공동 리뷰에서 확인할 내용 |
|---|---|---|
| Phase 1 분류기 비교 | 완료 | 세 모델 비교 범위와 해석이 과장되지 않았는지 |
| 최종 DistilBERT 연결 | 완료 | 별도 운영 체크포인트 없이 최종 모델에서 라우팅까지 연결했는지 |
| Calibration·routing | 완료 | 온도, 임계값, coverage와 error capture 설명 |
| Reddit·Mixed Emotion 12조건 | 완료 | Llama 2·3 × Direct·CoT·SELF-DISCOVER 결과 |
| 추가 9,000건 평가 | 완료 | 최종 Llama 3 세 프로토콜 고정 평가와 행별 검증 |
| 통계 검정 | 완료 | Phase 1 대비 개선과 방법 간 비교를 구분했는지 |
| 본문·부록 업데이트 | 완료 | 수치, 프롬프트, 계획, 실패 처리와 해석 |
| Figure 1 | 남음 | 기존 원본을 최소 수정해 최종 실험 구조와 맞추기 |
| 참고문헌 최종 감사 | 남음 | 최신 문장과 46개 참고문헌을 다시 연결하고 누락 인용 정리 |
| 저자·제출 정보 | 남음 | 저자, 소속, 교신저자, 연구비, 윤리, 공개 범위 |
| 재현 패키지 | 남음 | 최종 코드·결과·계획·manifest·파일 목록 정리 |
| 최종 PDF·Git | 남음 | 통합 검수 후 커밋과 원격 푸시 |

## 지금까지 완료한 연구와 실험

### 1. Phase 1 분류기 비교

- Reddit 120,000건을 클래스별로 균형 있게 구성했다.
- train/validation/calibration/test를 70/10/10/10으로 분리했다.
- DistilBERT, Llama 2 7B, Mistral 7B를 같은 12,000건 test에서 seeds 42, 43, 44로 비교했다.
- DistilBERT는 full fine-tuning, 두 7B 모델은 frozen-backbone linear probe라는 실제 비교 범위를 원고에 명시했다.
- 세 모델을 최대한 최적화한 절대 순위가 아니라, 공통 데이터와 제한된 탐색 예산 아래의 실용적 비교로 해석했다.

| 모델 | Accuracy, mean ± SD | Macro F1, mean ± SD |
|---|---:|---:|
| DistilBERT | 96.88 ± 0.04% | 96.88 ± 0.04% |
| Mistral 7B | 95.36 ± 0.08% | 95.36 ± 0.08% |
| Llama 2 7B | 95.17 ± 0.07% | 95.17 ± 0.07% |

### 2. Phase 1 효율성과 최종 모델 연결

- 동일 A100 환경에서 DistilBERT와 7B 비교 모델의 throughput, single-post latency, peak memory를 측정했다.
- DistilBERT의 성능뿐 아니라 첫 단계 분류기로서의 계산 효율성도 근거로 제시했다.
- 검증 기반으로 선택된 하이퍼파라미터의 최종 seed-42 DistilBERT 모델이 예측, calibration, routing, Phase 2 입력을 연속해서 공급하도록 통일했다.
- 과거 원고의 비교용 모델과 별도 운영 체크포인트 구분은 제거했다.
- 최종 온도는 1.480195, routing threshold는 0.70이다.

### 3. Calibration과 selective routing

- 별도 calibration split에서 temperature scaling을 수행했다.
- NLL, Brier score, ECE, adaptive ECE로 calibration 품질을 확인했다.
- one-sided selective-risk upper bound를 만족하는 후보 중 coverage가 가장 큰 threshold를 선택했다.
- test label을 threshold 선택에 사용하지 않았다.
- confidence가 threshold보다 낮은 사례만 Phase 2로 전달했다.

### 4. Phase 2 비교 설계

- Reddit과 Mixed Emotion 모두에서 Llama 2와 Llama 3를 평가했다.
- 각 모델에 Direct, CoT, 최종 SELF-DISCOVER를 적용해 총 12조건을 비교했다.
- 각 데이터셋 안에서 모든 방법은 동일한 Phase 1 결과와 routed ID를 사용한다.
- Phase 2는 Phase 1의 정제 입력이 아니라 제한적으로 비식별 처리한 원문 title과 selftext를 사용한다.
- 파싱에 실패하면 해당 사례를 제외하거나 정답으로 복구하지 않고 Phase 1 label을 유지한다.
- Llama 2와 Llama 3의 각 설정은 별도 비교 조건이며 ensemble이 아니다.

### 5. 최종 SELF-DISCOVER 방법

- 최종 방식은 v6c이며, researcher-authored task policy와 module bank를 입력으로 모델별 task-level plan을 한 번 생성한다.
- Llama 2와 Llama 3가 각각 만든 계획을 해당 모델의 모든 routed post에 재사용한다.
- 게시글별 실행은 emotional subject, temporal context, 세 label의 경쟁 근거, 대안 해석을 점검한다.
- Direct와 CoT도 같은 routed post에 적용해 모델 효과와 prompting protocol 효과를 함께 비교했다.
- 이전 independent-countercheck SD와 최종 SD 비교는 부록에 design-variant 비교로 제시했다.
- 여러 요소가 동시에 달라졌기 때문에 단일 구성요소 causal ablation으로 주장하지 않는다.

### 6. 추가 9,000건 평가

- 원래 120,000건에 포함되지 않은 같은 source corpus의 글에서 클래스별 3,000건, 총 9,000건을 구성했다.
- 정규화된 exact-text overlap과 후보군 내부 정규화 중복을 제거했다.
- 최종 DistilBERT 모델, calibration temperature, routing threshold, Llama 3 Direct·CoT·SELF-DISCOVER prompt, 저장된 SD plan을 고정한 뒤 평가했다.
- 이 9,000건으로 재학습, recalibration, threshold 재탐색, SD plan 재생성, prompt 수정은 하지 않았다.
- 세 프로토콜은 같은 111개 routed post를 평가했고 최종 지표는 전체 9,000건을 분모로 계산했다.
- 9,000개 ID, 111개 routed post, 세 방법의 전체 예측, 777개 저장 generation call을 검증했다.

## 최종 핵심 결과

| 데이터 | 전체 글 | Routed | Phase 1 Accuracy | Llama 3 최종 SD Accuracy | Corrected | Introduced | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reddit | 12,000 | 218 | 96.9167% | 97.2000% | 72 | 38 | +34 |
| Mixed Emotion | 300 | 86 | 83.6667% | 92.6667% | 28 | 1 | +27 |
| 추가 9,000건 | 9,000 | 111 | 97.5778% | 97.7667% | 30 | 13 | +17 |

해석의 중심은 다음과 같다.

- Confidence-guided routing은 적은 비율의 사례에 Phase 1 오류를 집중시킨다.
- LLM re-evaluation은 자동으로 좋아지는 단계가 아니며 모델과 prompting protocol에 따라 corrected와 introduced error가 달라진다.
- Llama 3 최종 SELF-DISCOVER는 세 평가셋에서 가장 높은 관측 accuracy와 macro F1을 보였다.
- Mixed Emotion에서는 두 모델 모두 SD가 Direct와 CoT보다 paired comparison에서 우수했다.
- Reddit에서 방법 간 차이는 더 작다. 따라서 관측 최고 성능과 통계적으로 확립된 방법 간 우위를 구분해 서술했다.
- 추가 9,000건에서 Llama 3 SD의 Phase 1 대비 개선은 Holm-adjusted p=0.0412다.
- 추가 9,000건에서 SD와 Direct·CoT 사이의 작은 차이는 통계적으로 확립되지 않았지만, Phase 1 대비 순개선은 확인됐다.

## 완료한 통계와 결과 검증

- 총 15조건을 example ID로 정렬해 Phase 1, routed output, final prediction, reference label을 대조했다.
- accuracy, macro F1, class-wise F1, confusion matrix, corrected, introduced, net correction을 원시 예측에서 재계산했다.
- exact McNemar test로 paired correctness를 비교했다.
- paired percentile-bootstrap confidence interval을 계산했다.
- Phase 1 대비 비교와 SD 대 Direct·CoT 비교에 서로 맞는 Holm family를 적용했다.
- 파싱 실패와 fallback을 모든 분모와 통계에 포함했다.
- 실패 사례만 다시 생성하거나 정답을 보고 label을 채우지 않았다.
- 표, 본문, 부록의 수치가 행별 검증 결과와 일치하도록 반영했다.

## 완료한 논문 수정

- 제목을 `Confidence-Guided Selective LLM Re-Evaluation for Emotion Classification in Social Media`로 정리했다.
- 초록, 서론, contributions, research questions, methodology, results, discussion, limitations, conclusion을 최종 실험 구조에 맞췄다.
- Phase 1 대비 개선과 re-evaluator 간 비교를 분리해 해석했다.
- SELF-DISCOVER를 보편적으로 항상 우월한 방식으로 과장하지 않으면서, 최종 관측 성과와 일관된 net correction은 분명하게 제시했다.
- Phase 2 original-text 입력과 Phase 1 cleaned-text 입력의 역할을 구분했다.
- 9,000건 same-source evaluation의 구성, 고정 조건, 한계와 결과를 본문과 부록에 반영했다.
- Direct·CoT·SD prompt, 모델별 실제 생성 plan, parser/fallback, generation budget을 부록에 수록했다.
- 본문·표·부록·그림의 수치를 최종 결과로 교체했다.
- 문장 중복, AI식 표현, 과도한 방어 문구, 모호한 기술 용어를 전반적으로 다듬었다.
- 최신 PDF는 29쪽이며, Figure 1을 제외한 현재 표·그림·본문 배치를 검토했다.

## 참고문헌의 현재 상태

- 기존 46개 참고문헌은 실재 여부, 주요 서지 정보, DOI·URL, 인용-주장 적합성을 과거 원고 기준으로 감사했다.
- 잘못된 URL, 저자, 제목, 연도와 확인된 인용 범위 문제는 이전 교정에서 수정했다.
- 이후 최종 실험 구조, Llama 3 세 프로토콜, SELF-DISCOVER 설명, 추가 9,000건 평가와 통계 문장이 크게 늘었다.
- 따라서 참고문헌을 처음부터 다시 수집할 필요는 없지만, **최신 원고 문장 기준의 최종 citation-to-claim audit는 남아 있다.**
- 현재 정적 확인에서는 bibliography 46개 중 45개가 본문에서 인용되고, Llama 3 원 논문 `ref40`은 목록에는 있으나 본문 citation이 빠져 있다. Llama 3 모델 설명이나 Phase 2 설정 문장에 연결할지, 사용하지 않는다면 bibliography에서 제거할지 결정해야 한다.
- 새 참고문헌은 숫자를 늘리기 위해 추가하지 않는다. 최신 핵심 주장에 현재 출처가 부족할 때만 primary source를 추가한다.

## 제출 전 반드시 마무리할 작업

### 1. Figure 1 최종 교체

- 기존 drawio 레이아웃, 컬러, 글꼴과 연결 구조를 최대한 유지한다.
- Phase 2 박스에 Llama 2와 Llama 3가 각각 Direct, CoT, SELF-DISCOVER로 평가된다는 점을 표시한다.
- 여섯 조건이 별도 실행이며 ensemble이 아님을 오해하지 않도록 한다.
- parsing failure 시 Phase 1 label을 유지한다는 규칙을 반영한다.
- `Data-use scope` 문구는 Phase 1 개발·calibration과 평가셋의 관계를 정확히 표현한다.
- 최종 drawio 원본과 벡터 PDF를 모두 보관한다.

완료 기준: 본문 Figure 1, caption, methodology 설명이 서로 일치하고 PDF 확대 시 글자와 선이 선명하다.

### 2. 최신 원고 기준 참고문헌 최종 감사

다음 순서로 한 번 더 확인한다.

1. 본문의 모든 `\cite{}` key와 bibliography key의 누락·중복·미사용 여부를 검사한다.
2. 현재 확인된 `ref40` Llama 3 미인용 문제를 정리한다.
3. 새로 작성한 문장별로 현재 citation이 실제 주장 범위를 직접 지원하는지 대조한다.
4. DistilBERT, Llama 2, Llama 3, Mistral, CoT, SELF-DISCOVER는 원 논문 또는 공식 primary source에 연결한다.
5. Calibration, selective classification, risk-coverage와 주요 통계 설명에 필요한 근거가 충분한지 확인한다.
6. Reddit depression/emotion classification 관련 배경과 synthetic-data 관련 문장이 확보한 원문의 범위를 넘지 않는지 확인한다.
7. 저자, 제목, 저널·학회, 연도, 권·호·페이지, DOI·URL, access date를 최신 bibliography와 대조한다.
8. 추가 문헌이 정말 필요한 주장만 표시하고, 기존 문헌으로 충분하면 불필요하게 reference 수를 늘리지 않는다.
9. citation 번호와 본문 인용 순서를 최종 PDF에서 확인한다.

완료 기준: 미인용 bibliography, 존재하지 않는 key, 잘못된 DOI·URL, 출처보다 강한 본문 주장, 근거 없는 새 주장이 없다.

### 3. 공동 저자와 제출 정보 확정

- 저자명과 저자 순서
- 소속과 주소
- 교신저자와 이메일
- 연구비·지원기관·과제번호
- Conflict of Interest
- Data Availability
- Code Availability
- Ethics/IRB 또는 면제·비해당 설명
- acknowledgment
- IEEE Access 제출 시스템에서 요구하는 저자 역할과 추가 문구

완료 기준: 저자 placeholder와 `VOLUME XX` 같은 검토용 표시 중 저자가 채워야 하는 항목을 제거하거나 확정한다. 출판사가 부여하는 정보는 임의로 만들지 않는다.

### 4. 재현 자료 패키지 정리

- 최종 Colab notebooks
- Phase 1 전체 예측과 routed input
- 12조건 및 추가 9,000건 결과 CSV
- end-to-end prediction CSV와 confusion matrix
- 최종 SELF-DISCOVER plan JSON
- experiment manifest와 환경·모델 revision·파일 hash
- prompt 원문과 parser/fallback 규칙
- 통계 계산 코드와 검증 보고서
- 공개 가능한 데이터, 비공개 원문, 모델 가중치 링크의 경계를 설명한 README

완료 기준: 다른 연구자가 어떤 파일로 어느 표를 재현하는지 README만 읽고 이해할 수 있다. 민감한 원문이나 재배포가 허용되지 않은 모델 파일은 공개 저장소에 올리지 않는다.

### 5. 최종 통합 PDF 검수

- Figure 1 번호, caption, 본문 참조
- 표·그림·부록의 수치 일치
- 본문 citation과 reference 번호
- 페이지 넘김, 큰 공백, 겹침, 잘림, 흐린 이미지
- 수식 번호와 Algorithm 번호
- 약어의 최초 정의
- 저자 placeholder와 제출용 metadata
- PDF 글꼴 포함과 벡터 그림 선명도
- Overleaf와 로컬 PDF의 일치

완료 기준: 최종 PDF 한 파일을 처음부터 끝까지 읽고 내용·수치·형식 오류가 없다는 공동 확인을 남긴다.

### 6. Git 커밋과 원격 푸시

- 최신 `main.tex`와 빌드 스크립트
- 최종 Figure 1 drawio와 vector PDF
- 최종 notebooks와 통계·검증 코드
- 결과 감사 보고서와 이 현황 문서
- 재현 자료 README
- reference 최종 감사 문서

완료 기준: 공동연구자가 원격 저장소에서 최종 원고와 재현 자료의 위치를 바로 찾을 수 있고, 오래된 실험 결과를 최종본으로 오인하지 않는다.

## 제출을 막지는 않지만 향후 보강 가치가 있는 작업

| 항목 | 의미 | 현재 판단 |
|---|---|---|
| Phase 2 실제 시간·토큰·비용의 완전한 matched benchmark | 생성 비용을 정량 비교 | 있으면 강해지지만 현재 논문의 필수 조건은 아님 |
| Mixed Emotion 복수 연구자·전문가 검토 | synthetic label과 사례 해석의 신뢰도 강화 | 유용하지만 현재 실험 결과의 필수 재실행 조건은 아님 |
| author-disjoint 또는 temporal split | 같은 작성자·시기 의존성 검증 | 후속 검증 가치가 큼 |
| 외부 corpus 평가 | cross-domain 일반화 확인 | 후속 연구에 적합 |
| generation seed 반복 | 생성 변동성 정량화 | greedy decoding 결과의 추가 안정성 분석으로 유용 |
| 공정성·하위집단 분석 | deployment 전 편향 검토 | 실제 적용을 주장하려면 필요하지만 본 논문은 비임상 연구 범위 |

이 항목들은 limitations와 future work에 정직하게 남긴다. 현재 원고를 완성하기 위해 새로 반드시 수행해야 하는 실험으로 취급하지 않는다.

## 다시 하지 않아도 되는 작업

- DistilBERT, Llama 2, Mistral Phase 1 학습 재실행
- 기존 세 seed 분류 비교 재실행
- Reddit·Mixed Emotion 12조건 재생성
- 추가 9,000건 분류 또는 Llama 3 세 프로토콜 재실행
- 이미 실패로 계산된 응답만 골라 재생성
- 모든 탐색 prompt를 본문에 나열
- 별도 운영 체크포인트를 다시 도입
- 결과를 더 높이기 위한 추가 prompt search

## 공동 리뷰 권장 순서

1. 이 문서의 완료·남은 상태에 합의한다.
2. 최신 29쪽 PDF의 제목, 초록, contributions, 핵심 결과, limitations, conclusion을 먼저 읽는다.
3. Figure 1 문구를 확정하고 최종 원본으로 교체한다.
4. 최신 원고 기준 reference audit를 마친다.
5. 저자·공개·윤리 정보를 확정한다.
6. 재현 패키지 README를 확인한다.
7. 최종 PDF를 처음부터 끝까지 통합 검수한다.
8. Git에 최종 커밋하고 원격에 푸시한 뒤 제출본을 고정한다.

## 공동 리뷰에 사용할 파일

- 최신 PDF: `Paper_20260930_verified/main.pdf`
- 최신 LaTeX: `reports/verified_manuscript_20260930/main.tex`
- 원고 전체 검수: `reports/verified_manuscript_20260930/REVIEW_KO.md`
- 문장·내용 검수: `reports/verified_manuscript_20260930/LANGUAGE_AND_CONTENT_REVIEW_20261001_KO.md`
- 결과·통계 감사: `reports/final_results_audit_20260930/README.md`
- Figure 1 수정 지침: `reports/unified_v6c_revision_20260929/ARCHITECTURE_EDIT_GUIDE_KO.md`
- 과거 참고문헌 감사: `reports/reference_audit_2026_08_24/reference_full_audit_ko.md`
- 현재 문서: `reports/verified_manuscript_20260930/PROJECT_STATUS_AND_SUBMISSION_TASKS_20261001_KO.md`

## 최종 진행 순서

**Figure 1 확정 → 최신 원고 기준 reference audit → 저자·공개·윤리 정보 확정 → 재현 패키지 정리 → 최종 PDF 통합 검수 → Git 커밋·푸시 → 제출**
