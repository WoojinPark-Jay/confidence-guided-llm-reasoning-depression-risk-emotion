# IEEE Access 제출 준비 체크리스트

이 문서는 현재 완료된 실험을 기준으로, IEEE Access 제출 전에 실제로 마무리해야 할 작업을 공동 점검하기 위한 체크리스트다. 이미 완료된 실험을 반복하거나 논문의 핵심과 직접 관련이 없는 분석은 제외한다.

## 1. 현재 고정된 주요 결과

- Reddit Phase 1 DistilBERT 정확도: 96.69%
- Reddit primary routing policy: temperature 1.7706, threshold 0.70, routed 171건(1.42%)
- Reddit routed subset Phase 1 정확도: 49.12%; routed Phase 1 error 87건
- Reddit end-to-end Llama 2 CoT: 96.67%, net correction -3
- Reddit end-to-end Llama 3 SELF-DISCOVER: 96.94%, net correction +30
- Mixed Emotion 300건 Phase 1 정확도: 81.33%; routed 44건(14.67%)
- Mixed Emotion end-to-end Llama 2 CoT: 85.33%, net correction +12
- Mixed Emotion end-to-end Llama 3 SELF-DISCOVER: 87.33%, net correction +18
- Primary routing policy: 사전 정의한 alpha 5% 위험예산과 calibration-selected threshold 0.70을 고정해 held-out 및 Phase 2 결과를 평가함

## 2. 제출 전 필수 작업

### [ ] 2.1 Mistral 7B 및 Llama 2 7B Phase 1 분류 실험 완료

- DistilBERT와 동일한 Reddit 데이터와 train/validation/calibration/test 조건을 적용한다.
- Accuracy, macro Precision, macro Recall, macro F1을 저장한다.
- Temperature, NLL, Brier score, ECE, adaptive ECE를 저장한다.
- 모델별 threshold, coverage, routing rate를 저장한다.
- 원고 Table I과 관련 `Pending` 값을 실제 결과로 교체한다.
- 완료 전까지 DistilBERT가 세 모델 중 최고라고 단정하지 않는다.

### [ ] 2.2 Reddit held-out test 표본 수 최종 고정

- 최종 산출물과 원고에서 held-out test를 클래스별 4,000건, 총 12,000건으로 통일한다.
- 저장된 prediction CSV, summary CSV/JSON, confusion matrix의 표본 수를 함께 확인한다.
- Abstract, Methods, Results, Conclusion, Appendix의 표본 수가 모두 일치하는지 확인한다.

### [ ] 2.3 최종 재현 실행 및 산출물 고정

- Final 01: DistilBERT 학습, calibration, threshold, Reddit/Mixed Phase 1 결과
- Final 02.2: Mixed Emotion 최종 model-specific Phase 2 결과
- Final 03: Mixed Emotion end-to-end 논문용 결과
- Final 04.4: Reddit alpha risk-budget 정책 비교
- 최종 모델, best hyperparameters, temperature, threshold, seed, 모델 revision, 패키지 버전을 보존한다.
- CSV, JSON, Excel, PNG, run manifest를 Google Drive와 로컬에 함께 보관한다.

### [ ] 2.4 핵심 도식 제작 및 삽입

- 전체 confidence-guided two-phase architecture
- Reddit 데이터 분할과 실험 흐름
- Temperature scaling과 confidence routing 과정
- Phase 1 accepted/routed 분기와 Phase 2 label replacement 과정
- 도식의 수치와 본문 표의 수치가 일치하는지 확인한다.

### [ ] 2.5 원고 수치 및 용어 최종 검수

- Reddit: 96.69%, 171건, 1.42%, 49.12%, corrected/introduced/net 값 확인
- Mixed Emotion: 81.33%, 44건, 14.67%, 85.33%, 87.33% 확인
- Temperature 1.7706, threshold 0.70, alpha 5% 표기 확인
- `Depression`, `Neutral`, `Happy` label 표기를 코드와 논문에서 통일한다.
- `proxy emotion classification`과 임상 진단을 명확히 구분한다.

## 3. 강하게 권장하는 보강 작업

### [x] 3.1 High-confidence accepted error audit

- Phase 1이 accept했지만 틀린 Reddit 사례를 confidence 구간과 class별로 정리한다.
- 실제 Depression을 Neutral 또는 Happy로 예측한 false negative를 별도로 점검한다.
- 대표 성공 사례뿐 아니라 라우팅이 포착하지 못한 오류 사례도 Appendix 또는 한계에 반영한다.
- 정확한 12,000-row held-out protocol에서 accepted error 310건, accepted selective risk 2.62%, accepted Depression false negative 125건(3.17%)을 확인했다.
- Confidence 0.98 이상에서도 accepted error 34건이 남았으며, 실제 사례 6건을 유형화했다.
- 상세 결과는 `docs/high_confidence_accepted_error_audit_results_ko.md`에 정리했다.

### [ ] 3.2 Phase 2 비용 및 처리 효율 기록

- GPU 종류, 실행시간, routed sample 수, 추론 token 또는 호출 수를 기록한다.
- Reddit에서 전체의 1.42%, Mixed Emotion에서 14.67%만 LLM reasoning을 수행한 효과를 정리한다.
- 모든 샘플을 LLM에 보내는 방식과 비교한 호출 감소율을 계산한다.
- 실제 비용이 없으면 추정 비용을 단정하지 않고 호출 수와 처리시간만 보고한다.

### [ ] 3.3 최종 prompt 독립 확인

- 논문에 보고하는 Llama 2 CoT와 Llama 3 SELF-DISCOVER prompt를 더 이상 수정하지 않고 고정한다. 내부 실험 버전명은 재현 이력에서만 사용한다.
- 가능하면 prompt 수정에 사용하지 않은 별도 routed subset에서 방향성이 유지되는지 확인한다.
- 현재 Reddit prompt-policy 비교는 탐색적 분석으로 구분하고, 독립 확인 결과가 있을 때만 일반적 우월성을 주장한다.

### [ ] 3.4 Mixed Emotion 품질 및 scenario audit

- 중복 또는 과도하게 유사한 문장을 검사한다.
- 클래스별 길이, scenario 분포, 주요 표현의 편향을 확인한다.
- label이 한 개의 노골적인 키워드로 결정되지 않는지 점검한다.
- 가능하면 두 명 이상의 연구자가 독립적으로 일부 또는 전체 label을 검토하고 불일치 사례를 기록한다.
- 합성 stress test 결과를 실제 임상 또는 Reddit 모집단의 성능으로 일반화하지 않는다.

### [ ] 3.5 클래스별·오류 유형별 결과 정리

- Reddit 및 Mixed Emotion의 class별 Precision, Recall, F1을 최종 산출물에서 확인한다.
- Depression false negative의 Phase 1 대비 Phase 2 변화를 확인한다.
- Corrected, introduced, unchanged-correct, unchanged-wrong 사례를 모델별로 정리한다.
- Mixed Emotion scenario별 결과가 현재 Appendix 수치와 일치하는지 검증한다.

## 4. 원고 완성도 점검

### [ ] 4.1 표와 부록

- 모든 표가 IEEE 두 단 레이아웃 밖으로 넘치지 않는지 확인한다.
- 표 글자 크기, 열 간격, caption 형식을 통일한다.
- 본문에서 모든 표와 알고리즘을 순서대로 인용한다.
- Appendix A2/A3 prompt가 실제 최종 코드의 prompt와 일치하는지 대조한다.
- Appendix reasoning example이 실제 저장 CSV에서 가져온 값인지 확인한다.

### [ ] 4.2 References와 Related Work

- 본문 인용 번호와 bibliography 순서를 확인한다.
- DOI, 저자명, 논문명, 저널/학회명, 연도를 원 출처와 대조한다.
- Calibration, selective classification, mental-health NLP, CoT, SELF-DISCOVER 관련 선행연구와 본 연구의 차이를 명확히 기술한다.

### [ ] 4.3 재현성과 윤리

- Data/Code Availability에 실행 순서와 공개 가능한 산출물 위치를 적는다.
- W&B, Hugging Face 토큰이 코드나 출력 파일에 포함되지 않았는지 확인한다.
- Reddit proxy label과 synthetic data의 한계를 명확히 적는다.
- 실제 임상 진단, 치료 권고, 자동화된 의료 의사결정으로 해석되지 않도록 표현을 점검한다.

### [ ] 4.4 최종 PDF 검수

- 제목·초록·본문·표·부록·References의 페이지 넘김을 확인한다.
- 빈 페이지, 고립된 heading, 겹치는 표, 잘린 수식, 과도하게 작은 글자를 확인한다.
- 그림과 표를 포함한 최종 PDF의 모든 페이지를 실제 화면으로 검수한다.
- 제출본과 코드 저장소의 최종 결과 수치가 일치하는지 마지막으로 대조한다.

## 5. 권장 작업 순서

1. Mistral 7B 및 Llama 2 7B Phase 1 실험 완료
2. Reddit held-out test를 정확히 12,000건으로 고정하고 최종 산출물 재생성
3. Final workflow 재현 산출물 고정
4. 핵심 도식 제작 및 삽입
5. High-confidence accepted error audit 완료 결과를 원고와 supplementary artifact에 동기화
6. 비용·처리 효율 정리
7. 모든 수치, 표, Appendix, References 최종 검수
8. Overleaf 최종 PDF 페이지별 검토 후 제출

Paired exact McNemar test, paired bootstrap 95% confidence interval, Holm 보정은 완료되어 원고에 반영되었다. 해석은 p-value만으로 결정하지 않고 accuracy change, corrected/introduced/net corrections, confidence interval, 입력 regime을 함께 본다.
