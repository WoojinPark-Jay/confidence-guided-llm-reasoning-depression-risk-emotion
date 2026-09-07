# 제출 준비용 원고 전면 교정 보고서

- 작업일: 2026-09-07
- 기준본: `CGSLR_IEEE_Access_Overleaf_Final_2026_09_03_v4.zip`의 `main.tex`
- 수정본: `Paper_20260907_submission_polished/main.tex`
- 범위: 초록부터 결론 및 부록의 서술, 논리 연결, 중복, 용어, 수치 해석, 알고리즘 설명, 조판
- 제외: 미확보 참고문헌 7편의 원문 검증, 추가 모델 실행, 새 실험 결과 생성
- 성격: 제출에 가까운 교정본. 미확인 사항을 완료된 검증이나 실험으로 바꾸지 않았다.

## 1. 핵심 결론

연구 질문, 두 단계 파이프라인, 실험 수치, 그림, 참고문헌 번호는 유지했다. 새 결과를 만들지 않고 기존 결과의 설명을 더 정확하고 간결하게 만들었다. 전체 diff는 73개 변경 블록이며, 참고문헌 전 본문을 대상으로 한 영문 단어 근사 집계는 14,531개에서 12,301개로 약 15% 줄었다. 이는 LaTeX 표기까지 포함한 비교용 근사치다.

가장 중요한 수정은 문체보다 주장 범위다. 데이터 재사용, 학습 방식 차이, calibration fit과 독립 검증의 차이, 모델과 프롬프트 효과의 혼재를 본문의 결론과 일치시켰다. 단순한 긍정적 수사를 늘리는 방식으로 제출 준비도를 높이지 않았다.

## 2. 섹션별 변경

| 위치 | 수정 내용 | 이유 |
|---|---|---|
| Abstract | 문제를 case selection과 correction의 분리로 시작하고 모든 핵심 결과를 유지. Reddit prompt-development 재사용을 한 문장으로 명시 | 결과의 긍정적 해석과 연구 설계의 범위를 일치 |
| Introduction | 일반적인 위기·기술발전 수사와 반복 맥락을 축약 | 실제 연구 질문에 더 빨리 도달 |
| RQ1a/RQ1b | '충분히 calibrated인가', '선택을 뒤집는가' 대신 평가할 대상을 직접 기술 | 결과를 미리 전제하는 질문을 피함 |
| Contributions | 새로운 calibration estimator나 SELF-DISCOVER 자체를 발명한 것이 아님을 명확히 함 | 기여를 통합·평가 프로토콜에 위치시킴 |
| Literature Review | encoder와 generative model의 역할 구분, 반복적 발전 서사 축약, naive Bayes LaTeX 오기 수정 | 기술 용어 및 문체 정리 |
| Calibration 관련 문헌 | selective classification이 calibration 자체를 해결한다는 인상을 제거 | confidence calibration과 abstention은 다른 문제 |
| Text cleaning | sentence order를 제거했다는 서술 수정 | retained-token order와 punctuation/negation 삭제를 구분 |
| Mixed Emotion | mixed-cue 조건의 적용 범위를 affective scenarios로 한정하고 Neutral controls를 명시 | 75개 clear Neutral 사례와 기존 포함·제외 기준의 모순 해결 |
| Data partition | 'test는 final evaluation only'를 Phase 1의 학습·선택 경계로 한정 | Phase 2 prompt 개발에서 같은 routed IDs를 사용한 사실과 일치 |
| Methodology 도입 | 반복되는 파이프라인 및 bounded comparison 설명 통합 | 읽기 흐름 개선 |
| 모델 설명 | DistilBERT의 distillation 배경, Mistral GQA/SWA, Llama base/chat 구분을 유지 | 모델 배경을 무조건 2~3문장으로 지우지 않음 |
| Phase 1 비교 | practical configurations 비교임을 명시, 학습 방식 차이는 한 곳에서 설명하고 부록 설정표 유지 | 같은 학습 범위의 순수 architecture 비교라는 오해 방지 |
| Calibration | 비교 모델 모두 calibrated되었다는 문장을 operational DistilBERT에만 적용 | 실제 보고된 calibration 분석 범위와 일치 |
| Risk bound | calibration data에서 T fitting과 threshold search가 함께 수행됨을 명시 | pointwise CP bound를 최종 정책의 보편적 보장으로 해석하지 않음 |
| Algorithm 1 | minimum accepted count, feasible 후보 조건, infeasible 시 중단 명시 | 본문 수식과 알고리즘 일치 |
| Algorithm 2 | Phase 1 input과 routed original text를 구분 | 실제 두 단계의 입력 표현 차이를 반영 |
| Algorithm 3 | true labels를 reference labels로 수정 | proxy-label 연구의 용어 일관성 |
| Experimental Setup | 중복 Model Training 설명 삭제, 실제 four-trial sweep과 epoch budget 설명 정리 | sweep-selected budget과 retained best epoch를 혼동하지 않도록 함 |
| Calibration 결과 | calibration-set fit 진단임을 명시 | 같은 데이터에서 감소한 NLL/ECE를 독립적인 일반화 검증으로 과장하지 않음 |
| Phase 1 결과 | matched 3-seed 값과 고정 operational checkpoint의 역할을 간결하게 구분 | CI 안에 있다는 사실로 checkpoint 동등성을 주장하지 않음 |
| 통계 해석 | seed-level CI와 paired example-level CI의 의미 구분 | 다른 종류의 불확실성을 혼동하지 않도록 함 |
| 7B 오류 집계 | pooled 3-seed predictions가 36,000개 독립 test examples를 의미하지 않음을 명시 | 동일 12,000행 반복 평가에 대한 오해 방지 |
| Table IV | To Depression / To Neutral / To Happy로 단축하고 오류만 세는 열임을 note로 명시 | 전체 predicted count로 오독되는 문제 해결 |
| Reddit 결과표 | accuracy change는 반올림 전 count에서 계산한다고 note 추가 | 96.67 - 96.69와 -0.03 pp의 표시 차이 설명 |
| Mixed 결과 | 'same 18 errors'를 'also correcting 18 errors'로 수정 | 개수 일치가 사례 ID 일치의 증거는 아님 |
| 클래스별 결과 | class-wise accuracy를 recall로 수정하고 Mixed Emotion임을 명시 | 클래스별 정답/100의 정확한 의미 |
| 사례 분석 | 설명문만으로 Phase 1의 실제 cue attribution을 단정한 문장 완화 | generated rationale와 causal explanation 구분 |
| 계산량 | routing rate를 LLM calls 절감과 직접 동일시하던 문장 수정 | multistage prompting의 호출 수, 토큰, 비용은 별도 측정 필요 |
| Discussion | source-text preservation의 일반적인 인과 효과로 확대하지 않음 | 최종 rerun에서 관측된 결과 범위 유지 |
| Limitations | 반복적인 향후연구 문단을 5개 구체적 범주로 통합 | 관련 없는 multimodal 확장 나열 대신 직접적인 검증 과제 제시 |
| Phase 2 비교 한계 | model family와 prompting method가 함께 달라지는 점 추가 | Llama 3 SELF-DISCOVER의 효과를 SELF-DISCOVER 단독 효과로 주장하지 않음 |
| Conclusion | 핵심 수치와 'selection과 correction은 별도 요구사항'을 중심으로 재작성 | 초록·결과·한계와 일치하는 결론 |
| Appendix A1c | 과도한 Needspace를 줄여 앞 페이지에 배치 | 불필요한 공백 감소 |
| Appendix A2/A3 | 표는 설명용 요약이며 verbatim prompt가 아님을 명시 | publication-safe role 표현과 실제 실행 prompt를 구분 |
| Appendix A4g | 표 글씨를 한 단계 확대 | 숫자 및 열 이름의 가독성 개선 |
| Appendix A5 | 포함/제외 기준과 Neutral controls를 정합적으로 수정 | 생성 프로토콜 설명 내부 모순 제거 |

## 3. 그대로 보존한 내용

- DistilBERT matched mean accuracy/F1 96.90%, SD 0.12 및 보고된 CI
- Mistral 95.56%, Llama 2 95.09% 및 SD/CI
- operational DistilBERT 96.69%, T*=1.7706, tau*=0.70
- Reddit 12,000행, routed 171, routed Phase 1 errors 87, 전체 Phase 1 errors 397
- Reddit Llama 2 corrected/introduced/net = 47/50/-3, Llama 3 = 42/12/30
- Mixed Emotion 300행, routed 44, Phase 1 errors 56, routed errors 21
- Mixed Emotion Llama 2 = 18/6/12, Llama 3 = 18/0/18
- 기존 bootstrap CI, exact McNemar p, Holm p, conditional oracle 값
- 표의 모든 수치행, 그림 원본 전체, 아키텍처 draw.io 원본
- 46개 bibliography 항목 및 번호
- 실제 사례 인용과 저장된 모델 출력. 분석 해석 문장은 교정했지만 관측 데이터를 다시 쓰지 않음

## 4. 대표 문장 전후

### 4.1 처리하지 않은 사실을 주장하지 않기

이전: `removal of negation, punctuation, sentence order, and discourse markers`

수정: `Cleaning preserves the order of retained tokens but can remove negation, punctuation, and discourse markers.`

### 4.2 비용과 처리량 구분

이전: `These routing rates quantify the reduction in LLM calls ...`

수정: `These are fractions of inputs, not counts of individual LLM calls: CoT and SELF-DISCOVER can require multiple generation stages per routed post.`

### 4.3 이유를 알고 있다는 인상 제거

이전: `Phase 1 predicted Happy after emphasizing encouraging feedback ...`

수정: `Phase 1 predicted Happy for a post that begins with encouraging feedback but ends with unresolved sadness.`

출력 label만으로 내부 attention이나 실제 원인을 확인한 것처럼 쓰지 않는다.

### 4.4 결과의 실제 범위

추가: `In Phase 2, model family and prompting method vary together, so the observed advantage of Llama 3 SELF-DISCOVER cannot be attributed to SELF-DISCOVER alone.`

이는 실험 결과를 낮추기 위한 표현이 아니라, 비교한 단위가 model-plus-prompt configuration임을 정확히 밝히는 문장이다.

### 4.5 프롬프트 표의 재현성

수정: `This table summarizes the protocol ... it is not a verbatim prompt transcript.`

실제 실행 문자열은 저장된 notebook을 기준으로 한다. 이번 교정에서 실행 prompt를 새로 바꾸거나 실험을 다시 실행하지 않았다.

## 5. 검수 결과와 범위

- pdfLaTeX로 재컴파일하고 최종 결과는 28페이지로 정리했다. 기준 PDF는 30페이지였다.
- 전체 페이지 PNG를 렌더링하여 본문/그림/표/부록 배치를 검토했다.
- Table A1c가 Appendix A의 예시표 아래에 배치되도록 조정했다.
- accepted-error 사례 테이블의 중간 과도한 빈 페이지를 만들지 않도록 기존 분할 배치를 유지했다.
- 그림은 재래스터화하지 않았고 기존 vector PDF와 draw.io를 그대로 유지했다.
- PDF 텍스트 추출을 확인했다. 검출된 대체 문자, 누락된 내부 label 참조, 중복 label은 없다.
- 기준 ZIP과 비교해 numeric table rows와 bibliography, figures의 보존을 자동 확인했다.
- 전달 ZIP을 별도 cleanroom에 풀어 pdfLaTeX로 반복 컴파일했다. 28페이지 출력과 전체 페이지 추출 텍스트가 전달 PDF와 동일하고, unresolved reference 경고는 없었다.
- 기존 IEEE Access 클래스는 header/footer용 zero-width box, spot color 및 font substitution 경고를 남긴다. '컴파일 경고 0'이라고 주장하지 않는다. 렌더링에서 본문이나 표가 해당 폭만큼 넘치는 현상은 관찰되지 않았다.
- 일부 부록은 설명 단위와 표를 나누지 않기 위한 여백이 남는다. references 끝, 마지막 페이지의 여백은 의도적으로 억지로 채우지 않았다.

## 6. 제출 전 실제로 남은 사항

1. 미확보 참고문헌 [9], [20], [22], [27], [31], [38], [45]의 원문 확인 상태는 그대로 미완료다. 이번 교정이 이를 대체하지 않는다.
2. 저자명, 소속, 교신저자, funding/acknowledgment, 저자 약력 및 실제 AI 사용 내역의 최종 입력은 저자 확인이 필요하다. 템플릿의 출판 volume/year는 실험 연도를 의미하지 않는다.
3. Phase 2 prompt-development에 사용한 사례를 다시 평가한 한계가 남아 있다. 문장 교정으로 독립 confirmatory study가 된 것은 아니다.
4. 사람 검토와 비용 측정은 미실시 상태를 정직하게 유지했다. 이 교정에서 수행한 것으로 바꾸지 않았다.
5. 실제 생성 prompt, 역할 표현, 데이터 배포 범위와 공개 Reddit 인용의 재식별 위험은 저자가 최종 확인해야 한다.
6. 출판용 전체 보장이나 게재 확률을 제시하지 않는다. 제출 전 공동저자 검토와 최종 Overleaf clean compile이 필요하다.

## 7. 공동연구자 리뷰 방법

- 먼저 본 보고서의 섹션별 변경표와 남은 사항을 확인한다.
- `manuscript_submission_polish_2026_09_07_full_diff.md`에서 모든 변경의 정확한 전후 문장을 확인한다.
- 수치 자체가 변경된 것으로 보이면 기준 ZIP과 대조한다. 이번 작업은 결과 갱신이 아니라 문장·해석·조판 교정이다.
- 새 버전이 합의되기 전까지 2026-09-03 v4 원본은 그대로 유지한다.

Git 공유 대상은 본 보고서와 전체 문장 전후 비교 문서다. 교정 PDF와 Overleaf ZIP은 별도 전달 파일이며, 이 문서 커밋에는 포함하지 않는다.
