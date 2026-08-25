# 참고문헌 전체 감사 보고서

- 감사일: 2026-08-24
- 대상 원고: `Paper_260620_overleaf_original_text_phase2_final_self_discover_figure/main.tex`
- 대상 참고문헌: 46개 전부
- 감사 범위: 실재 여부, 서지정보, 링크/DOI, 원문 접근 수준, 본문 인용 문맥, 주장과 출처 내용의 일치 여부
- 감사 당시 원고 수정 여부: **수정하지 않음**. 이 폴더는 감사 결과와 확보 자료를 보관한다.
- 후속 조치일: 2026-08-24~25
- 후속 조치 상태: **감사에서 발견된 원고 수정 9건을 교정 원고에 모두 반영하고 독립 재컴파일 및 29페이지 전체 렌더링 검수를 완료함**

## 0. 후속 조치 및 현재 상태

이 보고서의 문제 설명은 감사 당시 원고를 기준으로 한 발견 기록이다. 이후 교정 원고
`CGSLR_IEEE_Access_2026_Full_Paper_Reference_Audited_Overleaf.zip`에 아래 수정 사항을 모두 반영했다.
따라서 아래 3~4절에 적힌 문제는 미해결 목록이 아니라, 무엇을 발견하고 어떻게 고쳤는지를 보존하는 감사 추적 기록이다.

| Ref./위치 | 감사에서 발견된 문제 | 교정 원고의 조치 | 상태 |
|---|---|---|---|
| [32] | eRisk overview가 아닌 `invited_paper_5.pdf`로 연결 | 올바른 `invited_paper_1.pdf`로 URL 교체 | 완료 |
| [36] | 공식 저자 목록과 불일치 | 공식 6인 저자 목록으로 교체 | 완료 |
| [44] | 공식 ACL 제목과 불일치 | *Depression Detection on Social Media with Large Language Models*로 교체 | 완료 |
| Related Work의 [31] | transformer 확산의 직접 근거로 부적절 | 해당 문맥에서 [31]을 제거하고 [8], [33], [34]로 교체 | 완료 |
| [8] 본문 | MentalBERT 출판연도를 2021로 표기 | `Ji et al. (2022)`로 수정 | 완료 |
| [18] | arXiv-only 서지 | NeurIPS 2024 정식 venue와 DOI로 갱신 | 완료 |
| [20] 본문 | 확보 근거보다 넓은 linguistic-marker 주장 | 공식 초록으로 확인된 first-person singular pronouns와 negative-emotion words만 남김 | 완료 |
| [41] | TextBlob 홈페이지 수준의 넓은 URL | polarity 범위를 직접 설명하는 sentiment-analysis 문서로 교체 | 완료 |
| [46] | publisher HTML/PDF 간 제목 충돌 | version-of-record PDF의 *Mental Illness Detection* 제목으로 통일하고 충돌 사실은 감사 기록에 유지 | 완료(외부 메타데이터 충돌 기록 유지) |

교정 원고는 46개 참고문헌 번호 체계를 유지한다. 위 수정으로 현재 확인된 잘못된 URL, 저자, 제목, 연도 및 인용-주장 불일치는 모두 해소됐다. 남은 사항은 아래 5절의 접근 제한 논문에 대한 전문 확인 완결성 보강이며, 현재 원고에서 새로 확인된 필수 서지 오류는 아니다.

## 1. 최종 결론

참고문헌 46개는 모두 실재하는 출처로 확인됐다. 완전히 존재하지 않거나 조작된 참고문헌은 발견되지 않았다. 감사 당시 제출 전에 고쳐야 할 추적성·서지·인용 문제를 발견했으며, 해당 문제는 위 0절과 같이 교정 원고에 반영했다.

가장 심각한 문제는 **참고문헌 [32]의 URL이 전혀 다른 논문을 가리킨다는 것**이었다. 감사 당시 원고의 `invited_paper_5.pdf`는 eRisk 2018 overview가 아니라 CENTRE@CLEF replicability task 논문이었다. 올바른 eRisk 2018 overview는 `invited_paper_1.pdf`이며, 교정 원고에는 이를 반영했다.

추가로 [36]의 저자 목록, [44]의 제목이 공식 원문과 다르며, 본문의 MentalBERT 연도와 [31]의 transformer 근거 사용도 수정이 필요하다. [18]은 내용상 정확하지만 arXiv-only 서지를 NeurIPS 2024 정식 출판정보로 바꾸는 편이 적절하다.

## 2. 감사 완결성

| 검증 수준 | 개수 | 비율 | 의미 |
|---|---:|---:|---|
| 전문 PDF 직접 확인 | 32 | 69.6% | 공식 또는 저자/학회 PDF를 내려받아 본문을 대조함 |
| 공식 웹 전문 직접 확인 | 7 | 15.2% | WHO, BMJ, Cambridge, TextBlob 등 공식 HTML 전문을 대조함 |
| 공식 초록까지만 확인 | 5 | 10.9% | paywall 또는 전달 제한으로 전문을 확보하지 못함 |
| 공식 서지만 확인 | 2 | 4.3% | publisher 차단으로 제목·저자·DOI까지만 직접 확인함 |
| **전체 항목 판정 완료** | **46** | **100%** | 모든 참고문헌에 접근 수준과 판정을 부여함 |

따라서 46개 모두를 감사했지만, **내용 전문까지 확인한 항목은 39개(84.8%)**다. 나머지 7개는 접근 제한을 숨기지 않고 아래에 명시했다. 초록·서지만 확인한 논문에 대해서는 전문을 읽은 것처럼 단정하지 않았다.

## 3. 감사 당시 발견한 즉시 수정 사항

### P0: 제출 전 반드시 수정 (교정 원고 반영 완료)

#### [32] 잘못 연결된 PDF

- 원고 표기: `https://ceur-ws.org/Vol-2125/invited_paper_5.pdf`
- 실제 파일: *CENTRE@CLEF2018: Overview of the Replicability Task*
- 올바른 파일: [Overview of eRisk 2018: Early Risk Prediction on the Internet](https://ceur-ws.org/Vol-2125/invited_paper_1.pdf)
- 영향: 참고문헌 제목은 맞지만 독자가 링크를 열면 다른 논문이 나온다. 단순 형식 오류가 아니라 source traceability 오류다.

### P1: 중요한 서지·인용 수정 (교정 원고 반영 완료)

#### [36] 공식 저자 목록과 불일치

공식 저자는 다음 6명이다.

1. Rafael Salas-Zárate
2. Giner Alor-Hernández
3. María del Pilar Salas-Zárate
4. Mario Andrés Paredes-Valverde
5. Maritza Bustos-López
6. José Luis Sánchez-Cervantes

감사 당시 원고에는 실제 논문에 없는 저자가 포함되고 실제 저자 3명이 누락되어 있었다. 제목, journal, year, DOI 및 본문 인용 취지는 맞았다. [MDPI 공식 페이지](https://www.mdpi.com/2227-9032/10/2/291)

#### [44] 공식 제목과 불일치

- 감사 당시 원고: *Medical knowledge-guided depression detection on social media with large language models*
- 공식 제목: *Depression Detection on Social Media with Large Language Models*

본문에서 medical criteria, temporal mood courses, interpretability를 설명한 내용은 실제 논문과 일치한다. 제목만 공식 ACL 표기로 교체해야 한다. [ACL Anthology](https://aclanthology.org/2025.emnlp-industry.151/)

#### [31] transformer 주장에 맞지 않는 인용

[31]은 2019년 Reddit depression-post detection 연구이며 transformer adoption의 직접 근거가 아니다. 데이터셋과 Reddit classification 문단에서의 [31]은 적절하지만, “more recent work has increasingly adopted transformer-based architectures” 문장의 [31]은 [33], [34], [8] 등 실제 transformer 논문으로 교체해야 한다. [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8681445)

## 4. 감사 당시 발견한 추가 수정 권고

### [8] 본문 연도

MentalBERT의 참고문헌은 LREC 2022이며 실제 출판연도도 2022다. 본문 “Ji et al. (2021)”을 “Ji et al. (2022)”로 고쳐야 한다. [ACL Anthology](https://aclanthology.org/2022.lrec-1.778/)

### [18] 정식 venue로 업그레이드

감사 당시 arXiv 인용도 가짜이거나 틀린 것은 아니었지만, SELF-DISCOVER는 NeurIPS 2024 정식 논문으로 출판됐다. `Advances in Neural Information Processing Systems 37 (2024)`, DOI `10.52202/079017-4004`를 사용하는 것이 제출본에 더 적합하다. SELECT/ADAPT/IMPLEMENT와 atomic reasoning-module 설명은 원문과 부합한다. [NeurIPS 공식 페이지](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e41efb03e20ca3c231940a3c6917ef6f-Abstract-Conference.html)

### [20] 확보한 근거보다 조금 넓은 문장

공식 초록에서는 depressed participants가 negative emotion words와 `I`를 더 많이 사용했다는 점은 확인했다. 그러나 감사 당시 문장의 “fewer references to others”까지는 확보한 공식 초록만으로 독립 검증하지 못했다. 교정 원고는 문장을 직접 확인된 두 특징으로 좁혔다.

### [41] 더 직접적인 문서 링크

TextBlob polarity가 `[-1.0, 1.0]`이라는 주장은 공식 문서와 정확히 일치한다. 감사 당시 homepage 링크보다 [Sentiment Analysis quickstart](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis)를 직접 연결하는 편이 검증성이 높아 교정 원고에서 교체했다.

### [46] publisher 내부 제목 충돌

DOI landing metadata는 *Machine Learning Approaches for Depression Detection on Social Media...*라고 표시하지만, 공식 publisher PDF의 제목은 *Machine Learning Approaches for Mental Illness Detection on Social Media...*이다. 본문 내용은 47개 연구를 대상으로 platform, sampling, language, preprocessing, reporting bias를 다룬다는 점에서 인용 취지와 부합한다. 교정 원고는 version-of-record PDF 제목을 우선해 통일했으며, 외부 메타데이터 충돌 자체는 감사 기록에 남긴다. [Publisher DOI](https://doi.org/10.35566/jbds/caoyc)

## 5. 다운로드하지 못한 출처

아래 7개는 원문 PDF를 직접 확보하지 못했다. 이 사실은 감사 CSV에도 그대로 표시했다. 이 7개를 모두 확보하면 전문 확인률을 84.8%에서 100%로 높일 수 있지만, **7개 모두가 현재 원고의 필수 오류 수정 조건이라는 뜻은 아니다.** [20]과 [31]은 확보 범위를 넘던 문맥을 이미 수정했으며, [9], [27], [38], [45]는 현재 원고에서 공식 초록이 직접 뒷받침하는 일반적 배경 주장에만 사용한다. 방법 세부를 더 강하게 검증하려면 [22]와 [31]의 전문 확보를 우선하고, 그다음 [9], [20], [27], [38], [45]를 보완하는 순서가 효율적이다.

| Ref. | 출처 | 확보 수준 | 이유 | 현재 판정 |
|---:|---|---|---|---|
| [9] | Information Fusion survey | 공식 초록 | Elsevier paywall/redirect | 본문 주장 대체로 부합, 전문 재확인 권장 |
| [20] | Cognition and Emotion | 공식 초록 | Taylor & Francis 403/paywall | 문장 일부만 직접 검증됨 |
| [22] | CHI 2015 | 공식 서지 | ACM DOI 접근 차단 | 제목·저자·DOI 확인, 방법 세부 전문 확인 필요 |
| [27] | Neural Computation LSTM | 공식 초록 | MIT Press paywall | 일반적인 LSTM 설명은 부합 |
| [31] | IEEE Access Reddit study | 공식 서지 | IEEE PDF endpoint가 HTML/JS 응답 | dataset 문맥은 부합, transformer 근거는 부적합 |
| [38] | Current Opinion in Psychology | 공식 초록 | Elsevier paywall/redirect | 윤리·privacy 배경에 부합 |
| [45] | Journal of Affective Disorders review | 공식 초록 | Elsevier paywall/redirect | LLM promise/caution 주장에 대체로 부합 |

[5] BMJ PDF도 직접 다운로드는 403이었지만, 공식 BMJ 웹 전문을 읽어 내용은 전문 수준으로 검증했다. [7] Cambridge 논문 역시 공식 웹 전문을 확보해 검증했다.

## 6. 참고문헌별 판정 요약

### [1]–[11]: 배경, 임상·사회적 필요성, mental-health NLP

- **[1] 적합:** WHO 2025의 약 332 million 수치와 일치한다.
- **[2] 적합:** depression이 leading cause of ill health and disability라는 WHO 표현과 일치한다.
- **[3] 적합:** depression/anxiety 관련 연간 US$1 trillion productivity loss와 일치한다.
- **[4] 적합:** pandemic first year의 25% prevalence increase와 일치한다.
- **[5] 적합하나 표현 주의:** PHQ-9 screening accuracy 연구다. clinical diagnosis 근거로 확대하지 않아야 한다.
- **[6] 적합:** accessible, high-quality, person-centred mental-health systems와 system gaps를 지지한다.
- **[7] 적합:** non-clinical text를 이용한 mental-health NLP 응용과 한계를 지지한다.
- **[8] 내용 적합, 본문 연도 수정:** domain-specific pretraining 효과를 지지하지만 연도는 2022다.
- **[9] 대체로 적합:** emotion/context fusion survey의 범위와 부합하지만 초록까지만 확인했다.
- **[10] 적합:** temporal representation과 chronological context 필요성을 지지한다.
- **[11] 적합:** clinician trust와 contextual explainability 필요성을 지지한다.

### [12]–[19]: 모델, calibration, selective routing, reasoning

- **[12] 적합:** DistilBERT architecture와 compression claims가 정확하다.
- **[13] 적합:** Mistral 7B의 GQA/SWA 설명이 정확하다.
- **[14] 적합:** Llama 2 family와 model-size 설명이 정확하다.
- **[15] 적합:** modern neural-network calibration과 temperature scaling 근거가 정확하다.
- **[16] 적합:** selective classification의 coverage-risk 관점과 일치한다.
- **[17] 적합하나 범위 주의:** CoT 배경 인용으로는 맞지만 원 연구를 zero-shot 성능 근거로 표현하면 안 된다.
- **[18] 내용 적합, venue 보강:** SELF-DISCOVER protocol 설명은 맞고 정식 NeurIPS citation으로 바꾸는 것이 좋다.
- **[19] 대체로 적합:** synthetic depression data precedent는 맞지만 원 연구와 본 stress-test dataset은 다른 맥락이다.

### [20]–[39]: depression language, social-media datasets, ethics

- **[20] 부분 적합:** negative words와 first-person singular 증가는 확인했으나 문장 전체를 전문으로 검증하지 못했다.
- **[21] 적합:** linguistic, engagement, ego-network features와 depression prediction을 지지한다.
- **[22] 조건부 적합:** 서지는 맞고 인용 취지도 알려진 초록과 부합하지만 전문 접근이 막혔다.
- **[23] 적합:** self-reported diagnoses와 matched controls 설명이 맞다.
- **[24] 적합:** data quality, bias, methodological-validity 한계를 지지한다.
- **[25] 적합:** word2vec 배경이 맞다.
- **[26] 적합:** GloVe 배경이 맞다.
- **[27] 적합:** LSTM의 long-range sequential dependency 배경과 맞다.
- **[28] 적합:** CNN sentence classification 배경과 맞다.
- **[29] 적합:** six feature groups와 multimodal dictionary learning 결과가 맞다.
- **[30] 적합:** 해당 연구에서 CNN이 SVM/BiLSTM보다 우수했고 annotated-data 한계도 언급됐다.
- **[31] 용도별 판정:** Reddit study 근거로는 맞고 transformer adoption 근거로는 틀리다.
- **[32] 링크 오류:** 인용하려는 논문은 실재하고 내용도 맞지만 현재 URL이 다른 논문이다.
- **[33] 적합:** hierarchical transformer 및 post/user representation 설명과 부합한다.
- **[34] 적합:** transformer ensemble 근거로 적절하다.
- **[35] 적합:** diary text와 LLM-based depression assessment 연구 설명이 맞다.
- **[36] 내용 적합, 저자 오류:** systematic-review 결과는 본문 주장과 맞지만 서지 저자가 틀렸다.
- **[37] 적합:** construct validity, label operationalization, reporting 문제를 지지한다.
- **[38] 적합:** privacy와 ethical implications 배경에 적절하다.
- **[39] 적합:** social-media health research protocol과 privacy 근거로 적절하다.

### [40]–[46]: Llama 3, sentiment implementation, explainable LLMs, reviews

- **[40] 적합:** Llama 3 family 근거로 적절하다.
- **[41] 적합:** polarity range와 API 설명이 정확하며 direct anchor로 링크만 정밀화하면 된다.
- **[42] 적합:** Pattern sentiment backend 설명을 보조한다.
- **[43] 적합:** Reddit depression-level detection과 explanation generation을 지지한다.
- **[44] 내용 적합, 제목 오류:** DORIS 내용은 정확히 인용했지만 공식 제목을 고쳐야 한다.
- **[45] 대체로 적합:** LLMs for depression의 promise와 validation caution을 지지하지만 초록까지만 확인했다.
- **[46] 내용 적합, 제목 충돌:** bias review 근거는 맞고 publisher HTML/PDF 간 제목을 통일해야 한다.

## 7. 감사 결과의 의미

이번 감사에서 핵심 방법론인 temperature scaling, selective classification, CoT, SELF-DISCOVER, DistilBERT/Mistral/Llama architecture에 사용된 기술적 근거는 원문과 대체로 정확하게 일치했다. 사회적 필요성, mental-health NLP 배경, construct-validity 및 ethics 근거도 대부분 적합했다.

따라서 참고문헌 체계를 전면 교체할 필요는 없다. [32], [36], [44], [31], [8]을 포함해 감사에서 확인된 원고 수정 사항은 교정 원고에 모두 반영했다. 접근 제한 7개는 전문 확인 완결성의 한계로 이 보고서에 계속 명시한다. 추가 전문을 확보할 수 있다면 방법 세부 검증 가치가 큰 [22]와 [31]을 먼저 확인하고, 나머지 5개를 순차적으로 보완하는 것이 가장 효율적이다.

## 8. 산출물 안내

- `reference_full_audit_ko.md`: 사람이 읽는 전체 감사 보고서
- `reference_full_audit.csv`: 46개 참고문헌별 검증 수준과 판정
- `reference_issues.csv`: 수정 우선순위가 있는 문제만 모은 목록
- `downloads/`: 내려받은 공식 PDF/HTML
- `extracted/`: PDF에서 추출한 검색 가능한 텍스트
- `logs/download_results.tsv`: 최초 다운로드 HTTP 결과

세부 판정은 `reference_full_audit.csv`가 기준이며, 원문 접근이 제한된 항목은 `전문 확인`으로 과장하지 않았다.

GitHub 공개본에는 저작권과 재배포 조건을 고려해 `downloads/`와 `extracted/`를 포함하지 않는다. 보고서, 판정 CSV, 공식 source target, 다운로드 상태 로그만 공개하며 원문 파일은 연구자의 로컬 감사 증거로 보관한다.
