# 46개 참고문헌 인용 용도 대조

2026-09-09 / 수정 원고 기준

**아래 번호는 기존 감사 및 PDF 파일 번호다.** 새 원고의 자동 인용 번호는 [번호 대조표](reference_number_map_2026_09_09_ko.md)에 대응시켰다. 자료와 주장의 연결은 그대로이며 번호만 최초 등장 순서에 맞춰 바뀌었다.

## 읽는 방법

아래의 **유지**는 논문 자체의 모든 결과가 독립 검증됐다는 뜻이 아니라, 이 원고에서 사용하는 범위의 설명을 확보본이 뒷받침한다는 뜻이다. **수정 반영**은 관련 문장을 실제 `main.tex`에 수정했다는 뜻이다. 위치는 확보본의 초록·절 등 확인 지점이며 출판본과 PDF의 페이지 수가 다를 수 있다.

확보 파일 목록과 해시는 기존 `reference_inventory.csv`에 있다. 이 표는 파일 확보 상태가 아닌 내용상 인용 판단을 기록한다. 원문 전체를 그대로 복제하거나 공개 저장소에 올리기 위한 자료가 아니다.

## 배경 및 기초 방법 [1]–[19]

| 번호 | 확인 지점 | 이 원고의 인용 용도 | 판단과 반영 |
|---|---|---|---|
| [1] WHO depression | 공식 fact sheet, Overview | 세계적 우울 부담, 약 332 million | 유지. 공식 페이지에서도 숫자 확인. 본 연구의 proxy label 유병률로 해석하지 않음 |
| [2] WHO 2017 release | 공식 뉴스 본문 | 질병·장애 부담의 배경 | 유지. 과거 발표의 규모를 최신 유병 인원으로 옮겨 쓰지 않음 |
| [3] WHO mental health at work | 공식 fact sheet, Key facts | 우울·불안의 연간 생산성 손실 | 유지. 두 질환 합산 US$1 trillion이며 우울 단독 비용이 아님 |
| [4] WHO COVID-19 | 공식 2022 뉴스 본문 | 팬데믹 첫해의 우울·불안 증가 | 유지. 기간을 첫해로 한정 |
| [5] Levis et al. | BMJ 초록, 진단 정확도 방법·결과 | PHQ-9의 screening 역할 | 유지. screening을 임상 확진과 구분 |
| [6] OECD | 보고서 개요 및 정신건강 체계 benchmark 설명 | 접근 가능하고 사람 중심인 서비스 | 유지. 우리 모델의 임상 성능 근거로 사용하지 않음 |
| [7] Calvo et al. | 초록, non-clinical text 검토 범위 | 비임상 텍스트에서 정신건강 관련 언어 연구 | 유지. 저자 이니셜은 서지 표기 차이 수준이며 연구 식별 일치 |
| [8] MentalBERT | Methods and Setup, downstream evaluation | 도메인 사전학습의 분류 활용 | 유지. 모든 문제에서 우월하다는 주장 아님 |
| [9] Emotion fusion survey | 초록, challenges 논의 | 데이터 품질과 해석 가능성 | 수정 반영. 시간 의존성은 [10]으로 분리. arXiv 확보본과 최종 출판본의 전체 동일성은 미확인 |
| [10] Kumar et al. | 초록, temporal representation 설계 | 독립 게시물/청크 처리로 손실되는 시간·게시물 간 정보 | 수정 반영. 원 연구의 사용자 단위 과제와 우리 게시물 단위 과제 구분 |
| [11] Tonekaboni et al. | 초록, clinician 요구사항 연구; 공식 PMLR 서지 | 설명의 임상 사용 맥락 | 수정 반영. 생성 rationale 자체가 신뢰·임상 타당성을 보장하지 않음. 공식 pp.359–380 유지 |
| [12] DistilBERT | 초록 및 distillation 방법 | 모델 규모·사전학습 목적·출시 모델 특성 | 유지. 원 논문이 보고한 40%/60%를 이번 실험의 실측치로 사용하지 않음 |
| [13] Mistral 7B | 초록 및 GQA/SWA 설명 | decoder 구조와 attention 특성 | 유지. 본 연구의 linear probe 설정은 우리 실험 설정이지 원 논문의 결과가 아님 |
| [14] Llama 2 | 모델군·base/chat 소개 | Phase 1 base와 Phase 2 chat 구분 | 유지. 7B 전체를 동일 checkpoint로 기술하지 않음 |
| [15] Guo et al. | calibration 정의 및 temperature scaling | NLL 기반 scalar temperature, confidence calibration | 유지. 우리 grid와 CP 선택 기준을 이 논문의 알고리즘이라고 하지 않음 |
| [16] Geifman and El-Yaniv | selective classification의 risk–coverage 정의 | abstention과 coverage/risk 관계 | 유지·연결 문구 정리. 원 방법의 보장을 우리 calibration 재사용 프로토콜에 그대로 가져오지 않음 |
| [17] Wei et al. | 초록 및 few-shot CoT demonstrations | intermediate reasoning 아이디어 | 수정 반영. 우리 zero-shot multistage prompting은 few-shot 원 설정의 복제가 아님 |
| [18] Zhou et al. | §2, §2.1–2.2 및 Fig.2; module pool | SELECT–ADAPT–IMPLEMENT 계획 생성 | 중요 수정 반영. 원 방법은 task-level plan 재사용, 우리는 routed input별 생성. 본문·부록 캡션·효율성 한계 연결 |
| [19] Kang et al. | 초록 및 synthetic interview-summary pipeline | depression 연구에서 LLM 합성 데이터 사용 사례 | 수정 반영. 원 연구의 training augmentation을 우리 stress-test 검증 근거와 구분 |

## 기존 응용 및 데이터 연구 [20]–[34]

| 번호 | 확인 지점 | 이 원고의 인용 용도 | 판단과 반영 |
|---|---|---|---|
| [20] Rude et al. | 첫 페이지 초록, 집단 비교 설명 | 현재 우울 집단 에세이에서의 I/negative-emotion words | 수정 반영. 대학생·에세이 범위 명시. 과거 우울 집단에 같은 차이가 있었다고 쓰지 않음 |
| [21] De Choudhury et al. | feature 설명, Prediction Results | Twitter 행동·언어 특징 기반 예측 | 유지·축약. SVM과 특징 기반 연구의 구체적인 사례로 사용 |
| [22] Tsugawa et al. | 초록, 설문·SVM 평가 설명 | Twitter 활동과 questionnaire-based depression scores | 수정 반영. clinical confirmation 대신 questionnaire 기준으로 기술 |
| [23] Coppersmith et al. | 초록 및 shared-task 데이터 설명 | self-reported diagnosis, matched controls | 유지. depression/PTSD의 shared task를 임상 진단 데이터셋으로 바꾸지 않음 |
| [24] Wongkoblap et al. | 초록, dataset·consent·bias 논의 | 수집 편향·품질·방법론적 한계 | 유지·문장 구체화. 근거 없이 전통 모델을 열등하게 묘사하는 발전사 문구 축소 |
| [25] Mikolov et al. | 초록 및 vector representation 방법 | dense word embeddings | 유지. depression 응용의 직접 성능 근거가 아님 |
| [26] Pennington et al. | 초록 및 GloVe 방법 | 의미 관계를 포착하는 word vectors | 유지. 일반 표현 학습의 근거 |
| [27] Hochreiter and Schmidhuber | 첫 페이지 원문 이미지, 초록·Introduction | extended dependencies를 다루는 LSTM 설계 | 수정 반영. eRisk/우울 분류 성능을 입증하는 출처로 쓰지 않음. 원 출판연도 1997 유지 |
| [28] Kim | 초록, sentence classification 실험 | word-vector 기반 CNN의 분류 활용 | 수정 반영. 일반 sentence classification과 depression 응용 구분 |
| [29] Shen et al. | 초록 및 six-feature-group 설계 | multimodal dictionary-learning 연구 | 수정 반영. handcrafted/behavioral feature 단락으로 이동 |
| [30] Orabi et al. | 모델 비교 및 CNN/BiLSTM/SVM 실험 | Twitter depression detection의 신경망 비교 | 유지. 해당 실험의 CNN 결과로 한정하며 보편적 CNN 우월성 주장 없음 |
| [31] Tadesse et al. | III Evaluation Datasets, IV Methodology; PDF p.4 이후 | 1,841개 Reddit 게시물, 특징·분류기 비교 | 수정 반영. sizable와 대규모 심층 문맥 모델이라는 연결 제거 |
| [32] Losada et al. | eRisk depression task의 chronological chunk 설명 | 시간순 사용자 기록을 이용한 early prediction | 수정 반영. 2018 과제에 맞춰 기술. 원고 과제와 직접 수치 비교하지 않음 |
| [33] Zhang et al. | 초록 Methods, hierarchical transformer | Sina Weibo 사용자 수준 transformer 연구 | 유지. 문장·게시물·사용자 수준 구성과 본 연구의 독립 게시물 과제가 다름 |
| [34] Tavchioski et al. | 초록 및 transformer/ensemble 설정 | transformer classifier ensemble 연구 | 유지. 확보본 arXiv 표기를 정식 게재 정보로 임의 승격하지 않음 |

## 최근 응용·윤리·소프트웨어 [35]–[46]

| 번호 | 확인 지점 | 이 원고의 인용 용도 | 판단과 반영 |
|---|---|---|---|
| [35] Shin et al. | 정정 반영 PDF 초록·Methods; 공식 correction | diary text, GPT-3.5/GPT-4, 설문 참조 | 수정 반영. semi-structured diaries로 정확화. 2025 정정은 grant number이며 결과 정정 아님 |
| [36] Salas-Zárate et al. | 초록 및 reviewed-platform 결과 | Twitter/Reddit/Facebook 등 연구 플랫폼 | 유지. 리뷰의 조사 범위 내 빈도이지 현재 모든 연구의 시장점유율 아님 |
| [37] Chancellor and De Choudhury | 초록 및 construct validity/reporting 논의 | labels, sampling, evaluation의 한계 | 유지. proxy와 임상 상태를 구분하는 근거 |
| [38] Conway and O'Connor | Ethical implications, User expectations/privacy | 공개 게시물의 사용·개인정보 위험 | 유지. 현행 법규의 근거로 사용하지 않음 |
| [39] Benton et al. | 초록 및 연구윤리 프로토콜 논의 | health-related social data의 연구윤리 | 유지. 이 문헌을 인용했다는 사실이 우리 연구의 IRB 승인/면제를 대신하지 않음 |
| [40] The Llama 3 Herd of Models | 모델군 개요 | Llama 3 계열 배경 | 유지. 실제 8B-Instruct checkpoint 명칭은 실험 설정에 유지하며 405B/128K 결과를 전용하지 않음 |
| [41] TextBlob | 공식 Quickstart의 Sentiment Analysis | polarity API 및 [-1,1] 범위 | 수정 반영. API 근거는 이 문서로 한정. 문서에 임의로 붙인 2018 제거 |
| [42] Pattern for Python | 모듈 설명 및 sentiment 기능 | 관련 NLP 소프트웨어 배경 | 수정 반영. TextBlob의 실제 호출·집계 공식을 이 논문이 직접 정의한 것으로 쓰지 않음 |
| [43] Wang et al. | 초록, questionnaire prediction 및 explanation; ACL 서지 | Reddit 기반 설문 문항·수준 예측과 설명 | 수정 반영. 공식 export의 compound surname 사용. PDF의 Buddhitha 표기와 차이는 보고서에 기록 |
| [44] Lan et al. | 초록 및 DORIS framework | medical features + temporal features + GBT | 수정 반영. LLM이 최종 분류까지 모두 수행한다고 단순화하지 않음 |
| [45] Omar and Levkovich | 최종 PDF 초록, 34-study review 및 한계 | LLM 적용 가능성과 검증 필요성 | 수정 반영. encoder 포함 범위 명시, chatbot 치료 효능으로 확대하지 않음 |
| [46] Cao et al. | 초록 및 bias/methodological discussion | 표본·플랫폼·보고 편향 | 유지. 전반적 연구 한계의 근거이며 본 연구가 해당 편향을 제거했다는 근거가 아님 |

## 이번에 별도로 대조한 공식 페이지

- [WHO depression fact sheet](https://www.who.int/news-room/fact-sheets/detail/depression): 332 million 배경 수치.
- [PMLR Tonekaboni et al.](https://proceedings.mlr.press/v106/tonekaboni19a.html): pp.359–380 및 공식 서지. 확보 PDF의 내부 임시 페이지 표기와 구분.
- [NeurIPS SELF-DISCOVER](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e41efb03e20ca3c231940a3c6917ef6f-Abstract-Conference.html): 2024 proceedings 및 DOI 10.52202/079017-4004 확인.
- [ACL Wang et al.](https://aclanthology.org/2024.clpsych-1.8/): 현재 저자명과 BibTeX 성 구분.
- [JMIR correction](https://www.jmir.org/2025/1/e79198): [35]의 정정 범위 확인.
- [TextBlob Quickstart](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis): 정확한 API와 polarity 범위 재확인. 기존 보관 HTML은 일반 소개 페이지이므로 이번 대조에서는 해당 Quickstart 절을 직접 확인했다.

## 해석상의 한계

원문의 주장을 정확히 소개하는 것과 원문의 연구 품질을 보증하는 것은 다르다. 이 검수는 원고의 인용 적합성 검토이지 46편의 별도 systematic risk-of-bias 평가나 철회 이력 전수 조사, 임상 타당성 보증이 아니다. 최종 저널본과 다른 확보본은 필요한 범위에서 사용했으며, 모든 버전이 완전히 같다고 표시하지 않았다.
