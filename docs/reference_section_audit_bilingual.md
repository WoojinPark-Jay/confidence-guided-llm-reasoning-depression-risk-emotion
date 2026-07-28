# 논문 Introduction / Related Work / References 검증 정리

Audit date: 2026-07-28  
대상 원고: `/Users/woojinpark/Downloads/Paper_260620_FULL_overleaf_ieee/main.tex`  
범위: Introduction, Literature Review, Dataset-related citation, References 전체

---

## 1. 전체 결론

현재 레퍼런스는 **대부분 방향은 맞지만, 그대로 제출하기에는 몇 가지 방어 취약점이 있음**. 특히 reviewer가 볼 때 문제 삼을 수 있는 부분은 다음 네 가지다.

| 구분 | 판단 | 이유 |
|---|---|---|
| 전체 레퍼런스 방향 | 대체로 적절 | WHO, OECD, BMJ, JMIR, ACL, NeurIPS, PMLR 등 주요 근거는 있음 |
| 반드시 고칠 부분 | 있음 | [36] citation mismatch, [5] PHQ-9 citation 불완전, 일부 DOI/URL 누락 |
| 최신성 보강 | 필요 | LLM/social media depression detection 최신 논문 2-3개 추가 권장 |
| 논문 방어력 | 보강 필요 | proxy label, social-media bias, confidence routing 이론 설명을 더 명확히 해야 함 |

핵심은 **기존 레퍼런스를 다 갈아엎는 것이 아니라, 틀린 것 교정 + 최신 LLM 관련 논문 몇 개 추가 + citation claim matching 정리**다.

---

## 2. 반드시 고쳐야 하는 항목

| 우선순위 | 위치 | 현재 문제 | 왜 문제인지 | 수정 방향 |
|---|---|---|---|---|
| 필수 | Literature Review에서 `[36]` 사용 부분 | [36] Harrer et al.은 clinical trial design 논문인데, 현재 문장은 LLM/social-media generalization 문제를 설명하는 데 사용됨 | 주장과 논문 주제가 맞지 않음. reviewer가 보면 citation padding처럼 보일 수 있음 | [36]을 제거하거나 다른 주장에만 쓰고, 현재 문장에는 Chancellor and De Choudhury [38], Cao et al. 2025, LLM depression systematic review 등을 사용 |
| 필수 | References [36] | DOI가 틀림. 현재 `10.1016/j.tips.2019.06.006`로 되어 있으나, 해당 논문의 DOI는 `10.1016/j.tips.2019.05.005` | 명백한 서지 오류 | [36]을 유지한다면 DOI 수정. 가능하면 현재 위치에서는 제거 권장 |
| 필수 | References [5] | PHQ-9 BMJ 논문 citation이 불완전함 | BMJ 논문은 DEPRESsion Screening Data Collaboration 포함 표기가 더 정확함 | author 표기를 보강 |
| 필수 | Related Work LLM 부분 | LLM depression/social media 관련 최신 논문이 부족함 | 논문 주제가 LLM reasoning인데 LLM 관련 최신 prior work가 약해 보임 | Wang et al. 2024 CLPsych, Lan et al. 2025 EMNLP Industry 중 최소 1-2개 추가 |
| 필수 | Confidence routing 설명 | 본문에는 [15], [16]이 있지만 Related Work에서 calibration/selective classification 연결 설명이 약함 | 본 논문의 핵심 방법론인 confidence-guided routing의 이론적 정당성이 약해 보임 | Related Work 또는 Methodology 앞부분에 calibration/selective classification bridge paragraph 추가 |

---

## 3. 고치면 좋은 항목

| 우선순위 | 위치 | 현재 상태 | 수정 권장 |
|---|---|---|---|
| 상 | [34] Tavchioski et al. | arXiv ID/URL 없음 | `arXiv:2305.05325`와 URL 추가 |
| 상 | [37] Salas-Zarate et al. | author list가 실제 논문과 다를 가능성 있음 | 실제 author list 확인 후 정정 |
| 중 | [12]-[14], [17]-[19], [41] | arXiv reference는 맞지만 DOI-style URL 없음 | `https://doi.org/10.48550/arXiv.xxxxx` 추가 |
| 중 | [39], [40] | 좋은 ethics reference지만 DOI가 빠짐 | DOI 추가 |
| 중 | Early ML/DL Related Work | 내용은 맞지만 약간 길고 novelty와 거리가 있음 | 조금 줄이고 LLM/reasoning/confidence routing 설명을 늘리는 편이 좋음 |
| 중 | WHO 332 million claim | 현재 WHO 2025 기준으로 맞음 | 다만 reviewer가 280 million 수치를 알고 있을 수 있으므로 “WHO 2025 fact sheet” 기준임을 명확히 하면 좋음 |

---

## 4. 안 고쳐도 되는 항목

| Reference | 판단 | 이유 |
|---|---|---|
| [1] WHO depression fact sheet | 유지 가능 | 332 million 수치와 depression burden 설명을 뒷받침함 |
| [2] WHO 2017 news release | 유지 가능 | “leading cause of ill health/disability” framing에 사용 가능. 다만 [1]과 일부 중복 |
| [3] WHO mental health at work | 유지 가능 | US$1 trillion productivity loss claim에 적절 |
| [4] WHO COVID-19 25% increase | 유지 가능 | COVID-19 관련 mental health burden claim에 적절 |
| [6] OECD 2021 | 유지 가능 | mental health system benchmark 설명에 적절 |
| [7] Calvo et al. 2017 | 유지 가능 | non-clinical text NLP mental health application의 foundational review |
| [8] MentalBERT | 유지 가능 | mental health domain-specific pretraining 설명에 적절 |
| [9], [10] | 유지 가능 | emotion/social-media/temporal context 필요성 설명에 적절 |
| [11] Tonekaboni et al. | 유지 가능 | explainability and clinician trust 근거로 적절 |
| [15], [16] | 유지 가능 | calibration/selective classification 근거로 매우 중요 |
| [17], [18] | 유지 가능 | CoT / SELF-DISCOVER 근거로 적절 |
| [20]-[31] | 유지 가능 | classical ML, DL, social media depression detection 흐름 설명에 적절 |
| [33], [35] | 유지 가능 | 2024 JMIR/JMIR Mental Health 최신 근거로 적절 |
| [38]-[40] | 유지 가능 | social media validity/privacy/ethics 근거로 중요 |
| [42], [43] | 유지 가능 | TextBlob polarity와 Pattern sentiment backend 설명에 적절 |

---

## 5. 기존 Reference별 검수 결과

| Ref | 현재 논문 내 역할 | 검수 결과 | 해야 할 일 |
|---:|---|---|---|
| [1] WHO depression fact sheet | depression prevalence, global burden | 맞음 | 유지 |
| [2] WHO 2017 depression news | depression as leading cause | 대체로 맞음, 다만 오래됨 | 유지 가능. 필요시 [1]로 통합 |
| [3] WHO mental health at work | productivity cost | 맞음 | 유지 |
| [4] WHO COVID-19 25% increase | pandemic burden | 맞음 | 유지 |
| [5] PHQ-9 BMJ meta-analysis | PHQ-9 screening | 내용은 맞지만 author/collaboration 표기 보강 필요 | 수정 |
| [6] OECD mental health systems | health-system benchmark | 맞음 | DOI 링크로 정리 권장 |
| [7] Calvo et al. | non-clinical text NLP review | 맞음 | 유지 |
| [8] MentalBERT | mental-health domain pretraining | 맞음 | 유지 |
| [9] Emotion fusion survey | emotion/social media mental illness | 맞음 | 유지 |
| [10] Temporal representation | temporal/context limitation | 맞음 | 유지 |
| [11] Tonekaboni et al. | explainability/trust | 맞음 | 유지 |
| [12] DistilBERT | model background | 맞음 | DOI-style arXiv URL 추가 권장 |
| [13] Mistral 7B | model background | 맞음 | DOI-style arXiv URL 추가 권장 |
| [14] Llama 2 | model background | 맞음 | DOI-style arXiv URL 추가 권장 |
| [15] Calibration | softmax confidence limitation | 맞음 | 유지 |
| [16] Selective classification | risk-coverage/routing background | 맞음 | 유지 |
| [17] CoT | reasoning prompt | 맞음 | 유지 |
| [18] SELF-DISCOVER | structured reasoning | 맞음 | 유지 |
| [19] Synthetic depression data | synthetic dataset motivation | 관련 있음 | 유지 가능 |
| [20] Rude et al. | language markers in depression | 맞음 | 유지 |
| [21] De Choudhury et al. | social media depression prediction | 맞음 | 유지 |
| [22] Tsugawa et al. | Twitter activity depression recognition | 맞음 | 유지 |
| [23] CLPsych 2015 | shared task/self-reported labels | 맞음 | DOI 추가 권장 |
| [24] Wongkoblap et al. | systematic review, limitations | 맞음 | 유지 |
| [25] Mikolov et al. | word embeddings | 맞음 | URL 추가 권장 |
| [26] GloVe | word embeddings | 맞음 | 유지 |
| [27] LSTM | sequence modeling | 맞음 | 유지 |
| [28] Kim CNN | CNN text classification | 맞음 | 유지 |
| [29] Shen et al. | multimodal depression detection | 맞음 | 유지 |
| [30] Orabi et al. | deep learning depression detection | 맞음 | 유지 |
| [31] Tadesse et al. | Reddit depression detection | 맞음 | 유지 |
| [32] eRisk 2018 | early risk prediction benchmark | 맞음 | stable URL 추가 권장 |
| [33] Zhang et al. JMIR Mental Health | transformer depression prediction | 맞음 | 유지 |
| [34] Tavchioski et al. | transformer/ensemble depression detection | citation 정보 부족 | arXiv ID/URL 추가 |
| [35] Shin et al. JMIR | LLM diary depression detection | 맞음 | 유지 |
| [36] Harrer et al. | currently used for generalization concerns | 현재 문맥과 맞지 않음 + DOI 오류 | 현재 문장에서는 교체 권장 |
| [37] Salas-Zarate et al. | systematic review | 주제는 맞음, author list 점검 필요 | author list 수정 |
| [38] Chancellor and De Choudhury | construct validity/social media methods | 매우 적절 | 유지 |
| [39] Conway and O'Connor | social media ethics | 적절 | DOI 추가 |
| [40] Benton et al. | social media health research ethics | 적절 | DOI 추가 |
| [41] Llama 3 Herd | Llama 3 background | 맞음 | DOI-style arXiv URL 추가 |
| [42] TextBlob docs | polarity score range | 맞음 | 유지 |
| [43] Pattern for Python | TextBlob/Pattern sentiment basis | 맞음 | 유지 |

---

## 6. 추가하면 좋은 논문

### 필수 보강 후보

| 목적 | 추가 논문 | 왜 필요한가 | 링크 |
|---|---|---|---|
| LLM + social media depression + explanation | Y. Wang, D. Inkpen, and P. K. Gamaarachchige, “Explainable Depression Detection Using Large Language Models on Social Media Data,” CLPsych 2024 | 현재 논문 Phase 2의 explanation/reasoning 목적과 직접 연결됨 | https://aclanthology.org/2024.clpsych-1.8/ |
| 최신 LLM depression detection framework | X. Lan et al., “Depression Detection on Social Media with Large Language Models,” EMNLP Industry 2025 | 최신성 방어에 매우 좋음. medical knowledge, social media, explainability가 모두 관련됨 | https://aclanthology.org/2025.emnlp-industry.151/ |
| LLM depression review | “Exploring the efficacy and potential of large language models for depression: A systematic review,” Journal of Affective Disorders, 2025 | “LLMs are promising but require careful validation” 주장에 적합 | https://doi.org/10.1016/j.jad.2024.11.052 |

### 선택 보강 후보

| 목적 | 추가 논문 | 왜 필요한가 | 링크 |
|---|---|---|---|
| proxy label / bias / methodology defense | Cao et al., “Machine Learning Approaches for Depression Detection on Social Media: A Systematic Review of Biases and Methodological Challenges,” Journal of Behavioral Data Science, 2025 | proxy label, sampling bias, preprocessing, validation 문제 방어에 좋음 | https://doi.org/10.35566/jbds/caoyc |
| social media text depression meta-analysis | “Text-Based Depression Prediction on Social Media Using Machine Learning: Systematic Review and Meta-Analysis,” JMIR 2025 | social media text가 depression prediction에 유용하지만 제한이 있다는 균형 잡힌 근거 | https://doi.org/10.2196/59002 |
| LLM-assisted depression symptom modeling | Farruque et al., “Depression symptoms modelling from social media text: an LLM driven semi-supervised learning approach,” Language Resources and Evaluation, 2024 | LLM-assisted labeling/symptom modeling 맥락 보강 | https://doi.org/10.1007/s10579-024-09720-4 |

---

## 7. 실제 본문 수정 제안

### 7.1 [36] 들어간 문장 교체

현재 문장:

> In addition, reliance on large-scale text data raises concerns about model generalization across diverse demographic or linguistic groups [36].

수정안:

> In addition, social-media-based mental health prediction studies face persistent concerns regarding construct validity, sampling bias, demographic generalizability, and inconsistent reporting practices [38]. Recent reviews further emphasize that models trained on social media data may not generalize reliably across platforms, languages, demographic groups, or labeling protocols without more transparent validation and bias assessment.

사용 citation:

- [38] Chancellor and De Choudhury
- 추가 후보: Cao et al. 2025 또는 JMIR 2025 meta-analysis

### 7.2 LLM 최신 연구 보강 문단

삽입 위치: `Large Language Models for Mental Health Analysis` 섹션에서 Shin et al. [35] 문단 뒤

추가 문단:

> Closely related work has begun to examine LLMs for explainable depression detection on social media. Wang et al. applied LLMs to Reddit-based depression-level detection and emphasized explanation generation alongside predictive performance. Lan et al. further proposed an LLM-based framework for social-media depression detection that uses medical-knowledge-informed annotations and temporally summarized mood-course features to support both accuracy and interpretability. These studies support the relevance of LLM-based reasoning and explanation, while also highlighting the need for careful validation in mental-health-related applications.

사용 citation:

- Wang et al. 2024 CLPsych
- Lan et al. 2025 EMNLP Industry

### 7.3 Confidence routing 이론 연결 문단

삽입 위치: Literature Review 마지막 또는 Methodology 시작 전

추가 문단:

> The routing component of the proposed framework is also related to confidence calibration and selective classification. Modern neural networks can be poorly calibrated, meaning that their predicted probabilities do not necessarily reflect true correctness likelihood [15]. Selective classification addresses this issue by allowing a model to abstain from making predictions on uncertain inputs, thereby trading coverage for lower risk on accepted predictions [16]. In this study, we adapt this principle to depression-risk-related emotion classification by using validation-based risk-coverage analysis to decide which samples should be accepted by the Phase 1 classifier and which should be routed to Phase 2 reasoning.

이 문단은 논문 방어에 중요함. 현재 논문의 confidence filtering이 단순 heuristic이 아니라 selective classification/risk-coverage 계열이라는 점을 명확히 해줌.

---

## 8. Reference 표기 수정안

아래는 현재 bibliography에서 표기 교정이 필요한 항목이다.

### [5] PHQ-9

```tex
\bibitem{ref5} B. Levis, A. Benedetti, B. D. Thombs, and the DEPRESsion Screening Data (DEPRESSD) Collaboration, ``Accuracy of Patient Health Questionnaire-9 (PHQ-9) for screening to detect major depression: Individual participant data meta-analysis,'' BMJ, vol. 365, l1476, 2019. doi: 10.1136/bmj.l1476
```

### [34] Tavchioski et al.

```tex
\bibitem{ref34} I. Tavchioski, M. Robnik-\v{S}ikonja, and S. Pollak, ``Detection of depression on social networks using transformers and ensembles,'' arXiv preprint arXiv:2305.05325, 2023.
```

### [36] Harrer et al. DOI correction

유지할 경우:

```tex
\bibitem{ref36} S. Harrer, P. Shah, B. Antony, and J. Hu, ``Artificial intelligence for clinical trial design,'' Trends in Pharmacological Sciences, vol. 40, no. 8, pp. 577--591, 2019. doi: 10.1016/j.tips.2019.05.005
```

하지만 현재 Literature Review의 generalization claim에는 이 reference를 쓰지 않는 것이 좋음.

### [39] Conway and O'Connor

```tex
\bibitem{ref39} M. Conway and D. O'Connor, ``Social media, big data, and mental health: Current advances and ethical implications,'' Current Opinion in Psychology, vol. 9, pp. 77--82, 2016. doi: 10.1016/j.copsyc.2016.01.004
```

### [40] Benton et al.

```tex
\bibitem{ref40} A. Benton, G. Coppersmith, and M. Dredze, ``Ethical research protocols for social media health research,'' in Proceedings of the First ACL Workshop on Ethics in Natural Language Processing, Valencia, Spain, pp. 94--102, 2017. doi: 10.18653/v1/W17-1612
```

### arXiv references DOI-style URL 권장

| Ref | DOI-style URL |
|---|---|
| [12] DistilBERT | https://doi.org/10.48550/arXiv.1910.01108 |
| [13] Mistral 7B | https://doi.org/10.48550/arXiv.2310.06825 |
| [14] Llama 2 | https://doi.org/10.48550/arXiv.2307.09288 |
| [17] Chain-of-Thought | https://doi.org/10.48550/arXiv.2201.11903 |
| [19] Synthetic data generation | https://doi.org/10.48550/arXiv.2411.17672 |
| [41] Llama 3 Herd | https://doi.org/10.48550/arXiv.2407.21783 |

---

## 9. English Summary for Co-Researchers

The current Introduction and Literature Review are broadly aligned with the manuscript topic, but several reference-level issues should be fixed before submission.

The most important correction is Ref. [36]. It currently supports a statement about generalization concerns in LLM-based mental-health text analysis, but the cited paper is about AI for clinical trial design. This is not an appropriate citation for the claim, and the DOI in the bibliography is also incorrect. The sentence should instead cite social-media mental-health validity and bias literature, such as Chancellor and De Choudhury, Cao et al. 2025, or a recent LLM depression systematic review.

The PHQ-9 citation [5] is relevant but should be formatted more accurately by including the DEPRESsion Screening Data (DEPRESSD) Collaboration. Several arXiv references are correct but should be made more traceable by adding DOI-style arXiv links. Ref. [34] should include arXiv:2305.05325.

To improve the novelty and defensibility of the paper, the Related Work should add 1-2 recent papers on LLM-based depression detection on social media, especially Wang et al. 2024 CLPsych and Lan et al. 2025 EMNLP Industry. A short paragraph connecting the proposed routing mechanism to confidence calibration and selective classification should also be added, using Guo et al. [15] and Geifman and El-Yaniv [16].

Overall, the reference section does not need a full rewrite. The recommended action is a targeted revision: correct inaccurate references, replace weak citation matches, add recent LLM/social-media depression work, and strengthen the confidence-routing theoretical bridge.

---

## 10. 최종 작업 순서

1. [36] 현재 위치에서 제거 또는 교체
2. [36] DOI 수정 또는 reference list에서 제거
3. [5], [34], [39], [40] bibliography 표기 수정
4. LLM/social-media depression 최신 논문 1-2개 추가
5. confidence calibration/selective classification 연결 문단 추가
6. proxy label / social media bias 방어용 최신 review 1개 추가
7. 전체 reference 번호 다시 정렬

이 정도만 반영하면 Introduction과 Related Work의 reference 방어력은 충분히 올라간다.
