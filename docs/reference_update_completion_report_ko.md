# Reference and Related-Work Update Completion Report

작성일: 2026-07-31  
대상 원고: `/Users/woojinpark/Downloads/Paper_260620_FULL_overleaf_ieee/main.tex`  
검토 PDF: `/Users/woojinpark/Downloads/Confidence_Guided_Two_Phase_LLM_Reasoning_for_Robust_and_Interpretable_Depression_Risk_Related_Emotion_Classification_in_Social_Media_Text__5_.pdf`

## 1. 목적

본 문서는 기존 reference audit에서 확인된 문제를 실제 논문 원고에 반영한 내역을 정리한다. 단순 검토 결과가 아니라, 원고의 Introduction, Related Work, Dataset Considerations, References 섹션에 실제로 반영된 변경사항을 기록한다.

주요 목적은 다음과 같다.

- 기존 reference 오류 및 불완전한 서지 정보 수정
- 최신 LLM 기반 depression/social-media 연구 보강
- social-media proxy label, sampling bias, construct validity 관련 방어 문헌 추가
- confidence-guided routing을 calibration 및 selective classification 문헌과 연결
- 본문 citation 번호와 References 출력 번호 일치 여부 확인
- 변경 전 문장과 변경 후 문장을 문장 단위로 추적 가능하게 정리

## 2. 최종 반영 상태

| 점검 항목 | 결과 |
|---|---|
| Reference 개수 | 43개에서 47개로 확장 |
| `thebibliography` 설정 | `{47}`로 수정 |
| References 출력 순서 | `[1]`부터 `[47]`까지 정상 정렬 |
| 본문 citation 번호 | PDF 기준 본문 citation과 References 번호 일치 확인 |
| 기존 `[36]` 오용 | 본문에서 제거 완료 |
| `[36]` DOI 오류 | 수정 완료 |
| 신규 LLM depression references | `[44]`-`[46]` 추가 완료 |
| social-media bias/methodology review | `[47]` 추가 완료 |
| PDF 반영 여부 | `__5_.pdf` 기준 본문 및 References 반영 확인 |
| 남은 확인 사항 | Overleaf 최종 제출 전 DOI/URL 줄바꿈 육안 확인 권장 |

## 3. 본문 문장별 변경 내역

| 번호 | 위치 | 변경 전 문장 | 변경 후 문장 | 변경 이유 | 근거 |
|---:|---|---|---|---|---|
| 1 | `main.tex:34` | According to the World Health Organization, approximately 332 million people globally live with depression [1]. | According to the World Health Organization's 2025 fact sheet, approximately 332 million people globally live with depression [1]. | WHO의 332 million 수치가 최신 2025 fact sheet 기준임을 명확히 하기 위함. 과거 280 million 추정치와의 혼동을 줄임. | [WHO depression fact sheet](https://www.who.int/news-room/fact-sheets/detail/depression) |
| 2 | `main.tex:90` | 없음 | Recent work has begun to examine LLMs more directly for explainable depression-related analysis on social media. | LLM 기반 depression/social-media 분석 연구 흐름을 Related Work에 명시적으로 추가. | [Wang et al. 2024](https://aclanthology.org/2024.clpsych-1.8/), [Lan et al. 2025](https://dblp.uni-trier.de/rec/conf/emnlp/LanHCS00025.html) |
| 3 | `main.tex:90` | 없음 | Wang et al. applied LLMs to Reddit-based depression-level detection and emphasized explanation generation alongside predictive performance [44]. | Reddit 기반 LLM depression detection 및 explanation generation 선행연구를 추가하여 Phase 2 reasoning의 근거를 보강. | [Wang et al. 2024 CLPsych](https://aclanthology.org/2024.clpsych-1.8/) |
| 4 | `main.tex:90` | 없음 | Lan et al. further proposed a medical-knowledge-guided LLM framework for social-media depression detection, highlighting the relevance of domain knowledge, temporal mood-course modeling, and interpretability in LLM-based mental health applications [45]. | 최신 LLM depression detection 연구와 medical knowledge, temporal modeling, interpretability 쟁점을 연결. | [Lan et al. 2025 metadata](https://dblp.uni-trier.de/rec/conf/emnlp/LanHCS00025.html) |
| 5 | `main.tex:90` | 없음 | A recent systematic review of LLMs for depression also concluded that LLMs show promise for depression-related assessment and support tasks, but require careful validation, bias assessment, and clinically responsible integration before practical use [46]. | LLM 사용의 가능성과 한계를 균형 있게 제시하여 clinical overclaim을 방지. | [Omar and Levkovich 2025](https://pubmed.ncbi.nlm.nih.gov/39581383/) |
| 6 | `main.tex:92` | Transformer-based models are commonly regarded as complex, which can pose challenges for interpretability and transparency in sensitive clinical settings. | Transformer-based and generative models are commonly regarded as complex, which can pose challenges for interpretability and transparency in sensitive clinical settings. | 기존 transformer 중심 표현을 generative LLM까지 포함하도록 확장. | [Omar and Levkovich 2025](https://doi.org/10.1016/j.jad.2024.11.052) |
| 7 | `main.tex:92` | In addition, reliance on large-scale text data raises concerns about model generalization across diverse demographic or linguistic groups [36]. | Social-media-based mental health prediction also raises concerns about construct validity, sampling bias, demographic generalizability, platform dependence, and inconsistent reporting practices [38], [47]. | `[36]`은 clinical trial design 논문이라 해당 claim의 근거로 부적절했음. social-media mental-health validity/bias 문헌으로 교체. | [Chancellor and De Choudhury 2020](https://doi.org/10.1038/s41746-020-0233-7), [Cao et al. 2025](https://doi.org/10.35566/jbds/caoyc) |
| 8 | `main.tex:92` | Systematic reviews of depression detection research provide a broad overview of existing datasets, methods, evaluation strategies, and reported limitations of current approaches [37]. | Systematic reviews of depression detection research provide a broad overview of existing datasets, methods, evaluation strategies, and reported limitations of current approaches [37], [47]. | 기존 systematic review에 최신 bias/methodology systematic review를 추가. | [Salas-Zarate et al. 2022](https://doi.org/10.3390/healthcare10020291), [Cao et al. 2025](https://doi.org/10.35566/jbds/caoyc) |
| 9 | `main.tex:94` | 없음 | The routing component of the proposed framework is also related to confidence calibration and selective classification. | confidence routing이 단순 heuristic이 아니라 calibration/selective classification 계열 방법론과 연결됨을 명시. | [Guo et al. 2017](https://proceedings.mlr.press/v70/guo17a.html), [Geifman and El-Yaniv 2017](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) |
| 10 | `main.tex:94` | 없음 | Modern neural networks can be poorly calibrated, meaning that their predicted probabilities do not necessarily reflect true correctness likelihood [15]. | temperature scaling과 confidence calibration의 필요성을 이론적으로 뒷받침. | [Guo et al. 2017](https://proceedings.mlr.press/v70/guo17a.html) |
| 11 | `main.tex:94` | 없음 | Selective classification addresses this problem by allowing a model to abstain from making predictions on uncertain inputs, thereby trading coverage for lower risk on accepted predictions [16]. | accepted/routed 구조를 selective classification의 risk-coverage trade-off와 연결. | [Geifman and El-Yaniv 2017](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) |
| 12 | `main.tex:94` | 없음 | In this study, we adapt this principle to depression-risk-related emotion classification by using temperature-scaled confidence and calibration-split risk-coverage analysis to decide which samples should be accepted by the Phase 1 classifier and which should be routed to Phase 2 reasoning. | 본 연구의 confidence-guided two-phase 구조를 calibration, risk-coverage, selective routing 관점에서 설명. | 원고 Methodology `main.tex:219` 이후 |
| 13 | `main.tex:102` | Chancellor and De Choudhury further identified inconsistencies in label operationalization, the lack of standardized evaluation practices, and the need for greater transparency in data curation and validation procedures in social media-based mental health research [38]. | Chancellor and De Choudhury identified inconsistencies in label operationalization, the lack of standardized evaluation practices, and the need for greater transparency in data curation and validation procedures in social media-based mental health research [38]. | 문장 톤을 정리하고 후속 문장과 자연스럽게 연결. | [Chancellor and De Choudhury 2020](https://doi.org/10.1038/s41746-020-0233-7) |
| 14 | `main.tex:102` | 없음 | More recent review evidence similarly emphasizes that social-media depression detection studies remain vulnerable to sampling bias, platform bias, language and demographic imbalance, inconsistent preprocessing, and incomplete reporting of model-development procedures [47]. | proxy label 및 social-media dataset 한계에 대한 최신 review 기반 방어 강화. | [Cao et al. 2025](https://doi.org/10.35566/jbds/caoyc) |
| 15 | `main.tex:102` | 없음 | These concerns motivate the present study's explicit treatment of subreddit-derived labels as proxy emotion labels rather than clinical labels, as well as its reporting of data construction, filtering, calibration, and threshold-selection procedures. | 본 논문이 clinical diagnosis claim을 피하고 proxy-label task로 제한한다는 방어 문장 추가. | 원고 Limitation 및 Dataset sections |

## 4. References 항목별 변경 내역

| Reference | 위치 | 변경 전 | 변경 후 | 근거 |
|---|---|---|---|---|
| `[5]` PHQ-9 | `main.tex:1002` | B. Levis, A. Benedetti, and B. D. Thombs | B. Levis, A. Benedetti, B. D. Thombs, and the DEPRESsion Screening Data (DEPRESSD) Collaboration | [BMJ DOI](https://doi.org/10.1136/bmj.l1476) |
| `[6]` OECD | `main.tex:1004` | OECD Publishing, 2021. Available URL | OECD Health Policy Studies, OECD Publishing, Paris, 2021. doi: 10.1787/4ed890f6-en | [OECD DOI](https://doi.org/10.1787/4ed890f6-en) |
| `[12]` DistilBERT | `main.tex:1016` | arXiv preprint only | doi: 10.48550/arXiv.1910.01108 추가 | [arXiv DOI](https://doi.org/10.48550/arXiv.1910.01108) |
| `[13]` Mistral 7B | `main.tex:1018` | arXiv preprint only | doi: 10.48550/arXiv.2310.06825 추가 | [arXiv DOI](https://doi.org/10.48550/arXiv.2310.06825) |
| `[14]` Llama 2 | `main.tex:1020` | arXiv preprint only | doi: 10.48550/arXiv.2307.09288 추가 | [arXiv DOI](https://doi.org/10.48550/arXiv.2307.09288) |
| `[15]` Calibration | `main.tex:1022` | URL 없음 | PMLR URL 추가 | [PMLR](https://proceedings.mlr.press/v70/guo17a.html) |
| `[16]` Selective classification | `main.tex:1024` | URL 없음 | NeurIPS URL 추가 | [NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) |
| `[17]` CoT | `main.tex:1026` | DOI 없음 | doi: 10.48550/arXiv.2201.11903 추가 | [arXiv DOI](https://doi.org/10.48550/arXiv.2201.11903) |
| `[18]` SELF-DISCOVER | `main.tex:1028` | NeurIPS vol. 37, 2024 | arXiv preprint arXiv:2402.03620, doi: 10.48550/arXiv.2402.03620 | [arXiv DOI](https://doi.org/10.48550/arXiv.2402.03620) |
| `[19]` Synthetic data | `main.tex:1030` | DOI 없음 | doi: 10.48550/arXiv.2411.17672 추가 | [arXiv DOI](https://doi.org/10.48550/arXiv.2411.17672) |
| `[23]` CLPsych 2015 | `main.tex:1038` | DOI 없음 | doi: 10.3115/v1/W15-1204 추가 | [ACL Anthology](https://aclanthology.org/W15-1204/) |
| `[25]` word2vec | `main.tex:1042` | arXiv only | doi: 10.48550/arXiv.1301.3781 추가 | [arXiv DOI](https://doi.org/10.48550/arXiv.1301.3781) |
| `[28]` Kim CNN | `main.tex:1048` | DOI 없음 | doi: 10.3115/v1/D14-1181 추가 | [ACL Anthology](https://aclanthology.org/D14-1181/) |
| `[32]` eRisk | `main.tex:1056` | stable URL 없음 | CEUR stable PDF URL 추가 | [CEUR PDF](https://ceur-ws.org/Vol-2125/invited_paper_5.pdf) |
| `[34]` Tavchioski et al. | `main.tex:1060` | arXiv preprint, 2023 | arXiv:2305.05325 및 doi: 10.48550/arXiv.2305.05325 추가 | [arXiv DOI](https://doi.org/10.48550/arXiv.2305.05325) |
| `[36]` Harrer et al. | `main.tex:1064` | doi: 10.1016/j.tips.2019.06.006 | doi: 10.1016/j.tips.2019.05.005 | [Correct DOI](https://doi.org/10.1016/j.tips.2019.05.005) |
| `[38]` Chancellor and De Choudhury | `main.tex:1068` | DOI 없음 | doi: 10.1038/s41746-020-0233-7 추가 | [DOI](https://doi.org/10.1038/s41746-020-0233-7) |
| `[39]` Conway and O'Connor | `main.tex:1070` | DOI 없음 | doi: 10.1016/j.copsyc.2016.01.004 추가 | [DOI](https://doi.org/10.1016/j.copsyc.2016.01.004) |
| `[40]` Benton et al. | `main.tex:1072` | location/DOI 부족 | Valencia, Spain 및 doi: 10.18653/v1/W17-1612 추가 | [ACL Anthology](https://aclanthology.org/W17-1612/) |
| `[41]` Llama 3 | `main.tex:1074` | DOI 없음 | doi: 10.48550/arXiv.2407.21783 추가 | [arXiv DOI](https://doi.org/10.48550/arXiv.2407.21783) |

## 5. 신규 추가 References

| 번호 | Reference | 원고 내 역할 | 근거 링크 |
|---:|---|---|---|
| `[44]` | Y. Wang, D. Inkpen, and P. K. Gamaarachchige, CLPsych 2024 | LLM 기반 explainable depression detection 및 Reddit/social-media depression analysis 근거 | [ACL Anthology](https://aclanthology.org/2024.clpsych-1.8/) |
| `[45]` | X. Lan et al., EMNLP Industry 2025 | medical-knowledge-guided LLM depression detection 최신성 보강 | [DBLP metadata](https://dblp.uni-trier.de/rec/conf/emnlp/LanHCS00025.html) |
| `[46]` | M. Omar and I. Levkovich, Journal of Affective Disorders 2025 | LLMs for depression systematic review 근거 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/39581383/) |
| `[47]` | Y. Cao et al., Journal of Behavioral Data Science 2025 | social-media depression detection의 bias, sampling, reporting, methodology limitation 방어 | [JBDS](https://jbds.isdsa.org/jbds/article/view/110) |

## 6. PDF 반영 확인

검토 PDF `Confidence_Guided_Two_Phase_LLM_Reasoning_for_Robust_and_Interpretable_Depression_Risk_Related_Emotion_Classification_in_Social_Media_Text__5_.pdf`에서 다음 항목을 확인했다.

| 확인 항목 | PDF 반영 여부 |
|---|---|
| WHO 2025 fact sheet 문장 | 반영 확인 |
| Wang et al. `[44]` 문단 | 반영 확인 |
| Lan et al. `[45]` 문단 | 반영 확인 |
| LLM systematic review `[46]` 문단 | 반영 확인 |
| social-media bias/proxy-label `[47]` 문단 | 반영 확인 |
| confidence calibration/selective classification 문단 | 반영 확인 |
| 기존 `[36]` generalization 문장 | 제거 확인 |
| 기존 잘못된 `[36]` DOI `10.1016/j.tips.2019.06.006` | 제거 확인 |
| 수정된 `[36]` DOI `10.1016/j.tips.2019.05.005` | 반영 확인 |
| References 번호 `[1]`-`[47]` | 순서 정상 확인 |
| `[44]`-`[47]` DOI | 반영 확인 |

## 7. 남은 확인 사항

내용 및 reference 번호 구조는 반영 완료되었다. 최종 제출 전에는 Overleaf PDF에서 다음 항목만 육안 확인하면 된다.

1. 긴 DOI와 URL이 IEEE 2단 레이아웃에서 과도하게 삐져나오지 않는지 확인
2. References의 저자 이니셜 spacing이 필요한 경우 `Y.~Wang`처럼 미세 조정
3. 최종 실험 수치가 들어간 뒤 신규 references 번호가 다시 밀리지 않았는지 확인
