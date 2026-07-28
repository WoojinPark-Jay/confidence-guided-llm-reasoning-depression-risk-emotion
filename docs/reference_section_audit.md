# Reference Section Audit for Current Manuscript

Audit date: 2026-07-28  
Target file checked: `/Users/woojinpark/Downloads/Paper_260620_FULL_overleaf_ieee/main.tex`  
Scope: Introduction, Literature Review, Dataset-related citation use, and full reference list.

## Executive Summary

The reference section is mostly usable, but it is not yet submission-safe. The largest risks are not formatting but reviewer-facing defensibility:

1. **Ref. [36] is incorrectly matched to the claim.** The manuscript cites Harrer et al. on AI for clinical trial design to support generalization concerns in LLM/mental-health analysis. This is weak and the DOI in the bibliography is also wrong.
2. **The PHQ-9 citation [5] is incomplete.** The cited BMJ article includes the DEPRESsion Screening Data (DEPRESSD) Collaboration and should be formatted more accurately.
3. **The Introduction needs one or two more recent LLM/depression references.** Current Intro relies on Llama 2/Llama 3, CoT, SELF-DISCOVER, and a 2024 JMIR paper, but it does not yet cite newer LLM-based social-media depression detection papers.
4. **Several references are missing DOI or stable official URLs.** This is fixable and should be done before IEEE/JMIR submission.
5. **Some statements are too broad for their current citations.** For example, “LLMs in mental health remain at an early stage” is defensible, but it should be supported by a recent LLM-specific systematic review, not by a clinical-trial AI paper.

Recommended minimum action: fix [5], replace or relocate [36], add 2-3 recent LLM/depression/social-media references, and update bibliography entries with DOI/official links.

## High-Priority Issues

| Priority | Location | Current Issue | Why It Matters | Recommended Fix |
|---|---|---|---|---|
| Critical | Literature Review, paragraph ending with `[36]` | [36] Harrer et al. is about AI for clinical trial design, not demographic/generalization concerns in LLM mental-health text analysis. The bibliography DOI is also incorrect. | Reviewer may see this as irrelevant citation padding. | Replace [36] with a social-media mental-health validity/bias paper, e.g., Chancellor and De Choudhury [38], Cao et al. 2025, or a recent LLM depression systematic review. |
| Critical | Bibliography [36] | DOI is listed as `10.1016/j.tips.2019.06.006`, but the correct DOI for “Artificial intelligence for clinical trial design” is `10.1016/j.tips.2019.05.005`. | Direct factual error. | Either correct DOI or remove this reference from the paper if not directly used. |
| High | Introduction first paragraph | WHO fact sheet now reports approximately 332 million people globally have depression. This is correct for the current WHO page, but older WHO pages used 280 million. | Because this number recently changed, reviewer may notice inconsistency with older literature. | Keep 332 million only if citing the 29 Aug 2025 WHO page. Add “as of the WHO 2025 fact sheet” if desired. |
| High | Introduction / Related Work | LLM-specific depression detection literature is thin. Current LLM discussion uses Shin et al. 2024 but misses social-media LLM papers. | The paper’s contribution is LLM reasoning for social-media depression-risk emotion classification. | Add at least Wang et al. 2024 CLPsych and Lan et al. 2025 EMNLP Industry, or one LLM depression systematic review. |
| High | Related Work | The paper says “generative LLMs have opened new directions,” but does not cite a survey/review focused specifically on LLMs for depression. | Makes the LLM motivation feel under-supported. | Add “Exploring the efficacy and potential of large language models for depression: A systematic review,” Journal of Affective Disorders, 2025. |
| Medium | Dataset limitations discussion | Proxy-label concerns are supported by Chancellor and De Choudhury [38], but this could be strengthened with a newer bias/methodology review. | Reviewer likely attacks proxy labels. | Add Cao et al. 2025 or JMIR 2025 meta-analysis as supporting evidence. |
| Medium | Bibliography [34] | “Detection of depression on social networks using transformers and ensembles” is listed as arXiv preprint, 2023, but no arXiv ID/URL is included. | Weak reference traceability. | Add arXiv ID `2305.05325` and URL `https://arxiv.org/abs/2305.05325`. |
| Medium | Bibliography [12]-[14], [17]-[19], [41] | arXiv references should include stable DOI-style links where available. | Improves reproducibility and link verification. | Use `https://doi.org/10.48550/arXiv.xxxxx`. |
| Medium | Literature Review | Early ML/DL history is credible but slightly long relative to the novel contribution. | Reviewers may prefer stronger focus on uncertainty/routing/reasoning. | Condense early ML paragraphs slightly and add a short subsection on selective classification/calibration. |

## Claim-by-Claim Audit

| Manuscript Claim | Current Citation | Audit Result | Recommendation |
|---|---:|---|---|
| Approximately 332 million people globally live with depression. | [1] WHO depression fact sheet | Supported by WHO 2025 fact sheet. | Keep, but ensure the date remains “29 Aug 2025.” |
| Depression is a leading cause of ill health/disability. | [2] WHO 2017 news release | Generally supported, but somewhat old. | Acceptable; could be replaced/combined with WHO depression fact sheet [1]. |
| Depression/anxiety cost US$1 trillion annually in lost productivity. | [3] WHO mental health at work | Supported by WHO mental health at work fact sheet. | Keep. |
| COVID-19 caused 25% increase in anxiety/depression prevalence in first year. | [4] WHO 2022 news release | Supported. | Keep. |
| PHQ-9 is a standard screening instrument. | [5] BMJ 2019 | Broadly supported, but citation formatting incomplete. | Fix author list and note PHQ-9 is a screening tool, not diagnosis. |
| OECD mental health systems need accessible, high-quality, person-centred services. | [6] OECD 2021 | Supported. | Keep; use official OECD DOI page. |
| NLP can analyze mental-health-related language in non-clinical text. | [7] Calvo et al. 2017 | Supported. | Keep. |
| MentalBERT/domain pretraining helps mental-health NLP tasks. | [8] Ji et al. 2022 | Supported. | Keep. |
| Social-media mental health classification needs context/temporal information. | [9], [10] | Supported. | Keep; [10] is especially relevant to temporal context. |
| Explainability matters for clinician trust. | [11] | Supported. | Keep. |
| CoT and SELF-DISCOVER support structured reasoning. | [17], [18] | Supported. | Keep. |
| SVM and classical ML were early/common approaches. | [20]-[24] | Mostly supported. | Keep but avoid overclaiming that SVM “frequently achieved strong performance” without citing a review or specific comparison. |
| CNN models outperformed BiLSTM in Orabi et al. | [30] | Supported by Orabi et al. | Keep. |
| Transformer approaches improved depression prediction on social media. | [31], [33], [34] | Supported, but [34] needs URL/arXiv ID. | Keep and fix [34]. |
| Generative LLMs can detect depression from diary text. | [35] | Supported. | Keep. |
| LLM/AI mental-health systems raise generalization concerns. | [36] | Weak/incorrect support. | Replace [36] with LLM/depression/social-media bias/validity references. |
| Social media datasets have construct-validity and proxy-label issues. | [38] | Strongly supported. | Keep; consider also citing Cao et al. 2025 or JMIR 2025. |
| Social-media mental-health data raises privacy/ethics issues. | [39], [40] | Supported. | Keep. |
| TextBlob returns polarity in [-1, 1]. | [42], [43] | Supported by TextBlob docs and Pattern paper. | Keep; cite TextBlob docs for API behavior and Pattern for sentiment backend if used. |

## Reference-by-Reference Audit

| Ref | Current Status | Verification / Problem | Action |
|---:|---|---|---|
| [1] WHO depression fact sheet | Valid | Official WHO page supports 332 million and 29 Aug 2025. | Keep. URL: https://www.who.int/news-room/fact-sheets/detail/depression |
| [2] WHO “Depression: let’s talk” | Valid but older | Good for global burden framing, but [1] may already cover current facts. | Keep or merge with [1]. URL: https://www.who.int/news/item/30-03-2017--depression-let-s-talk-says-who-as-depression-tops-list-of-causes-of-ill-health |
| [3] WHO mental health at work | Valid | Supports workplace/productivity burden framing. | Keep. URL: https://www.who.int/news-room/fact-sheets/detail/mental-health-at-work |
| [4] WHO COVID-19 25% increase | Valid | Supports COVID-19 increase claim. | Keep. URL: https://www.who.int/news/item/02-03-2022-covid-19-pandemic-triggers-25-increase-in-prevalence-of-anxiety-and-depression-worldwide |
| [5] Levis et al. BMJ 2019 | Partially incorrect/incomplete | Title and DOI valid, but author list should include DEPRESsion Screening Data Collaboration. | Revise entry. DOI: https://doi.org/10.1136/bmj.l1476 |
| [6] OECD 2021 | Valid | Official DOI page available. | Replace PDF URL with DOI. DOI: https://doi.org/10.1787/4ed890f6-en |
| [7] Calvo et al. 2017 | Valid | Natural Language Engineering, 23(5), 649-685. | Keep. DOI: https://doi.org/10.1017/S1351324916000383 |
| [8] MentalBERT | Valid | ACL/LREC metadata confirms pages 7184-7190. | Keep. URL: https://aclanthology.org/2022.lrec-1.778/ |
| [9] Emotion fusion survey | Valid | Information Fusion 92:231-246, DOI confirmed. | Keep. DOI: https://doi.org/10.1016/j.inffus.2022.11.031 |
| [10] Temporal representation | Valid | ACL Findings EMNLP 2024 metadata confirms DOI/pages. | Keep. URL: https://aclanthology.org/2024.findings-emnlp.639/ |
| [11] Tonekaboni et al. | Valid | PMLR 106:359-380 confirmed. | Keep. URL: https://proceedings.mlr.press/v106/tonekaboni19a.html |
| [12] DistilBERT | Valid | arXiv ID correct. | Add DOI URL: https://doi.org/10.48550/arXiv.1910.01108 |
| [13] Mistral 7B | Valid | GQA/SWA claims supported. | Add DOI URL: https://doi.org/10.48550/arXiv.2310.06825 |
| [14] Llama 2 | Valid | 7B-70B and chat model family supported. | Add DOI URL: https://doi.org/10.48550/arXiv.2307.09288 |
| [15] Calibration | Valid | PMLR source confirms pages. | Keep. URL: https://proceedings.mlr.press/v70/guo17a.html |
| [16] Selective classification | Valid but page range may differ by source | NeurIPS page confirms concept; some metadata gives 4878-4887 or 4885-4894. | Use NeurIPS official URL and avoid relying on exact page range if uncertain. URL: https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks |
| [17] CoT | Valid | arXiv/NeurIPS 2022. | Keep; add DOI URL: https://doi.org/10.48550/arXiv.2201.11903 |
| [18] SELF-DISCOVER | Valid | NeurIPS 2024 page confirms framework and compute-efficiency claims. | Keep. URL: https://papers.nips.cc/paper_files/paper/2024/hash/e41efb03e20ca3c231940a3c6917ef6f-Abstract-Conference.html |
| [19] Synthetic data generation | Valid but peripheral | Useful for synthetic mental-health data motivation. | Keep if Mixed Emotion Dataset is discussed; add arXiv DOI. https://doi.org/10.48550/arXiv.2411.17672 |
| [20] Rude et al. | Valid | DOI/title/pages confirmed. | Keep. DOI: https://doi.org/10.1080/02699930441000030 |
| [21] De Choudhury et al. | Valid | Microsoft/DBLP source confirms title/authors/venue. | Keep; add official page. URL: https://www.microsoft.com/en-us/research/publication/predicting-depression-via-social-media/ |
| [22] Tsugawa et al. | Valid | CHI 2015 metadata and DOI confirmed. | Keep. DOI: https://doi.org/10.1145/2702123.2702280 |
| [23] Coppersmith et al. | Valid | ACL page confirms DOI/pages. | Add DOI. URL: https://aclanthology.org/W15-1204/ |
| [24] Wongkoblap et al. | Valid | JMIR page confirms DOI/PMID/PMCID. | Keep. DOI: https://doi.org/10.2196/jmir.7215 |
| [25] Mikolov et al. | Valid | arXiv ID correct. | Add URL: https://arxiv.org/abs/1301.3781 |
| [26] GloVe | Valid | ACL page confirms DOI/pages. | Keep. URL: https://aclanthology.org/D14-1162/ |
| [27] LSTM | Valid | DOI/pages confirmed. | Keep. DOI: https://doi.org/10.1162/neco.1997.9.8.1735 |
| [28] Kim CNN | Valid | ACL page confirms DOI/pages. | Keep. URL: https://aclanthology.org/D14-1181/ |
| [29] Shen et al. | Valid | IJCAI page confirms DOI/pages. | Keep. URL: https://www.ijcai.org/proceedings/2017/536 |
| [30] Orabi et al. | Valid | ACL page confirms DOI/pages. | Keep. URL: https://aclanthology.org/W18-0609/ |
| [31] Tadesse et al. | Valid | IEEE Access DOI confirmed. | Keep. DOI: https://doi.org/10.1109/ACCESS.2019.2909180 |
| [32] eRisk 2018 | Valid | CEUR/IR Anthology confirms overview. | Add stable URL: https://ceur-ws.org/Vol-2125/invited_paper_1.pdf |
| [33] Zhang et al. JMIR Mental Health | Valid | JMIR page confirms DOI and details. | Keep. DOI: https://doi.org/10.2196/58259 |
| [34] Tavchioski et al. | Needs metadata | Current entry lacks arXiv ID/URL. | Revise as arXiv:2305.05325. URL: https://arxiv.org/abs/2305.05325 |
| [35] Shin et al. JMIR | Valid | JMIR page confirms DOI and methods/results. | Keep. DOI: https://doi.org/10.2196/54617 |
| [36] Harrer et al. | Incorrect DOI and weak relevance | Current DOI is wrong; article topic is clinical trial design. | Remove from this claim or correct only if used elsewhere. Correct DOI: https://doi.org/10.1016/j.tips.2019.05.005 |
| [37] Salas-Zarate et al. | Bibliography likely wrong authors | Current entry includes six authors but appears to omit Maria del Pilar Salas-Zarate, Maritza Bustos-Lopez, and Paredes-Valverde, and includes a different author list/order. | Correct author list. DOI: https://doi.org/10.3390/healthcare10020291 |
| [38] Chancellor and De Choudhury | Valid | Nature page confirms DOI and construct-validity critique. | Keep. DOI: https://doi.org/10.1038/s41746-020-0233-7 |
| [39] Conway and O'Connor | Valid | ScienceDirect/PMC confirms DOI and ethics framing. | Add DOI: https://doi.org/10.1016/j.copsyc.2016.01.004 |
| [40] Benton et al. | Valid | ACL page confirms DOI/pages. | Add DOI. URL: https://aclanthology.org/W17-1612/ |
| [41] Llama 3 Herd | Valid | arXiv DOI/source confirmed. | Add DOI URL: https://doi.org/10.48550/arXiv.2407.21783 |
| [42] TextBlob | Valid but software citation | Official docs confirm polarity range. | Keep; cite docs as software documentation. URL: https://textblob.readthedocs.io/en/dev/ |
| [43] Pattern for Python | Valid | JMLR page confirms citation. | Keep. URL: https://www.jmlr.org/papers/v13/desmedt12a.html |

## Suggested New References to Add

These are not all mandatory. For defense, add at least two from this list.

| Recommended Use | Suggested Reference | Why Add It | Link |
|---|---|---|---|
| LLM-based depression detection on social media | Y. Wang, D. Inkpen, and P. K. Gamaarachchige, “Explainable Depression Detection Using Large Language Models on Social Media Data,” CLPsych 2024. | Directly matches your Phase 2 interpretability/reasoning motivation. | https://aclanthology.org/2024.clpsych-1.8/ |
| Latest social-media depression + LLM framework | X. Lan et al., “Depression Detection on Social Media with Large Language Models,” EMNLP Industry 2025. | Very close to your topic: medical knowledge, explainability, social media, LLMs. | https://aclanthology.org/2025.emnlp-industry.151/ |
| LLM depression systematic review | “Exploring the efficacy and potential of large language models for depression: A systematic review,” Journal of Affective Disorders, 2025. | Strong support for “LLMs in depression are promising but need ethical/clinical caution.” | https://doi.org/10.1016/j.jad.2024.11.052 |
| Bias/methodological limitations in social-media depression ML | Cao et al., “Machine Learning Approaches for Depression Detection on Social Media: A Systematic Review of Biases and Methodological Challenges,” Journal of Behavioral Data Science, 2025. | Helps defend proxy-label, bias, sampling, preprocessing, and validation limitations. | https://doi.org/10.35566/jbds/caoyc |
| Text-based depression prediction meta-analysis | “Text-Based Depression Prediction on Social Media Using Machine Learning: Systematic Review and Meta-Analysis,” JMIR 2025. | Useful in Introduction/Related Work to justify social-media text as predictive but limited. | https://doi.org/10.2196/59002 |
| LLM-driven semi-supervised depression symptom modeling | Farruque et al., “Depression symptoms modelling from social media text: an LLM driven semi-supervised learning approach,” Language Resources and Evaluation, 2024. | Useful for LLM-assisted labeling/symptom modeling context. | https://doi.org/10.1007/s10579-024-09720-4 |

## Concrete Edits Recommended

### 1. Replace Weak Sentence Around Ref. [36]

Current:

> In addition, reliance on large-scale text data raises concerns about model generalization across diverse demographic or linguistic groups [36].

Recommended:

> In addition, social-media-based mental health prediction studies face persistent concerns regarding construct validity, sampling bias, demographic generalizability, and inconsistent reporting practices [38]. Recent reviews further emphasize that models trained on social media data may not generalize reliably across platforms, languages, demographic groups, or labeling protocols without more transparent validation and bias assessment.

Suggested citations:

- [38] Chancellor and De Choudhury, 2020
- Cao et al., 2025
- JMIR 2025 meta-analysis

### 2. Add Recent LLM Depression Work After Shin et al. [35]

Add after the Shin et al. paragraph:

> Closely related work has begun to examine LLMs for explainable depression detection on social media. Wang et al. applied LLMs to Reddit-based depression-level detection and emphasized explanation generation alongside predictive performance. Lan et al. further proposed an LLM-based framework for social-media depression detection that uses medical-knowledge-informed annotations and temporally summarized mood-course features to support both accuracy and interpretability. These studies support the relevance of LLM-based reasoning and explanation, while also highlighting the need for careful validation in mental-health-related applications.

Suggested citations:

- Wang et al., CLPsych 2024
- Lan et al., EMNLP Industry 2025

### 3. Add Selective Classification/Calibration Bridge in Related Work

The paper uses confidence routing as a core contribution, but the Literature Review does not currently give enough space to calibration/selective classification.

Add a short paragraph:

> The routing component of the proposed framework is also related to confidence calibration and selective classification. Modern neural networks can be poorly calibrated, meaning that their predicted probabilities do not necessarily reflect true correctness likelihood [15]. Selective classification addresses this issue by allowing a model to abstain from making predictions on uncertain inputs, thereby trading coverage for lower risk on accepted predictions [16]. In this study, we adapt this principle to depression-risk-related emotion classification by using validation-based risk-coverage analysis to decide which samples should be accepted by the Phase 1 classifier and which should be routed to Phase 2 reasoning.

This paragraph strongly defends the methodology.

### 4. Correct Bibliography Entries

Minimum corrections:

```tex
\bibitem{ref5} B. Levis, A. Benedetti, B. D. Thombs, and the DEPRESsion Screening Data (DEPRESSD) Collaboration, ``Accuracy of Patient Health Questionnaire-9 (PHQ-9) for screening to detect major depression: Individual participant data meta-analysis,'' BMJ, vol. 365, l1476, 2019. doi: 10.1136/bmj.l1476
```

```tex
\bibitem{ref36} S. Harrer, P. Shah, B. Antony, and J. Hu, ``Artificial intelligence for clinical trial design,'' Trends in Pharmacological Sciences, vol. 40, no. 8, pp. 577--591, 2019. doi: 10.1016/j.tips.2019.05.005
```

However, if [36] is not directly used after revision, remove it and replace it with a more relevant social-media/LLM mental-health reference.

```tex
\bibitem{ref34} I. Tavchioski, M. Robnik-\v{S}ikonja, and S. Pollak, ``Detection of depression on social networks using transformers and ensembles,'' arXiv preprint arXiv:2305.05325, 2023.
```

```tex
\bibitem{ref39} M. Conway and D. O'Connor, ``Social media, big data, and mental health: Current advances and ethical implications,'' Current Opinion in Psychology, vol. 9, pp. 77--82, 2016. doi: 10.1016/j.copsyc.2016.01.004
```

```tex
\bibitem{ref40} A. Benton, G. Coppersmith, and M. Dredze, ``Ethical research protocols for social media health research,'' in Proceedings of the First ACL Workshop on Ethics in Natural Language Processing, Valencia, Spain, pp. 94--102, 2017. doi: 10.18653/v1/W17-1612
```

## Submission-Risk Assessment

| Area | Risk Before Fix | Risk After Recommended Fix |
|---|---|---|
| Intro global burden references | Low | Low |
| PHQ-9 statement | Medium | Low |
| LLM novelty/latestness | Medium-High | Low-Medium |
| Confidence routing theoretical grounding | Medium | Low |
| Proxy-label/dataset validity defense | Medium-High | Medium |
| Ethics/privacy references | Low | Low |
| Reference metadata accuracy | Medium | Low |

## Final Recommendation

For a defensible IEEE/JMIR-style submission, do not overhaul the full reference section. Instead:

1. Correct factual bibliography errors.
2. Add 2-3 recent LLM/social-media depression references.
3. Replace [36] with a directly relevant source.
4. Add a short confidence calibration/selective-classification bridge paragraph.
5. Keep proxy-label and non-clinical framing explicit throughout the Introduction and Literature Review.

This is enough to make the reference section substantially stronger without expanding the paper into a broad survey.
