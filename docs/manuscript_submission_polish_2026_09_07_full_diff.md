# 원고 교정 전체 변경 기록

기준: 2026-09-03 v4 ZIP의 main.tex. 수정: 2026-09-07 polished revision.
총 73개 변경 블록. 표기한 줄 번호는 해당 버전의 LaTeX 소스 기준이다.
삭제 블록은 수정 후가 비어 있고, 추가 블록은 수정 전이 비어 있다. 수치표와 그림은 보존했다.

## 1. Preamble / Abstract

- 기준본 줄: 41-41; 수정본 줄: 41-41

### 수정 전
```latex
Transformer classifiers can process social-media text at scale, but their confidence scores are not automatically reliable and mixed or shifting affective cues remain difficult. We propose a confidence-guided two-phase framework for three-class proxy emotion classification (Depression, Neutral, and Happy). Phase~1 fine-tunes DistilBERT, calibrates its probabilities on a dedicated calibration split, and selects a routing threshold under a prespecified selective-risk constraint. In a matched three-seed comparison on a common 12,000-example Reddit test set, DistilBERT achieved 96.90 $\pm$ 0.12\% mean accuracy, versus 95.56 $\pm$ 0.11\% and 95.09 $\pm$ 0.06\% for frozen-backbone Mistral~7B and Llama~2~7B linear probes. This bounded comparison supported DistilBERT as the operational classifier. Its independently fixed checkpoint achieved 96.69\% accuracy and routed 171 posts (1.42\%); these posts contained 87 Phase~1 errors and had 49.12\% routed-only accuracy. Phase~2 re-evaluated only these low-confidence posts using minimally sanitized original title--body text. Llama~2 produced three negative net corrections and 96.67\% end-to-end accuracy, whereas Llama~3 produced 30 positive net corrections and 96.94\% accuracy, with a paired improvement that remained significant after Holm correction. On a supplementary 300-example synthetic Mixed Emotion stress test, Llama~2 and Llama~3 increased accuracy from 81.33\% to 85.33\% and 87.33\%, respectively. The results show that calibrated routing can concentrate difficult cases in a small review set, but realized benefit depends on the re-evaluator and preservation of source-text evidence. This study evaluates research-oriented proxy labels rather than clinical diagnoses and requires clinician-annotated validation before real-world decision-support use.
```

### 수정 후
```latex
Transformer classifiers can process social-media text at scale, but confidence-based selection does not ensure that a second model will correct the selected predictions. We evaluate a two-phase framework for three-class proxy emotion classification (Depression, Neutral, and Happy) that separates this selection problem from LLM re-evaluation. Phase~1 uses temperature-calibrated DistilBERT predictions and a calibration-set risk--coverage criterion to route low-confidence inputs. In a three-seed comparison on the same 12,000 Reddit test posts, DistilBERT achieved 96.90 $\pm$ 0.12\% accuracy, versus 95.56 $\pm$ 0.11\% for Mistral~7B and 95.09 $\pm$ 0.06\% for Llama~2~7B frozen-backbone linear probes. The independently fixed operational DistilBERT checkpoint achieved 96.69\% accuracy and routed 171 posts (1.42\%), including 87 of its 397 errors. Phase~2 used minimally sanitized original text and either Llama~2 Chain-of-Thought or Llama~3 SELF-DISCOVER. Llama~2 introduced three more errors than it corrected, yielding 96.67\% end-to-end accuracy. Llama~3 produced 30 net corrections and 96.94\% accuracy; its paired improvement remained significant after Holm correction. On a 300-example synthetic Mixed Emotion stress test, the two configurations increased accuracy from 81.33\% to 85.33\% and 87.33\%, respectively. Thus, routing concentrated errors in a small subset, but correction gains depended on the re-evaluation configuration. Because prompt development reused the routed Reddit cases, the end-to-end findings are exploratory. The study evaluates proxy emotion labels, not clinical diagnoses; independent data and expert review are needed to establish broader validity.
```

## 2. \section{Introduction}

- 기준본 줄: 53-53; 수정본 줄: 53-53

### 수정 전
```latex
Depression is a pervasive and debilitating mental health disorder that affects millions of people worldwide. According to the World Health Organization's 2025 fact sheet, approximately 332 million people globally live with depression [1]. WHO has identified depression as a leading cause of ill health and disability worldwide [2]. Beyond its clinical impact, depression and anxiety together impose substantial economic costs, with an estimated US\$1 trillion lost annually in productivity [3]. The COVID-19 pandemic further intensified global mental health challenges, contributing to a 25\% increase in the prevalence of anxiety and depression during the first year of the pandemic [4]. These figures motivate research on methods that can identify depression-risk-related emotional signals in non-clinical language while preserving clear boundaries between computational proxy labels and clinical assessment.
```

### 수정 후
```latex
The World Health Organization estimates that approximately 332 million people live with depression [1] and identifies it as a leading cause of ill health and disability [2]. Depression and anxiety together account for an estimated US\$1 trillion in annual productivity losses [3]; their prevalence also increased during the first year of the COVID-19 pandemic [4]. This burden motivates research on emotional signals in everyday language, provided that computational text labels remain distinct from clinical assessment.
```

## 3. \section{Introduction}

- 기준본 줄: 55-55; 수정본 줄: 55-55

### 수정 전
```latex
Despite this urgency, depression screening and assessment still rely heavily on self-report questionnaires, such as the Patient Health Questionnaire-9 (PHQ-9), as well as clinician-led evaluation [5]. OECD benchmarks for high-performing mental health systems emphasize accessible, high-quality, and person-centred services, while also highlighting persistent gaps across countries and healthcare systems [6]. At the same time, the growing volume of publicly available textual data, including online journals, forums, and social media posts, has emerged as a promising source for studying mental health-related signals in everyday language use. This has led to increasing interest in natural language processing (NLP) and artificial intelligence (AI) methods for analyzing depression-risk-related linguistic and emotional patterns in non-clinical text [7].
```

### 수정 후
```latex
Validated questionnaires such as the Patient Health Questionnaire-9 (PHQ-9) support depression screening [5], while accessible, person-centred care remains a health-system priority [6]. Social-media analysis serves a different purpose: it allows researchers to study linguistic and emotional patterns in non-clinical text [7]. A post-level emotion prediction neither replaces those instruments nor establishes the author's clinical condition.
```

## 4. \section{Introduction}

- 기준본 줄: 57-57; 수정본 줄: 57-57

### 수정 전
```latex
Recent transformer-based approaches, including domain-adapted pretrained language models such as MentalBERT, have improved performance on mental health text classification benchmarks and have shown that target-domain pretraining can strengthen downstream detection tasks [8]. However, text-based depression-related classification from non-clinical text remains challenging because online posts often contain incomplete or limited context, and relevant signals may unfold over time rather than being fully captured within a single post [9], [10]. These difficulties are especially important in real-world depression discourse, where emotional cues may be context-dependent and difficult to interpret [9], [10]. In addition, systems that may inform healthcare or public-health research should provide explanations that clinicians and domain experts can understand and trust, as explainability has been identified as an important factor in building clinician trust and supporting the effective translation of machine learning systems into clinical practice [11].
```

### 수정 후
```latex
Recent transformer-based approaches, including domain-adapted pretrained language models such as MentalBERT, have improved performance on mental health text classification benchmarks and have shown that target-domain pretraining can strengthen downstream detection tasks [8]. However, text-based depression-related classification from non-clinical text remains challenging because online posts often contain incomplete or limited context, and relevant signals may unfold over time rather than being fully captured within a single post [9], [10]. In addition, systems that may inform healthcare or public-health research should provide explanations that clinicians and domain experts can understand and trust, as explainability has been identified as an important factor in building clinician trust and supporting the effective translation of machine learning systems into clinical practice [11].
```

## 5. \section{Introduction}

- 기준본 줄: 63-63; 수정본 줄: 63-63

### 수정 전
```latex
The evaluation is organized around three research questions. \textbf{RQ1} has two linked components: RQ1a asks whether the operational Phase~1 classifier is accurate and sufficiently calibrated for confidence-based routing, and RQ1b asks whether matched 7B comparison classifiers overturn its selection when predictive performance and model scale are considered together. \textbf{RQ2} asks whether the fixed routing policy concentrates Phase~1 errors while preserving high coverage. \textbf{RQ3} asks whether model-specific Phase~2 reasoning produces positive net corrections on naturally occurring held-out Reddit posts and on a controlled mixed-emotion stress test. This decomposition prevents classifier selection, routing quality, and re-evaluation benefit from being treated as the same empirical claim.
```

### 수정 후
```latex
The evaluation addresses three research questions. \textbf{RQ1a} assesses the operational classifier's predictive and calibration performance; \textbf{RQ1b} compares the candidate classifiers under a shared data and search budget. \textbf{RQ2} asks whether the fixed routing policy concentrates Phase~1 errors while retaining high coverage. \textbf{RQ3} asks whether re-evaluation produces positive net corrections on Reddit posts and on a controlled synthetic stress test. Separating these questions allows a useful router and an ineffective re-evaluator to be identified within the same experiment.
```

## 6. \section{Study Contributions}

- 기준본 줄: 75-75; 수정본 줄: 75-75

### 수정 전
```latex
Together, these contributions distinguish three questions that are often conflated: whether Phase~1 predicts well overall, whether confidence isolates a harder subset, and whether a particular Phase~2 reasoning policy improves that subset without creating more errors.
```

### 수정 후
```latex
The contribution is the integration and evaluation protocol, not a new calibration estimator, language-model architecture, or reasoning method.
```

## 7. \subsection{Early Approaches: Machine Learning for Depression Detection}

- 기준본 줄: 81-81; 수정본 줄: 81-81

### 수정 전
```latex
Early computational approaches to depression-related text classification relied on traditional machine learning techniques with handcrafted linguistic features. Psycholinguistic studies have shown that individuals reporting depressive symptoms often exhibit distinctive language patterns, such as increased use of first-person pronouns and negative emotional words. Rude et al. [20] quantified such markers, finding that currently depressed participants used first-person singular pronouns and negative-emotion words more frequently than never-depressed participants. Building on these insights, initial NLP systems used lexical features, such as word counts, n-grams, and sentiment lexica such as LIWC categories, to represent text, applying classifiers like support vector machines (SVM) or Na\\\"ive Bayes to distinguish depression-indicative from non-depression-indicative language [21]-[23]. For example, De Choudhury et al. (2013) [21] analyzed social media posts with features such as linguistic style, engagement, and ego-network characteristics to predict risk of major depression, demonstrating the feasibility of depression prediction from Twitter data. Similarly, Tsugawa et al. (2015) [22] showed that a user's Twitter activity patterns, including posting frequency, timing, and linguistic cues, could be leveraged to identify depression-related signals, achieving promising accuracy with an SVM-based model.
```

### 수정 후
```latex
Early computational approaches to depression-related text classification relied on traditional machine learning techniques with handcrafted linguistic features. Psycholinguistic studies have shown that individuals reporting depressive symptoms often exhibit distinctive language patterns, such as increased use of first-person pronouns and negative emotional words. Rude et al. [20] quantified such markers, finding that currently depressed participants used first-person singular pronouns and negative-emotion words more frequently than never-depressed participants. Building on these insights, initial NLP systems used lexical features, such as word counts, n-grams, and sentiment lexica such as LIWC categories, to represent text, applying classifiers like support vector machines (SVM) or naive Bayes to distinguish depression-indicative from non-depression-indicative language [21]-[23]. For example, De Choudhury et al. (2013) [21] analyzed social media posts with features such as linguistic style, engagement, and ego-network characteristics to predict risk of major depression, demonstrating the feasibility of depression prediction from Twitter data. Similarly, Tsugawa et al. (2015) [22] showed that a user's Twitter activity patterns, including posting frequency, timing, and linguistic cues, could be leveraged to identify depression-related signals, achieving promising accuracy with an SVM-based model.
```

## 8. \subsection{Large Language Models for Mental Health Analysis}

- 기준본 줄: 97-97; 수정본 줄: 97-97

### 수정 전
```latex
Recent years have seen a growing shift toward large language models (LLMs) and transformer-based architectures for analyzing mental health-related text. Transformer-based and generative language models, such as BERT and GPT, are pre-trained on large-scale text corpora and capture rich linguistic and semantic representations that can be adapted to downstream tasks, including depression-related text classification. Compared with earlier studies that primarily employed NLP techniques with traditional machine learning classifiers, more recent work has increasingly adopted transformer-based architectures to model social media text for depression-related risk analysis [8], [33], [34].
```

### 수정 후
```latex
Transformer encoders and generative LLMs serve different roles in mental-health text analysis. Encoders provide contextual representations for supervised classification [8], [33], [34], whereas generative models can produce labels together with textual responses. Their use in the same application domain does not make predictive accuracy and explanation quality interchangeable outcomes.
```

## 9. \subsection{Large Language Models for Mental Health Analysis}

- 기준본 줄: 99-99; 수정본 줄: 99-99

### 수정 전
```latex
Recent studies have demonstrated the effectiveness of transformer-based representations for depression prediction on social media. For example, Zhang et al. (2024) [33] proposed a transformer-based framework to identify depression-related risk signals using Sina Weibo posts, showing that such architectures are well suited for large-scale depression-related text analysis on user-generated content. By modeling text representations at both post and user levels, transformer-based architectures provide a flexible framework for learning semantic representations from social media data. Beyond general-purpose transformers, researchers have explored domain-specific pretraining and model combination strategies to better address the characteristics of mental health language [8], [34]. Ji et al. (2022) [8] introduced MentalBERT, a language model pretrained on mental health-related corpora, and demonstrated that domain-specific pretraining can be beneficial for downstream mental health NLP tasks. In addition, ensemble approaches that combine multiple transformer-based classifiers have been investigated as a way to improve classification performance in depression detection on social media [34].
```

### 수정 후
```latex
Zhang et al. [33] studied transformer-based depression-risk detection using Sina Weibo posts. MentalBERT [8] instead emphasizes domain-specific pretraining, while ensemble approaches combine transformer classifiers [34]. These studies motivate contextual representations but do not determine whether a second-stage model should process every post or only a selected subset.
```

## 10. \subsection{Large Language Models for Mental Health Analysis}

- 기준본 줄: 105-105; 수정본 줄: 105-105

### 수정 전
```latex
Despite these advances, the application of LLMs to mental health analysis remains at an early stage, and several important challenges persist. Transformer-based and generative models are commonly regarded as complex, which can pose challenges for interpretability and transparency in sensitive clinical settings. Social-media-based mental health prediction also raises concerns about construct validity, sampling bias, demographic generalizability, platform dependence, and inconsistent reporting practices [37], [46]. Systematic reviews of depression detection research provide a broad overview of existing datasets, methods, evaluation strategies, and reported limitations of current approaches [36], [46]. Overall, transformer-based and generative language models represent a significant methodological advance for depression-related text classification, but their adoption in mental health applications requires careful validation and responsible integration.
```

### 수정 후
```latex
These approaches retain the construct-validity, sampling, and platform-dependence concerns documented in reviews [36], [37], [46]. For the present task, a generated explanation is therefore an inspectable output, not evidence that the proxy label or the prediction is clinically valid.
```

## 11. \subsection{Confidence Calibration, Selective Prediction, and Cascaded Re-Evaluation}

- 기준본 줄: 109-109; 수정본 줄: 109-109

### 수정 전
```latex
The routing component of the proposed framework is grounded in confidence calibration and selective classification. Modern neural networks can be poorly calibrated, meaning that their predicted probabilities do not necessarily reflect their empirical likelihood of correctness [15]. Selective classification addresses this limitation by allowing a model to abstain on uncertain inputs, thereby trading prediction coverage for lower risk among accepted predictions [16]. These methods establish how to decide which predictions should be retained, but they do not by themselves specify how abstained predictions should be corrected.
```

### 수정 후
```latex
Calibration addresses the correspondence between predicted confidence and observed correctness [15]. Selective classification addresses a related but distinct problem: trading coverage for accepted-set risk by abstaining on selected inputs [16]. Abstention does not itself calibrate probabilities or correct the rejected predictions. This distinction motivates separate evaluation of confidence quality, case selection, and downstream correction.
```

## 12. \subsection{Confidence Calibration, Selective Prediction, and Cascaded Re-Evaluation}

- 기준본 줄: 111-111; 수정본 줄: 111-111

### 수정 전
```latex
The present study turns that abstention decision into a fixed cascaded re-evaluation workflow. Temperature-scaled confidence and calibration-split risk--coverage analysis determine which Phase~1 predictions are accepted; only the complementary routed set is passed to a separately configured LLM reasoner. The study then evaluates the classifier, router, and re-evaluator with distinct outcomes: predictive and calibration metrics for Phase~1, coverage and error concentration for routing, and corrected, introduced, and net errors for Phase~2. This separation is important because a well-calibrated router can identify difficult cases even when a particular reasoner fails to improve them, and a plausible rationale does not establish a correct replacement decision.
```

### 수정 후
```latex
Here, rejected inputs are sent to an LLM rather than left unclassified. Temperature scaling [15] and selective prediction [16] supply the routing components; CoT [17] and SELF-DISCOVER [18] supply the re-evaluation protocols. The empirical question is whether recombining their outputs improves the original classifier after newly introduced errors are counted.
```

## 13. \section{Dataset and Preprocessing}

- 기준본 줄: 125-125; 수정본 줄: 125-125

### 수정 전
```latex
This section describes the construction of the Reddit dataset, preprocessing procedures, sentiment-aware sampling strategy, supplementary Mixed Emotion Dataset, and ethical considerations. The proposed framework is designed for depression-risk-related emotion classification from non-clinical social media text; therefore, the dataset should be interpreted as supporting research-oriented emotional signal analysis rather than clinical diagnosis.
```

### 수정 후
```latex
The primary Reddit corpus supplies model development and evaluation data. A separate synthetic dataset tests controlled affective scenarios and Neutral controls. Their construction and uses are described below.
```

## 14. \subsection{Text Cleaning and Normalization}

- 기준본 줄: 143-143; 수정본 줄: 143-143

### 수정 전
```latex
In addition, a Reddit-specific stop-word list was applied to filter platform-specific filler expressions. The resulting cleaned text was stored in the \path{Title_with_selftext_cleaned} field and used for Phase~1 training, calibration, routing, and held-out prediction. It was not used as the final Phase~2 reasoning input because removal of negation, punctuation, sentence order, and discourse markers can discard evidence needed for trajectory-sensitive re-evaluation.
```

### 수정 후
```latex
In addition, a Reddit-specific stop-word list filtered platform-specific filler expressions. The cleaned field, \path{Title_with_selftext_cleaned}, was used for Phase~1 training, calibration, and prediction. Cleaning preserves the order of retained tokens but can remove negation, punctuation, and discourse markers. Phase~2 therefore used the minimally sanitized original text to retain these cues and sentence boundaries.
```

## 15. \subsection{Mixed Emotion Dataset}

- 기준본 줄: 164-164; 수정본 줄: 164-164

### 수정 전
```latex
Synthetic examples were generated using controlled prompting and reviewed according to predefined inclusion and exclusion criteria, following the broader use of LLM-generated data for depression-related modeling while retaining a separate evaluation-only role in this study [19]. Examples were retained only if they contained mixed or shifting emotional cues while remaining within the three-class proxy emotion label space. Examples were excluded if they contained explicit clinical diagnosis claims, direct treatment recommendations, crisis language, personally identifying information, or content that did not clearly instantiate the target ambiguity scenario. The generation prompt and dataset construction protocol are reported in Appendix Table A5. Because the examples are synthetic, results on this dataset should be interpreted as supplementary evidence rather than as a substitute for expert-annotated or naturally occurring mixed-emotion data. All mixed-emotion numerical results in this manuscript are based on the final 300-example stress-test evaluation.
```

### 수정 후
```latex
Synthetic examples were generated using controlled prompting and checked against inclusion and exclusion criteria. This follows the use of LLM-generated data in depression-related modeling [19], but the data here serve only as a supplementary evaluation set. Inclusion required consistency with the assigned scenario and target label: mixed or shifting cues for the affective scenarios, and low-affect content for the Neutral controls. Exclusions covered explicit clinical diagnosis claims, treatment recommendations, crisis language, identifying information, and off-scenario content. Appendix Table~A5 records the generation protocol. These checks were not an independent expert-annotation study; results describe the final 300 synthetic examples rather than naturally occurring mixed-emotion performance.
```

## 16. \subsection{Data Partitioning and Model Inputs}

- 기준본 줄: 170-170; 수정본 줄: 170-170

### 수정 전
```latex
The training set is used for supervised fine-tuning of the Phase 1 classifiers. The model-validation set is used for hyperparameter selection, early stopping, and best-checkpoint selection. A separate calibration set is then used for temperature scaling, calibration-quality assessment, and confidence-routing threshold selection based on the risk-coverage trade-off. The held-out test set is reserved for final performance evaluation only. The cleaned title-body text is tokenized using the tokenizer corresponding to each language model and converted into the required input format for model fine-tuning.
```

### 수정 후
```latex
The training partition supplies supervised parameter updates; model validation supplies hyperparameter, epoch, and checkpoint selection. The calibration partition supplies temperature fitting and threshold selection. Test labels are excluded from these Phase~1 operations. However, the routed Reddit test cases were reused during Phase~2 prompt development, as disclosed in the experimental limitations. Each classifier uses its own tokenizer on the cleaned title--body field.
```

## 17. \section{Methodology}

- 기준본 줄: 188-192; 수정본 줄: 188-188

### 수정 전
```latex
This study proposes a confidence-guided two-phase framework that combines efficient transformer-based emotion classification with selective LLM-based reasoning. In Phase~1, a fine-tuned transformer produces an initial class prediction and confidence score. High-confidence predictions are accepted directly, whereas low-confidence predictions are routed to Phase~2 for reasoning-based re-evaluation.

In Phase~1, DistilBERT [12] was fully fine-tuned for three-class classification and used as the operational classifier. Mistral~7B [13] and Llama~2~7B [14] were prespecified as matched comparison architectures. For these substantially larger comparators, the pretrained backbone was frozen and only the task-specific classification parameters were optimized; LoRA, QLoRA, and other adapter updates were not used. This deliberately bounded comparison asks whether either 7B representation provides enough predictive benefit to displace the much smaller operational model, rather than attempting an exhaustive optimization study of 7B fine-tuning methods. The operational DistilBERT model outputs class probabilities, and predictions above a calibration-derived risk-controlled confidence threshold are accepted directly.

Low-confidence predictions are routed to Phase~2, where reasoning-capable LLMs are used for re-evaluation. Specifically, Llama~2-7B-Chat [14] was paired with Chain-of-Thought prompting [17], whereas Llama~3-8B-Instruct [40] was integrated with the SELF-DISCOVER framework [18]. This design adds a separately auditable re-evaluation path for uncertain predictions; whether that path improves accuracy is determined empirically rather than assumed from the use of reasoning prompts.
```

### 수정 후
```latex
The pipeline has two stages: an operational DistilBERT classifier [12] assigns a label and calibrated confidence, and a separately configured LLM re-evaluates low-confidence inputs. Phase~2 uses either Llama~2-7B-Chat [14] with Chain-of-Thought prompting [17] or Llama~3-8B-Instruct [40] with SELF-DISCOVER [18]. The configurations are evaluated separately, not ensembled.
```

## 18. \subsection{Confidence-Guided Hybrid Model Architecture}

- 기준본 줄: 196-196; 수정본 줄: 192-192

### 수정 전
```latex
The proposed framework adopts a confidence-guided two-phase architecture that combines an initial transformer-based classifier with a secondary reasoning-based review stage. In Phase 1, the classifier predicts one of three emotional classes and estimates prediction confidence using the maximum softmax probability. The updated workflow further calibrates these confidence scores using temperature scaling on a threshold-calibration split that is separate from both model validation and held-out testing. Predictions at or above the selected confidence threshold are accepted directly, whereas predictions below it are routed to Phase 2, where a reasoning-capable LLM re-evaluates the input and generates an explanatory rationale. This cascading design limits expensive reasoning to the routed subset and adds an inspectable rationale for those cases; robustness is assessed from the observed end-to-end corrections rather than asserted as an architectural guarantee.
```

### 수정 후
```latex
Development and inference are separate operations. Development fixes the classifier, calibration temperature, and routing threshold. At inference, predictions at or above the threshold retain their Phase~1 labels. Predictions below it receive a Phase~2 label and a stored rationale. Full-set evaluation then recombines the two paths, counting both corrected and newly introduced errors.
```

## 19. \subsection{Phase 1: Initial Emotion Classification}

- 기준본 줄: 209-209; 수정본 줄: 205-205

### 수정 전
```latex
In Phase~1, DistilBERT [12] was fully fine-tuned for three-class classification of Reddit posts into Depression, Neutral, and Happy. Mistral~7B [13] and Llama~2~7B [14] were evaluated as frozen-backbone comparison classifiers. For each 7B model, the final hidden representation was mapped to the three classes by a bias-free $4096\times3$ linear probe, yielding 12,288 trainable parameters while leaving the pretrained backbone unchanged. The selected configuration for each architecture was repeated with random seeds 42, 43, and 44, and the matched comparison reports mean $\pm$ standard deviation for held-out accuracy and macro F1. This design is an operational screening comparison under a common data and search budget, not an estimate of the maximum accuracy attainable through architecture-specific adapter tuning or full 7B fine-tuning. For each post, Phase~1 produces class logits, a predicted label, and a maximum-softmax confidence score. The probability transformation, post-hoc calibration, and routing rule are defined jointly below so that raw and calibrated confidence are not treated as interchangeable quantities.
```

### 수정 후
```latex
The Phase~1 comparison includes DistilBERT [12], Mistral~7B [13], and Llama~2~7B [14] under common data partitions and a four-trial search budget. DistilBERT is fully fine-tuned, whereas the 7B backbones use frozen representations with trainable linear heads. Thus, this is a comparison of practical classifier configurations, not an isolated test of architecture or maximum attainable performance. Each selected configuration is evaluated with seeds 42, 43, and 44. Appendix Table~A1c gives the selected settings; only the fixed operational DistilBERT checkpoint supplies downstream routing scores.
```

## 20. \subsection{Mistral}

- 기준본 줄: 217-217; 수정본 줄: 213-213

### 수정 전
```latex
Mistral~7B [13] is a decoder-only transformer that combines grouped-query attention (GQA) with sliding-window attention (SWA). GQA reduces key--value memory requirements during inference, while SWA bounds attention to a local window and supports efficient processing of longer sequences. These features explain why Mistral is a relevant larger-model comparator, but they are not treated as contributions of the present work. Here, the prespecified Mistral checkpoint is used as a frozen representation backbone and only a task-specific three-class classification head is optimized under the common partitions and label space. The experiment therefore asks whether the frozen 7B representation yields a sufficient predictive advantage to displace the much smaller operational classifier; it does not estimate the best performance attainable through full fine-tuning, LoRA, QLoRA, or architecture-specific adaptation.
```

### 수정 후
```latex
Mistral~7B [13] is a decoder-only transformer with grouped-query attention (GQA) and sliding-window attention (SWA). GQA reduces key--value memory requirements, while SWA restricts attention to a local window. Here, Mistral provides a larger pretrained representation for three-class classification. Its backbone remains frozen, and a bias-free linear head maps the 4,096-dimensional representation to three class logits. Only the head's 12,288 parameters are trained; no adapter is added.
```

## 21. \subsection{Llama 2}

- 기준본 줄: 221-221; 수정본 줄: 217-217

### 수정 전
```latex
Llama~2 [14] is a family of decoder-only autoregressive transformers released at several parameter scales. The 7B base checkpoint provides a second higher-capacity representation family for the Phase~1 comparison and is distinct from the chat-tuned Llama~2 checkpoint used later as a Phase~2 reasoner. For classification, the pretrained 7B backbone remains frozen and only the task-specific three-class parameters are optimized using the same prespecified partitions, label mapping, and evaluation metrics. This isolates the utility of the frozen representation under a bounded supervised protocol and avoids turning the study into an adapter-design or full-model fine-tuning benchmark. Accordingly, the comparison supports an operational model choice within the tested scope rather than a general ranking of DistilBERT and Llama~2.
```

### 수정 후
```latex
Llama~2 [14] is a family of decoder-only autoregressive transformers. Its 7B base checkpoint supplies a second larger representation family for Phase~1 and is distinct from the chat-tuned checkpoint used in Phase~2. Classification uses the same frozen-backbone, bias-free 12,288-parameter linear-head design as Mistral. The shared partitions and label mapping permit predictive comparison, while the differing adaptation scope precludes attributing differences to model size alone.
```

## 22. \subsection{Llama 2}

- 기준본 줄: 223-223; 수정본 줄: 219-219

### 수정 전
```latex
During Phase~1, each comparison classifier outputs a predicted class and confidence score for each post. The confidence score is defined as the maximum softmax probability among the three class probabilities after calibration. A confidence threshold $\tau$ is then used to determine whether the prediction is sufficiently reliable to be accepted directly or whether it should be routed to Phase~2 for reasoning-based re-evaluation. The threshold selection procedure is described in a later subsection. Only the selected operational classifier supplies the probabilities used by the final routing pipeline; comparison-model runs are never used to choose $T^*$ or $\tau^*$.
```

### 수정 후
```latex
All three classifiers output class logits. Post-hoc calibration and threshold selection are performed only for the operational DistilBERT checkpoint; no calibrated routing result is claimed for the comparison models.
```

## 23. \subsection{Calibrated Confidence-Based Routing and Risk-Coverage Threshold Selection}

- 기준본 줄: 318-318; 수정본 줄: 314-314

### 수정 전
```latex
where $\mathrm{Beta}^{-1}(\cdot;a,b)$ is the inverse cumulative distribution function of a Beta$(a,b)$ distribution. When every accepted prediction is incorrect, the bound is set to 1. This bound is not another accuracy metric; it is a conservative guard against selecting a threshold merely because a finite calibration sample happened to contain few errors. The risk budget $\alpha$ and tail probability $\delta$ have distinct roles: $\delta$ determines the confidence level of the upper bound, whereas $\alpha$ determines whether the resulting bound is operationally acceptable. The Clopper--Pearson quantity is evaluated pointwise for each candidate threshold and is used here as a conservative calibration criterion; no simultaneous post-selection finite-sample guarantee over the entire threshold grid is claimed.
```

### 수정 후
```latex
where $\mathrm{Beta}^{-1}(\cdot;a,b)$ is the inverse cumulative distribution function of a Beta$(a,b)$ distribution. When every accepted prediction is incorrect, the bound is set to 1; empty accepted sets are ineligible. The tail probability $\delta$ specifies the nominal confidence level, whereas $\alpha$ specifies the acceptable risk budget. This is a calibration-set selection criterion. Because temperature fitting and threshold search reuse the calibration data, pointwise Clopper--Pearson bounds do not establish a simultaneous post-selection guarantee for the resulting policy. Nor does the criterion guarantee risk control under distribution shift.
```

## 24. \subsection{Calibrated Confidence-Based Routing and Risk-Coverage Threshold Selection}

- 기준본 줄: 330-330; 수정본 줄: 326-326

### 수정 전
```latex
The threshold selection table also records accepted accuracy, routed Phase 1 errors, error-capture rate, routing precision, and depression false-negative risk among accepted predictions. Additional confidence diagnostics include expected calibration error (ECE), adaptive ECE, Brier score, and negative log-likelihood before and after temperature scaling. Auxiliary analyses compare raw MSP, temperature-scaled MSP, entropy-based certainty, and probability margin; estimate bootstrap confidence intervals for selective metrics; assess threshold stability through calibration-set resampling; and extract high-confidence accepted errors for qualitative auditing.
```

### 수정 후
```latex
The threshold-selection output records accepted accuracy, routed Phase~1 errors, error capture, routing precision, and proxy-Depression false-negative risk among accepted predictions. Additional confidence diagnostics include expected calibration error (ECE), adaptive ECE, Brier score, and negative log-likelihood before and after temperature scaling. Auxiliary analyses compare raw MSP, temperature-scaled MSP, entropy-based certainty, and probability margin; estimate bootstrap confidence intervals for selective metrics; assess threshold stability through calibration-set resampling; and extract high-confidence accepted errors for qualitative auditing.
```

## 25. \subsection{Calibrated Confidence-Based Routing and Risk-Coverage Threshold Selection}

- 기준본 줄: 338-338; 수정본 줄: 334-334

### 수정 전
```latex
\State \textbf{Input:} risk target $\alpha$, confidence level $1-\delta$
```

### 수정 후
```latex
\State \textbf{Input:} $\alpha$, $1-\delta$, grid $\mathcal{T}$, minimum count $m_{\min}$
```

## 26. \subsection{Calibrated Confidence-Based Routing and Risk-Coverage Threshold Selection}

- 기준본 줄: 348-348; 수정본 줄: 344-346

### 수정 전
```latex
\State Choose feasible $\tau$ with maximum coverage
```

### 수정 후
```latex
\State Retain candidates with $\overline{R}_{\delta}(\tau)\leq\alpha$ and $|A(\tau)|\geq m_{\min}$
\State If none remain, report infeasibility and stop
\State Choose a retained $\tau$ with maximum coverage
```

## 27. \subsection{Calibrated Confidence-Based Routing and Risk-Coverage Threshold Selection}

- 기준본 줄: 358-358; 수정본 줄: 356-356

### 수정 전
```latex
\State \textbf{Input:} post $x$, Phase~1 classifier $f_{\theta}$, fixed $T^*$ and $\tau^*$
```

### 수정 후
```latex
\State \textbf{Input:} post record $x$, classifier $f_{\theta}$, fixed $T^*$ and $\tau^*$
```

## 28. \subsection{Calibrated Confidence-Based Routing and Risk-Coverage Threshold Selection}

- 기준본 줄: 360-360; 수정본 줄: 358-359

### 수정 전
```latex
\State Compute logits $\mathbf{z}=f_{\theta}(x)$ and calibrated probabilities $q_k(T^*)$
```

### 수정 후
```latex
\State Obtain the Phase~1 input $x^{(1)}$ from record $x$
\State Compute $\mathbf{z}=f_{\theta}(x^{(1)})$ and probabilities $q_k(T^*)$
```

## 29. \subsection{Calibrated Confidence-Based Routing and Risk-Coverage Threshold Selection}

- 기준본 줄: 365-365; 수정본 줄: 364-365

### 수정 전
```latex
    \State Generate structured rationale and terminal label with $g(x,\hat{y}^{(1)})$
```

### 수정 후
```latex
    \State Retrieve the minimally sanitized original text $x^{(2)}$
    \State Generate rationale and terminal label with $g(x^{(2)},\hat{y}^{(1)})$
```

## 30. \subsection{Calibrated Confidence-Based Routing and Risk-Coverage Threshold Selection}

- 기준본 줄: 377-377; 수정본 줄: 377-377

### 수정 전
```latex
\State \textbf{Input:} true labels $y_i$, Phase~1 labels $\hat{y}^{(1)}_i$, route indicators $r_i$
```

### 수정 후
```latex
\State \textbf{Input:} reference labels $y_i$, Phase~1 labels $\hat{y}^{(1)}_i$, route indicators $r_i$
```

## 31. \subsection{Calibrated Confidence-Based Routing and Risk-Coverage Threshold Selection}

- 기준본 줄: 391-391; 수정본 줄: 391-391

### 수정 전
```latex
The implementation records an explicit infeasibility status if no candidate satisfies the risk constraint. A minimum-risk fallback exists only for notebook smoke tests and was not invoked in the reported paper-scale experiment; no fallback threshold is treated as satisfying the risk-control constraint. The selected threshold, confidence method, temperature, risk-control method, candidate-generation rule, acceptance boundary, tie-breaking rule, split ratios, random seed, feasibility status, and final selective metrics are saved in \texttt{threshold\_provenance.json}. This provenance prevents held-out behavior from influencing the routing rule and makes the fixed operating decision reproducible.
```

### 수정 후
```latex
The implementation records an explicit infeasibility status if no candidate satisfies the risk constraint. A minimum-risk fallback exists only for notebook smoke tests and was not invoked in the reported paper-scale experiment; no fallback threshold is treated as satisfying the risk-control constraint. The selected threshold, confidence method, temperature, risk-control method, candidate-generation rule, acceptance boundary, tie-breaking rule, split ratios, random seed, feasibility status, and final selective metrics are saved in \texttt{threshold\_provenance.json}. This provenance records the fixed operating decision for reproducibility; it does not by itself establish independence from earlier development choices.
```

## 32. \subsection{Llama 3 with SELF-DISCOVER}

- 기준본 줄: 415-415; 수정본 줄: 415-415

### 수정 전
```latex
The process is illustrated with an observed final-run correction rather than a hypothetical scenario. For \texttt{MEV2\_DEP\_022}, Phase~1 predicted Happy after emphasizing encouraging feedback near the beginning of the post. The stored \texttt{SELECT} artifact named Critical Thinking, Textual Analysis, Risk Analysis, and Reflective Thinking. \texttt{ADAPT} reframed these as emotion-focused analysis, textual-evidence judgment, classification-risk assessment, and reflective emotional understanding. \texttt{IMPLEMENT} then organized those operations into a JSON plan for identifying cues, comparing the early positive statement with the later decline, and producing a terminal class. Execution concluded that the unresolved sadness dominated the full trajectory and returned \texttt{Final label: Depression}. Appendix Table~A4b reports publication-length excerpts from these stored fields.
```

### 수정 후
```latex
For the observed case \texttt{MEV2\_DEP\_022}, Phase~1 predicted Happy for a post that begins with encouraging feedback but ends with unresolved sadness. The stored \texttt{SELECT} trace names Critical Thinking, Textual Analysis, Risk Analysis, and Reflective Thinking. The subsequent artifacts adapt these instructions and organize them into a JSON plan. The terminal response cites the later sadness and returns \texttt{Final label: Depression}. Appendix Table~A4b provides excerpts. This record shows the model's generated justification; it does not establish which cues caused either model's prediction.
```

## 33. \subsection{Operational Inference Paths}

- 기준본 줄: 427-427; 수정본 줄: 427-427

### 수정 전
```latex
With $T^*$ and $\tau^*$ fixed on the calibration split, the system follows one of two inference paths. \textit{Direct acceptance:} if calibrated Phase~1 confidence is at least $\tau^*$, the Phase~1 label is retained without LLM inference. This operational rule does not imply that every accepted post is semantically simple or correct. \textit{Secondary re-evaluation:} if confidence is below $\tau^*$, the minimally sanitized original title--body text and Phase~1 label are passed to the assigned reasoner, which returns an inspectable rationale and one terminal label. Only this routed subset incurs the additional Phase~2 computation. The architecture therefore reduces reasoning cost relative to all-input LLM inference, but any accuracy or safety benefit must be established by end-to-end evaluation.
```

### 수정 후
```latex
Only routed inputs undergo Phase~2 inference. Accepted labels remain unchanged even if they are wrong; routed labels are replaced by the parsed terminal output even if that replacement introduces an error. The final runs produced valid labels for all routed records. Reference labels are used for evaluation, not provided to the reasoner. Selecting fewer inputs reduces the number of posts re-evaluated, but does not by itself quantify latency or monetary savings.
```

## 34. \subsection{Model Selection and Calibration Protocol}

- 기준본 줄: 441-446; 수정본 줄: 441-440

### 수정 전
```latex
DistilBERT, which has approximately 66 million parameters, was fully fine-tuned. For Mistral~7B and Llama~2~7B, the pretrained backbone remained frozen and optimization was restricted to a bias-free $4096\times3$ linear classification head with 12,288 trainable parameters. No LoRA, QLoRA, or other low-rank adapter was trained. This scope keeps the comparison reproducible and computationally bounded, but it should not be interpreted as the maximum attainable performance of either 7B architecture.

\subsection{Model Training}

After dataset preparation, DistilBERT was trained as the operational Phase~1 classifier on the curated Reddit dataset. Mistral~7B and Llama~2~7B were trained as prespecified frozen-backbone comparison classifiers using the same label space and prespecified partitions. Hyperparameters were selected without access to calibration or held-out test labels, and each selected architecture-specific configuration was repeated with three random seeds. The Phase~2 reasoning models are configured separately and applied only to predictions selected through confidence-guided routing.

```

### 수정 후
```latex

```

## 35. \subsection{Model Selection and Calibration Protocol}

- 기준본 줄: 449-449; 수정본 줄: 443-443

### 수정 전
```latex
The updated experimental workflow separates model selection from threshold calibration. The completed DistilBERT model is trained using the predefined training split and selected using the model-validation split. Hyperparameter tuning and early stopping are therefore performed without access to the threshold-calibration or held-out test sets. After the final checkpoint is selected, the calibration split is used to fit the temperature parameter, evaluate calibration quality, and select the confidence-routing threshold. The held-out test split is used only after the temperature and routing threshold have been fixed. The final protocol uses exactly 40,000 examples per class and partitions each class into 28,000 training, 4,000 validation, 4,000 calibration, and 4,000 test examples, yielding 84,000/12,000/12,000/12,000 records overall.
```

### 수정 후
```latex
Model selection uses the training and validation partitions described above. Temperature fitting and threshold selection use the separate calibration partition after the operational checkpoint is fixed. Neither operation updates the classifier weights. Calibration diagnostics are measured on the temperature-fitting partition and therefore describe calibration fit, not independent evidence of calibration generalization. Routing performance is evaluated separately on the test partition.
```

## 36. \subsection{Hyperparameter Tuning and Training Optimization}

- 기준본 줄: 457-457; 수정본 줄: 451-451

### 수정 전
```latex
W\&B Bayesian search used validation macro F1 as the objective and varied learning rate, batch size, epoch count, and weight decay without access to calibration or test labels. All models shared the same batch-size, epoch-count, and weight-decay candidates; the fully fine-tuned encoder and frozen 7B linear probes used prespecified learning-rate ranges appropriate to their different trainable parameterizations. Each selected comparison configuration was rerun on the full development data for seeds 42, 43, and 44. The comparison therefore controls the data partitions, search budget, objective, and repetition count while permitting architecture-appropriate selected settings; it does not claim exhaustive optimization of either 7B model. Exact selected settings and the distinct fixed checkpoint used for downstream calibration and routing are reported in Appendix Table~A1c.
```

### 수정 후
```latex
Each W\&B Bayesian sweep comprised four trials maximizing validation macro F1. The shared candidates were batch sizes 16, 32, and 64; training budgets of two or three epochs; and weight decays $10^{-2}$, $10^{-3}$, and $10^{-4}$. Learning-rate ranges differed for encoder fine-tuning and frozen-backbone probing. The selected configuration was rerun with seeds 42, 43, and 44 using the training partition, with validation macro F1 determining the retained epoch. A selected three-epoch budget therefore does not demonstrate convergence beyond that search limit. Appendix Table~A1c reports the sweep-selected settings and, in its note, the fixed operational checkpoint used for downstream analyses.
```

## 37. \subsection{Loss Function and Optimizer}

- 기준본 줄: 461-463; 수정본 줄: 455-455

### 수정 전
```latex
Cross-entropy loss was used for the three-class classifiers, and AdamW was adopted as the optimizer for the trainable parameters. AdamW was selected for its stable convergence with decoupled weight decay. Together with gradient clipping, this optimization setup supported stable training for the fully fine-tuned DistilBERT model and the task-specific classification parameters of the frozen 7B backbones.

The following sections report the saved experimental outputs and separately evaluate predictive performance, calibration, routing, and Phase~2 corrections.
```

### 수정 후
```latex
The classifiers minimize three-class cross-entropy using AdamW and gradient clipping. Updates apply to all DistilBERT parameters or to the linear head alone for the frozen 7B models.
```

## 38. \subsection{Evaluation Metrics}

- 기준본 줄: 469-469; 수정본 줄: 461-461

### 수정 전
```latex
Because Phase~1 and end-to-end predictions are paired at the example level, accuracy changes were also assessed with two-sided exact McNemar tests over discordant pairs. A corrected error is a Phase~1 error changed to a correct end-to-end decision; an introduced error is a correct Phase~1 decision changed to an end-to-end error. Paired percentile-bootstrap 95\% confidence intervals for the accuracy change were estimated with 50,000 row-level resamples and seed 20260813. Holm adjustment was applied across the four prespecified dataset--reasoner comparisons. These analyses quantify uncertainty in the change attributable to Phase~2 without treating routed examples as independent from their Phase~1 counterparts.
```

### 수정 후
```latex
Phase~1 and end-to-end correctness are paired for each example. Two-sided exact McNemar tests use the discordant counts: corrected Phase~1 errors and newly introduced errors. Paired percentile-bootstrap 95\% confidence intervals use 50,000 row-level resamples and seed 20260813. Holm adjustment covers the four dataset--reasoner comparisons. These analyses characterize the observed prediction pairs conditional on the fitted models and selected prompts; they do not account for prompt selection, repeated LLM generation, or dependence between posts from the same author. The three-seed intervals in the classifier comparison instead describe variation across training runs on a common test set.
```

## 39. \subsection{Evaluation Metrics}

- 기준본 줄: 471-471; 수정본 줄: 463-463

### 수정 전
```latex
Residual accepted-set errors were audited separately using the fixed 12,000-example held-out protocol and the calibration-selected threshold. The audit reports accepted selective risk, reference-to-prediction transitions, confidence-band error rates, and Depression false-negative risk among accepted reference-Depression posts. For all 310 accepted errors, the stored model input was linked back to the retained Reddit title and selftext by normalized exact matching against \texttt{title\_with\_selftext\_cleaned}; all 310 rows were matched. Six high-confidence cases were then selected because the original post itself provides comparatively clear evidence for the stored proxy label and exposes a distinct model failure mechanism. These cases illustrate failure modes rather than estimate their prevalence. Appendix Tables~A4e--A4f report the reference label, prediction, calibrated confidence, de-identified original-post excerpt, and text-grounded audit rationale. Qualitative review does not overwrite any reference label or alter any reported metric.
```

### 수정 후
```latex
Residual accepted-set errors were audited separately using the fixed 12,000-example held-out protocol and the calibration-selected threshold. The audit reports accepted selective risk, reference-to-prediction transitions, confidence-band error rates, and Depression false-negative risk among accepted reference-Depression posts. For all 310 accepted errors, the stored model input was linked back to the retained Reddit title and selftext by normalized exact matching against \texttt{title\_with\_selftext\_cleaned}; all 310 rows were matched. Six high-confidence cases were then selected because the original post itself provides comparatively clear evidence for the stored proxy label and exposes a distinct error pattern. These cases illustrate failure modes rather than estimate their prevalence. Appendix Tables~A4e--A4f report the reference label, prediction, calibrated confidence, de-identified original-post excerpt, and text-grounded audit rationale. Qualitative review does not overwrite any reference label or alter any reported metric.
```

## 40. \subsection{Phase 1 Performance on Reddit Data}

- 기준본 줄: 481-481; 수정본 줄: 473-473

### 수정 전
```latex
\caption{Completed matched Phase~1 classifier comparison on the held-out Reddit test set}
```

### 수정 후
```latex
\caption{Matched Phase~1 classifier comparison on the held-out Reddit test set}
```

## 41. \subsection{Phase 1 Performance on Reddit Data}

- 기준본 줄: 508-508; 수정본 줄: 500-500

### 수정 전
```latex
The operational choice is supported by predictive performance and deployment scale rather than by a claim that every architecture was optimized to its ceiling. DistilBERT has approximately 66 million total parameters and produced the highest mean held-out score under the prespecified comparison. Each 7B comparator is roughly two orders of magnitude larger at inference, yet its frozen representation did not overturn the smaller model's predictive advantage. The very small seed-wise standard deviations also indicate that this ordering was stable across the three repetitions. The 7B error analysis was consistent across architectures: Depression--Happy confusions accounted for 75.4\% of Llama~2 errors and 74.2\% of Mistral errors over the pooled three-seed predictions, while Neutral class F1 exceeded 98\% for both models. These findings motivate retaining DistilBERT as the operational classifier while directing the main methodological evaluation toward probability calibration, selective routing, and evidence-preserving re-evaluation.
```

### 수정 후
```latex
DistilBERT had the highest mean score and approximately 66 million parameters, compared with about seven billion for each comparator. These results support retaining it for the tested pipeline, without establishing superiority over fully adapted 7B models. Depression--Happy confusions accounted for 75.4\% of Llama~2 errors and 74.2\% of Mistral errors across pooled seed predictions; Neutral F1 exceeded 98\% for both. Pooling summarizes error composition, not 36,000 independent test examples. Small probe seed deviations also reflect the frozen backbone and should not be interpreted as equivalent training variability across adaptation methods.
```

## 42. \subsection{Phase 1 Performance on Reddit Data}

- 기준본 줄: 510-510; 수정본 줄: 502-502

### 수정 전
```latex
This conclusion is deliberately scoped. Full 7B fine-tuning, LoRA/QLoRA, larger search budgets, and architecture-specific adaptation could change the comparison. The present result establishes only that the tested frozen-backbone linear probes did not provide a sufficient advantage under a matched, computationally bounded protocol. All downstream calibration, routing, and Phase~2 experiments retained the previously fixed operational DistilBERT checkpoint, which achieved 96.69\% held-out accuracy. That value falls within the three-seed DistilBERT accuracy interval of [96.60, 97.20]\%; the checkpoint was not replaced after observing the comparison outcomes.
```

### 수정 후
```latex
The matched comparison and the downstream experiment serve different purposes. The former summarizes three-seed classifier performance; the latter retains the independently fixed operational DistilBERT checkpoint with 96.69\% accuracy. Its predictions, temperature, and routed IDs were not replaced after the comparison. All subsequent results refer to that single operational checkpoint.
```

## 43. \subsection{Phase 1 Performance on Reddit Data}

- 기준본 줄: 512-512; 수정본 줄: 504-504

### 수정 전
```latex
The updated implementation exports the full calibration-based confidence analysis required for selective routing. Table~\ref{tab:calibration-quality} reports the completed DistilBERT results before and after temperature scaling. NLL evaluates the probability assigned to the true class and strongly penalizes confident mistakes; the Brier score evaluates the full three-class probability vector; and ECE summarizes the gap between confidence and observed accuracy across confidence groups. Because every reported diagnostic decreased after scaling, the improvement is not tied to only one view of calibration: NLL fell from 0.1411 to 0.1041, Brier score from 0.0559 to 0.0514, fixed-width ECE from 0.0244 to 0.0083, and adaptive ECE from 0.0244 to 0.0138.
```

### 수정 후
```latex
The operational checkpoint's calibration diagnostics are reported next. Table~\ref{tab:calibration-quality} reports the completed DistilBERT results before and after temperature scaling. NLL evaluates the probability assigned to the true class and strongly penalizes confident mistakes; the Brier score evaluates the full three-class probability vector; and ECE summarizes the gap between confidence and observed accuracy across confidence groups. On the calibration partition, NLL fell from 0.1411 to 0.1041, Brier score from 0.0559 to 0.0514, fixed-width ECE from 0.0244 to 0.0083, and adaptive ECE from 0.0244 to 0.0138.
```

## 44. \subsection{Phase 1 Performance on Reddit Data}

- 기준본 줄: 516-516; 수정본 줄: 508-508

### 수정 전
```latex
\caption{Completed DistilBERT calibration quality before and after temperature scaling}
```

### 수정 후
```latex
\caption{DistilBERT calibration quality before and after temperature scaling}
```

## 45. \subsection{High-Confidence Accepted-Error Audit}

- 기준본 줄: 579-579; 수정본 줄: 571-571

### 수정 전
```latex
\caption{Completed high-confidence accepted-error audit on the Reddit held-out test set}
```

### 수정 후
```latex
\caption{High-confidence accepted-error audit on the Reddit held-out test set}
```

## 46. \subsection{High-Confidence Accepted-Error Audit}

- 기준본 줄: 586-586; 수정본 줄: 578-578

### 수정 전
```latex
Reference proxy label & Accepted $n$ & Accepted errors & Accepted risk & Predicted Depression & Predicted Neutral & Predicted Happy \\
```

### 수정 후
```latex
Reference proxy label & Accepted $n$ & Accepted errors & Accepted risk & To Depression & To Neutral & To Happy \\
```

## 47. \subsection{High-Confidence Accepted-Error Audit}

- 기준본 줄: 596-596; 수정본 줄: 588-588

### 수정 전
```latex
\parbox{\textwidth}{\footnotesize \textit{Note.} ``Accepted risk'' is calculated within each accepted reference class. The Depression row therefore gives the completed accepted Depression false-negative risk: $125/3{,}939=3.17\%$. Reference labels are subreddit-derived proxies, so disagreements may include both model errors and proxy-label/content mismatches.}
```

### 수정 후
```latex
\parbox{\textwidth}{\footnotesize \textit{Note.} ``Accepted risk'' is calculated within each accepted reference class. The last three columns count erroneous predictions only; correct predictions are omitted. The Depression false-negative risk is $125/3{,}939=3.17\%$. Reference labels are subreddit-derived proxies, so disagreements may reflect model errors or proxy-label/content mismatches.}
```

## 48. \subsection{High-Confidence Accepted-Error Audit}

- 기준본 줄: 599-599; 수정본 줄: 591-591

### 수정 전
```latex
Confidence-band analysis further localized the residual risk. Accepted error rates were 36.43\%, 25.95\%, 13.31\%, 4.18\%, and 0.36\% in the 0.70--0.80, 0.80--0.90, 0.90--0.95, 0.95--0.98, and 0.98--1.00 bands, respectively. Thus, confidence remained strongly informative, but no thresholded accepted set was error-free. Original-post review identified trajectory reversal and sarcasm, acute-distress cues overridden by a short positive clause, technical context masking explicit pride or excitement, and topic-term shortcuts triggered by mental-health vocabulary. Appendix Table~A4e summarizes these observed mechanisms, and Appendix Table~A4f supplies an original-post excerpt and the complete decision record for each selected case. The purpose of this audit is residual-risk transparency and failure analysis, not post hoc relabeling or evidence that Phase~2 produced an explanation; these examples were accepted by Phase~1 and therefore were never routed.
```

### 수정 후
```latex
Confidence-band analysis further localized the residual risk. Accepted error rates were 36.43\%, 25.95\%, 13.31\%, 4.18\%, and 0.36\% in the 0.70--0.80, 0.80--0.90, 0.90--0.95, 0.95--0.98, and 0.98--1.00 bands, respectively. Thus, confidence remained strongly informative, but no thresholded accepted set was error-free. Original-post review identified trajectory reversal and sarcasm, acute-distress cues overridden by a short positive clause, technical context masking explicit pride or excitement, and topic-term shortcuts triggered by mental-health vocabulary. Appendix Table~A4e summarizes these candidate error patterns, and Appendix Table~A4f supplies an original-post excerpt and the complete decision record for each selected case. The purpose of this audit is residual-risk transparency and failure analysis, not post hoc relabeling or evidence that Phase~2 produced an explanation; these examples were accepted by Phase~1 and therefore were never routed.
```

## 49. \subsection{Reddit Held-Out Test: End-to-End Results}

- 기준본 줄: 626-626; 수정본 줄: 618-618

### 수정 전
```latex
\parbox{\textwidth}{\footnotesize \textit{Note.} Llama~2 CoT corrected 47 routed Phase~1 errors and introduced 50; Llama~3 SELF-DISCOVER corrected 42 and introduced 12. Both reasoners received the same minimally sanitized original title--body input. All end-to-end metrics are reported on the same 12,000-example held-out test set.}
```

### 수정 후
```latex
\parbox{\textwidth}{\footnotesize \textit{Note.} Llama~2 CoT corrected 47 routed Phase~1 errors and introduced 50; Llama~3 SELF-DISCOVER corrected 42 and introduced 12. Both received the same minimally sanitized original text. Changes are calculated from exact counts before rounding, not by subtracting displayed accuracies. All metrics use the same 12,000 test examples.}
```

## 50. \subsection{Handling Mixed Emotion Inputs: Quantitative Results}

- 기준본 줄: 635-635; 수정본 줄: 627-627

### 수정 전
```latex
The completed Llama~2 CoT end-to-end evaluation increased accuracy by 4.00 percentage points (81.33\% to 85.33\%): it corrected 18 of the 21 routed Phase~1 errors but introduced six routed errors. Llama~3 SELF-DISCOVER increased accuracy by 6.00 percentage points (81.33\% to 87.33\%): it corrected the same 18 routed Phase~1 errors (85.71\%) and introduced no routed errors. On the routed subset itself, accuracy therefore increased from 52.27\% to 79.55\% with Llama~2 and 93.18\% with Llama~3. All reported results use the same 300 inputs, the same fixed Reddit-selected temperature and threshold, and the same 44 routed examples.
```

### 수정 후
```latex
Llama~2 CoT increased end-to-end accuracy by 4.00 percentage points (81.33\% to 85.33\%), correcting 18 of the 21 routed Phase~1 errors while introducing six. Llama~3 SELF-DISCOVER increased accuracy by 6.00 percentage points (81.33\% to 87.33\%), also correcting 18 errors (85.71\%) but introducing none. Routed-only accuracy was 79.55\% and 93.18\%, respectively, versus 52.27\% for Phase~1. Both configurations used the same 300 inputs, fixed Reddit-calibrated policy, and 44 routed cases.
```

## 51. \subsection{Handling Mixed Emotion Inputs: Quantitative Results}

- 기준본 줄: 689-689; 수정본 줄: 681-681

### 수정 전
```latex
The class-wise Llama~3 changes provide an additional check that the aggregate improvement was not produced by only one favorable class. Depression accuracy increased from 66\% to 79\% (13 net corrections), Happy from 95\% to 99\% (four net corrections), and Neutral from 83\% to 84\% (one net correction). No class incurred an introduced error under the observed Llama~3 run. Scenario-level results are reported in Appendix Table~A4c to distinguish performance on clear informational examples from performance on mixed-cue and trajectory-shift cases.
```

### 수정 후
```latex
On Mixed Emotion, class-wise Llama~3 recall increased from 66\% to 79\% for Depression (13 net corrections), 95\% to 99\% for Happy (four), and 83\% to 84\% for Neutral (one). Each class has 100 examples. No class incurred an introduced error in this run. Appendix Table~A4c separates clear informational scenarios from mixed-cue and trajectory-shift scenarios.
```

## 52. \subsection{Observed Correction Cases and Reasoning Traceability}

- 기준본 줄: 704-704; 수정본 줄: 696-696

### 수정 전
```latex
These examples illustrate the intended role of Phase~2: not to reinterpret every positive phrase as Depression, but to distinguish a brief local cue from the dominant trajectory of the complete post. The free-text rationale is retained for human inspection, whereas only the parsed terminal label is used in quantitative evaluation. This separation prevents persuasive prose from being mistaken for evidence of correctness. Aggregate value is therefore established by corrected, introduced, and net errors, while the case-level trace explains how a particular correction was reached.
```

### 수정 후
```latex
The examples show how generated responses relate local cues to the complete post. Only the parsed terminal label enters quantitative evaluation; the rationale remains available for inspection. Corrected and introduced counts establish the observed predictive effect, whereas the trace records a generated justification rather than a verified causal account of the decision.
```

## 53. \subsection{Interpretation of Routing and Re-Evaluation Results}

- 기준본 줄: 710-710; 수정본 줄: 702-702

### 수정 전
```latex
The framework is computationally selective. Under the fixed primary policy, only 1.42\% of held-out Reddit posts and 14.67\% of Mixed Emotion examples require LLM inference. These routing rates quantify the reduction in LLM calls relative to re-evaluating every input; wall-clock time, energy use, and monetary cost were not directly benchmarked. The higher stress-test routing rate is consistent with its deliberate concentration of competing and trajectory-shifting cues, but it is not a population estimate. The Reddit operating point remains the calibration-selected policy and is not re-selected from Phase~2 outcomes.
```

### 수정 후
```latex
Only 1.42\% of Reddit posts and 14.67\% of Mixed Emotion examples underwent re-evaluation. These are fractions of inputs, not counts of individual LLM calls: CoT and SELF-DISCOVER can require multiple generation stages per routed post. Wall-clock time, token use, energy, and monetary cost were not benchmarked. The stress-test routing rate characterizes its constructed scenario mix, not an expected deployment workload.
```

## 54. \subsection{Interpretation of Routing and Re-Evaluation Results}

- 기준본 줄: 716-716; 수정본 줄: 708-708

### 수정 전
```latex
Taken together, the experiments support the Phase~1 and routing components of the framework for the constructed proxy-label task. They also show that Phase~2 value is model- and input-dependent: Llama~3 produced positive paired effects on both the natural Reddit test set and the Mixed Emotion stress test, whereas Llama~2 did not improve Reddit end-to-end accuracy. The synthetic stress test provides focused evidence about emotionally ambiguous inputs, while the original-text Reddit rerun shows that preserving discourse information can materially affect re-evaluation quality. Neither result constitutes broad evidence of real-world mental-health generalization.
```

### 수정 후
```latex
Taken together, the experiments support the Phase~1 and routing components of the framework for the constructed proxy-label task. They also show that Phase~2 value is model- and input-dependent: Llama~3 produced positive paired effects on both the natural Reddit test set and the Mixed Emotion stress test, whereas Llama~2 did not improve Reddit end-to-end accuracy. The synthetic stress test provides focused evidence about emotionally ambiguous inputs, while the Reddit rerun reports performance with the original-text input policy. Neither result constitutes broad evidence of real-world mental-health generalization.
```

## 55. \section{Limitations and Future Work}

- 기준본 줄: 720-720; 수정본 줄: 712-712

### 수정 전
```latex
Despite the promising findings, several limitations should be acknowledged.
```

### 수정 후
```latex
\textit{Label and population validity.} Reddit labels reflect subreddit membership and polarity filtering, not clinical assessments. Filtering may favor sentiment-aligned posts and make the task easier than classification of unfiltered discourse. The balanced, post-level split also does not establish performance on unseen authors, future time periods, or other platforms. The synthetic stress test has controlled labels and scenarios but may contain generator-specific phrasing and repeated narrative patterns. Neither source establishes clinical validity. Independent annotation is needed to assess post-level label consistency and rationale quality before considering any decision-support use.
```

## 56. \section{Limitations and Future Work}

- 기준본 줄: 722-722; 수정본 줄: 714-714

### 수정 전
```latex
First, both evaluation sources use proxy labels rather than clinician-annotated diagnoses or formal assessment outcomes. The Reddit labels were constructed through subreddit-based heuristics and sentiment-aware filtering, while the Mixed Emotion Dataset was generated through controlled LLM prompting. This dual-source design supports large-scale experimentation and targeted stress-testing, but it does not establish clinical validity. In particular, depression-related false negatives could have serious consequences if the system were used without human oversight. The outputs must therefore be interpreted as research signals for a constructed emotion-classification task, not as diagnoses, screening outcomes, or treatment guidance.
```

### 수정 후
```latex
\textit{Residual risk and calibration.} The router left 310 errors among 11,829 accepted predictions, including 34 at confidence 0.98 or above. Confidence therefore supports selection without certifying individual predictions. Temperature and threshold selection share a calibration partition, and the risk bound is a pointwise selection criterion rather than a post-selection guarantee. Results under the fixed Reddit policy do not establish risk control on other distributions. The accepted-error examples identify candidate failure patterns; without attribution tests or independent adjudication, they do not establish causal mechanisms or failure-mode prevalence.
```

## 57. \section{Limitations and Future Work}

- 기준본 줄: 724-724; 수정본 줄: 716-716

### 수정 전
```latex
Second, the confidence-guided routing mechanism still depends on a deterministic threshold derived from model confidence scores. The updated workflow reduces this limitation by using temperature-scaled maximum-softmax confidence, calibration-quality metrics, and risk-coverage threshold selection on a dedicated calibration split. Nevertheless, calibrated confidence is not a complete measure of true uncertainty. The completed audit found 310 errors among 11,829 accepted predictions, including 34 errors at confidence 0.98 or above. Several cases reflected mixed trajectories, possible sarcasm, topic-term shortcuts, or mismatch between subreddit-derived proxy labels and post-level meaning. These results motivate more robust uncertainty estimation and independent review of both model decisions and proxy labels.
```

### 수정 후
```latex
\textit{Comparison scope.} The Phase~1 study compares three practical configurations under a limited search budget, not equivalently adapted or maximally optimized architectures. Three seeds characterize a narrow range of training variability. In Phase~2, model family and prompting method vary together, so the observed advantage of Llama~3 SELF-DISCOVER cannot be attributed to SELF-DISCOVER alone. A crossed model--prompt evaluation would be needed to separate these effects. Similarly, the final original-text rerun supports the reported input policy but does not establish a general causal effect of preserving any particular textual feature.
```

## 58. \section{Limitations and Future Work}

- 기준본 줄: 726-726; 수정본 줄: 718-718

### 수정 전
```latex
Third, the Phase~1 architecture comparison is intentionally bounded. DistilBERT is fully fine-tuned, whereas the two 7B backbones are frozen and only 12,288-parameter linear probes are optimized; LoRA, QLoRA, full-model fine-tuning, larger hyperparameter budgets, and broader model families are outside the present scope. The completed three-seed comparison justifies the operational model choice under this protocol, but it cannot establish the best attainable Mistral or Llama classifier. Phase~2 end-to-end comparisons have been completed on the synthetic routed set and on the 12,000-example Reddit held-out test set. However, the generated rationales have not been reviewed by mental health professionals, and their reliability as decision-support explanations remains unverified. Further ablation studies and expert evaluation are needed to determine which model architectures, adaptation methods, and reasoning strategies contribute most strongly to performance, traceability, and practical reliability.
```

### 수정 후
```latex
\textit{Prompt development and statistical scope.} Prompt variants were examined on the routed Reddit cases under the earlier cleaned-input condition. Although prompts were frozen before the original-text rerun, the same IDs and labels were reused. The end-to-end comparisons are therefore exploratory, not independent confirmatory tests. Bootstrap intervals and Holm-adjusted McNemar tests quantify the final paired outcomes but do not remove this development dependence. The next validation should freeze prompts and input handling on a separate development subset, then evaluate once on independent data. This issue is distinct from the calibration-partition fitting of $T^*$ and $\tau^*$.
```

## 59. \section{Limitations and Future Work}

- 기준본 줄: 728-738; 수정본 줄: 720-720

### 수정 전
```latex
Fourth, the model-specific prompt variants were iteratively examined on the routed Reddit evaluation subset under the earlier cleaned-input condition. The protocols were then frozen before the final original-text rerun, but the same routed IDs and proxy labels were reused. Consequently, the prompt-development diagnostics and final Reddit comparison should still be treated as exploratory rather than as a fully independent confirmatory test. A future study should freeze the prompt and input policy on a separate development subset and evaluate them once on an independent routed test set. This limitation does not affect the calibration-only selection of $T^*$ and $\tau^*$, which was completed before held-out inference.

Future research should further strengthen the clinical relevance, methodological robustness, and practical reliability of the proposed framework.

First, future studies should incorporate expert-labeled datasets annotated by licensed clinicians or trained mental health professionals. Aligning labels with formal diagnostic standards or validated screening instruments would enable a more clinically meaningful evaluation of depression-risk-related emotion classification performance. In addition, the reasoning outputs generated in Phase 2 should be assessed by domain experts to determine whether they are accurate, interpretable, and appropriate for use in human-in-the-loop decision-support settings.

Second, future work should further improve the framework's uncertainty estimation and routing mechanism. Although the updated workflow incorporates temperature scaling, calibration metrics, risk-coverage threshold selection, and a completed accepted-set audit, more reliable uncertainty quantification may better distinguish genuinely ambiguous cases from confidently misclassified ones. Independent annotation is also needed to separate model failures from subreddit-derived proxy-label noise. The audit is a residual-risk diagnostic; it does not alter the calibration-selected threshold or the reported end-to-end performance values.

Third, future research should examine a broader range of model architectures, fine-tuning strategies, and reasoning mechanisms. The current framework demonstrates the potential of selective reasoning using representative transformer-based classifiers and LLM prompting methods, but additional ablation studies are needed to identify which components contribute most strongly to predictive performance, interpretability, and computational efficiency. Such analysis would also help determine whether the framework generalizes across different platforms, linguistic styles, and population groups.

Finally, future work should move beyond single-post text classification toward longitudinal and multimodal analysis. Depression-related signals often emerge over time and may be reflected not only in language, but also in behavioral patterns, voice, sleep, activity, or other contextual indicators. Incorporating such signals, while maintaining strict privacy safeguards and human oversight, may support a more comprehensive understanding of depression-related risk in future research-oriented human-in-the-loop systems.
```

### 수정 후
```latex
\textit{Computational and explanatory scope.} Routing rates show how many posts avoid re-evaluation, but no controlled GPU-time, token, or cost comparison was performed. Generated rationales remain unvalidated explanations, and multistage inference may have different overhead across configurations. Controlled resource measurements and independent review of both labels and rationales are the most direct extensions of the present evaluation.
```

## 60. \section{Conclusion}

- 기준본 줄: 742-742; 수정본 줄: 724-724

### 수정 전
```latex
This study presented a confidence-guided two-phase framework that combines three-class proxy emotion classification with selective LLM re-evaluation. In the completed matched three-seed Phase~1 comparison, fully fine-tuned DistilBERT achieved 96.90\% mean held-out accuracy and macro F1, exceeding the bounded frozen-backbone Mistral and Llama~2 probes while remaining far smaller at inference. Temperature scaling and calibration-only risk--coverage selection then routed 1.42\% of the Reddit held-out set and 14.67\% of the Mixed Emotion stress test, concentrating substantially lower-accuracy subsets without exposing every input to an LLM.
```

### 수정 후
```latex
This study evaluated selective LLM re-evaluation by separating classifier performance, calibration, error concentration, and net correction. The three-seed comparison supported retaining DistilBERT within the tested configurations. Its fixed operational checkpoint routed 1.42\% of Reddit posts, capturing 87 of 397 Phase~1 errors. Llama~3 SELF-DISCOVER corrected 42 errors and introduced 12, raising full-set accuracy from 96.69\% to 96.94\%; Llama~2 CoT produced no detectable improvement. On the synthetic stress test, the corresponding gains were 6.00 and 4.00 percentage points.
```

## 61. \section{Conclusion}

- 기준본 줄: 744-746; 수정본 줄: 726-726

### 수정 전
```latex
The end-to-end evidence is deliberately conditional. On Reddit, original-text Llama~2 changed accuracy by $-0.03$ percentage points and was statistically indistinguishable from Phase~1, whereas Llama~3 increased accuracy by 0.25 percentage points through 30 net corrections; its paired interval was positive and its effect remained significant after Holm correction. On the synthetic Mixed Emotion stress test, Llama~2 and Llama~3 improved accuracy by 4.00 and 6.00 percentage points, respectively, and both effects remained significant after Holm correction. Thus, confidence-guided routing is effective at concentrating difficult examples, whereas successful correction depends on the reasoning policy and preservation of the source-text evidence.

The framework is a research-oriented analysis pipeline, not a clinical diagnostic system. Its principal methodological contribution is the reproducible separation of classifier performance, calibration quality, routing effectiveness, and re-evaluation benefit, together with an inspectable audit trail for routed cases. The evidence shows that calibration identifies where correction is most needed, but does not guarantee that a reasoner will exploit that opportunity: successful re-evaluation requires a capable model, preservation of the source-text evidence, and control of newly introduced errors. Within this fixed protocol, evidence-preserving Llama~3 re-evaluation produced a statistically reliable paired improvement. Validation on independent, clinician-annotated data and a prospectively frozen prompting protocol remains necessary before considering real-world decision-support use.
```

### 수정 후
```latex
The central finding is that selecting an error-prone subset and correcting it are separate requirements. High routing concentration alone did not ensure a positive end-to-end effect, and accepted errors remained outside the correction path. The reported gains describe fixed prediction pairs under an exploratory prompt-development protocol. Independent evaluation with frozen prompts and expert-reviewed labels is needed to establish generalization. The framework remains a proxy emotion-classification pipeline, not a clinical diagnostic system.
```

## 62. \section{Supplementary Methodological Details}

- 기준본 줄: 903-903; 수정본 줄: 883-883

### 수정 전
```latex
\Needspace{18\baselineskip}
```

### 수정 후
```latex
\Needspace{11\baselineskip}
```

## 63. \section{Chain-of-Thought Prompting Protocol}

- 기준본 줄: 953-953; 수정본 줄: 933-933

### 수정 전
```latex
Note. The role description is reported in a publication-safe form. The prompt was used for text-based emotion classification and low-confidence prediction re-evaluation. It should not be interpreted as clinical assessment, diagnostic authority, medical condition inference, treatment recommendation, or professional mental health advice.
```

### 수정 후
```latex
\noindent\footnotesize\textit{Note.} This table summarizes the protocol and uses a non-clinical rendering of the role description; it is not a verbatim prompt transcript. The executable prompt strings are retained in the released notebooks. The model's assigned role does not confer diagnostic or treatment authority.\normalsize
```

## 64. \section{SELF-DISCOVER Prompting Protocol}

- 기준본 줄: 985-985; 수정본 줄: 965-965

### 수정 전
```latex
Note. The role description is reported in a publication-safe form. The prompt was used for research-oriented emotional re-evaluation of text, not for clinical diagnosis, treatment recommendation, or professional mental health advice.
```

### 수정 후
```latex
\noindent\footnotesize\textit{Note.} This table summarizes the protocol with non-clinical role wording; verbatim executable prompts are retained in the released notebooks.\normalsize
```

## 65. \section{Observed Routing and Reasoning Audit Trail}

- 기준본 줄: 1099-1099; 수정본 줄: 1079-1079

### 수정 전
```latex
Note. Tables~A4a and A4b are publication-readable renderings of actual stored routed-sample outputs. Output excerpts are shortened only for layout; the full text fields remain in the Llama~2 and Llama~3 result CSVs. The threshold shown is the current DistilBERT operating value selected from the prespecified calibrated-MSP grid (0.70--1.00, step 0.01). The observed Mixed Emotion results are not clinical diagnostic reports.
```

### 수정 후
```latex
Note. Tables~A4a and A4b are publication-readable renderings of actual stored routed-sample outputs. Output excerpts are shortened only for layout; the full text fields remain in the Llama~2 and Llama~3 result CSVs. The observed Mixed Emotion results are not clinical diagnostic reports.
```

## 66. \section{Observed Routing and Reasoning Audit Trail}

- 기준본 줄: 1157-1157; 수정본 줄: 1137-1137

### 수정 전
```latex
RED\_041439 & Depression $\rightarrow$ Happy & 0.983 & Acute-distress cue missed & The title explicitly states renewed suicidal ideation; a brief report that the preceding weeks were ``okay'' appears to dominate the prediction. \\
```

### 수정 후
```latex
RED\_041439 & Depression $\rightarrow$ Happy & 0.983 & Acute-distress cue missed & The title explicitly states renewed suicidal ideation; the preceding ``okay'' period is a possible conflicting cue, not a verified cause of the prediction. \\
```

## 67. \section{Observed Routing and Reasoning Audit Trail}

- 기준본 줄: 1159-1159; 수정본 줄: 1139-1139

### 수정 전
```latex
RED\_026909 & Happy $\rightarrow$ Neutral & 0.995 & Technical-context dominance & A programming narrative ends with explicit unexpected achievement and pride, but the technical topic dominates the prediction. \\
```

### 수정 후
```latex
RED\_026909 & Happy $\rightarrow$ Neutral & 0.995 & Technical-context dominance & A programming narrative ends with explicit unexpected achievement and pride, yet the prediction is Neutral; the role of the technical vocabulary remains a hypothesis. \\
```

## 68. \section{Observed Routing and Reasoning Audit Trail}

- 기준본 줄: 1218-1218; 수정본 줄: 1198-1198

### 수정 전
```latex
Text-grounded audit & The pricing and hardware terminology are factual, but the opening and stated evaluation are explicitly enthusiastic. The prediction appears dominated by topic rather than affect. \\
```

### 수정 후
```latex
Text-grounded audit & The pricing and hardware terminology are factual, but the opening and stated evaluation are explicitly enthusiastic. The error is consistent with topic sensitivity, but the output alone does not establish the classifier's cue attribution. \\
```

## 69. \section{Observed Routing and Reasoning Audit Trail}

- 기준본 줄: 1230-1230; 수정본 줄: 1210-1210

### 수정 전
```latex
Text-grounded audit & The post discusses depression-related concepts but does not express the author's emotional state. Its function is an informational question, indicating a shortcut based on topic vocabulary. \\
```

### 수정 후
```latex
Text-grounded audit & The post discusses depression-related concepts but does not express the author's emotional state. Its function is an informational question, consistent with a possible topic-vocabulary shortcut, rather than demonstrating that mechanism. \\
```

## 70. \section{Observed Routing and Reasoning Audit Trail}

- 기준본 줄: 1247-1247; 수정본 줄: 1227-1227

### 수정 전
```latex
\noindent\footnotesize\textit{Audit scope.} These six cases were selected from the 310 normalized-exact original-post matches to demonstrate clear, distinct failure mechanisms and must not be used to estimate their prevalence. Quotations retain source wording except for URL removal and bracketed shortening. The complete prediction-level accepted-error file is included with the reproducibility package.\normalsize
```

### 수정 후
```latex
\noindent\footnotesize\textit{Audit scope.} These six cases illustrate error patterns, not verified causal mechanisms or their prevalence. Quotations retain source wording except for URL removal and bracketed shortening. The complete prediction-level accepted-error file is included with the reproducibility package.\normalsize
```

## 71. \section{Observed Routing and Reasoning Audit Trail}

- 기준본 줄: 1264-1264; 수정본 줄: 1244-1244

### 수정 전
```latex
\scriptsize
```

### 수정 후
```latex
\footnotesize
```

## 72. \section{Synthetic Mixed Emotion Dataset Generation Protocol}

- 기준본 줄: 1294-1294; 수정본 줄: 1274-1274

### 수정 전
```latex
Purpose & Construct 300 controlled examples containing emotionally mixed or sentiment-shifting cues for supplementary stress-test evaluation. \\
```

### 수정 후
```latex
Purpose & Construct 300 controlled examples spanning mixed or shifting affect and Neutral controls for supplementary stress-test evaluation. \\
```

## 73. \section{Synthetic Mixed Emotion Dataset Generation Protocol}

- 기준본 줄: 1300-1301; 수정본 줄: 1280-1281

### 수정 전
```latex
Exclusion criteria & Exclude examples with explicit diagnosis claims, treatment advice, medication references, crisis language, identifying information, off-topic content, or insufficient ambiguity. \\
Prompt disclosure & Report the generation prompt, prompt version, generation model, generation date, and quality-control criteria. \\
```

### 수정 후
```latex
Exclusion criteria & Exclude explicit diagnosis claims, treatment advice, medication references, crisis language, identifying information, and off-scenario content. Mixed-cue requirements do not apply to the clear Neutral controls. \\
Prompt disclosure & Generation instructions are reproduced below; the version, model, date, and inclusion criteria are recorded here. \\
```
