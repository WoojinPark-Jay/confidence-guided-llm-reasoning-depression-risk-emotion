# 검증 완료 원고 변경 내역

```diff
--- 20260929/main.tex
+++ 20260930/main.tex
@@ -32,7 +32,7 @@
 \history{}
 \doi{}

-\title{Confidence-Guided Selective LLM Re-Evaluation for Depression-Risk-Related Emotion Classification in Social Media Text}
+\title{Confidence-Guided Selective LLM Re-Evaluation for Emotion Classification in Social Media}
 % REQUIRED BEFORE SUBMISSION: replace the review placeholder with the final
 % author list, numbered affiliations, corresponding author, and funding note.
 \author{\uppercase{Author Details Omitted for Professor Review}}
@@ -44,7 +44,8 @@
 \corresp{Corresponding-author details will be inserted in the submission version.}

 \begin{abstract}
-Confidence-based routing can concentrate classifier errors, but improvement also depends on how the selected inputs are re-evaluated. We study a two-phase pipeline for three-class proxy emotion classification of social-media posts. A sweep-selected DistilBERT configuration supplies both the final classifier and its calibrated routing outputs; high-confidence predictions are retained, and low-confidence posts are re-evaluated using their original text. We compare Direct, Chain-of-Thought (CoT), and an adapted task-level SELF-DISCOVER protocol with Llama~2 and Llama~3. On 12,000 Reddit posts, Phase~1 achieves 96.9167\% accuracy and routes 218 posts (1.82\%). Llama~3 SELF-DISCOVER achieves 97.2000\%, correcting 72 errors while introducing 38. On a balanced 300-example synthetic Mixed Emotion stress test, the same routing policy selects 86 posts; accuracy increases from 83.6667\% to 92.6667\%, with 28 corrected errors and one introduced error. Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 among the six final configurations on both sets, although method rankings differ for Llama~2. The results distinguish error selection from correction and support structured, evidence-oriented re-evaluation within the evaluated settings. Prompt-configuration comparisons are exploratory, and the synthetic stress test does not establish clinical or external-domain validity.
+Confidence-guided routing enables selective LLM re-evaluation of difficult inputs while retaining the predictions of an efficient first-stage classifier for most posts. We develop and evaluate a two-phase pipeline for three-class, non-clinical proxy emotion classification in social media. A validation-selected DistilBERT model supplies calibrated predictions, and low-confidence posts undergo original-text re-evaluation. We compare Direct, Chain-of-Thought (CoT), and task-adapted SELF-DISCOVER with Llama~2 and Llama~3. On 12,000 Reddit posts, Phase~1 achieves 96.9167\% accuracy and routes only 218 posts (1.82\%); Llama~3 SELF-DISCOVER raises accuracy to 97.2000\%, correcting 72 errors and introducing 38. On 300 synthetic Mixed Emotion examples, it increases accuracy from 83.6667\% to 92.6667\%, with 28 corrections and one introduced error. A separate 9,000-post same-source evaluation fixes the final classifier, prompts, and cached plan and routes 111 posts (1.23\%). SELF-DISCOVER improves accuracy from 97.5778\% to 97.7667\%, yielding 17 net corrections. Its gains over the Phase~1 classifier are statistically significant on all three evaluation sets after the stated Holm adjustments; the additional evaluation gives $p=0.0412$. Llama~3 SELF-DISCOVER also achieves the highest observed accuracy among the evaluated configurations on each set. These findings support confidence-guided selective re-evaluation as a means of improving classifier performance while limiting the number of posts sent to an LLM, and show the importance of model and prompting-protocol choice for balancing corrections against introduced errors.
+
 \end{abstract}
 \begin{keywords}
 Depression-risk-related emotion classification, social media text analysis, large language models, confidence-guided routing, Chain-of-Thought prompting, SELF-DISCOVER, mental health NLP, reasoning traceability
@@ -56,30 +57,29 @@

 \section{Introduction}

-The World Health Organization estimates that approximately 332 million people live with depression \cite{ref1} and identifies it as a leading cause of ill health and disability \cite{ref2}. Depression and anxiety together account for an estimated US\$1 trillion in annual productivity losses \cite{ref3}; their prevalence also increased during the first year of the COVID-19 pandemic \cite{ref4}. This burden motivates research on emotional signals in everyday language, provided that computational text labels remain distinct from clinical assessment.
-
-Validated questionnaires such as the Patient Health Questionnaire-9 (PHQ-9) support depression screening \cite{ref5}, while accessible, person-centred care remains a health-system priority \cite{ref6}. Social-media analysis serves a different purpose: it allows researchers to study linguistic and emotional patterns in non-clinical text \cite{ref7}. A post-level emotion prediction neither replaces those instruments nor establishes the author's clinical condition.
-
-Domain-adapted models such as MentalBERT demonstrate the value of target-domain pretraining for mental health text classification \cite{ref8}. Nevertheless, emotion-fusion research identifies dataset quality and interpretability as persistent challenges \cite{ref9}. For user-level prediction, processing posts independently can also discard temporal and inter-post information \cite{ref10}. Our post-level task does not model a user's longitudinal history. Research on clinical explainability further emphasizes that useful explanations depend on the intended setting and the information clinicians need \cite{ref11}; a generated rationale alone does not meet those requirements.
+The World Health Organization estimates that approximately 332 million people live with depression \cite{ref1} and identifies it as a leading cause of ill health and disability \cite{ref2}. Depression and anxiety together account for an estimated US\$1 trillion in annual productivity losses \cite{ref3}; their prevalence also increased during the first year of the COVID-19 pandemic \cite{ref4}. This burden motivates research on emotional signals in everyday language, including computational analysis of non-clinical social-media text.
+
+Validated questionnaires such as the Patient Health Questionnaire-9 (PHQ-9) support depression screening \cite{ref5}, while accessible, person-centred care remains a health-system priority \cite{ref6}. Social-media analysis offers a complementary research setting for studying linguistic and emotional patterns \cite{ref7}. We focus on post-level proxy emotion labels rather than clinical diagnosis.
+
+Domain-adapted models such as MentalBERT demonstrate the value of target-domain pretraining for mental health text classification \cite{ref8}. Nevertheless, emotion-fusion research identifies dataset quality and interpretability as persistent challenges \cite{ref9}. User-level prediction can additionally benefit from temporal and inter-post information \cite{ref10}. Research on clinical explainability emphasizes that explanations should match the intended setting and information needs \cite{ref11}. In our post-level setting, textual rationales accompany label decisions, while predictive value is assessed through observed corrections and introduced errors.

 Existing research establishes the value of transformer classification, confidence calibration, selective prediction, and LLM-generated explanations, but these components answer different questions and are not interchangeable. A calibrated classifier may identify difficult inputs without correcting them, while an LLM may generate a plausible rationale without improving the underlying decision. This leaves a practical evaluation gap: confidence-based case selection and reasoning-based correction should be connected in one fixed pipeline and assessed separately at the classifier, router, and re-evaluator levels.

 To address this gap, we propose a confidence-guided two-phase framework for depression-risk-related proxy emotion classification that combines transformer-based classification with selective, rationale-supported LLM re-evaluation. The completed Phase~1 implementation uses DistilBERT to produce initial predictions and calibrated maximum-softmax-probability confidence scores. A dedicated calibration split is held out from model selection and used for temperature scaling, calibration assessment, and routing-threshold selection. Samples whose calibrated confidence scores fall below the selected threshold are routed to Phase~2. The threshold maximizes Phase~1 coverage subject to a one-sided upper bound on accepted-set selective risk. In Phase~2, low-confidence samples are re-analyzed through model-specific structured reasoning processes \cite{ref17}, \cite{ref18}. The resulting end-to-end evaluation reports not only final accuracy, but also whether routing concentrates errors and whether each re-evaluator corrects more routed errors than it introduces.

-The evaluation addresses three research questions. \textbf{RQ1a} assesses the operational classifier's predictive and calibration performance; \textbf{RQ1b} compares the candidate classifiers under a shared data and search budget. \textbf{RQ2} asks whether the fixed routing policy concentrates Phase~1 errors while retaining high coverage. \textbf{RQ3} asks whether re-evaluation produces positive net corrections on Reddit posts and on a controlled synthetic stress test. Separating these questions allows a useful router and an ineffective re-evaluator to be identified within the same experiment.
+The evaluation addresses three research questions, with a separate same-source evaluation of the fixed final Llama~3 protocols. \textbf{RQ1a} assesses the operational classifier's predictive and calibration performance; \textbf{RQ1b} compares the candidate classifiers under a shared data and search budget. \textbf{RQ2} asks whether the fixed routing policy concentrates Phase~1 errors while retaining high coverage. \textbf{RQ3} asks whether re-evaluation produces positive net corrections on Reddit posts and on a controlled synthetic stress test. Separating these questions makes the contribution of case selection and subsequent correction measurable within the same pipeline.

 \section{Study Contributions}

 The study makes four concrete contributions:
 \begin{itemize}
-\item It operationalizes selective LLM re-evaluation as an end-to-end pipeline in which an efficient classifier handles high-confidence inputs and reasoning models receive only calibration-defined low-confidence inputs. The Phase~1 comparison supports operational model selection through predictive performance and measured inference throughput, latency, and memory on a common accelerator, rather than claiming exhaustive optimization of every 7B architecture.
+\item It operationalizes selective LLM re-evaluation as an end-to-end pipeline in which an efficient classifier handles high-confidence inputs and reasoning models receive only calibration-defined low-confidence inputs. The Phase~1 comparison grounds model selection in predictive performance and measured inference throughput, latency, and memory on a common accelerator under a defined search budget.
 \item It replaces a heuristic confidence cutoff with a reproducible calibration protocol: temperature scaling, a prespecified threshold grid, a one-sided selective-risk upper bound, and maximum-coverage selection among feasible thresholds. The selected policy is fixed before held-out and Phase~2 evaluation.
 \item It evaluates routing quality separately from re-evaluation quality. Coverage, selective risk, error capture, corrected errors, introduced errors, and net corrections reveal whether the router finds difficult inputs and whether the LLM actually improves them. The final comparison crosses two language models with three re-evaluation protocols on the same routed examples within each dataset.
-\item It develops a task-specific SELF-DISCOVER adaptation \cite{ref18} combining an emotion-oriented module bank, a reusable model-generated procedure, and fixed checks for emotional subject, temporal context, competing label evidence, and alternative interpretations. Its empirical evaluation includes a controlled Mixed Emotion stress test and recorded outputs for inspecting ambiguous or shifting affect. The stress test is explicitly separated from model training and threshold selection.
+\item It develops a task-specific SELF-DISCOVER adaptation \cite{ref18} combining an emotion-oriented module bank, a reusable model-generated plan, and fixed checks for emotional subject, temporal context, competing evidence, and alternative interpretations. The adaptation is evaluated on the primary Reddit test set, a controlled Mixed Emotion stress test excluded from training and threshold selection, and a separate 9,000-post evaluation with the final Llama~3 protocols fixed. Example-aligned predictions quantify corrected and introduced errors.
 \end{itemize}

-The contribution is the integration and evaluation protocol, together with a task-specific adaptation of structured re-evaluation, not a new calibration estimator or language-model architecture.
-Classifier performance and resource measurements support the first-stage choice; calibration and error concentration assess routing; and the model--protocol comparison measures correction gains against newly introduced errors. The task-level SELF-DISCOVER adaptation is evaluated within this comparison, not assumed superior to label-only prediction.
+Together, these contributions connect efficient classifier selection, calibrated routing, and task-adapted LLM re-evaluation in a unified, auditable pipeline. Llama~3 SELF-DISCOVER yields positive net corrections and statistically significant accuracy gains over Phase~1 on all three evaluation sets under the stated Holm adjustments. Within-model comparisons further characterize how prompting protocols affect correction yield across datasets; detailed paired results are reported in the results section and Appendix~\ref{app:statistics}.


 \section{Literature Review}
@@ -92,7 +92,7 @@

 \subsection{Deep Learning for Contextualized Depression Detection}

-Neural approaches learn text representations rather than relying exclusively on handcrafted features. Word embeddings provide dense representations that encode semantic relationships \cite{ref25}, \cite{ref26}. LSTMs were designed to learn dependencies over extended sequences \cite{ref27}, while CNNs operating on word vectors were evaluated for sentence classification \cite{ref28}. These are methodological foundations, not themselves evidence of depression-detection performance.
+Neural approaches learn text representations rather than relying exclusively on handcrafted features. Word embeddings provide dense representations that encode semantic relationships \cite{ref25}, \cite{ref26}. LSTMs were designed to learn dependencies over extended sequences \cite{ref27}, while CNNs operating on word vectors were evaluated for sentence classification \cite{ref28}.

 Orabi et al. (2018) \cite{ref30} compared several deep learning models for user-level depression detection on Twitter. Using a TF-IDF linear SVM as a baseline, they showed that CNN-based models, particularly those using their optimized embeddings, outperformed the baseline classifier. In their experiments, CNN models also achieved better performance than the BiLSTM model. The authors further noted that supervised deep neural approaches have seen limited adoption in this domain due to the difficulty of obtaining sufficiently annotated training data \cite{ref30}.

@@ -106,13 +106,13 @@

 Shin et al. \cite{ref35} evaluated GPT-3.5 and GPT-4 on semistructured emotional diaries, using questionnaire-based assessments as reference measures. Their study supports investigating diary text for depression-related screening, but its participants, reference measures, and input format differ from subreddit-derived post labels.

-Wang et al. \cite{ref43} used Reddit writings to predict questionnaire-related depression levels and generate explanations for individual questionnaire items. Lan et al.'s DORIS framework \cite{ref44} instead uses LLM-derived symptom annotations and temporal mood-course features with a gradient-boosting-tree classifier, alongside generated justifications. These are distinct ways of combining prediction and explanation, not evidence that a generated rationale necessarily corrects an initial label. Omar and Levkovich's review \cite{ref45} includes both encoder models, such as BERT and RoBERTa, and generative approaches; it identifies predictive potential while emphasizing further validation, privacy, and ethical safeguards. Its findings should not be read as established therapeutic efficacy of generative chatbots.
+Wang et al. \cite{ref43} used Reddit writings to predict questionnaire-related depression levels and generate explanations for individual questionnaire items. Lan et al.'s DORIS framework \cite{ref44} combines LLM-derived symptom annotations and temporal mood-course features with a gradient-boosting-tree classifier and generated justifications. These studies illustrate distinct ways of integrating prediction and explanation. Omar and Levkovich's review \cite{ref45} identifies predictive potential across encoder and generative models while emphasizing validation, privacy, and ethical safeguards.

 These approaches retain the construct-validity, sampling, and platform-dependence concerns documented in reviews \cite{ref36}, \cite{ref37}, \cite{ref46}. For the present task, a generated explanation is therefore an inspectable output, not evidence that the proxy label or the prediction is clinically valid.

 \subsection{Confidence Calibration, Selective Prediction, and Cascaded Re-Evaluation}

-Calibration addresses the correspondence between predicted confidence and observed correctness \cite{ref15}. Selective classification addresses a related but distinct problem: trading coverage for accepted-set risk by abstaining on selected inputs \cite{ref16}. Abstention does not itself calibrate probabilities or correct the rejected predictions. This distinction motivates separate evaluation of confidence quality, case selection, and downstream correction.
+Calibration addresses the correspondence between predicted confidence and observed correctness \cite{ref15}. Selective classification addresses a related but distinct problem: trading coverage for accepted-set risk by withholding an immediate prediction on selected inputs, a strategy termed abstention \cite{ref16}. Abstention does not itself calibrate probabilities or correct the rejected predictions. This distinction motivates separate evaluation of confidence quality, case selection, and downstream correction.

 Here, rejected inputs are sent to an LLM rather than left unclassified. Temperature scaling \cite{ref15} and selective prediction \cite{ref16} motivate the routing design; CoT \cite{ref17} and SELF-DISCOVER \cite{ref18} motivate the adapted re-evaluation protocols described below. The empirical question is whether recombining their outputs improves the original classifier after newly introduced errors are counted.

@@ -138,7 +138,7 @@

 \subsection{Data Features}

-For each Reddit post, both textual content and auxiliary metadata were retained during dataset construction. The Phase~1 modeling input was the cleaned title--body text (\path{Title_with_selftext_cleaned}), which was generated by concatenating the post title and body text. The retained original title and selftext were used only when a routed post entered Phase~2. Additional fields, including subreddit source, author-related metadata, adult-content flag, timestamp, polarity score, and class label, were used only to support preprocessing, sampling, linkage, and dataset organization. A complete description of the retained variables is provided in Appendix Table A1.
+For each Reddit post, the text fields and metadata needed for preprocessing, linkage, and evaluation were retained during dataset construction. The Phase~1 modeling input was the cleaned title--body text (\path{Title_with_selftext_cleaned}), which was generated by concatenating the post title and body text. The retained original title and selftext were used only when a routed post entered Phase~2. Additional fields, including subreddit source, author-related metadata, adult-content flag, timestamp, polarity score, and class label, were used only to support preprocessing, sampling, linkage, and dataset organization. A complete description of the retained variables is provided in Appendix Table A1.

 Class labels were numerically encoded for model training as Depression = 0, Neutral = 1, and Happy = 2. This encoding was used only for computational convenience and does not imply an ordinal relationship among the classes.

@@ -146,7 +146,7 @@

 A standardized preprocessing pipeline was applied to reduce noise and improve comparability across posts. The concatenated title and selftext fields were first converted to lowercase and tokenized. Non-alphabetic content, URLs, numbers, special characters, and common stop words were removed. Lemmatization was then applied to reduce inflected word forms to their base forms. For example, ``Sad'' was converted to ``sad,'' and ``running'' was lemmatized to ``run.''

-In addition, a Reddit-specific stop-word list filtered platform-specific filler expressions. The cleaned field, \path{Title_with_selftext_cleaned}, was used for Phase~1 training, calibration, and prediction. Cleaning preserves the order of retained tokens but can remove negation, punctuation, and discourse markers. Phase~2 therefore used the minimally sanitized original text to retain these cues and sentence boundaries.
+In addition, a Reddit-specific stop-word list filtered platform-specific filler expressions. The cleaned field, \path{Title_with_selftext_cleaned}, was used for Phase~1 training, calibration, and prediction. Cleaning preserves the order of retained tokens but can remove negation, punctuation, and discourse markers. For routed Reddit posts, Phase~2 used the linked original title and selftext, not the Phase~1-cleaned input. It decoded HTML, replaced URLs and direct usernames, and normalized whitespace while preserving wording, capitalization, punctuation, negation, sentence order, and sentence boundaries.

 \subsection{Sentiment-Aware Data Sampling}

@@ -163,19 +163,24 @@

 \subsection{Mixed Emotion Dataset}

-In addition to the primary Reddit dataset, we constructed a supplementary synthetic Mixed Emotion Dataset to examine model behavior under emotionally ambiguous conditions. The final dataset contains 300 controlled examples, balanced across the three proxy emotion labels, with 100 Depression, 100 Neutral, and 100 Happy examples. This dataset was not used for Phase~1 model training, hyperparameter tuning, temperature scaling, or routing-threshold selection. Instead, it was designed as a targeted stress-test set for cases in which positive, neutral, and depression-related cues co-occur or shift across the text. In the evaluation workflow, the temperature and routing threshold selected on the Reddit calibration split are fixed and then applied directly to the Mixed Emotion Dataset; thresholds are not re-selected on the synthetic stress-test data.
-
-The current dataset includes seven controlled scenario groups: blended-emotion co-occurrence (40), positive-to-distress shift (40), distress-to-recovery shift (40), neutral framing with subtle affect (40), conflicting cues with a dominant trajectory (40), clear technical or informational Neutral examples (75), and Neutral examples with mild factual or procedural ambiguity (25). Each example was assigned a target label according to the dominant overall emotional trajectory rather than isolated sentiment-bearing phrases. The two Neutral groups preserve the proxy-label definition while avoiding a stress set that is artificially dominated by either completely trivial or clinically suggestive Neutral content. This design is intended to test whether confidence-guided routing and reasoning-based re-evaluation can address cases that are plausibly more difficult for a single-stage classifier.
-
-Synthetic examples were generated using controlled prompting and checked against inclusion and exclusion criteria. Kang et al. \cite{ref19} used LLM-generated clinical-interview summaries for depression-prediction training augmentation. Here, synthetic posts serve a different purpose: supplementary stress-test evaluation, not training augmentation or clinical validation. Inclusion required consistency with the assigned scenario and target label: mixed or shifting cues for the affective scenarios, and low-affect content for the Neutral controls. Exclusions covered explicit clinical diagnosis claims, treatment recommendations, crisis language, identifying information, and off-scenario content. Appendix Table~F1 records the generation protocol. These checks were not an independent expert-annotation study; results describe the final 300 synthetic examples rather than naturally occurring mixed-emotion performance.
+In addition to the primary Reddit dataset, we constructed a controlled synthetic Mixed Emotion stress-test dataset to examine model behavior under emotionally ambiguous conditions. The final dataset contains 300 controlled examples, balanced across the three proxy emotion labels, with 100 Depression, 100 Neutral, and 100 Happy examples. This dataset was not used for Phase~1 model training, hyperparameter tuning, temperature scaling, or routing-threshold selection. Instead, it was designed as a targeted stress-test set for cases in which positive, neutral, and depression-related cues co-occur or shift across the text. In the evaluation workflow, the temperature and routing threshold selected on the Reddit calibration split are fixed and then applied directly to the Mixed Emotion Dataset; thresholds are not re-selected on the synthetic stress-test data.
+
+The current dataset includes seven controlled scenario groups: blended-emotion co-occurrence (40), positive-to-distress shift (40), distress-to-recovery shift (40), neutral framing with subtle affect (40), conflicting cues with a dominant trajectory (40), clear technical or informational Neutral examples (75), and Neutral examples with mild factual or procedural ambiguity (25). Each example was assigned a target label according to the dominant overall emotional trajectory rather than isolated sentiment-bearing phrases. The two Neutral groups preserve the proxy-label definition while avoiding a stress set that is artificially dominated by either completely trivial or clinically suggestive Neutral content. This design tests whether confidence-guided routing and reasoning-based re-evaluation can address cases containing competing or shifting emotional evidence.
+
+Synthetic examples were generated using controlled prompting and checked against inclusion and exclusion criteria. Kang et al. \cite{ref19} used LLM-generated clinical-interview summaries for depression-prediction training augmentation. Here, synthetic posts provide controlled stress-test evaluation, distinct from training augmentation or clinical validation. Inclusion required consistency with the assigned scenario and target label: mixed or shifting cues for the affective scenarios, and low-affect content for the Neutral controls. Exclusions covered explicit clinical diagnosis claims, treatment recommendations, crisis language, identifying information, and off-scenario content. Appendix Table~F1 records the generation protocol. These checks establish adherence to the dataset design criteria. Independent expert annotation and naturally occurring mixed-emotion evaluation remain outside the scope of this dataset.

 \subsection{Data Partitioning and Model Inputs}

 After preprocessing and label encoding, the primary Reddit dataset is divided into training, model-validation, threshold-calibration, and held-out test sets by independently shuffling and partitioning each class. The final protocol samples exactly 40,000 examples per class (120,000 in total) and assigns 28,000 examples per class to training and 4,000 examples per class to each of validation, calibration, and held-out testing. Thus, the final split sizes are 84,000 training, 12,000 validation, 12,000 calibration, and 12,000 test examples. This per-class construction avoids one-example rounding differences from successive proportion-based split operations.

-The training partition supplies supervised parameter updates; model validation supplies hyperparameter, epoch, and checkpoint selection. The calibration partition supplies temperature fitting and threshold selection. Test labels are excluded from these Phase~1 operations. However, the routed Reddit test cases were reused during Phase~2 prompt development, so final prompt-configuration comparisons are interpreted as exploratory. Each classifier uses its own tokenizer on the cleaned title--body field.
-
-Only the cleaned textual content was used as the direct Phase~1 classifier input. For routed Reddit examples, Phase~2 received the linked original title and selftext after minimal privacy-oriented sanitization. Auxiliary metadata, such as subreddit source, author-related metadata, timestamp, and adult-content flag, was not used as a direct predictive feature in either phase.
+The training partition supplies supervised parameter updates; model validation supplies hyperparameter, epoch, and checkpoint selection. The calibration partition supplies temperature fitting and threshold selection. Test labels are excluded from these Phase~1 operations. The routed Reddit test cases also informed Phase~2 prompt development; within-set prompt-configuration rankings are therefore interpreted as exploratory. Each classifier uses its own tokenizer on the cleaned title--body field.
+
+Only the cleaned textual content was used as the direct Phase~1 classifier input. For routed Reddit examples, Phase~2 received the linked original title and selftext after minimal privacy-oriented sanitization. The remaining metadata, such as subreddit source, author-related metadata, timestamp, and adult-content flag, was not used as a direct predictive feature in either phase.
+
+\subsection{Additional Same-Source Evaluation}
+A separate balanced evaluation set comprises 9,000 posts (3,000 per class) sampled from records in the same source corpus that were not included in the original 120,000-post sample. Selection excludes normalized exact-text matches to the original sample and duplicate normalized texts within the candidate pool using Unicode NFKC normalization, case folding, and whitespace normalization. Sampling uses a fixed random seed. These procedures remove normalized exact-text overlap under the specified rule; the set remains same-source and is not guaranteed to be disjoint by author or time.
+
+Before evaluating this set, we fixed the final seed-42 DistilBERT model trained under the selected hyperparameters, the calibration temperature, the routing threshold, the three Llama~3 prompting protocols, and the previously generated SELF-DISCOVER plan. No training, recalibration, threshold selection, SELF-DISCOVER plan generation, or prompt revision used these 9,000 posts. Direct, CoT, and SELF-DISCOVER evaluated the same routed cases, and all end-to-end metrics were computed over the full 9,000-post set.

 \subsection{Ethical Considerations}

@@ -193,16 +198,16 @@
 \subsection{Unified Classifier-to-Re-Evaluation Pipeline}
 The final DistilBERT configuration is selected through the Phase~1 validation-based search. A final training run under that configuration supplies the predictions, logits, calibration temperature, and routed IDs used throughout the downstream experiment. No separately configured operational classifier is substituted after the classifier comparison. Three-seed classifier summaries describe training variability; downstream results describe this single linked final run, not an average of three reasoning pipelines.

-At inference, accepted predictions retain their Phase~1 labels. Every routed post is independently evaluated under each of six configurations: Llama~2-7B-Chat or Llama~3-8B-Instruct, each with Direct, CoT, or final SELF-DISCOVER prompting. These are separate alternatives, not an ensemble or a dataset-specific selection rule. All configurations within a dataset use the same Phase~1 outputs and routed IDs. Figure~\ref{fig:architecture} retains the original overview layout; the final configuration details are specified in the text.
+At inference, accepted predictions retain their Phase~1 labels. Every routed post is independently evaluated under each of six configurations: Llama~2-7B-Chat or Llama~3-8B-Instruct, each with Direct, CoT, or final SELF-DISCOVER prompting. These are separate alternatives, not an ensemble or a dataset-specific selection rule. All configurations within a dataset use the same Phase~1 outputs and routed IDs. Figure~\ref{fig:architecture} summarizes this shared pipeline; Figure~\ref{fig:sd-task-level} details the SELF-DISCOVER procedure.

 \begin{figure*}[t]\centering
 \includegraphics[width=\textwidth]{figures/architecture_figure_publication_final.pdf}
-\caption{Confidence-guided two-phase pipeline. Accepted predictions retain the Phase 1 label, while routed posts enter re-evaluation. Original artwork is retained in this review copy pending label updates; the final experiment evaluates both models with Direct, CoT, and SELF-DISCOVER as specified in the text.}
+\caption{Confidence-guided two-phase pipeline. Accepted predictions retain the Phase 1 label, while routed posts enter re-evaluation. Llama~2 and Llama~3 are each evaluated with Direct, CoT, and SELF-DISCOVER on the same routed posts. Parsed labels replace the initial predictions; parsing failures retain Phase~1 labels.}
 \label{fig:architecture}
 \end{figure*}
 \subsection{Phase 1: Initial Emotion Classification}

-The Phase~1 comparison includes DistilBERT \cite{ref12}, Mistral~7B \cite{ref13}, and Llama~2~7B \cite{ref14} under common data partitions and a four-trial search budget. DistilBERT is fully fine-tuned, whereas the 7B backbones use frozen representations with trainable linear heads. Thus, this is a comparison of practical classifier configurations, not an isolated test of architecture or maximum attainable performance. Each selected configuration is evaluated with seeds 42, 43, and 44. Appendix Table~A1c gives the selected settings; the final DistilBERT run under the selected configuration supplies downstream routing scores.
+The Phase~1 comparison includes DistilBERT \cite{ref12}, Mistral~7B \cite{ref13}, and Llama~2~7B \cite{ref14} under common data partitions and a four-trial search budget. DistilBERT is fully fine-tuned, whereas the 7B backbones use frozen representations with trainable linear heads. The comparison therefore evaluates practical classifier configurations under shared data partitions and a bounded search budget. Each selected configuration is evaluated with seeds 42, 43, and 44. Appendix Table~A1c gives the selected settings; the final DistilBERT run under the selected configuration supplies downstream routing scores.

 \subsection{DistilBERT}

@@ -323,7 +328,7 @@

 Here, $m_{\min}$ is a minimum accepted-sample requirement used to avoid selecting thresholds supported by too few calibration examples. The implementation sets $m_{\min}=\lceil0.10|D_{cal}|\rceil=1{,}200$. The selected $\tau=0.70$ accepted 11,807 calibration examples, so this safeguard was satisfied but did not determine the selected operating point. In the final DistilBERT operating policy, candidates are swept from 0.70 to 1.00 in increments of 0.01. This lower bound was chosen before held-out testing to avoid treating a weak three-class majority probability near 0.50 as sufficiently reliable for direct acceptance. Unless otherwise stated, the primary operating point uses temperature-scaled maximum softmax probability, a target selective risk of $\alpha=0.05$, a one-sided risk confidence level of $1-\delta=0.95$, the acceptance rule $c_i \geq \tau$, and the routing rule $c_i < \tau$. The risk constraint is a feasibility condition: it excludes candidate thresholds whose accepted-set upper risk exceeds $\alpha$. Among the feasible candidates in the prespecified 0.70--1.00 grid, the candidate with the greatest Phase~1 coverage is selected. Thus, the completed DistilBERT value $\tau^*=0.70$ reflects both the prespecified conservative operating range and calibration-set risk--coverage evaluation; it was not optimized using held-out test results or Phase~2 outcomes. The 7B models serve as Phase~1 predictive comparators only; they do not define alternative routing policies in the reported end-to-end experiment.

-The threshold-selection output records accepted accuracy, routed Phase~1 errors, error capture, routing precision, and proxy-Depression false-negative risk among accepted predictions. Additional confidence diagnostics include expected calibration error (ECE), adaptive ECE, Brier score, and negative log-likelihood before and after temperature scaling. Auxiliary analyses compare raw MSP, temperature-scaled MSP, entropy-based certainty, and probability margin; estimate bootstrap confidence intervals for selective metrics; assess threshold stability through calibration-set resampling; and extract high-confidence accepted errors for qualitative auditing.
+The threshold-selection output records accepted accuracy, routed Phase~1 errors, error capture, routing precision, and proxy-Depression false-negative risk among accepted predictions. Additional confidence diagnostics include expected calibration error (ECE), adaptive ECE, Brier score, and negative log-likelihood before and after temperature scaling. Supporting analyses compare raw MSP, temperature-scaled MSP, entropy-based certainty, and probability margin; estimate bootstrap confidence intervals for selective metrics; assess threshold stability through calibration-set resampling; and extract high-confidence accepted errors for qualitative auditing.

 \begin{algorithm}[t]
 \caption{Calibrated confidence-guided routing}
@@ -389,7 +394,7 @@

 Algorithm~\ref{alg:end-to-end-audit} formalizes the evaluation boundary between routed-subset behavior and full-set performance. In particular, a Phase~2 model is not credited merely for correcting selected errors: any newly introduced errors are deducted, and the final accuracy is recomputed over the complete evaluation set after accepted Phase~1 labels and routed Phase~2 labels are recombined.

-The export notebook prints a warning and falls back to the lowest empirical risk, then highest coverage, if no candidate satisfies the constraint. That branch was not needed here: the selected calibration point satisfies the prescribed upper-bound criterion. A fallback, if invoked, would not be interpreted as a risk-feasible policy. The temperature, threshold, risk target, confidence level, and candidate grid are stored with the calibration and threshold tables. The model source and seed are recorded separately in the execution environment artifact.
+The selected calibration point satisfied the upper-bound criterion, so the export notebook's infeasibility fallback was not invoked. The stored calibration, threshold, and environment artifacts record the policy values and model and seed identifiers.

 \subsection{Relationship to Mixed-Emotion Evaluation}

@@ -451,12 +456,17 @@
 \subsection{Execution and Resource Scope}
 The retained Phase~1 benchmark compares the selected classifier configurations on NVIDIA A100-SXM4-40GB hardware, with fp16 inference, a 256-token cap, and batch size 32. Three timing repetitions are distinct from the three training seeds. These classifier-stage measurements are not measurements of the complete final reasoning pipeline; Appendix Table~A1d retains their original measurement scope.

-Phase~2 uses chat-tuned Llama~2~7B and instruction-tuned Llama~3~8B with quantized loading and greedy decoding. Direct and final CoT request up to 1,024 new tokens per call. Final SELF-DISCOVER requests up to 1,024 for Llama~2 and 1,536 for Llama~3 in its per-post execution call. Thus, neither total generation cost nor token budget is claimed to be equal across all protocols. Llama~2's shorter context also constrains multi-turn inference. Template text and nominal execution budgets are disclosed in the appendices; wall-clock or monetary superiority of the final Phase~2 configurations is not inferred from accuracy.
+Phase~2 uses chat-tuned Llama~2~7B and instruction-tuned Llama~3~8B with quantized loading and greedy decoding. Direct and final CoT request up to 1,024 new tokens per call. Final SELF-DISCOVER requests up to 1,024 for Llama~2 and 1,536 for Llama~3 in its per-post execution call. Generation budgets therefore remain part of each complete protocol, and Llama~2's shorter context also constrains multi-turn inference. The appendices disclose template text and nominal execution budgets; a matched end-to-end time and cost comparison is reserved for separate benchmarking rather than inferred from accuracy.

 \subsection{Evaluation and Statistical Scope}
 Accuracy and macro F1 use all examples, after recombining accepted and routed predictions. Corrected errors $C$ are Phase~1 mistakes made correct by re-evaluation; introduced errors $I$ are previously correct predictions made wrong. The full-set accuracy change in percentage points is $100(C-I)/N$. Wrong-to-wrong changes are not corrections. Routed accuracy, error capture, and accepted-set risk measure different aspects of the pipeline.

-Appendix~\ref{app:statistics} reports count-based paired accuracy summaries. For a comparison with Phase~1, the discordant pairs are exactly $C$ and $I$, so the two-sided exact McNemar test is a binomial test with $C+I$ trials and probability one half. Paired percentile-bootstrap intervals for accuracy change are reconstructed from the three correctness-difference categories $+1,-1,0$, with counts $C,I,N-C-I$, using 50,000 multinomial resamples. This is equivalent to a paired row bootstrap for that scalar accuracy difference. It does not reconstruct class-wise metrics or paired comparisons between two re-evaluators. Holm correction is applied across the 12 final configuration-versus-Phase~1 tests. These intervals and tests describe the reported final counts; prompt development and selection make them exploratory rather than selection-adjusted confirmation of the winning prompt.
+All 15 final conditions are aligned by example ID and checked against the stored Phase~1 predictions and reference labels. The checks cover missing or duplicate IDs, parsed outputs, fallback behavior, full-set accuracy, macro F1, class-wise metrics, and corrected/introduced counts. The additional evaluation also validates 777 saved generation calls against their run settings and request hashes. No failed response is regenerated or assigned a reference-derived label.
+
+Two-sided exact McNemar tests compare paired correctness. Against Phase~1, discordant counts are $C$ and $I$; between two re-evaluators, they are the numbers correct only under each method. We estimate 95\% percentile-bootstrap intervals for accuracy differences with 50,000 multinomial resamples of the paired correctness-difference categories $+1,-1,0$. This yields the same distribution as resampling paired rows for that scalar difference. Holm correction uses the 12 main configuration-versus-Phase~1 tests as one family and the three additional-evaluation tests as another. Within each dataset--model combination, the two SD-versus-Direct/CoT tests form a separate family. Appendix~\ref{app:statistics} reports unadjusted and adjusted values and all comparison scopes.
+
+These intervals characterize example-level accuracy differences conditional on the fitted models and protocols. They are not confidence intervals for macro F1 or training-seed variability, and they do not account for within-author dependence. The main prompt-development comparisons are exploratory; the additional evaluation applies the frozen final protocols without adapting them during that run.
+
 \section{Experimental Results}
 \subsection{Phase 1 Performance on Reddit Data}

@@ -505,7 +515,7 @@
 \label{fig:phase1-model-comparison}
 \end{figure*}

-DistilBERT had the highest mean score and approximately 67 million parameters, compared with about seven billion for each comparator. These results support retaining it for the tested pipeline, without establishing superiority over fully adapted 7B models. Depression--Happy confusions accounted for 72.8\% of DistilBERT errors, 75.6\% of Llama~2 errors, and 74.3\% of Mistral errors across pooled seed predictions; Neutral F1 exceeded 98\% for all three. Pooling summarizes error composition, not 36,000 independent test examples. Small probe seed deviations also reflect the frozen backbone and should not be interpreted as equivalent training variability across adaptation methods.
+DistilBERT had the highest mean score and approximately 67 million parameters, compared with about seven billion for each comparator. Under the tested configurations and shared budget, its predictive performance and measured inference efficiency support its selection as the first-stage classifier. Depression--Happy confusions accounted for 72.8\% of DistilBERT errors, 75.6\% of Llama~2 errors, and 74.3\% of Mistral errors across pooled seed predictions; Neutral F1 exceeded 98\% for all three. Pooling summarizes error composition, not 36,000 independent test examples. Small probe seed deviations also reflect the frozen backbone and should not be interpreted as equivalent training variability across adaptation methods.


 The final linked DistilBERT run achieves 96.9167\% accuracy and 96.9166\% macro F1 on Reddit. This run uses the same selected hyperparameter configuration as the three-seed comparison. Its calibration and routing outputs, rather than those of a separately configured model, define every downstream condition.
@@ -565,7 +575,7 @@
 \end{figure*}

 \subsection{End-to-End Model and Prompting Results}
-Table~\ref{tab:final-e2e} reports the complete final comparison. Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 among these six configurations on both datasets. The advantage over Llama~3 CoT on Reddit is six correct predictions (0.0500 percentage points); numerical ordering alone does not establish a significant between-method difference.
+Table~\ref{tab:final-e2e} reports the complete final comparison. Llama~3 SELF-DISCOVER produces the largest observed Phase~1 gain on both datasets: 34 net corrections on Reddit and 27 on Mixed Emotion, reaching 97.2000\% and 92.6667\% accuracy. It also leads the six configurations in accuracy and macro F1 on each dataset. On Mixed Emotion, SD exceeds both alternatives within each model after Holm adjustment; for Llama~3, adjusted $p=0.0020$ versus Direct and $p<0.0001$ versus CoT. On Reddit, its six-prediction lead over Llama~3 CoT is not statistically significant, nor is its difference from Direct (Holm $p=0.3616$ and $0.2163$).
 \begin{table*}[t]
 \centering\small
 \setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.14}
@@ -600,9 +610,9 @@
 \end{figure*}

 \subsection{Correction and Harm}
-On Reddit, Llama~3 SELF-DISCOVER corrects 72 errors and introduces 38, producing 34 net corrections. Direct and CoT yield 23 and 28 net corrections, respectively. Llama~2 yields smaller gains: Direct has nine net corrections, CoT one, and SELF-DISCOVER three. Thus, SELF-DISCOVER does not outperform Direct within every model and dataset combination.
-
-On Mixed Emotion, Llama~3 SELF-DISCOVER corrects 28 errors and introduces one. Total errors decrease from 49 to 22, a 55.10\% relative reduction, and accuracy increases by 9.00 percentage points. This low introduced-error count is an observed property of the 300-example stress test, not a population safety guarantee. Llama~2 SELF-DISCOVER also improves the baseline (17 net corrections), whereas its Direct and CoT configurations introduce more errors than they correct. The pattern supports evaluating both correction yield and damage to initially correct predictions.
+On Reddit, Llama~3 SELF-DISCOVER corrects 72 errors and introduces 38, producing 34 net corrections. Direct and CoT yield 23 and 28 net corrections, respectively. Llama~2 yields smaller gains: Direct has nine net corrections, CoT one, and SELF-DISCOVER three. For Llama~2, Direct, CoT, and SELF-DISCOVER yield nine, one, and three net corrections, showing a model--protocol interaction.
+
+On Mixed Emotion, Llama~3 SELF-DISCOVER corrects 28 errors and introduces one. Total errors decrease from 49 to 22, a 55.10\% relative reduction, and accuracy increases by 9.00 percentage points. The single introduced error indicates strong preservation of initially correct labels within this controlled 300-example stress test. Llama~2 SELF-DISCOVER also improves the baseline (17 net corrections), whereas its Direct and CoT configurations introduce more errors than they correct. The pattern supports evaluating both correction yield and damage to initially correct predictions.

 \begin{figure*}[t]\centering
 \includegraphics[width=\textwidth]{figures/corrections.pdf}
@@ -610,18 +620,41 @@
 \label{fig:final-errors}
 \end{figure*}

+\begin{figure*}[t]\centering
+\includegraphics[width=0.96\textwidth]{figures/final_sd_confusion.pdf}
+\caption{Final Llama~3 SELF-DISCOVER confusion matrices, reconstructed from all end-to-end predictions. Rows are reference labels and columns are predictions; counts and within-class percentages share the Phase~1 figure's scale. Unparsed routed responses retain the initial label.}
+\label{fig:final-sd-confusion}
+\end{figure*}
+
+\subsection{Additional Evaluation of the Fixed Pipeline}
+On the separate 9,000-post set, Phase~1 correctly classifies 8,782 posts (97.5778\%) and routes 111 (1.23\%). The routed subset contains 48 of the 218 Phase~1 errors: its error rate is 43.24\%, compared with 2.42\% overall. Thus, the fixed threshold again selects an error-enriched subset while retaining 98.77\% coverage.
+
+Table~\ref{tab:additional9000} reports the fixed-protocol results. SELF-DISCOVER yields 17 net corrections, increases accuracy from 97.5778\% to 97.7667\%, and significantly improves over Phase~1 by 0.1889 percentage points (95\% CI [0.0444, 0.3333]; Holm $p=0.0412$). Direct and CoT each yield 13 net corrections. SELF-DISCOVER retains the highest observed accuracy, although its four-correct-prediction lead over either alternative is not statistically significant (Holm $p=0.9614$ for both comparisons). There are two Direct parsing failures and none for CoT or SD; all use the same fallback rule. Appendix~\ref{app:additional} provides the fixed settings and class-level results.
+
+\begin{table*}[t]\centering\small
+\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.14}
+\caption{Additional 9,000-post evaluation with the final Llama~3 protocols fixed. Accuracy and macro F1 are percentages. Change and its 95\% interval are in percentage points relative to Phase~1; Holm $p$ adjusts the three comparisons with Phase~1.}
+\label{tab:additional9000}
+\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}\toprule
+Protocol & Accuracy & Macro F1 & $C$ & $I$ & Change & Paired 95\% CI & Holm $p$ \\\midrule
+Phase 1 & 97.5778 & 97.5821 & -- & -- & -- & -- & -- \\
+Direct & 97.7222 & 97.7242 & 32 & 19 & +0.1444 & [-0.0111, 0.3000] & 0.1320 \\
+CoT & 97.7222 & 97.7234 & 28 & 15 & +0.1444 & [0.0000, 0.2889] & 0.1320 \\
+SELF-DISCOVER & 97.7667 & 97.7688 & 30 & 13 & +0.1889 & [0.0444, 0.3333] & 0.0412 \\\bottomrule
+\end{tabular*}\end{table*}
+
 \subsection{Conditional Correction Opportunity}
-With the router held fixed, a conditional oracle corrects every routed error without changing any initially correct prediction. Its full-set ceiling is $\mathrm{Acc}_{P1}+E_R/N$, where $E_R$ is the number of routed Phase~1 errors. The ceiling is 97.7333\% on Reddit and 93.0000\% on Mixed Emotion. Final Llama~3 SELF-DISCOVER realizes $34/98=34.69\%$ and $27/28=96.43\%$ of these respective net-correction opportunities. These are descriptive ceilings under a fixed router, not achievable performance guarantees or alternative tuned baselines.
+With the router held fixed, a conditional oracle corrects every routed error without changing any initially correct prediction. Its full-set ceiling is $\mathrm{Acc}_{P1}+E_R/N$, where $E_R$ is the number of routed Phase~1 errors. The ceiling is 97.7333\% on Reddit and 93.0000\% on Mixed Emotion. Final Llama~3 SELF-DISCOVER realizes $34/98=34.69\%$ and $27/28=96.43\%$ of these respective net-correction opportunities. On the additional 9,000 posts, the corresponding ceiling is 98.1111\% and SD realizes $17/48=35.42\%$ of the net-correction opportunity. These are descriptive ceilings under a fixed router, not achievable performance guarantees or alternative tuned baselines.
 \Needspace{6\baselineskip}
 \section{Discussion}
 \subsection{An Efficient First Stage and a Selective Second Stage}
 The linked experiment separates three decisions: selecting a practical classifier, routing uncertain posts, and choosing a re-evaluation protocol. DistilBERT retains strong three-seed predictive performance and a measured classifier-stage inference advantage under the bounded Phase~1 comparison. The final run then supplies one consistent set of predictions and confidence scores for every Phase~2 condition. Classifier means and downstream single-run scores answer different questions without requiring different hyperparameter configurations.

 \subsection{Selection and Correction Are Different Outcomes}
-Confidence routing identifies an error-enriched subset, but enrichment alone is insufficient. Llama~2 CoT is nearly neutral on Reddit and harmful on Mixed Emotion, whereas final SELF-DISCOVER with Llama~3 improves both sets. On the stress test, searching for evidence for and against all three classes and distinguishing present distress from resolved or attributed events are plausible contributors to the observed pattern. The experiment evaluates the complete prompt package; it does not establish the separate causal effect of each instruction or demonstrate that the generated rationale faithfully reveals the model's computation.
+Confidence routing creates an error-enriched correction opportunity; the re-evaluator determines the resulting net gain. Llama~2 CoT is nearly neutral on Reddit and harmful on Mixed Emotion, whereas final SELF-DISCOVER with Llama~3 improves both sets. On the stress test, searching for evidence for and against all three classes and distinguishing present distress from resolved or attributed events are plausible contributors to the observed pattern. The experiment evaluates the complete prompt package; it does not establish the separate causal effect of each instruction or demonstrate that the generated rationale faithfully reveals the model's computation.

 \subsection{What the Model--Protocol Comparison Shows}
-Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 on both sets. However, the 0.2833-point Reddit gain uses a 1.82\% routing rate, whereas the 9.00-point Mixed Emotion gain uses 28.67\%. These are different correction opportunities, not directly comparable effect sizes. Likewise, one introduced stress-test error versus 38 Reddit errors warrants dataset-specific interpretation rather than a universal reliability claim.
+Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 on all three evaluation sets and significantly improves final accuracy over Phase~1 under the stated Holm adjustments. Within-model protocol comparisons answer a narrower question: SD's advantage over Direct and CoT is supported on Mixed Emotion, while the smaller differences on the Reddit evaluations remain uncertain. Moreover, the 0.2833-point Reddit gain uses a 1.82\% routing rate, whereas the 9.00-point Mixed Emotion gain uses 28.67\%. These are different correction opportunities, not directly comparable effect sizes. Likewise, one introduced stress-test error versus 38 Reddit errors warrants dataset-specific interpretation rather than a universal reliability claim.

 The methodological value of the adaptation is to specify what evidence a selected post should be checked against, rather than merely requesting a longer explanation. Emotional subject and time-frame checks address attribution and trajectory, while the three-label ledger and counterfactual check make competing interpretations explicit. The observed gains support this complete configuration within the evaluated settings; they do not identify which instruction caused the gain. Appendix~\ref{app:sd-variants} shows that the final design improves on the earlier independent-countercheck variant for Llama~3 on both sets, but not for every model--dataset combination.

@@ -630,14 +663,15 @@
 \section{Limitations and Future Work}
 Subreddit-derived, sentiment-filtered labels are proxy constructs and can reward topical or community-specific shortcuts. Random post-level splits do not establish author-disjoint or temporal independence. The Mixed Emotion data are synthetic, scenario-balanced, and not independently expert-labeled; their results should not be generalized to naturally occurring emotional ambiguity or clinical depression.

-Prompt variants were developed and compared using these evaluation cases. Consequently, the selected prompt's observed ranking and count-based significance summaries remain exploratory. A frozen final configuration should be assessed on unused data to estimate generalization after selection. Earlier evaluations of a different downstream configuration are not used as independent validation of the final pipeline in this manuscript.
-
-Reddit re-evaluation uses richer original text than the cleaned Phase~1 input. This is a deliberate system design but prevents attributing the entire gain to prompting alone. Residual parsing failures retain Phase~1 predictions; their presence and the rule must be considered when comparing protocols. The final SELF-DISCOVER summaries contain two such failures on Reddit and one on Mixed Emotion for Llama~3. No unresolved failure is manually relabeled from its reference answer.
+The main Reddit and Mixed Emotion cases informed prompt development, so their configuration rankings are exploratory and the reported corrections do not adjust for prompt selection. The separate 9,000-post evaluation applies the final Llama~3 protocols and retained plan without changes during evaluation, providing an additional same-source check. It does not establish author-disjoint, temporal, cross-platform, or clinical generalization, and its small between-protocol differences remain uncertain.
+
+Reddit re-evaluation uses richer original text than the cleaned Phase~1 input. This is a deliberate system design but prevents attributing the entire gain to prompting alone. Residual parsing failures retain Phase~1 predictions; their presence and the rule must be considered when comparing protocols. Final SELF-DISCOVER has 21 parsing failures for Llama~2 on Reddit and none on Mixed Emotion, and two and one, respectively, for Llama~3. The additional Llama~3 SD evaluation has no parsing failures. The 21 Llama~2 Reddit fallbacks are part of its measured configuration-level performance rather than excluded cases. No unresolved failure is manually relabeled from its reference answer.

 The current experiments do not quantify repeated-generation variation for all final configurations or supply a fully matched end-to-end cost benchmark. Greedy decoding does not guarantee bitwise equality across hardware and software environments. Independent expert review, broader evaluation, and generation-cost accounting are useful extensions, rather than reasons to repeat the completed Phase~1 model comparison.

 \section{Conclusion}
-A unified DistilBERT-to-LLM pipeline allows classifier performance, routing concentration, and correction quality to be evaluated without substituting a separately configured downstream classifier. Among the final Direct, CoT, and SELF-DISCOVER configurations for two language models, Llama~3 SELF-DISCOVER achieves the highest observed accuracy and macro F1 on both Reddit and the synthetic Mixed Emotion stress test. Its gains are 34 and 27 net corrections, respectively. The principal empirical finding is configuration-dependent correction: routing creates an opportunity, but the re-evaluation protocol determines whether the opportunity is used without introducing excessive new errors. These exploratory proxy-label results support further validation, not clinical deployment claims.
+A unified DistilBERT-to-LLM pipeline connects efficient classifier selection, calibrated routing, and selective original-text re-evaluation. Across the main Reddit test, the Mixed Emotion stress test, and the additional 9,000-post evaluation, Llama~3 SELF-DISCOVER converts the routed correction opportunity into 34, 27, and 17 net corrections, respectively, and significantly improves accuracy over Phase~1 under the stated Holm adjustments. It achieves the highest observed accuracy and macro F1 among the evaluated configurations on all three sets. Paired comparisons further show that SD exceeds Direct and CoT on Mixed Emotion within both models, while the smaller between-protocol differences on Reddit remain unresolved. The results support the task-specific reusable plan and evidence checks as an effective complete re-evaluation configuration and demonstrate that model and protocol choice determine how routing translates into final gains. These findings establish value for proxy-label text classification; clinical use requires separate validation and governance.
+
 \begin{thebibliography}{46}

 \bibitem{ref1} World Health Organization, ``Depressive disorder (depression),'' WHO Fact Sheet, Aug. 29, 2025. Available: \url{https://www.who.int/news-room/fact-sheets/detail/depression}
@@ -739,9 +773,9 @@
 % Keep appendix headings and their supporting tables compact without orphaning titles.
 \setlength{\medskipamount}{4pt plus 1pt minus 1pt}
 \Needspace{12\baselineskip}
-\section{Supplementary Methodological Details}
-
-Appendix Table~A1 summarizes the main variables retained from the Reddit dataset during data construction and preprocessing. Appendix Table~A1b complements the schema description with actual class-balanced examples from the supplementary Mixed Emotion stress test.
+\section{Additional Methodological Details}
+
+Appendix Table~A1 summarizes the main variables retained from the Reddit dataset during data construction and preprocessing. Appendix Table~A1b complements the schema description with class-balanced examples from the controlled Mixed Emotion stress test.

 \par\medskip
 \noindent\textsc{Appendix Table A1. Description of Reddit dataset variables}\par
@@ -789,7 +823,7 @@
 \par\smallskip
 \normalsize

-\noindent\footnotesize\textit{Note.} These are unmodified input texts from the final 300-example supplementary dataset. They were not used for Phase~1 training, temperature fitting, or threshold selection. The table illustrates co-occurring cues and final-trajectory labeling; it is not intended to establish population prevalence or clinical validity.\normalsize
+\noindent\footnotesize\textit{Note.} These are unmodified input texts from the final 300-example controlled stress-test dataset. They were not used for Phase~1 training, temperature fitting, or threshold selection. The table illustrates co-occurring cues and final-trajectory labeling; it is not intended to establish population prevalence or clinical validity.\normalsize
 \par\medskip

 \Needspace{11\baselineskip}
@@ -971,7 +1005,7 @@
 \subsection{Generation Budget and Failure Handling}
 Direct and CoT request 1,024 new tokens per generation call. Final SD execution requests 1,024 for Llama~2 and 1,536 for Llama~3. The SD discovery calls request up to 512 tokens for Llama~2 and 768 for Llama~3. The source uses a shared per-model cache for the task-level structures, allowing reuse across the Reddit and Mixed Emotion runs. The templates specify instructions, not guaranteed instruction compliance. Quantized model loading, model-specific chat formatting, context limits, and call structure are part of the evaluated implementation. A shorter plan instruction does not by itself prove lower measured cost.

-Remaining final SD parsing failures are zero for Llama~2 on both sets, two for Llama~3 Reddit, and one for Llama~3 Mixed Emotion. These remain failures and use Phase~1 fallback. No additional model call is added to resolve them in the reported final SD results.
+Remaining final SD parsing failures are 21 for Llama~2 Reddit, zero for Llama~2 Mixed Emotion, two for Llama~3 Reddit, and one for Llama~3 Mixed Emotion. There are no SD parsing failures in the additional 9,000-post evaluation. These remain failures and use Phase~1 fallback. No additional model call is added to resolve them in the reported final SD results.
 \section{SELF-DISCOVER Design-Variant Comparison}\label{app:sd-variants}
 The earlier independent-countercheck variant generated an instance-specific plan using general reasoning modules; the final version uses a domain-oriented task-level plan and an explicit evidence ledger. Both withhold the Phase~1 label. Multiple components change jointly, so this is a design-variant comparison rather than a single-component causal ablation. The earlier variant is not used in the main six-configuration comparison.

@@ -998,30 +1032,53 @@

 The final design improves three of these four model--dataset combinations; it does not uniformly dominate the earlier design. Its Llama~3 configuration improves both datasets. The change in input-independent planning also changes the number and scope of discovery calls, so a cost comparison requires measured generation records rather than accuracy alone.

-\Needspace{28\baselineskip}
-\section{Paired Accuracy Summaries}\label{app:statistics}
-Each row asks whether one complete re-evaluation configuration differs from Phase~1 on the same examples. It does not ask whether SD is better than Direct or CoT. ``Change'' is the full-set accuracy difference in percentage points; the interval estimates uncertainty in that difference; exact $p$ compares the numbers corrected and newly harmed; Holm $p$ adjusts the 12 displayed comparisons.
-
-These calculations use the final reported corrected/introduced counts, with all unparseable SD responses left at the implemented Phase~1 fallback. For accuracy change, the paired correctness-difference counts are sufficient for the exact test and paired bootstrap. This does not constitute a new audit of all raw responses, nor a paired SD-versus-CoT significance test. The latter requires matched per-example predictions from both configurations. Intervals are percentile bootstrap intervals in percentage points. Holm adjustment covers all 12 rows below; it does not account for selection over previously explored prompts.
-
-\par\medskip\noindent\textsc{Appendix Table E1. Paired accuracy comparisons with Phase 1}\par\smallskip
-{\small\renewcommand{\arraystretch}{1.18}\noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrr@{}}\toprule
+\section{Paired Accuracy Comparisons}\label{app:statistics}
+All rows below use ID-aligned final predictions, including the implemented Phase~1 fallback for parsing failures. Table~E1 compares each main configuration with Phase~1; Table~E2 instead compares SD with Direct or CoT on the same examples. A positive change favors the named re-evaluator in E1 and SD in E2. Intervals are 50,000-resample paired percentile-bootstrap intervals for accuracy change, in percentage points. Exact $p$ is a two-sided McNemar test. These tests do not measure macro F1 uncertainty or adjust for prompt development on the main evaluation cases.
+\Needspace{12\baselineskip}
+\par\medskip\noindent\textsc{Appendix Table E1. Main comparisons with Phase 1}\par\smallskip
+{\small\renewcommand{\arraystretch}{1.15}\noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrr@{}}\toprule
 Dataset & Model & Protocol & Change (pp) & Paired 95\% CI & Exact $p$ & Holm $p$ \\\midrule
 Reddit & Llama 2 & Direct & +0.0750 & [-0.075, 0.225] & 0.3742 & 1.0000 \\
 Reddit & Llama 2 & CoT & +0.0083 & [-0.150, 0.167] & 1.0000 & 1.0000 \\
-Reddit & Llama 2 & SELF-DISCOVER & +0.0250 & [-0.142, 0.192] & 0.8424 & 1.0000 \\
+Reddit & Llama 2 & SD & +0.0250 & [-0.142, 0.192] & 0.8424 & 1.0000 \\
 Reddit & Llama 3 & Direct & +0.1917 & [0.017, 0.367] & 0.0380 & 0.2281 \\
 Reddit & Llama 3 & CoT & +0.2333 & [0.075, 0.392] & 0.0061 & 0.0479 \\
-Reddit & Llama 3 & SELF-DISCOVER & +0.2833 & [0.117, 0.458] & 0.0015 & 0.0151 \\
+Reddit & Llama 3 & SD & +0.2833 & [0.117, 0.458] & 0.0015 & 0.0151 \\
 Mixed Emotion & Llama 2 & Direct & -7.3333 & [-11.667, -3.000] & 0.0013 & 0.0139 \\
 Mixed Emotion & Llama 2 & CoT & -4.6667 & [-9.333, 0.000] & 0.0704 & 0.3520 \\
-Mixed Emotion & Llama 2 & SELF-DISCOVER & +5.6667 & [2.333, 9.000] & 0.0015 & 0.0151 \\
+Mixed Emotion & Llama 2 & SD & +5.6667 & [2.333, 9.000] & 0.0015 & 0.0151 \\
 Mixed Emotion & Llama 3 & Direct & +5.6667 & [2.000, 9.667] & 0.0060 & 0.0479 \\
 Mixed Emotion & Llama 3 & CoT & +1.0000 & [-2.333, 4.333] & 0.7011 & 1.0000 \\
-Mixed Emotion & Llama 3 & SELF-DISCOVER & +9.0000 & [5.667, 12.667] & $<0.0001$ & $<0.0001$ \\
+Mixed Emotion & Llama 3 & SD & +9.0000 & [5.667, 12.333] & $<0.0001$ & $<0.0001$ \\
 \bottomrule\end{tabular*}}\par\medskip
-
-\clearpage
+Holm adjustment in Table~E1 covers all 12 rows.
+\Needspace{12\baselineskip}
+\par\medskip\noindent\textsc{Appendix Table E2. SD versus alternative re-evaluators}\par\smallskip
+{\small\renewcommand{\arraystretch}{1.15}\noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrr@{}}\toprule
+Dataset & Model & Comparison & Change (pp) & Paired 95\% CI & Exact $p$ & Holm $p$ \\\midrule
+Reddit & Llama 2 & SD -- Direct & -0.0500 & [-0.192, 0.092] & 0.5560 & 1.0000 \\
+Reddit & Llama 2 & SD -- CoT & +0.0167 & [-0.142, 0.175] & 0.9179 & 1.0000 \\
+Reddit & Llama 3 & SD -- Direct & +0.0917 & [-0.008, 0.192] & 0.1081 & 0.2163 \\
+Reddit & Llama 3 & SD -- CoT & +0.0500 & [-0.033, 0.142] & 0.3616 & 0.3616 \\
+Mixed Emotion & Llama 2 & SD -- Direct & +13.0000 & [9.333, 17.000] & $<0.0001$ & $<0.0001$ \\
+Mixed Emotion & Llama 2 & SD -- CoT & +10.3333 & [6.000, 14.667] & $<0.0001$ & $<0.0001$ \\
+Mixed Emotion & Llama 3 & SD -- Direct & +3.3333 & [1.333, 5.333] & 0.0020 & 0.0020 \\
+Mixed Emotion & Llama 3 & SD -- CoT & +8.0000 & [5.000, 11.333] & $<0.0001$ & $<0.0001$ \\
+Additional 9,000 & Llama 3 & SD -- Direct & +0.0444 & [-0.056, 0.144] & 0.5235 & 0.9614 \\
+Additional 9,000 & Llama 3 & SD -- CoT & +0.0444 & [-0.044, 0.133] & 0.4807 & 0.9614 \\
+\bottomrule\end{tabular*}}\par\medskip
+Holm adjustment in Table~E2 is applied to the two comparisons within each dataset--model combination, not to all ten rows as one family.
+\Needspace{12\baselineskip}
+\par\medskip\noindent\textsc{Appendix Table E3. Additional 9,000 posts versus Phase 1}\par\smallskip
+{\small\renewcommand{\arraystretch}{1.15}\noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}\toprule
+Protocol & Change (pp) & Paired 95\% CI & Exact $p$ & Holm $p$ \\\midrule
+Direct & +0.1444 & [-0.0111, 0.3000] & 0.0919 & 0.1320 \\
+CoT & +0.1444 & [0.0000, 0.2889] & 0.0660 & 0.1320 \\
+SD & +0.1889 & [0.0444, 0.3333] & 0.0137 & 0.0412 \\
+\bottomrule\end{tabular*}}\par\medskip
+Table~E3 treats its three comparisons as a separate Holm family. Non-significance of an SD-versus-alternative comparison is not evidence of equivalence.
+
+\Needspace{18\baselineskip}
 \section{Synthetic Mixed Emotion Dataset Generation Protocol}

 \par\medskip
@@ -1035,8 +1092,8 @@
 \toprule
 Component & Description \\
 \midrule
-Purpose & Construct 300 controlled examples spanning mixed or shifting affect and Neutral controls for supplementary stress-test evaluation. \\
-Use in pipeline & Not used for Phase 1 training, hyperparameter tuning, or routing-threshold selection; used only for supplementary robustness evaluation. \\
+Purpose & Construct 300 controlled examples spanning mixed or shifting affect and Neutral controls for controlled stress-test evaluation. \\
+Use in pipeline & Not used for Phase 1 training, hyperparameter tuning, or routing-threshold selection; used only for controlled stress-test evaluation. \\
 Class balance & Depression = 100, Neutral = 100, Happy = 100. \\
 Scenario balance & Seven controlled scenario groups: five 40-example mixed or shifting-emotion groups, a 75-example clear technical/informational Neutral group, and a 25-example mild factual/procedural-ambiguity Neutral group. \\
 Scenario types & Blended-emotion co-occurrence; positive-to-distress shift; distress-to-recovery shift; Neutral framing with subtle affect; conflicting cues with a dominant trajectory; clear technical or informational Neutral content; and mild factual or procedural Neutral ambiguity. \\
@@ -1063,5 +1120,36 @@
 \par\medskip


+\clearpage
+\section{Class-Level Results and Additional Evaluation}\label{app:additional}
+Table~G1 reports class-wise F1 for each complete pipeline. All metrics use the full dataset, not only routed posts. Failures count routed responses without a parsed label; those records retain their Phase~1 predictions and remain in every metric. Reddit and Mixed Emotion contain 4,000 and 100 reference examples per class, respectively; the additional set contains 3,000 per class.
+\Needspace{12\baselineskip}
+\par\medskip\noindent\textsc{Appendix Table G1. Class-wise F1 and retained parsing failures}\par\smallskip
+{\small\renewcommand{\arraystretch}{1.15}\noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrr@{}}\toprule
+Dataset & Model & Protocol & Depression F1 & Neutral F1 & Happy F1 & Failures \\\midrule
+Reddit & Llama 2 & Direct & 96.40 & 98.53 & 96.05 & 1 \\
+Reddit & Llama 2 & CoT & 96.24 & 98.55 & 95.98 & 0 \\
+Reddit & Llama 2 & SD & 96.47 & 98.37 & 95.97 & 21 \\
+Reddit & Llama 3 & Direct & 96.60 & 98.51 & 96.21 & 1 \\
+Reddit & Llama 3 & CoT & 96.64 & 98.49 & 96.31 & 1 \\
+Reddit & Llama 3 & SD & 96.66 & 98.67 & 96.27 & 2 \\
+Mixed Emotion & Llama 2 & Direct & 84.10 & 82.91 & 58.48 & 0 \\
+Mixed Emotion & Llama 2 & CoT & 75.65 & 88.24 & 72.91 & 0 \\
+Mixed Emotion & Llama 2 & SD & 87.23 & 96.04 & 84.76 & 0 \\
+Mixed Emotion & Llama 3 & Direct & 90.11 & 93.27 & 84.76 & 0 \\
+Mixed Emotion & Llama 3 & CoT & 90.11 & 87.39 & 76.53 & 0 \\
+Mixed Emotion & Llama 3 & SD & 90.11 & 97.98 & 90.00 & 1 \\
+Additional 9,000 & Llama 3 & Direct & 97.34 & 98.78 & 97.05 & 2 \\
+Additional 9,000 & Llama 3 & CoT & 97.34 & 98.73 & 97.10 & 0 \\
+Additional 9,000 & Llama 3 & SD & 97.34 & 98.86 & 97.10 & 0 \\
+\bottomrule\end{tabular*}}\par\medskip
+\subsection{Fixed Additional-Evaluation Protocol}
+The additional run loads the same final seed-42 DistilBERT weights and uses $T^*=1.480195$, $\tau^*=0.70$, a 256-token limit, and fp32 inference with batch size 64. Phase~1 training, calibration fitting, and threshold search are disabled. The Llama~3 protocols use the exact templates in Appendices~B--C and the retained model-specific SD plan, with no new discovery calls. All three methods receive the same 111 routed original texts; the 6,000-character input policy, greedy decoding, 4-bit NF4 loading, and per-call limits (1,024 for Direct/CoT; 1,536 for SD) remain fixed. Execution uses an NVIDIA L4 GPU. Runtime, model revision, input hashes, and the saved plan are recorded with the run.
+
+There are 111 Direct calls, 555 CoT calls, and 111 SD calls. These counts include the five-turn CoT dialogue but no new SD discovery. All 777 saved calls were checked against their request hashes and run configuration; final outputs were aligned by ID and recombined with accepted Phase~1 predictions. The separate example-level tests in Appendix~E use these complete predictions. Call counts alone do not establish a matched time or monetary-cost advantage.
+
+\begin{center}\includegraphics[width=\textwidth]{figures/additional9000_confusion.pdf}\end{center}
+\noindent\footnotesize\textbf{Appendix Figure G1.} End-to-end confusion matrices for Direct, CoT, and final SELF-DISCOVER on the same 9,000 posts. Rows are reference labels; columns are final predictions. Counts and within-class percentages use a common color scale.\normalsize
+
 \EOD
 \end{document}
```
