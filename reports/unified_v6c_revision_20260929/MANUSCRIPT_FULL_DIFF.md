# 원고 전체 변경 내역

```diff
--- 20260912/main.tex
+++ 20260929/main.tex
@@ -10,6 +10,7 @@
 \usepackage{needspace}
 \usepackage{float}
 \usepackage{balance}
+\usepackage{listings}
 \newcolumntype{Y}{>{\raggedright\arraybackslash}X}
 \usepackage{graphicx}
 \usepackage{microtype}
@@ -22,6 +23,11 @@
 \emergencystretch=3em
 \sloppy

+\definecolor{PromptRule}{HTML}{8DAAC2}
+\lstset{basicstyle=\rmfamily\small,language={},keywordstyle={},commentstyle={},stringstyle={},literate={*}{{\char42}}1 {-}{{\char45}}1,breaklines=true,columns=fullflexible,keepspaces=true,showstringspaces=false,frame=tb,rulecolor=\color{PromptRule},aboveskip=7pt,belowskip=10pt,framesep=5pt,linewidth=\dimexpr\linewidth-18pt\relax,xleftmargin=6pt,xrightmargin=0pt,breakatwhitespace=true}
+\setlength{\textfloatsep}{12pt plus 2pt minus 2pt}
+\setlength{\dbltextfloatsep}{12pt plus 2pt minus 2pt}
+\def\thevol{XX}\def\theyear{2026}\clubpenalty=10000\widowpenalty=10000
 \begin{document}
 \history{}
 \doi{}
@@ -38,15 +44,15 @@
 \corresp{Corresponding-author details will be inserted in the submission version.}

 \begin{abstract}
-Transformer classifiers can process social-media text at scale, but confidence-based selection does not ensure that a second model will correct the selected predictions. We evaluate a two-phase framework for three-class proxy emotion classification (Depression, Neutral, and Happy) that separates routing from LLM re-evaluation. Phase~1 uses temperature-calibrated DistilBERT predictions and a calibration-set risk--coverage criterion to route low-confidence inputs. In a three-seed comparison on the same 12,000 Reddit test posts, DistilBERT achieved 96.88 $\pm$ 0.04\% accuracy, versus 95.36 $\pm$ 0.08\% for Mistral~7B and 95.17 $\pm$ 0.07\% for Llama~2~7B frozen-backbone linear probes, with higher throughput and lower latency on a common A100 benchmark. The fixed operational checkpoint achieved 96.69\% accuracy and routed 171 posts (1.42\%). Re-evaluation using minimally sanitized original text yielded 96.67\% accuracy with Llama~2 Chain-of-Thought and 96.94\% with Llama~3 SELF-DISCOVER. On a 300-example synthetic Mixed Emotion stress test, accuracy increased from 81.33\% to 85.33\% and 87.33\%, respectively. Because prompt development reused the original routed Reddit cases, we additionally evaluated the frozen pipeline on 9,000 previously unused same-source posts. Routing 137 posts (1.52\%) raised accuracy from 96.73\% to 97.03\% with Llama~3, yielding 27 net corrections and a Holm-adjusted $p=0.000131$; the Llama~2 gain was not significant. These results support configuration-dependent correction with limited re-evaluation volume beyond the prompt-development sample. They concern proxy emotion labels, not clinical diagnoses; external validation and expert review remain necessary for broader use.
+Confidence-based routing can concentrate classifier errors, but improvement also depends on how the selected inputs are re-evaluated. We study a two-phase pipeline for three-class proxy emotion classification of social-media posts. A sweep-selected DistilBERT configuration supplies both the final classifier and its calibrated routing outputs; high-confidence predictions are retained, and low-confidence posts are re-evaluated using their original text. We compare Direct, Chain-of-Thought (CoT), and an adapted task-level SELF-DISCOVER protocol with Llama~2 and Llama~3. On 12,000 Reddit posts, Phase~1 achieves 96.9167\% accuracy and routes 218 posts (1.82\%). Llama~3 SELF-DISCOVER achieves 97.2000\%, correcting 72 errors while introducing 38. On a balanced 300-example synthetic Mixed Emotion stress test, the same routing policy selects 86 posts; accuracy increases from 83.6667\% to 92.6667\%, with 28 corrected errors and one introduced error. Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 among the six final configurations on both sets, although method rankings differ for Llama~2. The results distinguish error selection from correction and support structured, evidence-oriented re-evaluation within the evaluated settings. Prompt-configuration comparisons are exploratory, and the synthetic stress test does not establish clinical or external-domain validity.
 \end{abstract}
-
 \begin{keywords}
 Depression-risk-related emotion classification, social media text analysis, large language models, confidence-guided routing, Chain-of-Thought prompting, SELF-DISCOVER, mental health NLP, reasoning traceability
 \end{keywords}

 \titlepgskip=-21pt
 \maketitle
+\raggedbottom

 \section{Introduction}

@@ -68,11 +74,13 @@
 \begin{itemize}
 \item It operationalizes selective LLM re-evaluation as an end-to-end pipeline in which an efficient classifier handles high-confidence inputs and reasoning models receive only calibration-defined low-confidence inputs. The Phase~1 comparison supports operational model selection through predictive performance and measured inference throughput, latency, and memory on a common accelerator, rather than claiming exhaustive optimization of every 7B architecture.
 \item It replaces a heuristic confidence cutoff with a reproducible calibration protocol: temperature scaling, a prespecified threshold grid, a one-sided selective-risk upper bound, and maximum-coverage selection among feasible thresholds. The selected policy is fixed before held-out and Phase~2 evaluation.
-\item It evaluates routing quality separately from re-evaluation quality. Coverage, selective risk, error capture, corrected errors, introduced errors, and net corrections reveal whether the router finds difficult inputs and whether the LLM actually improves them. An additional 9,000-post same-source holdout tests the frozen pipeline beyond the sample used during prompt development.
-\item It contributes a controlled Mixed Emotion stress test and case-level reasoning audit trail for studying posts whose surface cues and dominant emotional trajectory diverge. The stress test is explicitly separated from model training and threshold selection.
+\item It evaluates routing quality separately from re-evaluation quality. Coverage, selective risk, error capture, corrected errors, introduced errors, and net corrections reveal whether the router finds difficult inputs and whether the LLM actually improves them. The final comparison crosses two language models with three re-evaluation protocols on the same routed examples within each dataset.
+\item It develops a task-specific SELF-DISCOVER adaptation \cite{ref18} combining an emotion-oriented module bank, a reusable model-generated procedure, and fixed checks for emotional subject, temporal context, competing label evidence, and alternative interpretations. Its empirical evaluation includes a controlled Mixed Emotion stress test and recorded outputs for inspecting ambiguous or shifting affect. The stress test is explicitly separated from model training and threshold selection.
 \end{itemize}

-The contribution is the integration and evaluation protocol, not a new calibration estimator, language-model architecture, or reasoning method.
+The contribution is the integration and evaluation protocol, together with a task-specific adaptation of structured re-evaluation, not a new calibration estimator or language-model architecture.
+Classifier performance and resource measurements support the first-stage choice; calibration and error concentration assess routing; and the model--protocol comparison measures correction gains against newly introduced errors. The task-level SELF-DISCOVER adaptation is evaluated within this comparison, not assumed superior to label-only prediction.
+

 \section{Literature Review}

@@ -159,13 +167,13 @@

 The current dataset includes seven controlled scenario groups: blended-emotion co-occurrence (40), positive-to-distress shift (40), distress-to-recovery shift (40), neutral framing with subtle affect (40), conflicting cues with a dominant trajectory (40), clear technical or informational Neutral examples (75), and Neutral examples with mild factual or procedural ambiguity (25). Each example was assigned a target label according to the dominant overall emotional trajectory rather than isolated sentiment-bearing phrases. The two Neutral groups preserve the proxy-label definition while avoiding a stress set that is artificially dominated by either completely trivial or clinically suggestive Neutral content. This design is intended to test whether confidence-guided routing and reasoning-based re-evaluation can address cases that are plausibly more difficult for a single-stage classifier.

-Synthetic examples were generated using controlled prompting and checked against inclusion and exclusion criteria. Kang et al. \cite{ref19} used LLM-generated clinical-interview summaries for depression-prediction training augmentation. Here, synthetic posts serve a different purpose: supplementary stress-test evaluation, not training augmentation or clinical validation. Inclusion required consistency with the assigned scenario and target label: mixed or shifting cues for the affective scenarios, and low-affect content for the Neutral controls. Exclusions covered explicit clinical diagnosis claims, treatment recommendations, crisis language, identifying information, and off-scenario content. Appendix Table~A5 records the generation protocol. These checks were not an independent expert-annotation study; results describe the final 300 synthetic examples rather than naturally occurring mixed-emotion performance.
+Synthetic examples were generated using controlled prompting and checked against inclusion and exclusion criteria. Kang et al. \cite{ref19} used LLM-generated clinical-interview summaries for depression-prediction training augmentation. Here, synthetic posts serve a different purpose: supplementary stress-test evaluation, not training augmentation or clinical validation. Inclusion required consistency with the assigned scenario and target label: mixed or shifting cues for the affective scenarios, and low-affect content for the Neutral controls. Exclusions covered explicit clinical diagnosis claims, treatment recommendations, crisis language, identifying information, and off-scenario content. Appendix Table~F1 records the generation protocol. These checks were not an independent expert-annotation study; results describe the final 300 synthetic examples rather than naturally occurring mixed-emotion performance.

 \subsection{Data Partitioning and Model Inputs}

 After preprocessing and label encoding, the primary Reddit dataset is divided into training, model-validation, threshold-calibration, and held-out test sets by independently shuffling and partitioning each class. The final protocol samples exactly 40,000 examples per class (120,000 in total) and assigns 28,000 examples per class to training and 4,000 examples per class to each of validation, calibration, and held-out testing. Thus, the final split sizes are 84,000 training, 12,000 validation, 12,000 calibration, and 12,000 test examples. This per-class construction avoids one-example rounding differences from successive proportion-based split operations.

-The training partition supplies supervised parameter updates; model validation supplies hyperparameter, epoch, and checkpoint selection. The calibration partition supplies temperature fitting and threshold selection. Test labels are excluded from these Phase~1 operations. However, the routed Reddit test cases were reused during Phase~2 prompt development, as disclosed in the experimental limitations. Each classifier uses its own tokenizer on the cleaned title--body field.
+The training partition supplies supervised parameter updates; model validation supplies hyperparameter, epoch, and checkpoint selection. The calibration partition supplies temperature fitting and threshold selection. Test labels are excluded from these Phase~1 operations. However, the routed Reddit test cases were reused during Phase~2 prompt development, so final prompt-configuration comparisons are interpreted as exploratory. Each classifier uses its own tokenizer on the cleaned title--body field.

 Only the cleaned textual content was used as the direct Phase~1 classifier input. For routed Reddit examples, Phase~2 received the linked original title and selftext after minimal privacy-oriented sanitization. Auxiliary metadata, such as subreddit source, author-related metadata, timestamp, and adult-content flag, was not used as a direct predictive feature in either phase.

@@ -182,25 +190,19 @@
 Finally, the framework was not designed to identify, monitor, or intervene with individual users. Any deployment beyond research use would require institutional review, domain-expert oversight, bias and fairness assessment, and compliance with relevant data governance requirements.

 \section{Methodology}
-
-The pipeline has two stages: an operational DistilBERT classifier \cite{ref12} assigns a label and calibrated confidence, and a separately configured LLM re-evaluates low-confidence inputs. Phase~2 uses either Llama~2-7B-Chat \cite{ref14} with Chain-of-Thought prompting \cite{ref17} or Llama~3-8B-Instruct \cite{ref40} with SELF-DISCOVER \cite{ref18}. The configurations are evaluated separately, not ensembled.
-
-\subsection{Confidence-Guided Hybrid Model Architecture}
-
-Development and inference are separate operations. Development fixes the classifier, calibration temperature, and routing threshold. At inference, predictions at or above the threshold retain their Phase~1 labels. Predictions below it receive a Phase~2 label and a stored rationale. Full-set evaluation then recombines the two paths, counting both corrected and newly introduced errors.
-
-Figure~\ref{fig:architecture} summarizes the complete data-to-output workflow. It separates model development and calibration from held-out evaluation, shows the fixed accept-or-route decision, and makes explicit that the two Phase~2 configurations are evaluated separately rather than ensembled.
-
-\begin{figure*}[t]
-\centering
+\subsection{Unified Classifier-to-Re-Evaluation Pipeline}
+The final DistilBERT configuration is selected through the Phase~1 validation-based search. A final training run under that configuration supplies the predictions, logits, calibration temperature, and routed IDs used throughout the downstream experiment. No separately configured operational classifier is substituted after the classifier comparison. Three-seed classifier summaries describe training variability; downstream results describe this single linked final run, not an average of three reasoning pipelines.
+
+At inference, accepted predictions retain their Phase~1 labels. Every routed post is independently evaluated under each of six configurations: Llama~2-7B-Chat or Llama~3-8B-Instruct, each with Direct, CoT, or final SELF-DISCOVER prompting. These are separate alternatives, not an ensemble or a dataset-specific selection rule. All configurations within a dataset use the same Phase~1 outputs and routed IDs. Figure~\ref{fig:architecture} retains the original overview layout; the final configuration details are specified in the text.
+
+\begin{figure*}[t]\centering
 \includegraphics[width=\textwidth]{figures/architecture_figure_publication_final.pdf}
-\caption{Confidence-guided two-phase pipeline. Phase~1 produces a calibrated confidence score using a temperature $T^*$ fitted on the calibration split and applies a fixed routing threshold $\tau^*$. Accepted predictions retain the Phase~1 label, while routed cases are re-evaluated by one separately configured Phase~2 reasoner and then recombined for end-to-end evaluation and audit.}
+\caption{Confidence-guided two-phase pipeline. Accepted predictions retain the Phase 1 label, while routed posts enter re-evaluation. Original artwork is retained in this review copy pending label updates; the final experiment evaluates both models with Direct, CoT, and SELF-DISCOVER as specified in the text.}
 \label{fig:architecture}
 \end{figure*}
-
 \subsection{Phase 1: Initial Emotion Classification}

-The Phase~1 comparison includes DistilBERT \cite{ref12}, Mistral~7B \cite{ref13}, and Llama~2~7B \cite{ref14} under common data partitions and a four-trial search budget. DistilBERT is fully fine-tuned, whereas the 7B backbones use frozen representations with trainable linear heads. Thus, this is a comparison of practical classifier configurations, not an isolated test of architecture or maximum attainable performance. Each selected configuration is evaluated with seeds 42, 43, and 44. Appendix Table~A1c gives the selected settings; only the fixed operational DistilBERT checkpoint supplies downstream routing scores.
+The Phase~1 comparison includes DistilBERT \cite{ref12}, Mistral~7B \cite{ref13}, and Llama~2~7B \cite{ref14} under common data partitions and a four-trial search budget. DistilBERT is fully fine-tuned, whereas the 7B backbones use frozen representations with trainable linear heads. Thus, this is a comparison of practical classifier configurations, not an isolated test of architecture or maximum attainable performance. Each selected configuration is evaluated with seeds 42, 43, and 44. Appendix Table~A1c gives the selected settings; the final DistilBERT run under the selected configuration supplies downstream routing scores.

 \subsection{DistilBERT}

@@ -319,7 +321,7 @@
 |A(\tau)| &\geq m_{\min}.
 \end{align}

-Here, $m_{\min}$ is a minimum accepted-sample requirement used to avoid selecting thresholds supported by too few calibration examples. The implementation sets $m_{\min}=\lceil0.10|D_{cal}|\rceil=1{,}200$. The selected $\tau=0.70$ accepted 11,830 calibration examples, so this safeguard was satisfied but did not determine the selected operating point. In the final DistilBERT operating policy, candidates are swept from 0.70 to 1.00 in increments of 0.01. This lower bound was chosen before held-out testing to avoid treating a weak three-class majority probability near 0.50 as sufficiently reliable for direct acceptance. Unless otherwise stated, the primary operating point uses temperature-scaled maximum softmax probability, a target selective risk of $\alpha=0.05$, a one-sided risk confidence level of $1-\delta=0.95$, the acceptance rule $c_i \geq \tau$, and the routing rule $c_i < \tau$. The risk constraint is a feasibility condition: it excludes candidate thresholds whose accepted-set upper risk exceeds $\alpha$. Among the feasible candidates in the prespecified 0.70--1.00 grid, the candidate with the greatest Phase~1 coverage is selected. Thus, the completed DistilBERT value $\tau^*=0.70$ reflects both the prespecified conservative operating range and calibration-set risk--coverage evaluation; it was not optimized using held-out test results or Phase~2 outcomes. The 7B models serve as Phase~1 predictive comparators only; they do not define alternative routing policies in the reported end-to-end experiment.
+Here, $m_{\min}$ is a minimum accepted-sample requirement used to avoid selecting thresholds supported by too few calibration examples. The implementation sets $m_{\min}=\lceil0.10|D_{cal}|\rceil=1{,}200$. The selected $\tau=0.70$ accepted 11,807 calibration examples, so this safeguard was satisfied but did not determine the selected operating point. In the final DistilBERT operating policy, candidates are swept from 0.70 to 1.00 in increments of 0.01. This lower bound was chosen before held-out testing to avoid treating a weak three-class majority probability near 0.50 as sufficiently reliable for direct acceptance. Unless otherwise stated, the primary operating point uses temperature-scaled maximum softmax probability, a target selective risk of $\alpha=0.05$, a one-sided risk confidence level of $1-\delta=0.95$, the acceptance rule $c_i \geq \tau$, and the routing rule $c_i < \tau$. The risk constraint is a feasibility condition: it excludes candidate thresholds whose accepted-set upper risk exceeds $\alpha$. Among the feasible candidates in the prespecified 0.70--1.00 grid, the candidate with the greatest Phase~1 coverage is selected. Thus, the completed DistilBERT value $\tau^*=0.70$ reflects both the prespecified conservative operating range and calibration-set risk--coverage evaluation; it was not optimized using held-out test results or Phase~2 outcomes. The 7B models serve as Phase~1 predictive comparators only; they do not define alternative routing policies in the reported end-to-end experiment.

 The threshold-selection output records accepted accuracy, routed Phase~1 errors, error capture, routing precision, and proxy-Depression false-negative risk among accepted predictions. Additional confidence diagnostics include expected calibration error (ECE), adaptive ECE, Brier score, and negative log-likelihood before and after temperature scaling. Auxiliary analyses compare raw MSP, temperature-scaled MSP, entropy-based certainty, and probability margin; estimate bootstrap confidence intervals for selective metrics; assess threshold stability through calibration-set resampling; and extract high-confidence accepted errors for qualitative auditing.

@@ -340,7 +342,7 @@
     \State Compute coverage, routing rate, and risk bound
 \EndFor
 \State Retain candidates with $\overline{R}_{\delta}(\tau)\leq\alpha$ and $|A(\tau)|\geq m_{\min}$
-\State If none remain, report infeasibility and stop
+\State If none remain, flag infeasibility; a fallback is not risk-feasible
 \State Choose a retained $\tau$ with maximum coverage
 \State Fix $T^*$ and $\tau^*$ before held-out testing
 \end{algorithmic}
@@ -352,7 +354,7 @@
 \footnotesize
 \begin{algorithmic}[1]
 \State \textbf{Input:} post record $x$, classifier $f_{\theta}$, fixed $T^*$ and $\tau^*$
-\State \textbf{Input:} Phase~2 reasoner $g$ (Llama~2 CoT or Llama~3 SELF-DISCOVER)
+\State \textbf{Input:} Phase~2 configuration $g$ (model and prompting protocol)
 \State Obtain the Phase~1 input $x^{(1)}$ from record $x$
 \State Compute $\mathbf{z}=f_{\theta}(x^{(1)})$ and probabilities $q_k(T^*)$
 \State Set $\hat{y}^{(1)}=\arg\max_k q_k(T^*)$ and $c=\max_k q_k(T^*)$
@@ -360,9 +362,10 @@
     \State \textbf{return} $\hat{y}^{(1)}$ as the accepted Phase~1 final label
 \Else
     \State Retrieve the minimally sanitized original text $x^{(2)}$
-    \State Generate rationale and terminal label with $g(x^{(2)},\hat{y}^{(1)})$
+    \State Generate a label with $g$ using its prescribed inputs
     \State Parse exactly one label from \{Depression, Neutral, Happy\}
-    \State \textbf{return} the parsed Phase~2 final label
+    \State If parsing fails, retain $\hat{y}^{(1)}$ and record failure
+    \State \textbf{return} the parsed label or recorded fallback
 \EndIf
 \end{algorithmic}
 \end{algorithm}
@@ -375,7 +378,7 @@
 \State \textbf{Input:} reference labels $y_i$, Phase~1 labels $\hat{y}^{(1)}_i$, route indicators $r_i$
 \State \textbf{Input:} parsed Phase~2 labels $\hat{y}^{(2)}_i$ for routed inputs
 \For{each example $i$}
-    \State $\hat{y}^{(E)}_i \gets \hat{y}^{(2)}_i$ if $r_i=1$; otherwise $\hat{y}^{(E)}_i \gets \hat{y}^{(1)}_i$
+    \State $\hat{y}^{(E)}_i \gets \hat{y}^{(2)}_i$ if $r_i=1$ and parsing succeeds; otherwise $\hat{y}^{(E)}_i \gets \hat{y}^{(1)}_i$
 \EndFor
 \State $C\gets\sum_i\mathbf{1}[r_i=1,\hat{y}^{(1)}_i\ne y_i,\hat{y}^{(E)}_i=y_i]$
 \State $I\gets\sum_i\mathbf{1}[r_i=1,\hat{y}^{(1)}_i=y_i,\hat{y}^{(E)}_i\ne y_i]$
@@ -386,91 +389,75 @@

 Algorithm~\ref{alg:end-to-end-audit} formalizes the evaluation boundary between routed-subset behavior and full-set performance. In particular, a Phase~2 model is not credited merely for correcting selected errors: any newly introduced errors are deducted, and the final accuracy is recomputed over the complete evaluation set after accepted Phase~1 labels and routed Phase~2 labels are recombined.

-The implementation records an explicit infeasibility status if no candidate satisfies the risk constraint. A minimum-risk fallback exists only for notebook smoke tests and was not invoked in the reported paper-scale experiment; no fallback threshold is treated as satisfying the risk-control constraint. The selected threshold, confidence method, temperature, risk-control method, candidate-generation rule, acceptance boundary, tie-breaking rule, split ratios, random seed, feasibility status, and final selective metrics are saved in \texttt{threshold\_provenance.json}. This provenance records the fixed operating decision for reproducibility; it does not by itself establish independence from earlier development choices.
+The export notebook prints a warning and falls back to the lowest empirical risk, then highest coverage, if no candidate satisfies the constraint. That branch was not needed here: the selected calibration point satisfies the prescribed upper-bound criterion. A fallback, if invoked, would not be interpreted as a risk-feasible policy. The temperature, threshold, risk target, confidence level, and candidate grid are stored with the calibration and threshold tables. The model source and seed are recorded separately in the execution environment artifact.

 \subsection{Relationship to Mixed-Emotion Evaluation}

 The routing mechanism is entirely confidence-based during inference. It does not directly detect blended emotion or sentiment shift. Instead, the Mixed Emotion Dataset serves a complementary evaluation role by stress-testing whether the Phase 1 classifier and the Phase 2 reasoning stage behave more reliably on examples where the one-label-per-input assumption is more difficult. This separation keeps the routing rule reproducible while allowing the evaluation to probe emotionally complex cases.

-\subsection{Phase 2: Reasoning-Based Re-Evaluation with Llama 2 and Llama 3}
-
-Phase~2 adds a reasoning step to re-evaluate and explain predictions that were flagged as uncertain or potentially incorrect in Phase~1. LLMs are deployed in a zero-shot prompting setup to perform this secondary analysis. Specifically, Llama~2-7B-Chat, referred to as Llama~2, is paired with Chain-of-Thought prompting \cite{ref17}, and Llama~3-8B-Instruct, referred to as Llama~3, is paired with SELF-DISCOVER reasoning \cite{ref18}. These models are not fine-tuned Phase~1 classifiers. Instead, they serve as secondary evaluators that consider the content of a post along with the initial model outcome, then provide a rationale and, when appropriate, a revised prediction. To preserve the evidence needed for this task, routed Reddit posts are linked back to their original title and selftext. The reasoning input replaces URLs and direct username patterns but retains sentence order, punctuation, capitalization, negation, and temporal transitions. The two prompting protocols use different intermediate reasoning formats, but both are constrained to end with exactly one explicit final label in the same three-class space: \texttt{Final label: Depression}, \texttt{Final label: Neutral}, or \texttt{Final label: Happy}. Thus, Phase~2 comparisons are performed at the final-label decision level rather than at the raw-output level. End-to-end correction results are reported only after the routed-sample runs are complete.
-
-\subsection{Llama 2 with Chain-of-Thought Prompting}
-
-Llama 2-7B-Chat, a chat-optimized variant of Llama 2, was used to produce a structured re-evaluation record for routed examples. Our zero-shot, multistage protocol follows the CoT idea of eliciting intermediate reasoning, rather than reproducing the few-shot demonstrations evaluated by Wei et al. \cite{ref17}. It elicits an independent assessment, comparison with Phase~1, and a terminal class label. The retained text is inspectable, but its presence alone is not evidence that the rationale is faithful or clinically valid.
-
-For the final Llama~2 configuration, the CoT protocol first requires an independent, text-grounded assessment before revealing the Phase~1 label. It then compares the two decisions and ends with one exact label from Depression, Neutral, and Happy. The protocol explicitly avoids numerical emotion percentages, because a largest percentage component can conflict with the qualitative final decision. The complete reported prompting structure is detailed in Appendix Table A2.
-
-The revised CoT protocol preserves a text-grounded rationale while avoiding the conflicting numerical-breakdown rule used in earlier exploratory prompting. Phase~2 output quality and correction behavior were evaluated empirically on the fixed routed subsets rather than inferred from prompt design alone.
-
-\subsection{Llama 3 with SELF-DISCOVER}
-
-The protocol is summarized visually in Appendix Figure~C1; the figure is provided as a reference aid rather than as a new model contribution.
-
-We use an input-specific adaptation of SELF-DISCOVER \cite{ref18} with Llama~3-8B-Instruct, abbreviated as Llama~3 SELF-DISCOVER below. The original method discovers a task-level plan from unlabeled task examples and reuses it across instances. Our implementation instead constructs and stores a plan for each routed post. Its prompt contains the 39-module reasoning pool: prompt-level instructions, not neural-network layers or separately trained components. For each post, \texttt{SELECT} identifies potentially useful instructions, \texttt{ADAPT} tailors them to the classification problem, and \texttt{IMPLEMENT} composes a structured plan, generally represented in JSON. The model then applies the plan and returns an assessment ending in one permitted terminal label. Appendix Tables~A3--A3c document this adaptation and its prompt-development diagnostics.
-
-For the observed case \texttt{MEV2\_DEP\_022}, Phase~1 predicted Happy for a post that begins with encouraging feedback but ends with unresolved sadness. The stored \texttt{SELECT} trace names Critical Thinking, Textual Analysis, Risk Analysis, and Reflective Thinking. The subsequent artifacts adapt these instructions and organize them into a JSON plan. The terminal response cites the later sadness and returns \texttt{Final label: Depression}. Appendix Table~A4b provides excerpts. This record shows the model's generated justification; it does not establish which cues caused either model's prediction.
-
-The module references in these fields are generated reasoning traces. They may paraphrase or group the canonical module wording, so module identifiers are retained for auditability but are not themselves treated as quantitative predictions. Quantitative evaluation uses only the parsed terminal label, whereas corrected, introduced, and net errors determine whether the longer reasoning process improves classification.
-
-\subsection{Model-Specific Prompt Protocols and Final Input Policy}
-
-Exploratory fixed-subset diagnostics showed that a shared cross-model prompt did not benefit both reasoners. The final Llama~2 protocol therefore uses a frozen CoT prompt, which requires independent assessment before Phase~1-label comparison and one exact terminal label, while Llama~3 uses the SELF-DISCOVER protocol. These model-specific protocols were frozen before the final original-text evaluation. Appendix Table~A3c reports the earlier cleaned-input development diagnostics separately from the final end-to-end results so that prompt development is not confused with the final input-policy evaluation.
-
-For the final Reddit run, the 171 routed records were linked to retained original title--selftext pairs by normalized exact matching of the cleaned Phase~1 field and proxy label. All 171 records were matched, no key mapped to conflicting original texts, and only one post exceeded the 6,000-character reasoning limit; for that post, the beginning and ending were preserved with an explicit middle-omission marker. This linkage audit was completed before LLM loading and is exported with the result package. The final policy therefore changes only the text representation supplied to Phase~2, not the Phase~1 model, temperature, threshold, routed IDs, LLM checkpoints, prompts, or generation settings.
-
-\subsection{Operational Inference Paths}
-
-Only routed inputs undergo Phase~2 inference. Accepted labels remain unchanged even if they are wrong; routed labels are replaced by the parsed terminal output even if that replacement introduces an error. The final runs produced valid labels for all routed records. Reference labels are used for evaluation, not provided to the reasoner. Selecting fewer inputs reduces the number of posts re-evaluated, but does not by itself quantify latency or monetary savings.
-
-\subsection{Reasoning Outputs as Audit Artifacts}
-
-A key benefit of the hybrid approach is that it adds an inspectable rationale layer to routed predictions. For each routed input, the final output is accompanied by a natural-language explanation generated by Llama~2 or Llama~3 that identifies the textual cues used to support the terminal label. This improves traceability, although a generated rationale should not be assumed to be faithful or clinically valid without independent review.
-
-For example, if a routed post is assigned Depression, the explanation can identify expressions of persistent sadness, hopelessness, or unresolved emotional decline and contrast them with any isolated positive cues. Domain experts can then examine whether the cited evidence supports the final class. The explanation is therefore treated as an audit artifact, not as proof that the prediction is correct. Retaining the full reasoning fields, parsed label, initial prediction, and reference label enables both quantitative error accounting and qualitative review of failure modes.
-
+\begin{figure*}[t]
+\centering
+\includegraphics[width=\textwidth]{figures/sd_task_level_overview.pdf}
+\caption{Task-specific SELF-DISCOVER design within selective re-evaluation. (a) Before post classification, the selected LLM constructs a model-specific procedure from researcher-authored task materials; the code saves it for reuse. (b) Each routed original post is evaluated anew using that procedure and fixed subject, time-frame, evidence, and alternative-interpretation checks. The dashed arrow denotes procedure reuse, not answer reuse. Discovery is skipped when a retained plan exists. Appendix~\ref{app:sd-templates} provides the detailed workflow and exact prompts; label parsing and Phase~1 fallback follow Algorithm~\ref{alg:sd-final}.}
+\label{fig:sd-task-level}
+\end{figure*}
+
+\subsection{Phase 2: Factorial Model and Protocol Comparison}
+Direct prompting requests one final label without a rationale and includes the Phase~1 prediction. CoT first requests an independent assessment, subsequently discloses the Phase~1 label for comparison, and finally requests a terminal label. The implemented CoT dialogue contains five generation calls, including the initial acknowledgment and text-response turns. Final SELF-DISCOVER withholds the Phase~1 label throughout. These are complete prompting protocols: their comparison does not isolate rationale length alone, because label disclosure and call structure also differ.
+
+\subsection{Task-Level SELF-DISCOVER}
+SELF-DISCOVER separates \emph{constructing a procedure} from \emph{applying it}, as summarized in Figure~\ref{fig:sd-task-level}. Each model generates a plan for the three-class task, stores it, and reuses it across routed posts. This \emph{cached plan} is a saved procedure, not stored post-level answers. Every post still requires a new model response.
+
+\textit{Task-specific design.} We retain the SELECT--ADAPT--IMPLEMENT framework of SELF-DISCOVER \cite{ref18} and specialize its inputs and execution guidance to proxy emotion classification. Researchers supply the class policy, 18 emotion-oriented reasoning modules, and fixed evidence-checking instructions; the LLM generates the procedure from those materials. The adaptation addresses distinctions that isolated affect words cannot resolve: whose emotion is expressed, whether distress is current or retrospectively resolved, and whether positive wording describes an actual state rather than a wish. The contribution is this task-specific specification and its integration with selective re-evaluation, not the invention of task-level planning or caching.
+
+\textit{Who constructs the plan, and when.} Before classifying a routed post, the code checks for a retained plan for the selected LLM. If none exists, that LLM receives only the researcher-authored task description, policy, and module bank. It does not receive the first post, any other evaluation post, a reference label, or a Phase~1 prediction during discovery. SELECT chooses relevant modules; ADAPT specializes them to the task; IMPLEMENT generates the reusable procedure, which the code stores. If a plan already exists, all three discovery calls are skipped. Llama~2 and Llama~3 generate separate plans; each model reuses its own plan across Reddit and Mixed Emotion.
+
+IMPLEMENT requests at most three one-line JSON steps, each returning to the original text and deferring label selection until the end. This format is requested, not enforced; the realized plans are reproduced in Appendix~\ref{app:sd-templates}. Access to the post is an instruction for subsequent execution, not an input to plan construction.
+
+\textit{Per-post execution.} The prompt combines the plan, policy, original post, and fixed checks. It asks whose emotion is expressed, whether it is current or retrospectively resolved, and which quotations support or oppose each label. A counterfactual check asks what evidence would favor the second-best label and whether it is present. The model emits a final classification line. SD never receives the Phase~1 label; the evaluator retains it as a fallback if parsing fails.
+
+This division keeps the procedure shared while requiring fresh, post-specific evidence and a new decision for every routed input. The evidence ledger and counterfactual check are researcher-authored execution instructions, not automatically discovered properties of the cached plan. Generated explanations do not establish faithful internal reasoning. Appendix~\ref{app:sd-templates} provides the workflow, templates, and actual plans; Appendix~\ref{app:sd-variants} compares the earlier per-post and final task-level designs without attributing their differences to any single component.
+
+\begin{algorithm}[t]
+\caption{Final task-level SELF-DISCOVER re-evaluation}
+\label{alg:sd-final}\footnotesize
+\begin{algorithmic}[1]
+\State \textbf{Input:} model $g$; researcher-authored task, policy, modules, execution guidance; routed posts
+\If{a retained plan for $g$ exists}
+\State Load that plan; skip discovery
+\Else
+\State Give $g$ task materials only; no post text or labels
+\State $g$: SELECT relevant modules and ADAPT them to the task
+\State $g$: IMPLEMENT the procedure; store it as the plan
+\EndIf
+\For{each routed post $x$}
+\State Execute the plan on the original post, without the Phase~1 label
+\State Request source quotes, three-label evidence, and a counterfactual check
+\State Parse the final label; retain Phase~1 if parsing fails
+\State Store the response, parsed status, and final prediction
+\EndFor
+\end{algorithmic}\end{algorithm}
+
+\subsection{Input and Output Policy}
+Reddit Phase~1 uses cleaned title--body text; Phase~2 uses the linked, minimally sanitized original title and selftext. The original-text policy preserves punctuation, negation, and discourse structure but changes the input representation as well as the prediction mechanism. Improvements are therefore attributed to the complete re-evaluation configuration, not solely to reasoning. Long posts follow the notebook's 6,000-character input policy. Mixed Emotion uses the complete synthetic text rather than a separately lemmatized version.
+
+The three permitted outputs are Depression, Neutral, and Happy. The parser prioritizes the terminal label marker and records unparseable responses. Such responses retain the Phase~1 prediction in the end-to-end evaluation; parsing failure is not automatically counted as a classification error. The remaining final SELF-DISCOVER failures are not regenerated or assigned labels from the reference answer. Label normalization and any recovery of an explicit answer must be applied without consulting the reference label.
 \section{Experimental Setup}
-
-\subsection{Computing Environment}
-
-The experiments were implemented in Python with PyTorch, Hugging Face Transformers, Datasets, scikit-learn, SciPy, and pandas; W\&B recorded the model-selection sweeps. Phase~1 used \texttt{distilbert-base-uncased} as the operational checkpoint and prespecified 7B base checkpoints for the Mistral and Llama~2 comparison classifiers. Routed Phase~2 inference used \texttt{NousResearch/Llama-2-7b-chat-hf} and \texttt{NousResearch/Meta-Llama-3-8B-Instruct}, loaded in 4-bit form with bitsandbytes in GPU-backed Google Colab sessions. Llama~2 CoT generation used a 256-token limit with sampling temperature 0.6 and top-$p$ 0.9; the Llama~3 SELF-DISCOVER structure stages used a 1,024-token limit and the terminal answer was decoded deterministically. The repository notebooks print installed package versions at runtime and save prediction-level outputs after every routed row.
-
-The three Phase~1 comparison configurations were trained and profiled on NVIDIA A100-SXM4-40GB accelerators in the same software environment. The inference benchmark used fp16, a 256-token cap, and batch size 32 over the 12,000-post held-out split; single-post latency was measured separately on 512 posts. Three warm-up batches were discarded, and each benchmark was repeated three times. CUDA synchronization bracketed timed regions, which included host-to-device transfers. NVML sampled GPU utilization and board power at 1~Hz. Appendix Table~A1d reports the measurement protocol and stage-level resource use. The additional 9,000-post evaluation ran on an NVIDIA L4 using the model-specific inference protocols described above.
-
-\subsection{Model Selection and Calibration Protocol}
-
-Model selection uses the training and validation partitions described above. Temperature fitting and threshold selection use the separate calibration partition after the operational checkpoint is fixed. Neither operation updates the classifier weights. Calibration diagnostics are measured on the temperature-fitting partition and therefore describe calibration fit, not independent evidence of calibration generalization. Routing performance is evaluated separately on the test partition.
-
-\subsection{Confidence Analysis and Provenance Outputs}
-
-For reproducibility, the advanced confidence workflow exports temperature-scaling metadata, calibration-quality metrics, threshold sweeps for each confidence score, bootstrap confidence intervals, threshold-stability summaries, high-confidence accepted errors, per-class selective metrics, fixed-threshold Mixed Emotion Dataset results, and Phase~2 end-to-end summaries. The results in this manuscript were derived from the saved \texttt{threshold\_provenance.json} file and prediction-level CSV files rather than manually copied notebook output. Phase~2 evaluation reports both corrected Phase~1 errors and introduced errors because a reasoner can improve some routed cases while worsening others.
-
-\subsection{Hyperparameter Tuning and Training Optimization}
-
-Each W\&B Bayesian sweep comprised four trials maximizing validation macro F1. The shared candidates were batch sizes 16, 32, and 64; training budgets of two or three epochs; and weight decays $10^{-2}$, $10^{-3}$, and $10^{-4}$. Learning-rate ranges differed for encoder fine-tuning and frozen-backbone probing. The selected configuration was rerun with seeds 42, 43, and 44 using the training partition, with validation macro F1 determining the retained epoch. A selected three-epoch budget therefore does not demonstrate convergence beyond that search limit. Appendix Table~A1c reports the sweep-selected settings and, in its note, the fixed operational checkpoint used for downstream analyses.
-
-\subsection{Loss Function and Optimizer}
-
-The classifiers minimize three-class cross-entropy using AdamW and gradient clipping. Updates apply to all DistilBERT parameters or to the linear head alone for the frozen 7B models.
-
-\subsection{Evaluation Metrics}
-
-The metrics were organized by the research question they answer rather than treated as interchangeable indicators. For RQ1a, predictive performance was evaluated with accuracy and class-wise precision, recall, and F1, with unweighted macro averages reported across Depression, Neutral, and Happy; calibration was evaluated with NLL, Brier score, ECE, and adaptive ECE. For RQ1b, matched classifier performance is reported as mean $\pm$ standard deviation across three seeds and interpreted together with model parameter scale, trainable scope, and the common-hardware inference benchmark. Timing variability is reported separately as mean $\pm$ SD across three measurement repetitions. For RQ2, routing was evaluated with coverage, routing rate, accepted-set risk and its one-sided upper bound, error capture, and routing precision. For RQ3, the primary end-to-end outcome was the example-level change in correctness after accepted Phase~1 predictions were recombined with routed Phase~2 labels. This change was summarized by full-set accuracy change and by corrected, introduced, and net errors. Calibration and routing measures are supporting diagnostics: they establish whether the confidence policy is credible and whether it concentrates difficult cases, but they do not by themselves establish that Phase~2 improves the final system.
-
-Phase~1 and end-to-end correctness are paired for each example. Two-sided exact McNemar tests use the discordant counts: corrected Phase~1 errors and newly introduced errors. For the original Reddit and Mixed Emotion evaluations, paired percentile-bootstrap 95\% confidence intervals use 50,000 row-level resamples and seed 20260813, with Holm adjustment across the four dataset--reasoner comparisons. The additional holdout uses 10,000 paired correctness resamples and seed 20260911, with its two prespecified comparisons against Phase~1 treated as a separate Holm family. These analyses characterize the observed prediction pairs conditional on the fitted models and selected prompts; they do not account for prompt selection, repeated LLM generation, or dependence between posts from the same author. The three-seed intervals in the classifier comparison instead describe variation across training runs on a common test set.
-
-Residual accepted-set errors were audited separately using the fixed 12,000-example held-out protocol and the calibration-selected threshold. The audit reports accepted selective risk, reference-to-prediction transitions, confidence-band error rates, and Depression false-negative risk among accepted reference-Depression posts. For all 310 accepted errors, the stored model input was linked back to the retained Reddit title and selftext by normalized exact matching against \texttt{title\_with\_selftext\_cleaned}; all 310 rows were matched. Six high-confidence cases were then selected because the original post itself provides comparatively clear evidence for the stored proxy label and exposes a distinct error pattern. These cases illustrate failure modes rather than estimate their prevalence. Appendix Tables~A4e--A4f report the reference label, prediction, calibrated confidence, minimally sanitized original-post excerpt, and text-grounded audit rationale. Qualitative review does not overwrite any reference label or alter any reported metric.
-
-\subsection{Additional Same-Source Holdout Protocol}
-\label{sec:additional-holdout-protocol}
-
-After freezing the model-specific prompts and input policy, we formed an additional balanced holdout of 9,000 posts from unused records in the same source corpus. We excluded text matching any post in the previously used 120,000-post sample after Unicode NFKC normalization, case folding, and whitespace normalization, and removed within-pool normalized-text duplicates. Sampling used seed 20260911 and retained 3,000 posts per class. No selected text overlapped the original sample under this matching rule. This is a same-source holdout, not an author-disjoint, temporal, or external-domain test.
-
-The saved operational DistilBERT checkpoint, temperature $T^*=1.7706$, threshold $\tau^*=0.70$, and model-specific prompt and input policies were retained without fitting or selection on these 9,000 posts. Both reasoners processed the same routed IDs in separate configurations. Accepted predictions remained unchanged, and any unparseable routed output would be counted as incorrect rather than silently replaced by Phase~1. Evaluation was a single fixed run per configuration, not a new three-seed training comparison. Detailed execution settings are retained in the reproducibility records.
-
+\subsection{Classifier Selection and Calibration}
+All classifier configurations use the common 84,000/12,000/12,000/12,000 train/validation/calibration/test partitions, a maximum length of 256 tokens, and a four-trial Bayesian search maximizing validation macro F1. The selected DistilBERT settings are learning rate $4.0368\times10^{-5}$, batch size 64, an epoch budget of 3, and weight decay $10^{-3}$. The epoch budget is a selected training hyperparameter, not a statement that every seed's best validation checkpoint occurred at that epoch. Seeds 42, 43, and 44 characterize classifier variation; the final downstream run uses seed 42. Appendix Table~A1c records the selected settings for all three classifiers.
+
+The downstream export loads the saved reference seed-42 model from this selected configuration; it does not perform a new training run inside the export notebook. Its environment record identifies fp32 evaluation, a 256-token maximum input length, and the same selected hyperparameters. The final model is calibrated with $T^*=1.480195$ and routed at $\tau^*=0.70$. The calibration partition has 11,807 accepted and 193 routed posts, accepted-set empirical risk 1.9395\%, and one-sided upper bound 2.1615\%, below the 5\% target. These calibration counts are distinct from the 218 routed Reddit test posts. No Mixed Emotion labels are used for temperature fitting or threshold selection.
+
+\subsection{Execution and Resource Scope}
+The retained Phase~1 benchmark compares the selected classifier configurations on NVIDIA A100-SXM4-40GB hardware, with fp16 inference, a 256-token cap, and batch size 32. Three timing repetitions are distinct from the three training seeds. These classifier-stage measurements are not measurements of the complete final reasoning pipeline; Appendix Table~A1d retains their original measurement scope.
+
+Phase~2 uses chat-tuned Llama~2~7B and instruction-tuned Llama~3~8B with quantized loading and greedy decoding. Direct and final CoT request up to 1,024 new tokens per call. Final SELF-DISCOVER requests up to 1,024 for Llama~2 and 1,536 for Llama~3 in its per-post execution call. Thus, neither total generation cost nor token budget is claimed to be equal across all protocols. Llama~2's shorter context also constrains multi-turn inference. Template text and nominal execution budgets are disclosed in the appendices; wall-clock or monetary superiority of the final Phase~2 configurations is not inferred from accuracy.
+
+\subsection{Evaluation and Statistical Scope}
+Accuracy and macro F1 use all examples, after recombining accepted and routed predictions. Corrected errors $C$ are Phase~1 mistakes made correct by re-evaluation; introduced errors $I$ are previously correct predictions made wrong. The full-set accuracy change in percentage points is $100(C-I)/N$. Wrong-to-wrong changes are not corrections. Routed accuracy, error capture, and accepted-set risk measure different aspects of the pipeline.
+
+Appendix~\ref{app:statistics} reports count-based paired accuracy summaries. For a comparison with Phase~1, the discordant pairs are exactly $C$ and $I$, so the two-sided exact McNemar test is a binomial test with $C+I$ trials and probability one half. Paired percentile-bootstrap intervals for accuracy change are reconstructed from the three correctness-difference categories $+1,-1,0$, with counts $C,I,N-C-I$, using 50,000 multinomial resamples. This is equivalent to a paired row bootstrap for that scalar accuracy difference. It does not reconstruct class-wise metrics or paired comparisons between two re-evaluators. Holm correction is applied across the 12 final configuration-versus-Phase~1 tests. These intervals and tests describe the reported final counts; prompt development and selection make them exploratory rather than selection-adjusted confirmation of the winning prompt.
 \section{Experimental Results}
-
 \subsection{Phase 1 Performance on Reddit Data}

 The matched Phase~1 comparison used the same class-balanced 120,000-post sample, prespecified 70/10/10/10 partitions, 12,000-example held-out test set, maximum input length of 256 tokens, four-trial Bayesian search budget, and random seeds 42, 43, and 44 for all three architectures. Table~\ref{tab:phase1-performance} reports predictive performance and inference efficiency for these configurations; training configurations are documented in Appendix Table~A1c.
@@ -520,266 +507,137 @@

 DistilBERT had the highest mean score and approximately 67 million parameters, compared with about seven billion for each comparator. These results support retaining it for the tested pipeline, without establishing superiority over fully adapted 7B models. Depression--Happy confusions accounted for 72.8\% of DistilBERT errors, 75.6\% of Llama~2 errors, and 74.3\% of Mistral errors across pooled seed predictions; Neutral F1 exceeded 98\% for all three. Pooling summarizes error composition, not 36,000 independent test examples. Small probe seed deviations also reflect the frozen backbone and should not be interpreted as equivalent training variability across adaptation methods.

-The matched comparison and the downstream experiment serve different purposes. The former summarizes three-seed classifier performance; the latter retains the independently fixed operational DistilBERT checkpoint with 96.69\% accuracy. Its predictions, temperature, and routed IDs were not replaced after the comparison. All calibration, routing, and end-to-end results refer to that single operational checkpoint.
-
-\textit{Phase~1 inference efficiency.} Table~\ref{tab:phase1-performance}(b) adds a measured computational basis for this choice. On the common A100 benchmark, DistilBERT processed 4,959.5 posts/s, giving 89.4- and 95.9-fold higher batched throughput than the Llama~2 and Mistral probes. Its median single-post latency was 4.80~ms, compared with 34.05 and 36.62~ms, and its peak inference memory was 0.29~GB versus 14.05 and 15.22~GB. These are classifier inference measurements, not speedups of the complete two-phase system. Because Phase~1 processes every input, its low per-post cost complements selective use of the more expensive reasoners. The benchmark profiles the matched comparison configurations; it does not replace or remeasure the fixed downstream checkpoint.
-
-The one-off classifier build costs were much closer: 0.59, 0.68, and 0.70 GPU-hours for DistilBERT, Llama~2, and Mistral, respectively. The probe totals include 0.64--0.67 GPU-hours of backbone feature extraction before head fitting. Thus, the main computational advantage of DistilBERT in this comparison is repeated inference rather than a large reduction in model-development cost. Appendix Table~A1d separates these stages and reports the repeated latency measurements.
-
-The operational checkpoint's calibration diagnostics are reported next. Table~\ref{tab:calibration-quality} reports the completed DistilBERT results before and after temperature scaling. NLL evaluates the probability assigned to the true class and strongly penalizes confident mistakes; the Brier score evaluates the full three-class probability vector; and ECE summarizes the gap between confidence and observed accuracy across confidence groups. On the calibration partition, NLL fell from 0.1411 to 0.1041, Brier score from 0.0559 to 0.0514, fixed-width ECE from 0.0244 to 0.0083, and adaptive ECE from 0.0244 to 0.0138.
-
-\begin{table*}[t]
-\centering
-\caption{DistilBERT calibration quality before and after temperature scaling}
-\label{tab:calibration-quality}
-\footnotesize
-\setlength{\tabcolsep}{4pt}
-\renewcommand{\arraystretch}{1.18}
-\begin{tabularx}{\textwidth}{@{}l c c c c c@{}}
-\toprule
-Confidence score & Temperature & NLL & Brier & ECE & Adaptive ECE \\
-\midrule
-Raw MSP & 1.0000 & 0.1411 & 0.0559 & 0.0244 & 0.0244 \\
-Temperature-scaled MSP & 1.7706 & 0.1041 & 0.0514 & 0.0083 & 0.0138 \\
-\bottomrule
-\end{tabularx}
-\end{table*}
-
-Figure~\ref{fig:calibration-routing} connects the three calibration decisions that support the operating policy. Panel (a) shows how the observed-accuracy curve moves toward the ideal confidence--accuracy diagonal after scaling; the vertical segments visualize the bin-wise calibration gaps summarized by ECE. Panel (b) places NLL, Brier score, and ECE on a common relative scale while retaining their original values in the annotations. Panel (c) then shows the risk--coverage behavior used to characterize the fixed routing policy.
-
+
+The final linked DistilBERT run achieves 96.9167\% accuracy and 96.9166\% macro F1 on Reddit. This run uses the same selected hyperparameter configuration as the three-seed comparison. Its calibration and routing outputs, rather than those of a separately configured model, define every downstream condition.
+\subsection{Class-Level Error Structure}
+Figure~\ref{fig:phase1-confusion} shows the final linked classifier's errors on both datasets. Cell counts and row-normalized percentages distinguish the balanced Reddit test from the smaller stress test. These are the final Phase~1 predictions, not pooled three-seed counts or re-evaluation outputs.
 \begin{figure*}[t]
 \centering
-\includegraphics[width=\textwidth]{figures/figure_a_calibration_and_routing.pdf}
-\caption{DistilBERT calibration and selective-routing diagnostics. (a) Reliability before and after temperature scaling; ECE is the sample-weighted mean absolute gap between confidence and observed accuracy across bins, not a single plotted coordinate. (b) Raw and scaled NLL, Brier score, and ECE shown relative to each raw value (100\%); lower values indicate better calibration. (c) Risk--coverage curve with the fixed $\tau^*=0.70$ operating point.}
-\label{fig:calibration-routing}
+\includegraphics[width=0.96\textwidth]{figures/phase1_confusion.pdf}
+\caption{Final Phase 1 confusion matrices on Reddit ($N=12{,}000$) and Mixed Emotion ($N=300$). Rows are reference labels; columns are predictions. Each cell shows a count and its percentage within the reference class. Both panels use the same row-normalized color scale.}
+\label{fig:phase1-confusion}
 \end{figure*}

-Using the calibrated confidence scores, candidates from 0.70 to 1.00 are swept on the dedicated calibration split and evaluated with the risk-coverage rule described in the Methodology section. The primary DistilBERT operating policy uses $T^*=1.7706$, $\alpha=0.05$, and $\tau^*=0.70$. The calibration-side upper-risk bound at this operating point was 2.82\%, satisfying the prespecified feasibility condition. Because $\tau=0.70$ was the lowest feasible candidate in the prespecified grid, it retained the greatest Phase~1 coverage; it was not chosen by examining held-out or Phase~2 outcomes.
-
-Table~\ref{tab:routing-policy} connects the calibration-selected operating point to its held-out routing behavior. The final original-text Phase~2 experiment uses this prespecified policy at $\tau^*=0.70$; no threshold is chosen or revised using Phase~2 accuracy.
-
+\subsection{Calibration and Routing}
+Temperature scaling reduces all four calibration diagnostics (Table~\ref{tab:calibration-final}). The fixed test router selects 218 of 12,000 Reddit posts and 86 of 300 Mixed Emotion examples. The difference in routing volume is an observed response to the stress-test inputs, not a dataset-specific threshold adjustment.
 \begin{table*}[t]
-\centering
-\caption{Calibration-selected primary routing policy and held-out error concentration}
-\label{tab:routing-policy}
-\scriptsize
-\setlength{\tabcolsep}{2.2pt}
-\renewcommand{\arraystretch}{1.18}
-\begin{tabularx}{\textwidth}{@{}c c c c c c c c c@{}}
-\toprule
-Risk $\alpha$ & $\tau^*$ & Cal. upper risk & Routed $n$ & Coverage & Routed P1 errors & Error capture & Routed P1 acc. & Error enrichment \\
-\midrule
-5.0\% & 0.70 & 2.82\% & 171 & 98.58\% & 87 & 21.91\% & 49.12\% & 15.38$\times$ \\
-\bottomrule
-\end{tabularx}
-\vspace{2pt}
-\parbox{\textwidth}{\footnotesize \textit{Note.} The risk budget and candidate grid were fixed before held-out and Phase~2 evaluation. The threshold was selected only on the 12,000-example calibration split. The fixed policy concentrated 87 of 397 Phase~1 errors in 171 routed posts. Error enrichment is the routed error rate (50.88\%) divided by the full-test error rate (3.31\%).}
+\centering\small
+\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.14}
+\caption{Final DistilBERT calibration diagnostics (calibration split)}
+\label{tab:calibration-final}
+\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}
+\toprule
+Score & Temperature & NLL & Brier & ECE & Adaptive ECE \\
+\midrule
+Raw MSP & 1.0000 & 0.0933 & 0.0443 & 0.0176 & 0.0171 \\
+Temperature-scaled MSP & 1.4802 & 0.0795 & 0.0416 & 0.0064 & 0.0128 \\
+\bottomrule
+\end{tabular*}
 \end{table*}

-Routing effectiveness was assessed independently of the subsequent LLM outcomes. Under the primary policy, 171 posts (1.42\%) were routed. The routed subset contained 87 of 397 Phase~1 errors, giving 21.91\% error capture and 50.88\% routing precision. The corresponding full-test error rate was only 3.31\%, so the routed subset was approximately 15.4 times as error-dense as the full held-out set. Equivalently, a random subset of 171 posts would contain about 5.7 Phase~1 errors in expectation at the full-test error rate, whereas confidence routing concentrated 87 observed errors. Its 49.12\% Phase~1 accuracy was therefore far below the 96.69\% full-test accuracy, demonstrating that calibrated confidence concentrated an error-prone subset while sending only a small fraction of posts to Phase~2. Appendix Table~A4g reports the same decomposition for both evaluation sets.
-
-Figure~\ref{fig:routing-concentration} visualizes this concentration mechanism on both evaluation sets. The confidence distributions explain why the same fixed threshold routes different proportions of Reddit and Mixed Emotion inputs. The error-capture curve compares confidence routing with a random-routing reference, while the final panel reports how much more error-dense each routed subset is than its complete evaluation set.
+\begin{figure*}[t]\centering
+\includegraphics[width=\textwidth]{figures/calibration.pdf}
+\caption{Calibration diagnostics for the final linked DistilBERT model. Reliability, relative metric reduction, and risk--coverage curves use the calibration split; the selected point is not selected on test outcomes.}
+\label{fig:calibration-final}
+\end{figure*}
+\begin{table*}[t]
+\centering\small
+\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.14}
+\caption{Fixed-policy routing outcomes; rates are percentages}
+\label{tab:routing-final}
+\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}
+\toprule
+Dataset & $N$ & Routed & Route rate & Coverage & Error capture & Routed error & Accepted risk \\
+\midrule
+Reddit & 12000 & 218 & 1.82 & 98.18 & 26.49 & 44.95 & 2.31 \\
+Mixed Emotion & 300 & 86 & 28.67 & 71.33 & 57.14 & 32.56 & 9.81 \\
+\bottomrule
+\end{tabular*}
+\end{table*}
+
+On Reddit, the routed subset contains 98 of the 370 Phase~1 errors (26.49\% error capture). Its error rate is 44.95\%, compared with 3.08\% over the full set, an approximately 14.58-fold enrichment. The accepted subset still contains 272 errors, beyond the scope of routed-only correction. On Mixed Emotion, routing captures 28 of 49 errors (57.14\%); 21 errors remain accepted. Routed-set Phase~1 accuracy is 55.05\% on Reddit and 67.44\% on Mixed Emotion.

 \begin{figure*}[t]
 \centering
-\includegraphics[width=\textwidth]{figures/figure_d_routing_concentration.pdf}
-\caption{Confidence-based error concentration under the fixed routing policy. (a) Distribution of calibrated Phase~1 confidence with the same $\tau^*=0.70$ threshold applied to both evaluation sets. (b) Fraction of Phase~1 errors captured as increasingly large low-confidence subsets are routed; the dashed diagonal denotes random routing. (c) Overall and routed-subset Phase~1 error rates, with error-enrichment ratios shown above the routed values.}
+\includegraphics[width=\textwidth]{figures/routing_concentration.pdf}
+\caption{Fixed-policy routing concentration. (a) Cumulative calibrated-confidence distributions. (b) Errors captured when posts are ordered from lowest to highest confidence; dots identify the fixed threshold and the diagonal is a proportional-capture reference. (c) Full-set and routed error rates, with enrichment ratios. Curves describe the evaluation sets; they are not used to select the threshold.}
 \label{fig:routing-concentration}
 \end{figure*}

-\subsection{High-Confidence Accepted-Error Audit}
-
-The complementary accepted-set audit examined the 11,829 predictions retained by Phase~1. Of these, 310 were errors, corresponding to 2.62\% accepted selective risk and 97.38\% accepted accuracy. Among 3,939 accepted reference-Depression posts, 125 were predicted as Happy or Neutral, yielding an accepted Depression false-negative risk of 3.17\%. Error frequency declined sharply with confidence, but did not vanish: 34 accepted errors, including 16 Depression false negatives, had calibrated confidence of at least 0.98.
-
+\subsection{End-to-End Model and Prompting Results}
+Table~\ref{tab:final-e2e} reports the complete final comparison. Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 among these six configurations on both datasets. The advantage over Llama~3 CoT on Reddit is six correct predictions (0.0500 percentage points); numerical ordering alone does not establish a significant between-method difference.
 \begin{table*}[t]
-\centering
-\caption{High-confidence accepted-error audit on the Reddit held-out test set}
-\label{tab:accepted-error-audit}
-\footnotesize
-\setlength{\tabcolsep}{4pt}
-\renewcommand{\arraystretch}{1.16}
-\begin{tabularx}{\textwidth}{@{}l c c c c c c@{}}
-\toprule
-Reference proxy label & Accepted $n$ & Accepted errors & Accepted risk & To Depression & To Neutral & To Happy \\
-\midrule
-Depression & 3,939 & 125 & 3.17\% & -- & 10 & 115 \\
-Neutral & 3,978 & 41 & 1.03\% & 17 & -- & 24 \\
-Happy & 3,912 & 144 & 3.68\% & 112 & 32 & -- \\
-\midrule
-All classes & 11,829 & 310 & 2.62\% & 129 & 42 & 139 \\
-\bottomrule
-\end{tabularx}
-\vspace{2pt}
-\parbox{\textwidth}{\footnotesize \textit{Note.} ``Accepted risk'' is calculated within each accepted reference class. The last three columns count erroneous predictions only; correct predictions are omitted. The Depression false-negative risk is $125/3{,}939=3.17\%$. Reference labels are subreddit-derived proxies, so disagreements may reflect model errors or proxy-label/content mismatches.}
+\centering\small
+\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.14}
+\caption{Final full-set outcomes on 12,000 Reddit posts and 300 Mixed Emotion examples. Accuracy and macro F1 are percentages; change is relative to Phase 1 in percentage points. $C$ counts corrected errors and $I$ newly introduced errors; $C-I$ is their net difference. Unparsed responses retain Phase 1 predictions.}
+\label{tab:final-e2e}
+\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrrrr@{}}
+\toprule
+Dataset & Model & Protocol & Accuracy & Macro F1 & $C$ & $I$ & $C-I$ & Change \\
+\midrule
+Reddit & DistilBERT & Phase 1 & 96.9167 & 96.9166 & -- & -- & -- & -- \\
+Reddit & Llama 2 & Direct & 96.9917 & 96.9898 & 45 & 36 & +9 & +0.0750 \\
+Reddit & Llama 2 & CoT & 96.9250 & 96.9240 & 48 & 47 & +1 & +0.0083 \\
+Reddit & Llama 2 & SELF-DISCOVER & 96.9417 & 96.9391 & 52 & 49 & +3 & +0.0250 \\
+Reddit & Llama 3 & Direct & 97.1083 & 97.1056 & 68 & 45 & +23 & +0.1917 \\
+Reddit & Llama 3 & CoT & 97.1500 & 97.1471 & 63 & 35 & +28 & +0.2333 \\
+Reddit & Llama 3 & SELF-DISCOVER & 97.2000 & 97.1990 & 72 & 38 & +34 & +0.2833 \\
+Mixed Emotion & DistilBERT & Phase 1 & 83.6667 & 84.0005 & -- & -- & -- & -- \\
+Mixed Emotion & Llama 2 & Direct & 76.3333 & 75.1627 & 11 & 33 & -22 & -7.3333 \\
+Mixed Emotion & Llama 2 & CoT & 79.0000 & 78.9298 & 19 & 33 & -14 & -4.6667 \\
+Mixed Emotion & Llama 2 & SELF-DISCOVER & 89.3333 & 89.3452 & 22 & 5 & +17 & +5.6667 \\
+Mixed Emotion & Llama 3 & Direct & 89.3333 & 89.3803 & 26 & 9 & +17 & +5.6667 \\
+Mixed Emotion & Llama 3 & CoT & 84.6667 & 84.6760 & 15 & 12 & +3 & +1.0000 \\
+Mixed Emotion & Llama 3 & SELF-DISCOVER & 92.6667 & 92.6966 & 28 & 1 & +27 & +9.0000 \\
+\bottomrule
+\end{tabular*}
 \end{table*}

-Confidence-band analysis further localized the residual risk. Accepted error rates were 36.43\%, 25.95\%, 13.31\%, 4.18\%, and 0.36\% in the 0.70--0.80, 0.80--0.90, 0.90--0.95, 0.95--0.98, and 0.98--1.00 bands, respectively. Thus, confidence remained strongly informative, but no thresholded accepted set was error-free. Original-post review identified trajectory reversal and sarcasm, acute-distress cues overridden by a short positive clause, technical context masking explicit pride or excitement, and topic-term shortcuts triggered by mental-health vocabulary. Appendix Table~A4e summarizes these candidate error patterns, and Appendix Table~A4f supplies an original-post excerpt and the complete decision record for each selected case. The purpose of this audit is residual-risk transparency and failure analysis, not post hoc relabeling or evidence that Phase~2 produced an explanation; these examples were accepted by Phase~1 and therefore were never routed.
-
-\subsection{Two-Phase System Efficacy and Confidence-Guided Routing}
-
-With the classifier and routing policy fixed, RQ3 evaluates whether the assigned Phase~2 reasoner improves the complete system rather than merely producing plausible explanations on selected cases. Llama~2 CoT and Llama~3 SELF-DISCOVER therefore receive the same routed IDs within each evaluation set, and every corrected error is considered together with any newly introduced error. The following Reddit and Mixed Emotion subsections report the routed-only behavior, recombined full-set metrics, and paired statistical results without treating low confidence as a guarantee of successful correction.
-
-\subsection{Reddit Held-Out Test: End-to-End Results}
-
-Table~\ref{tab:reddit-e2e-results} reports the end-to-end comparison on the same 12,000-example Reddit held-out test set used for the routing analysis. Accepted examples retain the Phase~1 DistilBERT label, and only the 171 routed examples are eligible for Phase~2 replacement. The two reasoners receive the same minimally sanitized original title--body input. Within the routed subset, accuracy changed from 49.12\% under Phase~1 to 47.37\% with Llama~2 and 66.67\% with Llama~3. Accordingly, Llama~2 CoT produced a negligible $-0.03$-percentage-point full-test change (47 corrected versus 50 introduced routed errors), whereas Llama~3 SELF-DISCOVER increased full-test accuracy by 0.25 percentage points through 42 corrected and 12 introduced routed errors, or 30 net corrections. This increase is modest in full-test percentage points because only 1.42\% of the test set is routed, but the routed-only change is larger and the paired full-test effect is statistically supported.
-
-\begin{table*}[t]
-\centering
-\caption{Reddit held-out end-to-end comparison under the final model-specific prompt policy}
-\label{tab:reddit-e2e-results}
-\scriptsize
-\setlength{\tabcolsep}{2.1pt}
-\renewcommand{\arraystretch}{1.16}
-\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X c c c c c c c@{}}
-\toprule
-System & Test $n$ & Routed $n$ & Phase~1 acc. & Routed-only acc. & End-to-end acc. & Change & Net corrections \\
-\midrule
-DistilBERT Phase~1 only & 12,000 & 171 & 96.69\% & 49.12\% & 96.69\% & -- & 0 \\
-DistilBERT $\rightarrow$ Llama~2 CoT & 12,000 & 171 & 96.69\% & 47.37\% & 96.67\% & $-0.03$ pp & $-3$ \\
-DistilBERT $\rightarrow$ Llama~3 SELF-DISCOVER & 12,000 & 171 & 96.69\% & 66.67\% & 96.94\% & $+0.25$ pp & $+30$ \\
-\bottomrule
-\end{tabularx}
-\vspace{2pt}
-\parbox{\textwidth}{\footnotesize \textit{Note.} Llama~2 CoT corrected 47 routed Phase~1 errors and introduced 50; Llama~3 SELF-DISCOVER corrected 42 and introduced 12. Both received the same minimally sanitized original text. Changes are calculated from exact counts before rounding, not by subtracting displayed accuracies. All metrics use the same 12,000 test examples.}
-\end{table*}
-
-\subsection{Handling Mixed Emotion Inputs: Quantitative Results}
-
-To further examine model behavior under emotionally complex conditions, we evaluated the completed DistilBERT Phase~1 model on the 300-example supplementary synthetic Mixed Emotion Dataset. This dataset is not a primary benchmark or evidence of general clinical validity. Instead, it is a targeted stress test for cases in which positive, neutral, and depression-related cues co-occur or shift across the text. The Reddit-calibrated temperature and routing threshold were applied unchanged; neither was fitted or re-selected on the synthetic data.
-
-DistilBERT achieved 81.33\% Phase~1 accuracy (244/300) on the final stress test. The model correctly classified 66 Depression examples, 95 Happy examples, and 83 Neutral examples. Of the 44 routed examples, 21 were Phase~1 errors, corresponding to 52.27\% routed-subset accuracy and 37.50\% capture of the 56 total Phase~1 errors. The routed error rate was 47.73\%, compared with 18.67\% across all 300 examples, a 2.56-fold concentration. This behavior is consistent with the intended role of the set: it concentrates evaluation on emotionally mixed and trajectory-sensitive examples rather than estimating prevalence-level performance.
-
-Llama~2 CoT increased end-to-end accuracy by 4.00 percentage points (81.33\% to 85.33\%), correcting 18 of the 21 routed Phase~1 errors while introducing six. Llama~3 SELF-DISCOVER increased accuracy by 6.00 percentage points (81.33\% to 87.33\%), also correcting 18 errors (85.71\%) but introducing none. Routed-only accuracy was 79.55\% and 93.18\%, respectively, versus 52.27\% for Phase~1. Both configurations used the same 300 inputs, fixed Reddit-calibrated policy, and 44 routed cases.
-
-\begin{table*}[t]
-\centering
-\caption{Mixed Emotion stress-test end-to-end results on the shared routed subset}
-\label{tab:mixed-emotion-results}
-\scriptsize
-\setlength{\tabcolsep}{2.1pt}
-\renewcommand{\arraystretch}{1.16}
-\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X c c c c c c c@{}}
-\toprule
-System & Routed $n$ & Routed-only acc. & Corrected & Introduced & Net & End-to-end acc. & Change \\
-\midrule
-DistilBERT Phase~1 only & 44 & 52.27\% & -- & -- & -- & 81.33\% & -- \\
-DistilBERT $\rightarrow$ Llama~2 CoT & 44 & 79.55\% & 18 & 6 & 12 & 85.33\% & $+4.00$ pp \\
-DistilBERT $\rightarrow$ Llama~3 SELF-DISCOVER & 44 & 93.18\% & 18 & 0 & 18 & 87.33\% & $+6.00$ pp \\
-\bottomrule
-\end{tabularx}
-\vspace{2pt}
-\parbox{\textwidth}{\footnotesize \textit{Note.} All systems use the same 300 supplementary inputs and the same 44 routed examples. Phase~1 accuracy is 81.33\%. ``Corrected'' and ``introduced'' count changes only within the routed subset; ``net'' is corrected minus introduced.}
-\end{table*}
-
-Figure~\ref{fig:end-to-end-effect} summarizes the distinction between identifying difficult inputs and correcting them. Panel (a) reports accuracy only on the routed subsets, panel (b) shows the resulting change after routed predictions are recombined with accepted Phase~1 predictions, and panel (c) separates corrected from introduced errors. This decomposition prevents a positive correction count from obscuring errors newly introduced by a reasoner.
-
-\begin{figure*}[t]
-\centering
-\includegraphics[width=\textwidth]{figures/figure_b_end_to_end_effect.pdf}
-\caption{Effect of selective LLM re-evaluation on the shared routed subsets and complete evaluation sets. (a) Phase~1 and Phase~2 accuracy on routed examples. (b) Full-set accuracy change after recombination with accepted Phase~1 predictions. (c) Corrected and introduced routed errors; net improvement depends on their difference rather than on corrections alone.}
-\label{fig:end-to-end-effect}
+\begin{figure*}[t]\centering
+\includegraphics[width=\textwidth]{figures/end_to_end.pdf}
+\caption{Full-set accuracy for the final six configurations. Points show observed accuracy; horizontal segments connect each point to the linked Phase 1 baseline (dashed line). The two panels use different axis ranges. SD denotes the final task-level compact-plan SELF-DISCOVER protocol.}
+\label{fig:final-effect}
 \end{figure*}

-Table~\ref{tab:paired-statistics} reports paired uncertainty analyses for the same end-to-end comparisons. On Reddit, the Llama~2 confidence interval includes zero and its exact McNemar test does not distinguish the end-to-end result from the Phase~1 baseline. In contrast, the original-text Llama~3 interval is entirely positive, and its improvement remains significant after Holm adjustment across the four comparisons. On the Mixed Emotion stress test, both paired bootstrap intervals are positive and both changes remain significant after Holm adjustment; Llama~3 again provides the larger net improvement and introduces no routed error.
-
-\begin{table*}[t]
-\centering
-\caption{Paired statistical analysis of Phase~1 and end-to-end correctness}
-\label{tab:paired-statistics}
-\footnotesize
-\setlength{\tabcolsep}{3pt}
-\renewcommand{\arraystretch}{1.16}
-\begin{tabularx}{\textwidth}{@{}l >{\raggedright\arraybackslash}X c c c c c c@{}}
-\toprule
-Dataset & Reasoner & Change & Corrected & Introduced & Paired 95\% CI & Exact $p$ & Holm $p$ \\
-\midrule
-Reddit & Llama~2 CoT & $-0.03$ pp & 47 & 50 & [$-0.18$, 0.13] & 0.839 & 0.839 \\
-Reddit & Llama~3 SELF-DISCOVER & $+0.25$ pp & 42 & 12 & [0.13, 0.38] & $<0.0001$ & 0.0002 \\
-Mixed Emotion & Llama~2 CoT & $+4.00$ pp & 18 & 6 & [1.00, 7.33] & 0.0227 & 0.0453 \\
-Mixed Emotion & Llama~3 SELF-DISCOVER & $+6.00$ pp & 18 & 0 & [3.33, 8.67] & $<0.0001$ & $<0.0001$ \\
-\bottomrule
-\end{tabularx}
-\vspace{2pt}
-\parbox{\textwidth}{\footnotesize \textit{Note.} Confidence intervals are paired percentile-bootstrap intervals from 50,000 resamples. Exact $p$-values are two-sided McNemar tests over corrected and introduced pairs. Holm $p$ adjusts the four comparisons shown. The Reddit analysis follows the manuscript's exact 12,000-row balanced protocol.}
-\end{table*}
-
-On Mixed Emotion, class-wise Llama~3 recall increased from 66\% to 79\% for Depression (13 net corrections), 95\% to 99\% for Happy (four), and 83\% to 84\% for Neutral (one). Each class has 100 examples. No class incurred an introduced error in this run. Appendix Table~A4c separates clear informational scenarios from mixed-cue and trajectory-shift scenarios.
-
-Figure~\ref{fig:error-structure} shows the corresponding class-level error structure. The Reddit matrices reveal reductions in both Depression-to-Happy and Happy-to-Depression confusion after Llama~3 re-evaluation. The Mixed Emotion matrices show that the largest change is recovery of Depression examples initially assigned to Happy, while the Neutral-to-Happy residual remains unchanged under the fixed router.
-
-\begin{figure*}[t]
-\centering
-\includegraphics[width=0.86\textwidth]{figures/figure_c_error_structure.pdf}
-\caption{Confusion matrices before and after selective Llama~3 re-evaluation. Cells report counts and row-normalized percentages. (a)--(b) Reddit held-out test. (c)--(d) Mixed Emotion stress test. Accepted predictions remain unchanged; differences arise only from the fixed routed subsets.}
-\label{fig:error-structure}
+\subsection{Correction and Harm}
+On Reddit, Llama~3 SELF-DISCOVER corrects 72 errors and introduces 38, producing 34 net corrections. Direct and CoT yield 23 and 28 net corrections, respectively. Llama~2 yields smaller gains: Direct has nine net corrections, CoT one, and SELF-DISCOVER three. Thus, SELF-DISCOVER does not outperform Direct within every model and dataset combination.
+
+On Mixed Emotion, Llama~3 SELF-DISCOVER corrects 28 errors and introduces one. Total errors decrease from 49 to 22, a 55.10\% relative reduction, and accuracy increases by 9.00 percentage points. This low introduced-error count is an observed property of the 300-example stress test, not a population safety guarantee. Llama~2 SELF-DISCOVER also improves the baseline (17 net corrections), whereas its Direct and CoT configurations introduce more errors than they correct. The pattern supports evaluating both correction yield and damage to initially correct predictions.
+
+\begin{figure*}[t]\centering
+\includegraphics[width=\textwidth]{figures/corrections.pdf}
+\caption{Corrected (blue, right) and introduced (gray, left) errors for each final configuration. Introduced counts are plotted to the left for comparison, not as negative error counts. Right-hand labels show net corrections. The panels use different count scales.}
+\label{fig:final-errors}
 \end{figure*}

-\subsection{Additional Same-Source Holdout Results}
-\label{sec:additional-holdout-results}
-
-On the additional 9,000 posts, the unchanged operational classifier achieved 96.73\% accuracy and routed 137 posts (1.52\%). The routed subset contained 67 of 294 Phase~1 errors (22.79\% error capture), with a 48.91\% error rate versus 3.27\% overall. The remaining 8,863 accepted posts had a 2.56\% error rate. Thus, the fixed policy again concentrated errors in a small subset, while leaving 227 accepted errors outside the correction path.
-
-Table~\ref{tab:additional-holdout} reports the complete-set effects. Llama~3 SELF-DISCOVER corrected 36 errors and introduced nine, giving 27 net corrections and 97.03\% accuracy. Its $+0.30$ percentage-point gain remained significant after Holm adjustment ($p=0.000131$). Llama~2 CoT corrected 41 errors but introduced 26, yielding 96.90\% accuracy; its paired interval included zero. This distinction supports the evaluated Llama~3 configuration without implying that any LLM re-evaluator will improve the classifier or establishing a direct statistical difference between the two reasoners.
-
-\begin{table*}[t]
-\centering
-\caption{Additional same-source holdout: fixed-policy performance on 9,000 posts}
-\label{tab:additional-holdout}
-\footnotesize
-\setlength{\tabcolsep}{4pt}
-\renewcommand{\arraystretch}{1.16}
-\begin{tabularx}{\textwidth}{@{}Y c c c c c c c@{}}
-\toprule
-Configuration & Accuracy & Macro F1 & Corrected & Introduced & Change & 95\% CI (pp) & Holm $p$ \\
-\midrule
-Phase~1 & 96.73\% & 96.73\% & -- & -- & -- & -- & -- \\
-+ Llama~2 CoT & 96.90\% & 96.90\% & 41 & 26 & $+0.17$ pp & [$-0.01$, 0.34] & 0.0864 \\
-+ Llama~3 SELF-DISCOVER & 97.03\% & 97.03\% & 36 & 9 & $+0.30$ pp & [0.16, 0.46] & 0.000131 \\
-\bottomrule
-\end{tabularx}
-\vspace{2pt}
-\parbox{\textwidth}{\footnotesize\textit{Note.} Both reasoners re-evaluated the same 137 posts; all other predictions were retained. No parsing failures occurred. Changes and paired percentile-bootstrap intervals use all 9,000 posts (10,000 resamples). Holm adjustment covers the two exact McNemar comparisons against Phase~1 in this additional evaluation, separately from Table~\ref{tab:paired-statistics}.}
-\end{table*}
-
-The Llama~3 improvement corresponds to a 9.18\% reduction in total errors (294 to 267). This evaluation extends the evidence beyond the prompt-development sample, while remaining conditional on the same source and proxy-label construction. Appendix~F reports class-wise results; improvement was not uniform across all classes.
-
-\subsection{Observed Correction Cases and Reasoning Traceability}
-
-Appendix~D reports one observed Mixed Emotion Depression correction for each Phase~2 model using the stored output columns rather than an invented illustration. It also reports an observed Reddit original-text Llama~3 correction in which Phase~1 predicted Happy for a post describing numbness and exhaustion from trying to remain positive. Llama~2 CoT retains independent assessment, Phase~1 comparison, and terminal-label fields, while Llama~3 SELF-DISCOVER retains an auditable SELECT--ADAPT--IMPLEMENT trace before the final answer.
-
-The examples show how generated responses relate local cues to the complete post. Only the parsed terminal label enters quantitative evaluation; the rationale remains available for inspection. Corrected and introduced counts establish the observed predictive effect, whereas the trace records a generated justification rather than a verified causal account of the decision.
-
-\subsection{Interpretation of Routing and Re-Evaluation Results}
-
-The three research questions receive distinct answers. For \textbf{RQ1a}, DistilBERT achieved strong held-out predictive performance, and temperature scaling reduced NLL, Brier score, and fixed-bin ECE on the calibration split. For \textbf{RQ1b}, the completed three-seed comparison retained DistilBERT: its mean held-out accuracy and macro F1 were both 96.88\%, compared with 95.36\% for the Mistral frozen-backbone probe and 95.17\% for the Llama~2 probe. This is evidence for the operational choice under the bounded comparison, reinforced by the measured inference advantage, not a ranking of maximally tuned architectures. For \textbf{RQ2}, the fixed confidence policy preserved 98.58\% Reddit coverage while concentrating 87 of the 397 Phase~1 errors in the 171 routed posts; routed-sample accuracy was 49.12\%, compared with 96.69\% overall. The same fixed policy identified a 44-example Mixed Emotion subset with 52.27\% Phase~1 accuracy, compared with 81.33\% overall. For \textbf{RQ3}, re-evaluation benefit was model-dependent. Llama~2 remained statistically indistinguishable from Phase~1 on Reddit, whereas Llama~3 raised routed-only accuracy to 66.67\% and produced a positive paired full-test effect. On the controlled stress test, Llama~2 and Llama~3 raised routed-only accuracy to 79.55\% and 93.18\%, respectively, with Llama~3 avoiding the six introduced errors observed for Llama~2. Confidence therefore identifies a correction opportunity, but the reasoning policy determines how much of that opportunity is realized.
-
-Only 1.42\% of the original Reddit test posts, 14.67\% of Mixed Emotion examples, and 1.52\% of additional holdout posts underwent re-evaluation. These are fractions of inputs, not counts of individual LLM calls: CoT and SELF-DISCOVER require multiple generation stages per routed post. The Phase~1 benchmark supports the classifier's inference advantage, but the fraction avoiding re-evaluation is not a measured GPU-time saving because every input incurs Phase~1 computation and no all-input LLM timing baseline was run. The stress-test routing rate characterizes its constructed scenario mix, not an expected deployment workload.
-
-A conditional routing oracle further clarifies the scale of the observed gains without introducing a new tuned baseline. If every routed Phase~1 error were corrected and no correct routed prediction were changed, Reddit accuracy could rise from 96.69\% to at most 97.42\% under the fixed 171-post routing policy; the 30 Llama~3 net corrections realize 34.5\% of the 87-error correction opportunity. On Mixed Emotion, the corresponding ceiling is 88.33\%, and the 18 Llama~3 net corrections realize 85.7\% of the 21-error opportunity. These ceilings are conditional on the already selected router and are not claims about an attainable model. Their purpose is to distinguish a limited routing opportunity from a failure to exploit routed cases. Definitions and the complete decomposition are reported in Appendix Table~A4g.
-
-In addition to selective computation, the framework adds an inspectable rationale channel for routed cases. Unlike a single-stage classifier that outputs only a class label or probability score, the Phase 2 record retains the generated rationale and terminal label. This improves case-level traceability, but the rationale is not treated as proof of faithfulness, clinical validity, or prediction correctness.
-
-Taken together, the experiments support the Phase~1 and routing components of the framework for the constructed proxy-label task. Llama~3 produced positive paired effects on the original Reddit test, the Mixed Emotion stress test, and the additional same-source holdout. Llama~2 showed no statistically detectable gain on either Reddit evaluation. The additional holdout strengthens evidence for the frozen Llama~3 configuration beyond the prompt-development sample, but does not establish broad real-world mental-health generalization or isolate the effect of the reasoning method from the underlying model.
+\subsection{Conditional Correction Opportunity}
+With the router held fixed, a conditional oracle corrects every routed error without changing any initially correct prediction. Its full-set ceiling is $\mathrm{Acc}_{P1}+E_R/N$, where $E_R$ is the number of routed Phase~1 errors. The ceiling is 97.7333\% on Reddit and 93.0000\% on Mixed Emotion. Final Llama~3 SELF-DISCOVER realizes $34/98=34.69\%$ and $27/28=96.43\%$ of these respective net-correction opportunities. These are descriptive ceilings under a fixed router, not achievable performance guarantees or alternative tuned baselines.
+\Needspace{6\baselineskip}
+\section{Discussion}
+\subsection{An Efficient First Stage and a Selective Second Stage}
+The linked experiment separates three decisions: selecting a practical classifier, routing uncertain posts, and choosing a re-evaluation protocol. DistilBERT retains strong three-seed predictive performance and a measured classifier-stage inference advantage under the bounded Phase~1 comparison. The final run then supplies one consistent set of predictions and confidence scores for every Phase~2 condition. Classifier means and downstream single-run scores answer different questions without requiring different hyperparameter configurations.
+
+\subsection{Selection and Correction Are Different Outcomes}
+Confidence routing identifies an error-enriched subset, but enrichment alone is insufficient. Llama~2 CoT is nearly neutral on Reddit and harmful on Mixed Emotion, whereas final SELF-DISCOVER with Llama~3 improves both sets. On the stress test, searching for evidence for and against all three classes and distinguishing present distress from resolved or attributed events are plausible contributors to the observed pattern. The experiment evaluates the complete prompt package; it does not establish the separate causal effect of each instruction or demonstrate that the generated rationale faithfully reveals the model's computation.
+
+\subsection{What the Model--Protocol Comparison Shows}
+Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 on both sets. However, the 0.2833-point Reddit gain uses a 1.82\% routing rate, whereas the 9.00-point Mixed Emotion gain uses 28.67\%. These are different correction opportunities, not directly comparable effect sizes. Likewise, one introduced stress-test error versus 38 Reddit errors warrants dataset-specific interpretation rather than a universal reliability claim.
+
+The methodological value of the adaptation is to specify what evidence a selected post should be checked against, rather than merely requesting a longer explanation. Emotional subject and time-frame checks address attribution and trajectory, while the three-label ledger and counterfactual check make competing interpretations explicit. The observed gains support this complete configuration within the evaluated settings; they do not identify which instruction caused the gain. Appendix~\ref{app:sd-variants} shows that the final design improves on the earlier independent-countercheck variant for Llama~3 on both sets, but not for every model--dataset combination.
+
+The final model-by-protocol comparison is more informative than contrasting different models with different prompts only. Nevertheless, different context windows, generation budgets, and exposure to the Phase~1 label remain part of the evaluated configurations. Llama~3's advantage is therefore a configuration-level finding, not an isolated causal estimate of model generation or parameter count. Likewise, Direct is a label-only re-evaluation baseline: its benefit does not prove that an explicit reasoning trace is necessary.

 \section{Limitations and Future Work}
-
-\textit{Label and population validity.} Reddit labels reflect subreddit membership and polarity filtering, not clinical assessments. Filtering may favor sentiment-aligned posts and make the task easier than classification of unfiltered discourse. The balanced, post-level split also does not establish performance on unseen authors, future time periods, or other platforms. The synthetic stress test has controlled labels and scenarios but may contain generator-specific phrasing and repeated narrative patterns. Neither source establishes clinical validity. Independent annotation is needed to assess post-level label consistency and rationale quality before considering any decision-support use.
-
-\textit{Residual risk and calibration.} The router left 310 errors among 11,829 accepted predictions, including 34 at confidence 0.98 or above. Confidence therefore supports selection without certifying individual predictions. Temperature and threshold selection share a calibration partition, and the risk bound is a pointwise selection criterion rather than a post-selection guarantee. Results under the fixed Reddit policy do not establish risk control on other distributions. The accepted-error examples identify candidate failure patterns; without attribution tests or independent adjudication, they do not establish causal mechanisms or failure-mode prevalence.
-
-\textit{Comparison scope.} The Phase~1 study compares three practical configurations under a limited search budget, not equivalently adapted or maximally optimized architectures. Three seeds characterize a narrow range of training variability. In Phase~2, model family and prompting method vary together, so the observed advantage of Llama~3 SELF-DISCOVER cannot be attributed to SELF-DISCOVER alone. A crossed model--prompt evaluation would be needed to separate these effects. Similarly, the final original-text rerun supports the reported input policy but does not establish a general causal effect of preserving any particular textual feature.
-
-\textit{Prompt development and statistical scope.} Prompt variants were examined on the original routed Reddit cases under the earlier cleaned-input condition. Freezing prompts before the original-text rerun did not remove that development dependence, so those original end-to-end comparisons remain exploratory. The additional 9,000-post holdout evaluates the frozen pipeline on previously unused same-source text and provides separate evidence of improvement, without retroactively changing the status of the original analyses. Normalized exact-text exclusion does not establish author independence or remove all near-duplicates, and the paired intervals do not quantify repeated-generation variability. External or author-disjoint evaluation and expert-reviewed labels remain important extensions. This issue is distinct from calibration-partition fitting of $T^*$ and $\tau^*$.
-
-\textit{Computational and explanatory scope.} The controlled Phase~1 benchmark compares our implementations on one accelerator model and serving configuration, not optimized inference systems across hardware platforms. Three timing repetitions characterize within-session variation; they do not capture all variation across Colab sessions. GPU-board energy estimates exclude CPU and system-level consumption and have limited temporal resolution for short timed regions. Total cascade energy and monetary savings were not established. Our per-post SELF-DISCOVER adaptation does not reuse a task-level plan, so efficiency results from the original method cannot be transferred to this implementation. Generated rationales still require independent assessment of faithfulness and label support.
+Subreddit-derived, sentiment-filtered labels are proxy constructs and can reward topical or community-specific shortcuts. Random post-level splits do not establish author-disjoint or temporal independence. The Mixed Emotion data are synthetic, scenario-balanced, and not independently expert-labeled; their results should not be generalized to naturally occurring emotional ambiguity or clinical depression.
+
+Prompt variants were developed and compared using these evaluation cases. Consequently, the selected prompt's observed ranking and count-based significance summaries remain exploratory. A frozen final configuration should be assessed on unused data to estimate generalization after selection. Earlier evaluations of a different downstream configuration are not used as independent validation of the final pipeline in this manuscript.
+
+Reddit re-evaluation uses richer original text than the cleaned Phase~1 input. This is a deliberate system design but prevents attributing the entire gain to prompting alone. Residual parsing failures retain Phase~1 predictions; their presence and the rule must be considered when comparing protocols. The final SELF-DISCOVER summaries contain two such failures on Reddit and one on Mixed Emotion for Llama~3. No unresolved failure is manually relabeled from its reference answer.
+
+The current experiments do not quantify repeated-generation variation for all final configurations or supply a fully matched end-to-end cost benchmark. Greedy decoding does not guarantee bitwise equality across hardware and software environments. Independent expert review, broader evaluation, and generation-cost accounting are useful extensions, rather than reasons to repeat the completed Phase~1 model comparison.

 \section{Conclusion}
-
-This study evaluated selective LLM re-evaluation by separating classifier performance, calibration, error concentration, and net correction. The three-seed comparison and common-hardware inference benchmark supported retaining DistilBERT within the tested configurations: it combined the highest mean predictive scores with substantially higher batched throughput, lower single-post latency, and lower inference memory. Its fixed operational checkpoint routed 1.42\% of Reddit posts, capturing 87 of 397 Phase~1 errors. Llama~3 SELF-DISCOVER corrected 42 errors and introduced 12, raising full-set accuracy from 96.69\% to 96.94\%; Llama~2 CoT produced no detectable improvement. On the synthetic stress test, the corresponding gains were 6.00 and 4.00 percentage points.
-
-The central finding is that selecting an error-prone subset and correcting it are separate requirements. On an additional 9,000-post same-source holdout, the frozen pipeline re-evaluated 1.52\% of posts and improved accuracy from 96.73\% to 97.03\% with Llama~3, with a significant paired gain. This strengthens support beyond the original prompt-development sample while leaving accepted errors outside the correction path. External validation and expert-reviewed labels are needed to establish broader applicability. The framework remains a proxy emotion-classification pipeline, not a clinical diagnostic system.
-
-\balance
+A unified DistilBERT-to-LLM pipeline allows classifier performance, routing concentration, and correction quality to be evaluated without substituting a separately configured downstream classifier. Among the final Direct, CoT, and SELF-DISCOVER configurations for two language models, Llama~3 SELF-DISCOVER achieves the highest observed accuracy and macro F1 on both Reddit and the synthetic Mixed Emotion stress test. Its gains are 34 and 27 net corrections, respectively. The principal empirical finding is configuration-dependent correction: routing creates an opportunity, but the re-evaluation protocol determines whether the opportunity is used without introducing excessive new errors. These exploratory proxy-label results support further validation, not clinical deployment claims.
 \begin{thebibliography}{46}

 \bibitem{ref1} World Health Organization, ``Depressive disorder (depression),'' WHO Fact Sheet, Aug. 29, 2025. Available: \url{https://www.who.int/news-room/fact-sheets/detail/depression}
@@ -876,7 +734,7 @@
 \end{thebibliography}

 \clearpage
-\onecolumn
+\onecolumn\raggedbottom
 \appendices
 % Keep appendix headings and their supporting tables compact without orphaning titles.
 \setlength{\medskipamount}{4pt plus 1pt minus 1pt}
@@ -953,12 +811,12 @@
 \par\smallskip
 \normalsize

-\noindent\footnotesize\textit{Note.} Each row reports the configuration selected by a four-trial Bayesian sweep that maximized validation macro F1. ``Epochs'' is the training budget selected by the sweep; each complete configuration was subsequently evaluated with seeds 42, 43, and 44. The downstream calibration and routing analyses retained the independently fixed operational DistilBERT checkpoint (learning rate $8.996\times10^{-5}$, batch 32, three epochs, weight decay $10^{-2}$), which achieved 96.69\% held-out accuracy and was not replaced after the matched comparison.\normalsize
-\par\medskip
-
-
-
-\Needspace{39\baselineskip}
+\noindent\footnotesize\textit{Note.} Each row reports the configuration selected by a four-trial Bayesian sweep that maximized validation macro F1. ``Epochs'' is the training budget selected by the sweep; each complete configuration was subsequently evaluated with seeds 42, 43, and 44. The final downstream DistilBERT run uses the selected configuration in this table, with temperature 1.480195 and threshold 0.70. The single-run downstream score is not the three-seed mean.\normalsize
+\par\medskip
+
+
+
+\Needspace{12\baselineskip}
 \noindent\textsc{Appendix Table A1d. Phase~1 computational measurements}\par
 \label{tab:appendix-a1d-efficiency}
 \smallskip
@@ -1006,371 +864,171 @@
 \normalsize
 \par\medskip

+\Needspace{4\baselineskip}
+\section{Final Prompting Protocols}
+The following templates are extracted from the executed notebook code. Placeholder fields are substituted at runtime. The shared policy is identical in the supplied final SD and Llama~3 baseline templates; the final Llama~2 CoT instruction sequence matches the Llama~3 sequence. Direct discloses the Phase~1 label immediately, CoT discloses it after an independent assessment, and final SD never discloses it. No reference label is inserted into any prompt.
+
+
+\par\medskip
+\noindent\textsc{Appendix Table B1. Final re-evaluation protocols}\par\smallskip
+{\small\renewcommand{\arraystretch}{1.18}
+\noindent\begin{tabularx}{\textwidth}{@{}p{0.13\textwidth}YYY@{}}\toprule
+Protocol & Information supplied & Processing structure & Scored output \\\midrule
+Direct & Original post, shared policy, Phase 1 label & One label-only generation call & One parsed label; otherwise Phase 1 fallback \\
+CoT & Original post; Phase 1 label revealed after independent assessment & Five dialogue calls, including acknowledgment and text-response turns & Terminal label from the final response; otherwise fallback \\
+SELF-DISCOVER & Original post, policy, cached model-specific plan; no Phase 1 label & Task-level discovery, followed by one execution call per routed post & Terminal label after evidence and counterfactual checks; otherwise fallback \\\bottomrule
+\end{tabularx}}\par\medskip
+The sections below reproduce the literal prompt text. Placeholder names remain as written in the executable templates.
+
+\Needspace{13\baselineskip}
+\subsection{Shared Classification Policy}
+\lstinputlisting{prompts/policy.txt}
+\Needspace{18\baselineskip}
+\subsection{Direct Re-Evaluation}
+One user message is formed by inserting the policy, original text, and Phase~1 label into the following template. No explicit rationale is requested.
+\lstinputlisting{prompts/direct.txt}
+\subsection{Chain-of-Thought Dialogue}
+The following system message and user instructions are sent in order. Assistant responses are retained in the conversation. After the first user instruction the model generates an acknowledgment; after the text is provided it generates a text response; then independent analysis, comparison, and final-decision calls follow. There are five generation calls per routed post, not one.
+\par\smallskip\Needspace{6\baselineskip}\noindent\textbf{System message}\par
+\lstinputlisting{prompts/cot_0.txt}
+\par\smallskip\Needspace{6\baselineskip}\noindent\textbf{Initial user instruction}\par
+\lstinputlisting{prompts/cot_1.txt}
+\par\smallskip\Needspace{6\baselineskip}\noindent\textbf{Post input}\par
+\lstinputlisting{prompts/cot_text.txt}
+\par\smallskip\Needspace{6\baselineskip}\noindent\textbf{Independent analysis}\par
+\lstinputlisting{prompts/cot_2.txt}
+\par\smallskip\Needspace{6\baselineskip}\noindent\textbf{Comparison with Phase 1}\par
+\lstinputlisting{prompts/cot_3.txt}
+\par\smallskip\Needspace{6\baselineskip}\noindent\textbf{Final decision}\par
+\lstinputlisting{prompts/cot_4.txt}
+
+\section{Final SELF-DISCOVER Templates}\label{app:sd-templates}
+This appendix documents the final task-level SELF-DISCOVER protocol (artifact identifier v6c). Read it in two parts: \emph{discovery} generates and stores a model-specific procedure, whereas \emph{execution} applies that procedure to each routed original post. Figure~C1 separates these scopes. The source templates below are reproduced verbatim; the explanatory paragraphs describe how they connect, rather than modifying the prompts.
+
+The chronology and authorship are distinct. Researchers first write the task description, class policy, module bank, and execution guidance. When no retained plan exists, the selected LLM performs SELECT, ADAPT, and IMPLEMENT using the task materials only; the code saves its generated procedure before classifying any post. No individual post or reference label enters these discovery calls. Only then does the LLM receive a routed original post together with the saved procedure and fixed execution guidance. A retained plan bypasses all three discovery calls.
+
+The stored plan is a reusable text artifact, not a set of cached predictions. Each model uses its own plan for both datasets. Reusing it does not mean learning a procedure from the first evaluation post, reusing that post's answer, updating model weights, or bypassing per-post inference.
+
 \Needspace{24\baselineskip}
-\section{Chain-of-Thought Prompting Protocol}
-
-This table summarizes the final Chain-of-Thought prompting protocol used with Llama~2-7B-Chat to re-evaluate low-confidence Phase~1 predictions. The protocol requires independent text-grounded assessment before Phase~1-label comparison, uses trajectory reasoning only when a shift is supported by the text, and ends with one extractable final label. The earlier numerical emotion-breakdown requirement is not used.
-
-\par\medskip
-\noindent\textsc{Appendix Table A2. Chain-of-Thought prompting protocol for emotional re-evaluation}\par
-\label{tab:appendix-a2-cot-prompt}
+\begin{center}
+\includegraphics[width=0.96\textwidth]{figures/appendix_sd_workflow.pdf}
+\end{center}
+\noindent\footnotesize\textbf{Appendix Figure C1.} Final task-level SELF-DISCOVER workflow, separating authorship, timing, and input scope. (a) Researchers supply task materials; before post classification, the selected LLM generates a plan through SELECT, ADAPT, and IMPLEMENT if no retained plan exists. No evaluation post or reference label enters discovery. (b) The same LLM applies the saved plan and fixed researcher-authored checks to each routed original post, generating a fresh response. The dashed arrow denotes plan reuse, not reuse of an answer. Parsing and fallback are code operations, not additional LLM calls.\normalsize\par\medskip
+
+\Needspace{15\baselineskip}\noindent\textsc{Appendix Table C1. Final SELF-DISCOVER stages}\par\smallskip
+{\small\renewcommand{\arraystretch}{1.18}
+\noindent\begin{tabularx}{\textwidth}{@{}p{0.15\textwidth}YYY@{}}\toprule
+Stage & Input & Requested operation & Retained output \\\midrule
+SELECT & Task policy and 18 domain modules & Select task-relevant reasoning operations & Selected modules \\
+ADAPT & Task description and selected modules & Specialize operations to the three-class task & Adapted modules \\
+IMPLEMENT & Adapted modules & Request a compact plan of at most three one-line steps & Actual model-generated plan, cached without a conformance validator \\
+Execution & Original post, policy, cached plan & Compare evidence for all three labels and check an alternative interpretation & Response text, parsed label, and failure status \\\bottomrule
+\end{tabularx}}\par\medskip
+
+\Needspace{12\baselineskip}
+\subsection{Task Description and Module Bank}
+\textit{Purpose and scope.} These are researcher-authored inputs to model-level discovery, not an individual evaluation post. The task description defines the three-label annotation problem, and the module bank supplies 18 candidate operations concerning emotional subject, time frame, trajectory, topic, and competing evidence. The policy placeholder expands to the shared classification policy in Appendix~B. The title--body wording is retained verbatim in the Mixed Emotion run, whose supplied input is the synthetic post text.
+\lstinputlisting{prompts/sd_task.txt}
+\lstinputlisting{prompts/sd_modules.txt}
+\Needspace{12\baselineskip}
+\subsection{SELECT}
+\textit{Input:} the task description and full module bank. \textit{Output:} the model's selected module descriptions, passed to ADAPT. This selection occurs during plan construction, not separately for each evaluated post. Placeholder spelling, including \texttt{resonining\_modules}, is retained from the code.
+\lstinputlisting{prompts/sd_select.txt}
+\Needspace{12\baselineskip}
+\subsection{ADAPT}
+\textit{Input:} the task description and SELECT output. \textit{Output:} task-specialized module descriptions, passed to IMPLEMENT. This step asks the model to translate the selected operations into the emotion-classification setting; it does not yet assign a post-level label.
+\lstinputlisting{prompts/sd_adapt.txt}
+\Needspace{12\baselineskip}
+\subsection{Compact IMPLEMENT}
+\textit{Input:} the task description and ADAPT output. \textit{Output:} a reusable reasoning plan, retained for later execution. Here IMPLEMENT means constructing the procedure, not executing it on every post. The prompt requests at most three one-line steps and access to the full post at each step. There is no conformance validator or retry enforcing those requests; the realized plans are reported below.
+\lstinputlisting{prompts/sd_implement.txt}
+\Needspace{12\baselineskip}
+\subsection{Per-Post Execution}
+\textit{Input:} one routed original post, the shared policy, the model's saved plan, and the fixed execution instructions below. \textit{Output:} a newly generated response ending in a classification line. The system message precedes the user message; \texttt{structure} is replaced by the saved plan and \texttt{text} by this post. No Phase~1 label is supplied to the model.
+
+The evidence ledger is a three-way comparison: for each label the model is asked to identify the strongest supporting and opposing quotation. The subsequent counterfactual check evaluates a competing interpretation rather than changing the post. These are explicit instructions attached to every execution call, not properties guaranteed by caching or by the discovered plan alone.
+\lstinputlisting{prompts/sd_system.txt}
+\lstinputlisting{prompts/sd_execute.txt}
+
+\Needspace{15\baselineskip}
+\subsection{Illustrative Passage Through the Execution Stage}
+Consider the constructed sentence, ``Last year I felt isolated; now I enjoy meeting my friends again and feel relieved.'' This is an explanatory example, not a dataset record, model response, or additional evaluation result. The same cached plan would be inserted regardless of this sentence's eventual label. A policy-consistent reading would separate past isolation from current relief, use the quoted time markers as evidence, and examine whether unresolved current distress is actually stated. Under the stated policy, the resolved positive trajectory supports Happy. The example illustrates what the instructions request; it does not show that every generated response follows them.
+
+The evaluator then reads the terminal label from the actual response. A permitted label replaces the Phase~1 prediction for this routed post. If no label is parsed, the evaluator retains Phase~1 instead. Unrouted posts never enter this execution stage. The scored prediction is therefore distinct from the generated explanation, and a parsing failure is not automatically a wrong classification.
+
+\Needspace{12\baselineskip}
+\subsection{Realized Cached Plans}
+The following plans are reproduced from the recorded notebook outputs, not rewritten to make them conform to the instructions. The Mixed Emotion runs explicitly reused the cached plans. Llama~3 produced three evidence-oriented steps plus a separate final step. Llama~2 produced four prose steps and included provisional label assignment inside early steps. These differences show why the requested plan constraints should not be reported as enforced behavior; they do not independently establish the cause of the performance difference.
+\par\medskip\noindent\begin{minipage}{\textwidth}
+\textbf{Llama 2 generated plan}
+\lstinputlisting{prompts/sd_plan_llama2.txt}
+\end{minipage}\par\medskip
+\par\medskip\noindent\begin{minipage}{\textwidth}
+\textbf{Llama 3 generated plan}
+\lstinputlisting{prompts/sd_plan_llama3.txt}
+\end{minipage}\par\medskip
+
+\subsection{Generation Budget and Failure Handling}
+Direct and CoT request 1,024 new tokens per generation call. Final SD execution requests 1,024 for Llama~2 and 1,536 for Llama~3. The SD discovery calls request up to 512 tokens for Llama~2 and 768 for Llama~3. The source uses a shared per-model cache for the task-level structures, allowing reuse across the Reddit and Mixed Emotion runs. The templates specify instructions, not guaranteed instruction compliance. Quantized model loading, model-specific chat formatting, context limits, and call structure are part of the evaluated implementation. A shorter plan instruction does not by itself prove lower measured cost.
+
+Remaining final SD parsing failures are zero for Llama~2 on both sets, two for Llama~3 Reddit, and one for Llama~3 Mixed Emotion. These remain failures and use Phase~1 fallback. No additional model call is added to resolve them in the reported final SD results.
+\section{SELF-DISCOVER Design-Variant Comparison}\label{app:sd-variants}
+The earlier independent-countercheck variant generated an instance-specific plan using general reasoning modules; the final version uses a domain-oriented task-level plan and an explicit evidence ledger. Both withhold the Phase~1 label. Multiple components change jointly, so this is a design-variant comparison rather than a single-component causal ablation. The earlier variant is not used in the main six-configuration comparison.
+
+\par\medskip\noindent\textsc{Appendix Table D1. Earlier and final SELF-DISCOVER designs}\par\smallskip
+{\small\renewcommand{\arraystretch}{1.18}
+\noindent\begin{tabularx}{\textwidth}{@{}p{0.20\textwidth}YY@{}}\toprule
+Component & Earlier independent-countercheck & Final task-level compact plan \\\midrule
+Discovery scope & A plan constructed for each post & A model-specific plan constructed for the task and reused \\
+Module bank & General reasoning operations & 18 emotion-oriented operations \\
+Execution guidance & Independent assessment and countercheck & Explicit subject/time checks, three-label quotation ledger, and counterfactual check \\
+Phase 1 label & Withheld during SD reasoning & Withheld during SD reasoning \\
+Evaluation role & Earlier design comparator in this appendix & SELF-DISCOVER condition in the main comparison \\\bottomrule
+\end{tabularx}}\par\medskip
+
+\par\medskip\noindent\textsc{Appendix Table D2. Full-set outcomes of the two SD designs}\par\smallskip
+{\small\renewcommand{\arraystretch}{1.18}\noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{}}\toprule
+Dataset & Model & Earlier acc. & Final acc. & Earlier net & Final net \\\midrule
+Reddit & Llama 2 & 97.0500\% & 96.9417\% & 16 & 3 \\
+Reddit & Llama 3 & 97.1583\% & 97.2000\% & 29 & 34 \\
+Mixed Emotion & Llama 2 & 87.6667\% & 89.3333\% & 12 & 17 \\
+Mixed Emotion & Llama 3 & 90.3333\% & 92.6667\% & 20 & 27 \\\bottomrule
+\end{tabular*}}\par\medskip
+\noindent\textit{Table note.} Accuracy uses all 12,000 Reddit posts or all 300 Mixed Emotion examples, not only routed posts. Net denotes corrected minus introduced errors relative to the same final Phase~1 predictions. ``Earlier'' refers only to the independent-countercheck design, not another intermediate compact-plan version. Table~D2 reports observed outcomes, not a paired significance test between the two SD designs.
+
+The final design improves three of these four model--dataset combinations; it does not uniformly dominate the earlier design. Its Llama~3 configuration improves both datasets. The change in input-independent planning also changes the number and scope of discovery calls, so a cost comparison requires measured generation records rather than accuracy alone.
+
+\Needspace{28\baselineskip}
+\section{Paired Accuracy Summaries}\label{app:statistics}
+Each row asks whether one complete re-evaluation configuration differs from Phase~1 on the same examples. It does not ask whether SD is better than Direct or CoT. ``Change'' is the full-set accuracy difference in percentage points; the interval estimates uncertainty in that difference; exact $p$ compares the numbers corrected and newly harmed; Holm $p$ adjusts the 12 displayed comparisons.
+
+These calculations use the final reported corrected/introduced counts, with all unparseable SD responses left at the implemented Phase~1 fallback. For accuracy change, the paired correctness-difference counts are sufficient for the exact test and paired bootstrap. This does not constitute a new audit of all raw responses, nor a paired SD-versus-CoT significance test. The latter requires matched per-example predictions from both configurations. Intervals are percentile bootstrap intervals in percentage points. Holm adjustment covers all 12 rows below; it does not account for selection over previously explored prompts.
+
+\par\medskip\noindent\textsc{Appendix Table E1. Paired accuracy comparisons with Phase 1}\par\smallskip
+{\small\renewcommand{\arraystretch}{1.18}\noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrr@{}}\toprule
+Dataset & Model & Protocol & Change (pp) & Paired 95\% CI & Exact $p$ & Holm $p$ \\\midrule
+Reddit & Llama 2 & Direct & +0.0750 & [-0.075, 0.225] & 0.3742 & 1.0000 \\
+Reddit & Llama 2 & CoT & +0.0083 & [-0.150, 0.167] & 1.0000 & 1.0000 \\
+Reddit & Llama 2 & SELF-DISCOVER & +0.0250 & [-0.142, 0.192] & 0.8424 & 1.0000 \\
+Reddit & Llama 3 & Direct & +0.1917 & [0.017, 0.367] & 0.0380 & 0.2281 \\
+Reddit & Llama 3 & CoT & +0.2333 & [0.075, 0.392] & 0.0061 & 0.0479 \\
+Reddit & Llama 3 & SELF-DISCOVER & +0.2833 & [0.117, 0.458] & 0.0015 & 0.0151 \\
+Mixed Emotion & Llama 2 & Direct & -7.3333 & [-11.667, -3.000] & 0.0013 & 0.0139 \\
+Mixed Emotion & Llama 2 & CoT & -4.6667 & [-9.333, 0.000] & 0.0704 & 0.3520 \\
+Mixed Emotion & Llama 2 & SELF-DISCOVER & +5.6667 & [2.333, 9.000] & 0.0015 & 0.0151 \\
+Mixed Emotion & Llama 3 & Direct & +5.6667 & [2.000, 9.667] & 0.0060 & 0.0479 \\
+Mixed Emotion & Llama 3 & CoT & +1.0000 & [-2.333, 4.333] & 0.7011 & 1.0000 \\
+Mixed Emotion & Llama 3 & SELF-DISCOVER & +9.0000 & [5.667, 12.667] & $<0.0001$ & $<0.0001$ \\
+\bottomrule\end{tabular*}}\par\medskip
+
+\clearpage
+\section{Synthetic Mixed Emotion Dataset Generation Protocol}
+
+\par\medskip
+\noindent\textsc{Appendix Table F1. Synthetic Mixed Emotion Dataset generation protocol}\par
+\label{tab:appendix-a5-mixed-emotion-protocol}
 \smallskip
 \small
-\setlength{\tabcolsep}{3.5pt}
-\renewcommand{\arraystretch}{1.2}
-\noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.18\textwidth}YYY@{}}
-\toprule
-Prompt Section & Prompt Instruction & Purpose in the Framework & Expected Output \\
-\midrule
-Role & You are an expert annotator for research-oriented, non-clinical emotion classification. Do not make a clinical diagnosis, infer a medical condition, or provide treatment advice. & Defines a text-classification role without clinical diagnostic authority. & A non-clinical, research-oriented analysis. \\
-Independent assessment & Before seeing the Phase~1 label, identify the dominant emotional meaning of the full text and cite the strongest textual evidence. & Intended to limit anchoring on the initial prediction; this effect was not measured separately. & An initial Depression, Neutral, or Happy assessment. \\
-Classification and trajectory rule & Depression requires dominant unresolved distress; Neutral requires mainly factual, routine, balanced, or emotionally mild content; Happy requires dominant happiness, relief, gratitude, accomplishment, fulfillment, or clear positive resolution. Use trajectory reasoning only when a temporal shift is explicitly supported. & Specifies full-text decision criteria and discourages unsupported temporal interpretations. & A text-grounded dominant-emotion decision. \\
-Phase~1 comparison & Reveal the Phase~1 label only after independent assessment. Confirm it only when it is supported by the full-text evidence; otherwise explain the correction. & Separates independent assessment from confirm-or-correct reasoning. & A supported confirmation or correction. \\
-Output constraint & Use only Depression, Neutral, or Happy. Do not use synonyms or a percentage breakdown. End with exactly one line in the format \texttt{Final label: [label]}. & Produces a stable, machine-extractable final Phase~2 label. & A response ending with one permitted final label. \\
-\bottomrule
-\end{tabularx}
-\par\medskip
-
-
-\noindent\footnotesize\textit{Note.} This table summarizes the protocol and uses a non-clinical rendering of the role description; it is not a verbatim prompt transcript. The executable prompt strings are retained in the released notebooks. The model's assigned role does not confer diagnostic or treatment authority.\normalsize
-
-
-\Needspace{34\baselineskip}
-\section{SELF-DISCOVER Prompting Protocol}
-
-This table presents the input-specific SELF-DISCOVER adaptation used with Llama~3-8B-Instruct. Discovery is repeated for each routed post rather than once for the whole task as in the original method \cite{ref18}. The protocol requests an emotional assessment, comparison with Phase~1, textual justification, and one final label within the three-class label space.
-
-\par\medskip
-\noindent\textsc{Appendix Table A3. SELF-DISCOVER prompting protocol for nuanced emotion classification}\par
-\label{tab:appendix-a3-self-discover-prompt}
-\smallskip
-\footnotesize
-\setlength{\tabcolsep}{3pt}
-\renewcommand{\arraystretch}{1.18}
-\noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.18\textwidth}YYY@{}}
-\toprule
-Prompt Section & Prompt Instruction & Purpose in the Framework & Expected Output \\
-\midrule
-Role & You are an expert annotator for mental-health-related emotion classification. Your task is to analyze the emotional content of text data using structured reasoning. This task is intended for research-oriented text classification, not clinical diagnosis. & Assigns the model a structured annotation role for deeper emotional interpretation without implying clinical diagnostic authority. & The model performs structured emotional classification within a non-clinical framework. \\
-Task Context & I will provide you with text data and an AI-generated emotional classification label. Your task is to determine the dominant emotion that best represents the overall sentiment of the text. The text may contain multiple emotions, but your goal is to determine the most representative emotion that captures the overall tone. Do not make a clinical diagnosis, infer a medical condition, or provide treatment advice. & Defines the input information, classification objective, and need to handle mixed emotional cues while limiting the task to non-clinical text classification. & The model identifies the dominant emotion while considering the overall emotional trajectory of the text. \\
-Question 1: Independent Emotion Analysis & Objectively analyze the given text.  Identify the dominant emotion that best represents the text's overall sentiment. & Encourages independent assessment of the input text before comparing it with the AI-generated label. & An initial dominant emotion based on the text itself. \\
-Question 2: Comparison with AI Label & Compare your analysis from Question 1 with the AI's classified label. & Evaluates whether the independent analysis aligns with the Phase 1 prediction. & A comparison between the independent analysis and the AI-generated label. \\
-Question 3: Evidence-Based Re-Evaluation & Evaluate the AI's classification using textual evidence. If it aligns with your independent assessment, confirm it. If it does not, determine the correct dominant emotion and justify the decision using only evidence from the text. & Enables reasoning-based correction when the Phase 1 prediction is inconsistent with the textual evidence. & A confirmed or corrected label with a text-grounded justification. \\
-Question 4: Final Label Selection & Select the option that most accurately represents your analysis. Do not create or explain additional choices beyond the provided options. End the response with exactly one final label using the format \texttt{Final label: [label]}. Options: Depression, Neutral, Happy. & Restricts the final output to the predefined three-class label space and supports deterministic label extraction. & A structured response ending with one final label selected from Depression, Neutral, and Happy. \\
-Classification Guidelines & If the text expresses ongoing sadness, hopelessness, emotional distress, or a strongly negative final emotional trajectory, classify it as Depression. If the text is mainly factual, balanced, informational, or low-affect, classify it as Neutral. If the text expresses happiness, accomplishment, relief, gratitude, or fulfillment as the dominant final sentiment, classify it as Happy. When multiple emotions are present, choose the class that best reflects the overall message and final takeaway rather than isolated phrases; do not default to Neutral merely because cues are mixed. & Provides label-specific decision rules and supports consistent handling of mixed or shifting emotional trajectories. & Consistent application of the three-class emotion-label criteria. \\
-Output Constraint & Do not create labels outside the provided options. When explaining the decision, base the justification on textual evidence rather than clinical assumptions. The response should end with exactly one final label in the format Final label: Depression, Final label: Neutral, or Final label: Happy. & Ensures that the output remains aligned with the study's three-class classification setting and can be parsed into a final Phase 2 label. & A structured output that includes a rationale and a clearly extractable final label. \\
-\bottomrule
-\end{tabularx}
-\par\medskip
-
-
-\noindent\footnotesize\textit{Note.} This table summarizes the protocol with non-clinical role wording; verbatim executable prompts are retained in the released notebooks.\normalsize
-
-\par\medskip
-\Needspace{30\baselineskip}
-\begin{center}
-\includegraphics[width=0.82\textwidth]{figures/appendix_c_self_discover_workflow_vector_final.pdf}
-\end{center}
-\vspace{-0.5\baselineskip}
-\noindent\footnotesize\textbf{Appendix Figure C1.} Per-input SELECT--ADAPT--IMPLEMENT plan construction in the study's SELF-DISCOVER adaptation.\normalsize\par
-\smallskip
-\noindent\footnotesize\textit{Note.} The diagram depicts the study implementation, not a new reasoning method. Unlike the task-level plan reuse in \cite{ref18}, a plan is constructed for each routed post. Modules are prompt instructions, not trained neural components. The plan is retained for inspection; quantitative evaluation uses the terminal label.\normalsize
-\par\medskip
-\Needspace{20\baselineskip}
-\noindent\textsc{Appendix Table A3b. Canonical 39-module pool used by SELF-DISCOVER}\par
-\label{tab:appendix-a3b-module-pool}
-\smallskip
-\footnotesize
-\setlength{\tabcolsep}{3.5pt}
-\renewcommand{\arraystretch}{1.14}
-\noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.25\textwidth}YY@{}}
-\toprule
-Module ID(s) & Canonical family or named style & Function in the supplied pool & Relevance to routed emotion re-evaluation \\
-\midrule
-1--9 & General problem framing & Experiment design, candidate ideas, progress measurement, simplification, assumptions, risks, alternative perspectives, long-term implications, and decomposition. & Offers generic operations from which the model may select a smaller task-relevant subset. \\
-10 & Critical Thinking & Examine evidence from multiple perspectives, question assumptions, and identify bias or logical weaknesses. & Supports comparison of isolated emotional cues with evidence from the complete post. \\
-11 & Creative Thinking & Generate unconventional alternatives beyond a default solution. & Provides alternative interpretations when the initial label is poorly supported. \\
-12 & Collaborative Thinking & Seek diverse perspectives and external expertise. & Included in the canonical pool, although the single-model implementation does not contact external reviewers. \\
-13 & Systems Thinking & Examine interdependencies, underlying causes, feedback, and the larger system. & Can connect local statements with the broader narrative context of the post. \\
-14 & Risk Analysis & Evaluate uncertainty, trade-offs, and consequences of competing choices. & Can prompt caution when evidence for the candidate labels is incomplete or conflicting. \\
-15 & Reflective Thinking & Reconsider assumptions, biases, and prior interpretations. & Supports re-examination of the Phase~1 prediction after an independent reading. \\
-16--24 & Diagnosis and evidence planning & Identify the core issue, causes, prior attempts, obstacles, data, affected stakeholders, resources, progress measures, and metrics. & Supplies problem-diagnosis and evidence-checking operations that can be adapted to emotional text. \\
-25--32 & Problem-type and constraint checks & Determine whether a problem is technical, physical, behavioral, decision-based, analytical, design-oriented, systemic, or urgent. & Helps the model decide which reasoning style is relevant; these prompts are not clinical assessments. \\
-33--37 & Solution-family and counterfactual revision & Recall typical solutions, propose alternatives, challenge the current best solution, modify it, or replace it. & Encourages alternatives to simply accepting the Phase~1 label. \\
-38--39 & Step-by-step planning and execution & Construct and implement an explicit sequential plan. & Supports the structured \texttt{IMPLEMENT} artifact used before the final answer. \\
-\bottomrule
-\end{tabularx}
-\par\smallskip
-\noindent\footnotesize\textit{Note.} These are prompt-level reasoning instructions, not learned model components. The table groups the verbatim 39-entry pool for readability; the complete wording and numbering are embedded in the released final Colab notebooks. For each routed post, \texttt{SELECT} is model-generated and may paraphrase or group canonical entries. The study therefore audits the trace but scores only the parsed final class label.\normalsize
-
-\par\medskip
-\Needspace{18\baselineskip}
-\noindent\textsc{Appendix Table A3c. Exploratory cleaned-input prompt-policy diagnostics}\par
-\label{tab:appendix-a3c-prompt-ablation}
-\smallskip
-\footnotesize
-\setlength{\tabcolsep}{3pt}
-\renewcommand{\arraystretch}{1.16}
-\noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.20\textwidth}>{\raggedright\arraybackslash}p{0.28\textwidth}Y>{\raggedright\arraybackslash}p{0.15\textwidth}@{}}
-\toprule
-Prompt policy & Defining instruction & Routed-only development result & Decision \\
-\midrule
-Llama~2 final CoT prompt & Analyze the text independently, compare with the Phase~1 label only afterward, and end with one exact class label. & 42.69\%; 38 corrected and 49 introduced (net $-11$); no parse failure. & Retained for final rerun \\
-Llama~3 shared cross-model prompt & Apply the same trajectory-aware instruction template considered for both LLM families. & 43.90\%; 32 introduced errors; seven terminal labels were not parsed reliably. & Rejected \\
-Llama~3 error-aware refinement & Add explicit evidence checks and a stricter terminal-label contract to the shared prompt. & 50.29\%; parsing was repaired, but routed-only accuracy remained below SELF-DISCOVER. & Rejected \\
-Llama~3 SELF-DISCOVER & Dynamically SELECT, ADAPT, and IMPLEMENT reasoning modules before producing one final class label. & 52.63\%; 13 corrected and seven introduced (net $+6$); no parse failure. & Retained for final rerun \\
-\bottomrule
-\end{tabularx}
-\par\smallskip
-\noindent\footnotesize\textit{Note.} These are exploratory development diagnostics from the earlier cleaned-input condition, not the final original-text end-to-end result. Internal prompt version identifiers are omitted because they are experiment-management labels rather than published methods. All rows used the same Phase~1 model, calibrated temperature, routing threshold, 171 routed IDs, LLM checkpoint family, generation settings, and evaluation definitions. The final original-text results are reported in Tables~\ref{tab:reddit-e2e-results} and~\ref{tab:paired-statistics}.\normalsize
-\par\medskip
-\normalsize
-
-
-\Needspace{12\baselineskip}
-\section{Observed Routing and Reasoning Audit Trail}
-
-This appendix retains one observed Mixed Emotion routed correction for each Phase~2 model and one observed Reddit original-text correction from the final Llama~3 run. The tables present stored output fields rather than reconstructed narratives, so a reader can trace the original Phase~1 label, text-grounded reasoning, terminal label, and evaluation reference. These are research artifacts for proxy emotion classification, not clinical diagnostic reports.
-
-For every routed Llama~3 example, the result CSV retains \texttt{LLaMA3\_SELECT}, \texttt{LLaMA3\_ADAPT}, \texttt{LLaMA3\_IMPLEMENT}, \texttt{LLaMA3\_Answer}, and the parsed \texttt{LLaMA3\_final\_label}; analogous Llama~2 records retain its independent analysis, comparison, and final-response fields. Thus, aggregate results can be traced back to case-level reasoning artifacts without treating generated text as clinical evidence.
-
-\par\medskip
-\Needspace{14\baselineskip}
-\noindent\textsc{Appendix Table A4a. Llama~2 CoT output-column example for an observed correction}\par
-\label{tab:appendix-a4-llama2-output-columns}
-\smallskip
-\footnotesize
-\setlength{\tabcolsep}{3.5pt}
-\renewcommand{\arraystretch}{1.16}
-\noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.18\textwidth}>{\raggedright\arraybackslash}p{0.43\textwidth}Y@{}}
-\toprule
-Stored column & Observed output excerpt for MEV2\_DEP\_062 & Function and interpretation \\
-\midrule
-Input and reference & ``At first, a workday felt manageable because there was encouraging feedback from someone I respect. Still, I kept track of dates, forms, and reminders, and as the day went on, there was a sharp drop in my mood once I had time to think. I tried to stay honest about the mixed nature of the experience, especially before going to sleep. What stayed longest was not the good moment, but the sadness that returned and remained unresolved.'' Reference: Depression; Phase~1: Happy. & Establishes the exact evidence, reference label, and initial error being audited. The Phase~2 model must correct the label from the text rather than from hidden metadata. \\
-\texttt{LLaMA2\_1} & ``The dominant emotion is sadness.'' The output cites the encouraging feedback, subsequent sharp mood drop, and sadness that remained unresolved, concluding that the negative trajectory dominates the initial positive cue. & Records an assessment before Phase~1 comparison and its stated emphasis on the final trajectory; it does not establish the cause of the prediction. \\
-\texttt{LLaMA2\_2} & ``I have to disagree with the Phase 1 classifier's prediction of Happy.'' The output again cites the mood drop and unresolved sadness as evidence that the initial label is inconsistent with the complete text. & Audits agreement with Phase~1 by naming the disputed label and the text-grounded reason for rejecting it. \\
-\texttt{LLaMA2\_3} & \texttt{Final label: Depression} & Supplies the canonical terminal response. The parser accepts only Depression, Neutral, or Happy and uses this label to replace the routed Phase~1 prediction. \\
-\texttt{LLaMA2\_final\_label} & Depression & Stores the machine-extractable result used in end-to-end metrics and the confusion matrix; free-form prose is retained only for audit. \\
-\bottomrule
-\end{tabularx}
-\par\medskip
-\normalsize
-
-\Needspace{26\baselineskip}
-\noindent\textsc{Appendix Table A4b. Llama~3 SELF-DISCOVER output-column example for an observed correction}\par
-\label{tab:appendix-a4-llama3-output-columns}
-\smallskip
-\footnotesize
-\setlength{\tabcolsep}{3.5pt}
-\renewcommand{\arraystretch}{1.16}
-\noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.18\textwidth}>{\raggedright\arraybackslash}p{0.43\textwidth}Y@{}}
-\toprule
-Stored column & Observed output excerpt for MEV2\_DEP\_022 & Function and interpretation \\
-\midrule
-Input and reference & ``At first, a workday felt manageable because there was encouraging feedback from someone I respect. Still, I kept track of dates, forms, and reminders, and as the day went on, there was a sharp drop in my mood once I had time to think. I realized that the ending of the day mattered more than the beginning, especially during a slow part of the evening. What stayed longest was not the good moment, but the sadness that returned and remained unresolved.'' Reference: Depression; Phase~1: Happy. & Establishes the original mixed-trajectory input and the Phase~1 error. It lets the reader compare the later reasoning trace directly with the evidence available at inference time. \\
-\texttt{LLaMA3\_SELECT} & The stored trace selects Critical Thinking, Textual Analysis, Risk Analysis, and Reflective Thinking. Its preliminary analysis notes that the post begins positively but ends with unresolved sadness and emotional distress. & Records the actual model-generated selection trace. The generated names may paraphrase entries in the canonical pool and are retained for inspection rather than treated as scored predictions. \\
-\texttt{LLaMA3\_ADAPT} & The selected operations are rewritten as Emotion-Focused Analysis, Textual Evidence-Based Judgment, Emotion Classification Risk Assessment, and Reflective Emotional Understanding. & Tailors the selected operations to the three-class task while requiring the decision to consider the full trajectory and alternative interpretations. \\
-\texttt{LLaMA3\_IMPLEMENT} & The JSON plan defines the three class guidelines, then sequences emotional-cue identification, textual-evidence review, classification-risk assessment, reflective review, and terminal-label production. & Records the generated plan before execution; the plan can be inspected, but only the final parsed label contributes to accuracy. \\
-\texttt{LLaMA3\_Answer} & ``The text starts with a positive note ... but this is short-lived. The speaker's mood drops sharply ... and the text concludes with unresolved sadness.'' It rejects Happy and ends with \texttt{Final label: Depression}. & Provides the human-readable, text-grounded assessment and directly explains why the initial Happy prediction is corrected. \\
-\texttt{LLaMA3\_final\_label} & Depression & Stores the parsed terminal label used only for routed examples; accepted examples retain their Phase~1 label. \\
-\bottomrule
-\end{tabularx}
-\par\medskip
-\normalsize
-
-Note. Tables~A4a and A4b are publication-readable renderings of actual stored routed-sample outputs. Output excerpts are shortened only for layout; the full text fields remain in the Llama~2 and Llama~3 result CSVs. The observed Mixed Emotion results are not clinical diagnostic reports.
-
-\noindent\textsc{Appendix Table A4c. Mixed Emotion end-to-end accuracy by controlled scenario}\par
-\label{tab:appendix-a4c-scenario-results}
-\smallskip
-\footnotesize
-\setlength{\tabcolsep}{3.5pt}
-\renewcommand{\arraystretch}{1.14}
-\noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X c c c c c@{}}
-\toprule
-Controlled scenario & $n$ & Routed & Phase~1 & Llama~2 e2e & Llama~3 e2e \\
-\midrule
-Blended-emotion co-occurrence & 40 & 8 & 60.0\% & 70.0\% & 72.5\% \\
-Positive-to-distress shift & 40 & 9 & 65.0\% & 77.5\% & 77.5\% \\
-Distress-to-recovery shift & 40 & 12 & 90.0\% & 95.0\% & 97.5\% \\
-Neutral framing with subtle affect & 40 & 10 & 87.5\% & 92.5\% & 97.5\% \\
-Conflicting cues, dominant trajectory & 40 & 0 & 100.0\% & 100.0\% & 100.0\% \\
-Clear technical/informational Neutral & 75 & 0 & 100.0\% & 100.0\% & 100.0\% \\
-Mild factual/procedural Neutral ambiguity & 25 & 5 & 32.0\% & 28.0\% & 36.0\% \\
-\bottomrule
-\end{tabularx}
-\par\smallskip
-\noindent\footnotesize\textit{Note.} Accepted examples retain their Phase~1 labels, so rows with zero routed examples have identical end-to-end accuracy. These scenario results are descriptive because the controlled groups are small and unequal in size. They show where the aggregate gain originates without treating the synthetic scenario distribution as representative of real-world prevalence.\normalsize
-
-\par\medskip
-\noindent\textsc{Appendix Table A4d. Observed Reddit original-text Llama~3 correction record}\par
-\label{tab:appendix-a4d-reddit-original-correction}
-\smallskip
-\footnotesize
-\setlength{\tabcolsep}{3.5pt}
-\renewcommand{\arraystretch}{1.15}
-\noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.19\textwidth}Y@{}}
-\toprule
-Stored field & Observed record for RED\_077578 \\
-\midrule
-Original Phase~2 input & ``GF is `numb' and is tired of trying to stay positive. What can I do?'' The stored proxy reference is Depression, while Phase~1 predicted Happy. \\
-\texttt{LLaMA3\_SELECT} & Selected evidence review and reflective operations; the trace identified emotional numbness, exhaustion, and the absence of a positive resolution before comparing with the Happy prediction. \\
-\texttt{LLaMA3\_ADAPT} & Reframed the selected operations as emotion extraction, analysis, comparison, reconciliation, and three-class classification. \\
-\texttt{LLaMA3\_IMPLEMENT} & Generated a JSON plan that identifies emotional cues, evaluates the overall trajectory, compares the independent assessment with Phase~1, and produces one terminal label. \\
-\texttt{LLaMA3\_Answer} & ``The text expresses feelings of being `numb' and being tired of trying to stay positive, which indicates a sense of emotional exhaustion and hopelessness.'' It rejects Happy and ends with \texttt{Final label: Depression}. \\
-Evaluation record & Phase~1: Happy (incorrect); Phase~2: Depression (correct); counted as one corrected routed error in the final original-text Reddit run. \\
-\bottomrule
-\end{tabularx}
-\par\smallskip
-\noindent\footnotesize\textit{Note.} This is an observed stored output, shortened only for publication layout. URLs and direct username patterns were masked before inference; no such pattern occurred in this excerpt. The rationale is an audit artifact and is not treated as a clinical explanation.\normalsize
-
-\par\medskip
-\Needspace{22\baselineskip}
-\noindent\textsc{Appendix Table A4e. Audit classification of representative accepted high-confidence disagreements}\par
-\label{tab:appendix-a4e-accepted-errors}
-\smallskip
-\footnotesize
-\setlength{\tabcolsep}{3.2pt}
-\renewcommand{\arraystretch}{1.16}
-\noindent\begin{tabularx}{\textwidth}{@{}p{0.10\textwidth}p{0.15\textwidth}c p{0.24\textwidth}Y@{}}
-\toprule
-ID & Reference $\rightarrow$ prediction & Conf. & Observed failure mode & Evidence in the original post \\
-\midrule
-RED\_041439 & Depression $\rightarrow$ Happy & 0.983 & Acute-distress cue missed & The title explicitly states renewed suicidal ideation; the preceding ``okay'' period is a possible conflicting cue, not a verified cause of the prediction. \\
-RED\_000879 & Depression $\rightarrow$ Happy & 0.980 & Negative-state cue missed & Blackmail, alcohol dependence, imprisonment, family separation, and loss of custody provide no textual support for Happy. \\
-RED\_026909 & Happy $\rightarrow$ Neutral & 0.995 & Technical-context dominance & A programming narrative ends with explicit unexpected achievement and pride, yet the prediction is Neutral; the role of the technical vocabulary remains a hypothesis. \\
-RED\_103680 & Happy $\rightarrow$ Neutral & 0.989 & Factual-detail dominance & Explicit excitement about a successful purchase and new project is obscured by product and pricing details. \\
-RED\_010855 & Neutral $\rightarrow$ Depression & 0.982 & Topic-term shortcut & A factual question about computerized diagnosis is classified as Depression after repeated mental-health terminology. \\
-RED\_000438 & Neutral $\rightarrow$ Happy & 0.986 & Question-form/context error & A factual question about a sound from a cup of tea contains no positive affect but receives a highly confident Happy prediction. \\
-\bottomrule
-\end{tabularx}
-\par\smallskip
-\noindent\footnotesize\textit{Note.} All six rows are accepted errors because prediction $\ne$ stored reference proxy label and confidence $\geq0.70$. Cases were selected only after exact original-post linkage and textual review; they do not estimate failure-mode prevalence.\normalsize
-
-\par\medskip
-\Needspace{20\baselineskip}
-\noindent\textsc{Appendix Table A4f. Verifiable case records for the accepted high-confidence audit}\par
-\label{tab:appendix-a4f-accepted-error-records}
-\smallskip
-\noindent\footnotesize Each excerpt was recovered from the retained \texttt{title} and \texttt{selftext} fields by normalized exact matching to the stored model input. Capitalization, punctuation, and discourse order are preserved; URLs and unnecessary identifying details are omitted. The classifier itself received the corresponding cleaned field.\normalsize
-
-\par\smallskip
-\noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.18\textwidth}Y@{}}
-\toprule
-\multicolumn{2}{@{}l}{\textbf{RED\_041439 --- explicit acute distress missed}} \\
-\midrule
-Original-post excerpt & ``Aaaannd I'm suicidal again... The past two weeks were going pretty okay too....'' \\
-Recorded decision & Reference proxy: Depression; Phase~1 prediction: Happy; calibrated confidence: 0.983; accepted at $\tau=0.70$. \\
-Text-grounded audit & The present-state title directly expresses suicidal ideation. The statement that the preceding two weeks were ``okay'' does not reverse the current distress, so Happy is inconsistent with the post's final state. \\
-Evaluation treatment & Counted as one accepted Depression $\rightarrow$ Happy error; no Phase~2 output exists because the row was not routed. \\
-\bottomrule
-\end{tabularx}
-
-\par\medskip
-\noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.18\textwidth}Y@{}}
-\toprule
-\multicolumn{2}{@{}l}{\textbf{RED\_000879 --- severe negative state predicted as Happy}} \\
-\midrule
-Original-post excerpt & ``My ex wife blackmailed me into signing child support paperwork and made me leave my home. Today I am an alcoholic headed to prison and [...] she has custody of my little boy.'' \\
-Recorded decision & Reference proxy: Depression; Phase~1 prediction: Happy; calibrated confidence: 0.980; accepted at $\tau=0.70$. \\
-Text-grounded audit & The post describes coercion, alcohol dependence, imprisonment, separation from home, and loss of custody. No positive resolution or Happy cue appears in the source text. \\
-Evaluation treatment & Counted as one accepted Depression $\rightarrow$ Happy error under the unchanged reference-label evaluation. \\
-\bottomrule
-\end{tabularx}
-
-\par\medskip
-\noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.18\textwidth}Y@{}}
-\toprule
-\multicolumn{2}{@{}l}{\textbf{RED\_026909 --- explicit pride masked by technical context}} \\
-\midrule
-Original-post excerpt & ``I'm Teaching Myself Java. My husband and I are trying to learn how to program [...] I never expected to grasp it as quickly as I am, and I'm kinda proud of myself.'' \\
-Recorded decision & Reference proxy: Happy; Phase~1 prediction: Neutral; calibrated confidence: 0.995; accepted at $\tau=0.70$. \\
-Text-grounded audit & Programming details dominate most of the post, but the explicitly stated takeaway is unexpected accomplishment and pride. Neutral misses the positive evaluative conclusion. \\
-Evaluation treatment & Counted as one accepted Happy $\rightarrow$ Neutral error. \\
-\bottomrule
-\end{tabularx}
-
-\clearpage
-\noindent\textsc{Appendix Table A4f (continued). Verifiable case records for the accepted high-confidence audit}\par
-\smallskip
-\noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.18\textwidth}Y@{}}
-\toprule
-\multicolumn{2}{@{}l}{\textbf{RED\_103680 --- excitement obscured by product details}} \\
-\midrule
-Original-post excerpt & ``I just saved \$170 off the cost of an EEPROM programmer by shopping on eBay---my new project/hobby couldn't start off with a better investment! I'm so excited! [...] Compare \$28 [...] to \$200 plus shipping.'' \\
-Recorded decision & Reference proxy: Happy; Phase~1 prediction: Neutral; calibrated confidence: 0.989; accepted at $\tau=0.70$. \\
-Text-grounded audit & The pricing and hardware terminology are factual, but the opening and stated evaluation are explicitly enthusiastic. The error is consistent with topic sensitivity, but the output alone does not establish the classifier's cue attribution. \\
-Evaluation treatment & Counted as one accepted Happy $\rightarrow$ Neutral error. \\
-\bottomrule
-\end{tabularx}
-
-\par\medskip
-\noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.18\textwidth}Y@{}}
-\toprule
-\multicolumn{2}{@{}l}{\textbf{RED\_010855 --- mental-health terminology in a factual question}} \\
-\midrule
-Original-post excerpt & ``Sorry if this is a common question but, why can't computers diagnose mental health disorders? [...] why can't an online test diagnose it? Is it just the authority they have to officially diagnose things and prescribe medicine and treatment?'' \\
-Recorded decision & Reference proxy: Neutral; Phase~1 prediction: Depression; calibrated confidence: 0.982; accepted at $\tau=0.70$. \\
-Text-grounded audit & The post discusses depression-related concepts but does not express the author's emotional state. Its function is an informational question, consistent with a possible topic-vocabulary shortcut, rather than demonstrating that mechanism. \\
-Evaluation treatment & Counted as one accepted Neutral $\rightarrow$ Depression error. \\
-\bottomrule
-\end{tabularx}
-
-\noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.18\textwidth}Y@{}}
-\toprule
-\multicolumn{2}{@{}l}{\textbf{RED\_000438 --- affect inferred from a factual science question}} \\
-\midrule
-Original-post excerpt & ``Why do I hear a high-pitched tone coming from my tea? I boiled some water, and poured it onto a tea bag, inside of a ceramic mug. [...] What's going on? And is it safe to drink?'' \\
-Recorded decision & Reference proxy: Neutral; Phase~1 prediction: Happy; calibrated confidence: 0.986; accepted at $\tau=0.70$. \\
-Text-grounded audit & The post reports an observation and asks for an explanation and safety information. It contains no explicit positive emotional evaluation, making Happy unsupported by the original text. \\
-Evaluation treatment & Counted as one accepted Neutral $\rightarrow$ Happy error. \\
-\bottomrule
-\end{tabularx}
-
-\par\smallskip
-\noindent\footnotesize\textit{Audit scope.} These six cases illustrate error patterns, not verified causal mechanisms or their prevalence. Quotations retain source wording except for URL removal and bracketed shortening. The complete prediction-level accepted-error file is included with the reproducibility package.\normalsize
-
-\par\medskip
-\noindent\textsc{Appendix Table A4g. Routing concentration and conditional correction opportunity}\par
-\label{tab:appendix-a4g-routing-decomposition}
-\smallskip
-\footnotesize
-For an evaluation set of size $N$ with $E$ Phase~1 errors, let $n_R$ be the number routed and $E_R$ the number of Phase~1 errors among them. The routed-error enrichment and conditional oracle accuracy are
-\[
-\mathrm{Enrichment}=\frac{E_R/n_R}{E/N},
-\qquad
-\mathrm{Acc}_{\mathrm{oracle}\mid R}=\mathrm{Acc}_{\mathrm{P1}}+\frac{E_R}{N}.
-\]
-Accuracies in the equation are proportions; the table reports percentages. For reasoner $m$, the realized share of the fixed routing opportunity is
-\[
-\rho_m=\frac{\mathrm{Corrected}_m-\mathrm{Introduced}_m}{E_R}.
-\]
-This share is negative when introduced errors outnumber corrections.
-\par\smallskip
-\footnotesize
-\setlength{\tabcolsep}{1.5pt}
-\renewcommand{\arraystretch}{1.15}
-\noindent\begin{tabularx}{0.98\textwidth}{@{}l c c c c c c c@{}}
-\toprule
-Dataset & Full err. & Routed err. & Enrichment & Routed P1 & Llama~2 routed & Llama~3 routed & Oracle e2e \\
-\midrule
-Reddit & 3.31\% & 50.88\% & 15.38$\times$ & 49.12\% & 47.37\% & 66.67\% & 97.42\% \\
-Mixed Emotion & 18.67\% & 47.73\% & 2.56$\times$ & 52.27\% & 79.55\% & 93.18\% & 88.33\% \\
-\bottomrule
-\end{tabularx}
-\par\smallskip
-\noindent\footnotesize\textit{Note.} Reddit has $N=12{,}000$, $E=397$, $n_R=171$, and $E_R=87$; Mixed Emotion has $N=300$, $E=56$, $n_R=44$, and $E_R=21$. Under the fixed router, Llama~2 and Llama~3 realize $-3.45\%$ and 34.48\% of the Reddit correction opportunity, and 57.14\% and 85.71\% of the Mixed Emotion opportunity. The oracle assumes perfect correction of routed Phase~1 errors with no introduced errors. It is a descriptive upper bound under the fixed routing policy, not a trained comparator and not a basis for threshold selection.\normalsize
-
-
-\par\medskip
-\Needspace{16\baselineskip}
-\section{Synthetic Mixed Emotion Dataset Generation Protocol}
-
-\par\medskip
-\noindent\textsc{Appendix Table A5. Synthetic Mixed Emotion Dataset generation protocol}\par
-\label{tab:appendix-a5-mixed-emotion-protocol}
-\smallskip
-\footnotesize
 \setlength{\tabcolsep}{4pt}
 \renewcommand{\arraystretch}{1.08}
 \noindent\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.20\textwidth}Y@{}}
@@ -1392,49 +1050,18 @@
 \par\medskip


-\noindent\footnotesize\textit{Generation prompt used for synthetic stress-test examples.}\normalsize\par
+\noindent\small\textit{Generation prompt used for synthetic stress-test examples.}\normalsize\par
 \smallskip
 \noindent
-\begin{minipage}[t]{0.49\textwidth}\scriptsize
+\begingroup\small
 Generate synthetic Reddit-style posts for a controlled mixed-emotion stress-test dataset. Use one of three target labels: Depression, Neutral, or Happy. For Depression and Happy examples, include mixed or shifting cues but make the final emotional trajectory clear to a trained reviewer. For Neutral examples, retain low affective intensity: include either technical/informational content or mild factual/procedural ambiguity without sustained distress, relief, accomplishment, or other dominant affect. When multiple emotions occur, the target label must follow the overall message and final takeaway rather than an isolated phrase.
-\end{minipage}\hfill
-\begin{minipage}[t]{0.49\textwidth}\scriptsize
+\par\endgroup\medskip
+\begingroup\small
 For each example, write one realistic post between 60 and 90 words. Do not include explicit diagnosis claims, medication names, therapy claims, suicide, self-harm, crisis language, usernames, URLs, subreddit names, hashtags, or personally identifying information. Avoid duplicated phrasing and avoid making the label obvious through a single keyword. Return structured records with: example identifier, target label, scenario type, primary context, dominant emotional trajectory, text, and brief label rationale.
-\end{minipage}
-
-\par\medskip
-\Needspace{19\baselineskip}
-\section{Additional Holdout Class-Wise Results}
-\label{app:additional-holdout}
-
-Table~A6 supplements the full-set results in Table~\ref{tab:additional-holdout} with class-wise F1 and accuracy on the shared routed subset. Both configurations improved Depression and Happy F1, whereas Neutral F1 decreased slightly. The evaluation protocol is described in Section~\ref{sec:additional-holdout-protocol}.
-
-\par\medskip
-\Needspace{13\baselineskip}
-\noindent\textsc{Appendix Table A6. Additional holdout: class-wise F1 and routed accuracy}\par
-\smallskip
-\small
-\setlength{\tabcolsep}{5pt}
-\renewcommand{\arraystretch}{1.16}
-\noindent\begin{tabularx}{\textwidth}{@{}Y c c c c@{}}
-\toprule
-Configuration & Depression F1 & Neutral F1 & Happy F1 & Routed accuracy \\
-\midrule
-Phase~1 & 95.87\% & 98.82\% & 95.52\% & 51.09\% \\
-+ Llama~2 CoT & 96.23\% & 98.53\% & 95.94\% & 62.04\% \\
-+ Llama~3 SELF-DISCOVER & 96.36\% & 98.67\% & 96.07\% & 70.80\% \\
-\bottomrule
-\end{tabularx}
-\par\smallskip
-\noindent\footnotesize\textit{Note.} Class F1 uses all 9,000 posts; routed accuracy uses the same 137 posts for each configuration. Both re-evaluators increased overall macro F1 but slightly reduced Neutral F1. These are fixed-run results, not means across training seeds.\normalsize
-
-% REQUIRED BEFORE SUBMISSION:
-% 1. Add the final Acknowledgment text, including funding and the IEEE-required
-%    disclosure of any AI-generated manuscript text and the affected sections.
-% 2. Add one IEEEbiographynophoto or IEEEbiography environment for every author.
-% Example:
-% \begin{IEEEbiographynophoto}{Full Author Name}
-% Short biography required by IEEE Access.
-% \end{IEEEbiographynophoto}
+\par\endgroup
+
+\par\medskip
+
+
 \EOD
 \end{document}
```
