# 원고 전체 변경 내역

```diff
--- 2026-09-09/main.tex
+++ 2026-09-11/main.tex
@@ -38,7 +38,7 @@
 \corresp{Corresponding-author details will be inserted in the submission version.}
 
 \begin{abstract}
-Transformer classifiers can process social-media text at scale, but confidence-based selection does not ensure that a second model will correct the selected predictions. We evaluate a two-phase framework for three-class proxy emotion classification (Depression, Neutral, and Happy) that separates this selection problem from LLM re-evaluation. Phase~1 uses temperature-calibrated DistilBERT predictions and a calibration-set risk--coverage criterion to route low-confidence inputs. In a three-seed comparison on the same 12,000 Reddit test posts, DistilBERT achieved 96.90 $\pm$ 0.12\% accuracy, versus 95.56 $\pm$ 0.11\% for Mistral~7B and 95.09 $\pm$ 0.06\% for Llama~2~7B frozen-backbone linear probes. The independently fixed operational DistilBERT checkpoint achieved 96.69\% accuracy and routed 171 posts (1.42\%), including 87 of its 397 errors. Phase~2 used minimally sanitized original text and either Llama~2 Chain-of-Thought or Llama~3 SELF-DISCOVER. Llama~2 introduced three more errors than it corrected, yielding 96.67\% end-to-end accuracy. Llama~3 produced 30 net corrections and 96.94\% accuracy; its paired improvement remained significant after Holm correction. On a 300-example synthetic Mixed Emotion stress test, the two configurations increased accuracy from 81.33\% to 85.33\% and 87.33\%, respectively. Thus, routing concentrated errors in a small subset, but correction gains depended on the re-evaluation configuration. Because prompt development reused the routed Reddit cases, the end-to-end findings are exploratory. The study evaluates proxy emotion labels, not clinical diagnoses; independent data and expert review are needed to establish broader validity.
+Transformer classifiers can process social-media text at scale, but confidence-based selection does not ensure that a second model will correct the selected predictions. We evaluate a two-phase framework for three-class proxy emotion classification (Depression, Neutral, and Happy) that separates this selection problem from LLM re-evaluation. Phase~1 uses temperature-calibrated DistilBERT predictions and a calibration-set risk--coverage criterion to route low-confidence inputs. In a three-seed comparison on the same 12,000 Reddit test posts, DistilBERT achieved 96.88 $\pm$ 0.04\% accuracy, versus 95.36 $\pm$ 0.08\% for Mistral~7B and 95.17 $\pm$ 0.07\% for Llama~2~7B frozen-backbone linear probes. A common-A100 benchmark also showed substantially higher batched throughput and lower single-post latency for DistilBERT. The independently fixed operational DistilBERT checkpoint achieved 96.69\% accuracy and routed 171 posts (1.42\%), including 87 of its 397 errors. Phase~2 used minimally sanitized original text and either Llama~2 Chain-of-Thought or Llama~3 SELF-DISCOVER. Llama~2 introduced three more errors than it corrected, yielding 96.67\% end-to-end accuracy. Llama~3 produced 30 net corrections and 96.94\% accuracy; its paired improvement remained significant after Holm correction. On a 300-example synthetic Mixed Emotion stress test, the two configurations increased accuracy from 81.33\% to 85.33\% and 87.33\%, respectively. Thus, routing concentrated errors in a small subset, but correction gains depended on the re-evaluation configuration. Because prompt development reused the routed Reddit cases, the end-to-end findings are exploratory. The study evaluates proxy emotion labels, not clinical diagnoses; independent data and expert review are needed to establish broader validity.
 \end{abstract}
 
 \begin{keywords}
@@ -66,7 +66,7 @@
 
 The study makes four concrete contributions:
 \begin{itemize}
-\item It operationalizes selective LLM re-evaluation as an end-to-end pipeline in which an efficient classifier handles high-confidence inputs and reasoning models receive only calibration-defined low-confidence inputs. The Phase~1 comparison is designed to support operational model selection, not to claim exhaustive optimization of every 7B architecture.
+\item It operationalizes selective LLM re-evaluation as an end-to-end pipeline in which an efficient classifier handles high-confidence inputs and reasoning models receive only calibration-defined low-confidence inputs. The Phase~1 comparison supports operational model selection through predictive performance and measured inference throughput, latency, and memory on a common accelerator, rather than claiming exhaustive optimization of every 7B architecture.
 \item It replaces a heuristic confidence cutoff with a reproducible calibration protocol: temperature scaling, a prespecified threshold grid, a one-sided selective-risk upper bound, and maximum-coverage selection among feasible thresholds. The selected policy is fixed before held-out and Phase~2 evaluation.
 \item It evaluates routing quality separately from re-evaluation quality. Coverage, selective risk, error capture, corrected errors, introduced errors, and net corrections reveal whether the router finds difficult inputs and whether the LLM actually improves them.
 \item It contributes a controlled Mixed Emotion stress test and case-level reasoning audit trail for studying posts whose surface cues and dominant emotional trajectory diverge. The stress test is explicitly separated from model training and threshold selection.
@@ -434,7 +434,9 @@
 
 \subsection{Computing Environment}
 
-The experiments were implemented in Python with PyTorch, Hugging Face Transformers, Datasets, scikit-learn, SciPy, and pandas; W\&B recorded the model-selection sweeps. Phase~1 used \texttt{distilbert-base-uncased} as the operational checkpoint and prespecified 7B base checkpoints for the Mistral and Llama~2 comparison classifiers. Routed Phase~2 inference used \texttt{NousResearch/Llama-2-7b-chat-hf} and \texttt{NousResearch/Meta-Llama-3-8B-Instruct}, loaded in 4-bit form with bitsandbytes in GPU-backed Google Colab sessions. Llama~2 CoT generation used a 256-token limit with sampling temperature 0.6 and top-$p$ 0.9; the Llama~3 SELF-DISCOVER structure stages used a 1,024-token limit and the terminal answer was decoded deterministically. The repository notebooks print installed package versions at runtime and save prediction-level outputs after every routed row. Hardware-dependent timing is not reported as a comparative result because the runs were not executed under a controlled latency benchmark.
+The experiments were implemented in Python with PyTorch, Hugging Face Transformers, Datasets, scikit-learn, SciPy, and pandas; W\&B recorded the model-selection sweeps. Phase~1 used \texttt{distilbert-base-uncased} as the operational checkpoint and prespecified 7B base checkpoints for the Mistral and Llama~2 comparison classifiers. Routed Phase~2 inference used \texttt{NousResearch/Llama-2-7b-chat-hf} and \texttt{NousResearch/Meta-Llama-3-8B-Instruct}, loaded in 4-bit form with bitsandbytes in GPU-backed Google Colab sessions. Llama~2 CoT generation used a 256-token limit with sampling temperature 0.6 and top-$p$ 0.9; the Llama~3 SELF-DISCOVER structure stages used a 1,024-token limit and the terminal answer was decoded deterministically. The repository notebooks print installed package versions at runtime and save prediction-level outputs after every routed row.
+
+The three Phase~1 comparison configurations were trained and profiled on NVIDIA A100-SXM4-40GB accelerators in the same software environment. The inference benchmark used fp16, a 256-token cap, and batch size 32 over the 12,000-post held-out split; single-post latency was measured separately on 512 posts. Three warm-up batches were discarded, and each benchmark was repeated three times. CUDA synchronization bracketed timed regions, which included host-to-device transfers. NVML sampled GPU utilization and board power at 1~Hz. Appendix Table~A1d reports the measurement protocol and stage-level resource use. Phase~2 generation time, tokens, and energy were not measured in these experiments; routing volume is reported separately from computational savings.
 
 \subsection{Model Selection and Calibration Protocol}
 
@@ -454,7 +456,7 @@
 
 \subsection{Evaluation Metrics}
 
-The metrics were organized by the research question they answer rather than treated as interchangeable indicators. For RQ1a, predictive performance was evaluated with accuracy and class-wise precision, recall, and F1, with unweighted macro averages reported across Depression, Neutral, and Happy; calibration was evaluated with NLL, Brier score, ECE, and adaptive ECE. For RQ1b, matched classifier performance is reported as mean $\pm$ standard deviation across three seeds and interpreted together with model parameter scale and trainable scope; no uncontrolled wall-clock comparison is claimed. For RQ2, routing was evaluated with coverage, routing rate, accepted-set risk and its one-sided upper bound, error capture, and routing precision. For RQ3, the primary end-to-end outcome was the example-level change in correctness after accepted Phase~1 predictions were recombined with routed Phase~2 labels. This change was summarized by full-set accuracy change and by corrected, introduced, and net errors. Calibration and routing measures are supporting diagnostics: they establish whether the confidence policy is credible and whether it concentrates difficult cases, but they do not by themselves establish that Phase~2 improves the final system.
+The metrics were organized by the research question they answer rather than treated as interchangeable indicators. For RQ1a, predictive performance was evaluated with accuracy and class-wise precision, recall, and F1, with unweighted macro averages reported across Depression, Neutral, and Happy; calibration was evaluated with NLL, Brier score, ECE, and adaptive ECE. For RQ1b, matched classifier performance is reported as mean $\pm$ standard deviation across three seeds and interpreted together with model parameter scale, trainable scope, and the common-hardware inference benchmark. Timing variability is reported separately as mean $\pm$ SD across three measurement repetitions. For RQ2, routing was evaluated with coverage, routing rate, accepted-set risk and its one-sided upper bound, error capture, and routing precision. For RQ3, the primary end-to-end outcome was the example-level change in correctness after accepted Phase~1 predictions were recombined with routed Phase~2 labels. This change was summarized by full-set accuracy change and by corrected, introduced, and net errors. Calibration and routing measures are supporting diagnostics: they establish whether the confidence policy is credible and whether it concentrates difficult cases, but they do not by themselves establish that Phase~2 improves the final system.
 
 Phase~1 and end-to-end correctness are paired for each example. Two-sided exact McNemar tests use the discordant counts: corrected Phase~1 errors and newly introduced errors. Paired percentile-bootstrap 95\% confidence intervals use 50,000 row-level resamples and seed 20260813. Holm adjustment covers the four dataset--reasoner comparisons. These analyses characterize the observed prediction pairs conditional on the fitted models and selected prompts; they do not account for prompt selection, repeated LLM generation, or dependence between posts from the same author. The three-seed intervals in the classifier comparison instead describe variation across training runs on a common test set.
 
@@ -464,40 +466,58 @@
 
 \subsection{Phase 1 Performance on Reddit Data}
 
-The matched Phase~1 comparison used the same class-balanced 120,000-post sample, prespecified 70/10/10/10 partitions, 12,000-example held-out test set, maximum input length of 256 tokens, four-trial Bayesian search budget, and random seeds 42, 43, and 44 for all three architectures. Table~\ref{tab:phase1-performance} reports the completed predictive results; training configurations are documented in Appendix Table~A1c.
+The matched Phase~1 comparison used the same class-balanced 120,000-post sample, prespecified 70/10/10/10 partitions, 12,000-example held-out test set, maximum input length of 256 tokens, four-trial Bayesian search budget, and random seeds 42, 43, and 44 for all three architectures. Table~\ref{tab:phase1-performance} reports predictive performance and inference efficiency for these configurations; training configurations are documented in Appendix Table~A1c.
 
 \begin{table*}[t]
 \centering
-\caption{Matched Phase~1 classifier comparison on the held-out Reddit test set}
+\caption{Matched Phase~1 predictive performance and inference efficiency}
 \label{tab:phase1-performance}
 \footnotesize
 \setlength{\tabcolsep}{4.2pt}
 \renewcommand{\arraystretch}{1.16}
-\begin{tabularx}{\textwidth}{@{}l c c c c@{}}
+\textbf{(a) Predictive performance on the common 12,000-post test set}\par\smallskip
+\begin{tabularx}{\textwidth}{@{}l *{4}{>{\centering\arraybackslash}X}@{}}
 \toprule
 Model & Accuracy, mean $\pm$ SD & Accuracy 95\% CI & Macro F1, mean $\pm$ SD & Macro F1 95\% CI \\
 \midrule
-DistilBERT & 96.90 $\pm$ 0.12\% & [96.60, 97.20]\% & 96.90 $\pm$ 0.12\% & [96.60, 97.19]\% \\
-Mistral~7B & 95.56 $\pm$ 0.11\% & [95.29, 95.83]\% & 95.56 $\pm$ 0.11\% & [95.30, 95.82]\% \\
-Llama~2~7B & 95.09 $\pm$ 0.06\% & [94.94, 95.24]\% & 95.09 $\pm$ 0.06\% & [94.94, 95.24]\% \\
+DistilBERT & 96.88 $\pm$ 0.04\% & [96.79, 96.98]\% & 96.88 $\pm$ 0.04\% & [96.79, 96.98]\% \\
+Mistral~7B & 95.36 $\pm$ 0.08\% & [95.15, 95.57]\% & 95.36 $\pm$ 0.08\% & [95.16, 95.56]\% \\
+Llama~2~7B & 95.17 $\pm$ 0.07\% & [95.00, 95.34]\% & 95.17 $\pm$ 0.07\% & [94.99, 95.34]\% \\
 \bottomrule
 \end{tabularx}
 \vspace{2pt}
-\parbox{\textwidth}{\footnotesize \textit{Note.} Values are percentages over seeds 42, 43, and 44 on the same 12,000 test examples. Confidence intervals are two-sided Student-$t$ intervals across the three seed-level results. The comparison evaluates a bounded operational screening protocol; it does not estimate fully tuned 7B performance.}
+\parbox{\textwidth}{\footnotesize \textit{Note.} Panel (a) reports percentages over training seeds 42, 43, and 44. Confidence intervals are two-sided Student-$t$ intervals across the three seed-level results. The comparison evaluates a bounded operational screening protocol; it does not estimate fully tuned 7B performance.}
+\par\medskip
+\textbf{(b) Inference on NVIDIA A100-SXM4-40GB}\par\smallskip
+\begin{tabularx}{\textwidth}{@{}l >{\centering\arraybackslash}X >{\centering\arraybackslash}X >{\centering\arraybackslash}X@{}}
+\toprule
+Model & Throughput (posts/s) & Single-post p50 (ms) & Peak memory (GB) \\
+\midrule
+DistilBERT & 4,959.5 $\pm$ 15.5 & 4.80 $\pm$ 0.03 & 0.29 \\
+Mistral~7B & 51.7 $\pm$ 0.0 & 36.62 $\pm$ 0.06 & 15.22 \\
+Llama~2~7B & 55.5 $\pm$ 0.0 & 34.05 $\pm$ 0.10 & 14.05 \\
+\bottomrule
+\end{tabularx}
+\par\smallskip
+\parbox{\textwidth}{\footnotesize \textit{Note.} Panel (b) reports mean $\pm$ SD across three timing repetitions, distinct from the training seeds in panel (a). Throughput uses batch size 32 and all 12,000 test posts; p50 is the median latency for 512 posts processed individually. All models use fp16 and a 256-token cap. Memory refers to batched inference. An SD displayed as 0.0 reflects rounding. Appendix Table~A1d provides p95 latency and the full protocol.}
 \end{table*}
 
-Figure~\ref{fig:phase1-model-comparison} visualizes the same observed three-seed results and their 95\% confidence intervals. DistilBERT achieved 96.90 $\pm$ 0.12\% accuracy, compared with 95.56 $\pm$ 0.11\% for Mistral and 95.09 $\pm$ 0.06\% for Llama~2. Macro F1 followed the same ordering. Under this fixed comparison protocol, DistilBERT exceeded Mistral and Llama~2 by 1.34 and 1.81 percentage points in mean accuracy, respectively.
+Figure~\ref{fig:phase1-model-comparison} visualizes the same observed three-seed results and their 95\% confidence intervals. DistilBERT achieved 96.88 $\pm$ 0.04\% accuracy, compared with 95.36 $\pm$ 0.08\% for Mistral and 95.17 $\pm$ 0.07\% for Llama~2. Macro F1 showed the same pattern. Under this fixed comparison protocol, DistilBERT exceeded Mistral and Llama~2 by 1.52 and 1.71 percentage points in mean accuracy, respectively. The two probes differed by 0.19 points; their relative ordering is not the basis of the operational choice. 
 
 \begin{figure*}[t]
 \centering
 \includegraphics[width=0.88\textwidth]{figures/figure_e_phase1_model_comparison.pdf}
-\caption{Observed Phase~1 classifier performance on the common 12,000-example held-out Reddit test set. Columns show three-seed means and whiskers show Student-$t$ 95\% confidence intervals for accuracy and macro F1. The vertical axis is truncated to make the between-model differences legible.}
+\caption{Observed Phase~1 classifier performance on the common 12,000-example held-out Reddit test set. Columns show three-seed means and whiskers show Student-$t$ 95\% confidence intervals for accuracy and macro F1. These intervals describe training-run variation, not repeated timing measurements. The vertical axis is truncated to make the between-model differences legible.}
 \label{fig:phase1-model-comparison}
 \end{figure*}
 
-DistilBERT had the highest mean score and approximately 66 million parameters, compared with about seven billion for each comparator. These results support retaining it for the tested pipeline, without establishing superiority over fully adapted 7B models. Depression--Happy confusions accounted for 75.4\% of Llama~2 errors and 74.2\% of Mistral errors across pooled seed predictions; Neutral F1 exceeded 98\% for both. Pooling summarizes error composition, not 36,000 independent test examples. Small probe seed deviations also reflect the frozen backbone and should not be interpreted as equivalent training variability across adaptation methods.
-
-The matched comparison and the downstream experiment serve different purposes. The former summarizes three-seed classifier performance; the latter retains the independently fixed operational DistilBERT checkpoint with 96.69\% accuracy. Its predictions, temperature, and routed IDs were not replaced after the comparison. All subsequent results refer to that single operational checkpoint.
+DistilBERT had the highest mean score and approximately 67 million parameters, compared with about seven billion for each comparator. These results support retaining it for the tested pipeline, without establishing superiority over fully adapted 7B models. Depression--Happy confusions accounted for 72.8\% of DistilBERT errors, 75.6\% of Llama~2 errors, and 74.3\% of Mistral errors across pooled seed predictions; Neutral F1 exceeded 98\% for all three. Pooling summarizes error composition, not 36,000 independent test examples. Small probe seed deviations also reflect the frozen backbone and should not be interpreted as equivalent training variability across adaptation methods.
+
+The matched comparison and the downstream experiment serve different purposes. The former summarizes three-seed classifier performance; the latter retains the independently fixed operational DistilBERT checkpoint with 96.69\% accuracy. Its predictions, temperature, and routed IDs were not replaced after the comparison. All calibration, routing, and end-to-end results refer to that single operational checkpoint.
+
+\textit{Phase~1 inference efficiency.} Table~\ref{tab:phase1-performance}(b) adds a measured computational basis for this choice. On the common A100 benchmark, DistilBERT processed 4,959.5 posts/s, giving 89.4- and 95.9-fold higher batched throughput than the Llama~2 and Mistral probes. Its median single-post latency was 4.80~ms, compared with 34.05 and 36.62~ms, and its peak inference memory was 0.29~GB versus 14.05 and 15.22~GB. These are classifier inference measurements, not speedups of the complete two-phase system. Because Phase~1 processes every input, its low per-post cost complements selective use of the more expensive reasoners. The benchmark profiles the matched comparison configurations; it does not replace or remeasure the fixed downstream checkpoint.
+
+The one-off classifier build costs were much closer: 0.59, 0.68, and 0.70 GPU-hours for DistilBERT, Llama~2, and Mistral, respectively. The probe totals include 0.64--0.67 GPU-hours of backbone feature extraction before head fitting. Thus, the main computational advantage of DistilBERT in this comparison is repeated inference rather than a large reduction in model-development cost. Appendix Table~A1d separates these stages and reports the repeated latency measurements.
 
 The operational checkpoint's calibration diagnostics are reported next. Table~\ref{tab:calibration-quality} reports the completed DistilBERT results before and after temperature scaling. NLL evaluates the probability assigned to the true class and strongly penalizes confident mistakes; the Brier score evaluates the full three-class probability vector; and ECE summarizes the gap between confidence and observed accuracy across confidence groups. On the calibration partition, NLL fell from 0.1411 to 0.1041, Brier score from 0.0559 to 0.0514, fixed-width ECE from 0.0244 to 0.0083, and adaptive ECE from 0.0244 to 0.0138.
 
@@ -695,9 +715,9 @@
 
 \subsection{Interpretation of Routing and Re-Evaluation Results}
 
-The three research questions receive distinct answers. For \textbf{RQ1a}, DistilBERT achieved strong held-out predictive performance, and temperature scaling reduced NLL, Brier score, and fixed-bin ECE on the calibration split. For \textbf{RQ1b}, the completed three-seed comparison retained DistilBERT: its mean held-out accuracy and macro F1 were both 96.90\%, compared with 95.56\% for the Mistral frozen-backbone probe and 95.09\% for the Llama~2 probe. This is evidence for the operational choice under the bounded comparison, not a ranking of maximally tuned architectures. For \textbf{RQ2}, the fixed confidence policy preserved 98.58\% Reddit coverage while concentrating 87 of the 397 Phase~1 errors in the 171 routed posts; routed-sample accuracy was 49.12\%, compared with 96.69\% overall. The same fixed policy identified a 44-example Mixed Emotion subset with 52.27\% Phase~1 accuracy, compared with 81.33\% overall. For \textbf{RQ3}, re-evaluation benefit was model-dependent. Llama~2 remained statistically indistinguishable from Phase~1 on Reddit, whereas Llama~3 raised routed-only accuracy to 66.67\% and produced a positive paired full-test effect. On the controlled stress test, Llama~2 and Llama~3 raised routed-only accuracy to 79.55\% and 93.18\%, respectively, with Llama~3 avoiding the six introduced errors observed for Llama~2. Confidence therefore identifies a correction opportunity, but the reasoning policy determines how much of that opportunity is realized.
-
-Only 1.42\% of Reddit posts and 14.67\% of Mixed Emotion examples underwent re-evaluation. These are fractions of inputs, not counts of individual LLM calls: CoT and SELF-DISCOVER can require multiple generation stages per routed post. Wall-clock time, token use, energy, and monetary cost were not benchmarked. The stress-test routing rate characterizes its constructed scenario mix, not an expected deployment workload.
+The three research questions receive distinct answers. For \textbf{RQ1a}, DistilBERT achieved strong held-out predictive performance, and temperature scaling reduced NLL, Brier score, and fixed-bin ECE on the calibration split. For \textbf{RQ1b}, the completed three-seed comparison retained DistilBERT: its mean held-out accuracy and macro F1 were both 96.88\%, compared with 95.36\% for the Mistral frozen-backbone probe and 95.17\% for the Llama~2 probe. This is evidence for the operational choice under the bounded comparison, reinforced by the measured inference advantage, not a ranking of maximally tuned architectures. For \textbf{RQ2}, the fixed confidence policy preserved 98.58\% Reddit coverage while concentrating 87 of the 397 Phase~1 errors in the 171 routed posts; routed-sample accuracy was 49.12\%, compared with 96.69\% overall. The same fixed policy identified a 44-example Mixed Emotion subset with 52.27\% Phase~1 accuracy, compared with 81.33\% overall. For \textbf{RQ3}, re-evaluation benefit was model-dependent. Llama~2 remained statistically indistinguishable from Phase~1 on Reddit, whereas Llama~3 raised routed-only accuracy to 66.67\% and produced a positive paired full-test effect. On the controlled stress test, Llama~2 and Llama~3 raised routed-only accuracy to 79.55\% and 93.18\%, respectively, with Llama~3 avoiding the six introduced errors observed for Llama~2. Confidence therefore identifies a correction opportunity, but the reasoning policy determines how much of that opportunity is realized.
+
+Only 1.42\% of Reddit posts and 14.67\% of Mixed Emotion examples underwent re-evaluation. These are fractions of inputs, not counts of individual LLM calls: CoT and SELF-DISCOVER can require multiple generation stages per routed post. The Phase~1 benchmark establishes the inference-cost advantage of the selected classifier configuration, whereas total cascade time and monetary savings require Phase~2 measurements. Every input still incurs Phase~1 computation, so the fraction avoiding re-evaluation is not a measured GPU-time saving. The stress-test routing rate characterizes its constructed scenario mix, not an expected deployment workload.
 
 A conditional routing oracle further clarifies the scale of the observed gains without introducing a new tuned baseline. If every routed Phase~1 error were corrected and no correct routed prediction were changed, Reddit accuracy could rise from 96.69\% to at most 97.42\% under the fixed 171-post routing policy; the 30 Llama~3 net corrections realize 34.5\% of the 87-error correction opportunity. On Mixed Emotion, the corresponding ceiling is 88.33\%, and the 18 Llama~3 net corrections realize 85.7\% of the 21-error opportunity. These ceilings are conditional on the already selected router and are not claims about an attainable model. Their purpose is to distinguish a limited routing opportunity from a failure to exploit routed cases. Definitions and the complete decomposition are reported in Appendix Table~A4g.
 
@@ -715,11 +735,11 @@
 
 \textit{Prompt development and statistical scope.} Prompt variants were examined on the routed Reddit cases under the earlier cleaned-input condition. Although prompts were frozen before the original-text rerun, the same IDs and labels were reused. The end-to-end comparisons are therefore exploratory, not independent confirmatory tests. Bootstrap intervals and Holm-adjusted McNemar tests quantify the final paired outcomes but do not remove this development dependence. The next validation should freeze prompts and input handling on a separate development subset, then evaluate once on independent data. This issue is distinct from the calibration-partition fitting of $T^*$ and $\tau^*$.
 
-\textit{Computational and explanatory scope.} Routing rates show how many posts avoid re-evaluation, but no controlled GPU-time, token, or cost comparison was performed. Our per-post SELF-DISCOVER adaptation does not reuse a task-level plan, so efficiency results from the original method cannot be transferred to this implementation. Generated rationales remain unvalidated explanations, and multistage inference may have different overhead across configurations. Controlled resource measurements and independent review of both labels and rationales are the most direct extensions of the present evaluation.
+\textit{Computational and explanatory scope.} The controlled Phase~1 benchmark compares our implementations on one accelerator model and serving configuration, not optimized inference systems across hardware platforms. Three timing repetitions characterize within-session variation; they do not capture all variation across Colab sessions. GPU-board energy estimates exclude CPU and system-level consumption and have limited temporal resolution for short timed regions. Phase~2 generation was not instrumented, so the results do not establish total cascade runtime, energy, or monetary savings. Our per-post SELF-DISCOVER adaptation does not reuse a task-level plan, so efficiency results from the original method cannot be transferred to this implementation. Generated rationales remain unvalidated explanations, and multistage inference may have different overhead across configurations. Phase~2 resource measurements and independent review of both labels and rationales are the next steps.
 
 \section{Conclusion}
 
-This study evaluated selective LLM re-evaluation by separating classifier performance, calibration, error concentration, and net correction. The three-seed comparison supported retaining DistilBERT within the tested configurations. Its fixed operational checkpoint routed 1.42\% of Reddit posts, capturing 87 of 397 Phase~1 errors. Llama~3 SELF-DISCOVER corrected 42 errors and introduced 12, raising full-set accuracy from 96.69\% to 96.94\%; Llama~2 CoT produced no detectable improvement. On the synthetic stress test, the corresponding gains were 6.00 and 4.00 percentage points.
+This study evaluated selective LLM re-evaluation by separating classifier performance, calibration, error concentration, and net correction. The three-seed comparison and common-hardware inference benchmark supported retaining DistilBERT within the tested configurations: it combined the highest mean predictive scores with substantially higher batched throughput, lower single-post latency, and lower inference memory. Its fixed operational checkpoint routed 1.42\% of Reddit posts, capturing 87 of 397 Phase~1 errors. Llama~3 SELF-DISCOVER corrected 42 errors and introduced 12, raising full-set accuracy from 96.69\% to 96.94\%; Llama~2 CoT produced no detectable improvement. On the synthetic stress test, the corresponding gains were 6.00 and 4.00 percentage points.
 
 The central finding is that selecting an error-prone subset and correcting it are separate requirements. High routing concentration alone did not ensure a positive end-to-end effect, and accepted errors remained outside the correction path. The reported gains describe fixed prediction pairs under an exploratory prompt-development protocol. Independent evaluation with frozen prompts and expert-reviewed labels is needed to establish generalization. The framework remains a proxy emotion-classification pipeline, not a clinical diagnostic system.
 
@@ -889,9 +909,9 @@
 \toprule
 Model & Training & Learning rate & Batch & Epochs & Weight decay \\
 \midrule
-DistilBERT & Full fine-tuning & $5.717\times10^{-5}$ & 32 & 2 & $10^{-3}$ \\
-Llama~2~7B & Linear probe & $2.824\times10^{-4}$ & 64 & 3 & $10^{-4}$ \\
-Mistral~7B & Linear probe & $2.292\times10^{-4}$ & 16 & 2 & $10^{-2}$ \\
+DistilBERT & Full fine-tuning & $4.037\times10^{-5}$ & 64 & 3 & $10^{-3}$ \\
+Llama~2~7B & Linear probe & $4.151\times10^{-4}$ & 32 & 3 & $10^{-4}$ \\
+Mistral~7B & Linear probe & $2.399\times10^{-4}$ & 64 & 2 & $10^{-4}$ \\
 \bottomrule
 \end{tabularx}
 \par\smallskip
@@ -901,6 +921,54 @@
 \par\medskip
 
 
+
+\Needspace{39\baselineskip}
+\noindent\textsc{Appendix Table A1d. Phase~1 computational measurements}\par
+\label{tab:appendix-a1d-efficiency}
+\smallskip
+\small
+\setlength{\tabcolsep}{4pt}
+\renewcommand{\arraystretch}{1.14}
+\noindent\textbf{(a) One-off classifier development}\par\smallskip
+\noindent\begin{tabularx}{\textwidth}{@{}l *{5}{>{\centering\arraybackslash}X}@{}}
+\toprule
+Model & Feature extraction (GPU-h) & Search (GPU-h) & Three-seed fitting (GPU-h) & Build total (GPU-h) & Peak memory: extraction / fitting (GB) \\
+\midrule
+DistilBERT & --- & 0.32 & 0.27 & 0.59 & --- / 4.35 \\
+Mistral~7B & 0.67 & 0.02 & 0.003 & 0.70 & 29.11 / 1.11 \\
+Llama~2~7B & 0.64 & 0.03 & 0.01 & 0.68 & 27.01 / 1.11 \\
+\bottomrule
+\end{tabularx}
+\par\medskip
+\noindent\textbf{(b) Repeated inference measurements}\par\smallskip
+\noindent\begin{tabularx}{\textwidth}{@{}l *{5}{>{\centering\arraybackslash}X}@{}}
+\toprule
+Model & Batched time (min) & Single-post p95 (ms) & Single-post peak memory (GB) & GPU-h per 1,000 posts & Batched GPU energy (Wh) \\
+\midrule
+DistilBERT & 0.040 & 5.11 $\pm$ 0.12 & 0.16 & 0.000056 & 0.16 \\
+Mistral~7B & 3.87 & 38.37 $\pm$ 0.18 & 14.28 & 0.005376 & 25.5 \\
+Llama~2~7B & 3.61 & 36.23 $\pm$ 0.40 & 13.27 & 0.005008 & 23.8 \\
+\bottomrule
+\end{tabularx}
+\par\medskip
+\noindent\textbf{(c) Complete measured run, including evaluation and repeated benchmarks}\par\smallskip
+\noindent\begin{tabularx}{\textwidth}{@{}l *{3}{>{\centering\arraybackslash}X}@{}}
+\toprule
+Model & Total GPU time (h) & GPU-board energy (Wh) & Batched GPU utilization (\%) \\
+\midrule
+DistilBERT & 0.61 & 124.6 & 57.0 \\
+Mistral~7B & 0.91 & 335.7 & 99.7 \\
+Llama~2~7B & 0.87 & 310.8 & 99.7 \\
+\bottomrule
+\end{tabularx}
+\par\smallskip
+\noindent\textit{Protocol.} All three configurations were measured on NVIDIA A100-SXM4-40GB accelerators in the same software environment. Training used bf16 and the inference benchmark used fp16. The input-length cap was 256 tokens. Batched inference used batch size 32 on all 12,000 held-out posts; single-post latency used 512 posts at batch size one. Each inference benchmark was repeated three times after three warm-up batches. Table~\ref{tab:phase1-performance}(b) reports mean throughput, mean p50 latency, and their across-repetition SDs; panel (b) above reports mean p95 latency and its SD. Displayed zero SDs result from rounding, not an assumption of zero variation.
+\par\smallskip
+\noindent\textit{Timing and resource accounting.} CUDA was synchronized before and after each timed region, including host-to-device copies; this is a model-inference benchmark rather than full application latency. Non-padding input tokens were counted. NVML sampled utilization and board power at 1~Hz, and energy was computed by trapezoidal integration of board-power samples. Energy covers the GPU board only; the short DistilBERT timing windows limit the temporal resolution of its energy and utilization estimates. Batched time and energy in panel (b) describe one repetition, not the sum of three. The GPU-hours per 1,000 posts are normalized from batched throughput. Values are rounded independently.
+\par\smallskip
+\noindent\textit{Build and evaluation scope.} Probe feature extraction processed 108,000 posts before fitting cached-feature linear heads. The three-seed fitting and fitting-memory entries therefore exclude the backbone, whose cost and peak memory appear separately in panel (a). The search used four trials per model. Cached-feature head scoring is not classifier inference throughput; neither those timings nor the DistilBERT training-pipeline scoring rate are used in Table~\ref{tab:phase1-performance}(b). Panel (c) includes search, fitting, scoring, and all timing repetitions, and is not a per-deployment workload estimate. Monetary conversion and Phase~2 generation costs are not included.
+\normalsize
+\par\medskip
 
 \Needspace{24\baselineskip}
 \section{Chain-of-Thought Prompting Protocol}
```
