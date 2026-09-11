# 이번 추가 평가 반영 전후 전체 diff

```diff
--- before/main.tex
+++ after/main.tex
@@ -38,7 +38,7 @@
 \corresp{Corresponding-author details will be inserted in the submission version.}
 
 \begin{abstract}
-Transformer classifiers can process social-media text at scale, but confidence-based selection does not ensure that a second model will correct the selected predictions. We evaluate a two-phase framework for three-class proxy emotion classification (Depression, Neutral, and Happy) that separates this selection problem from LLM re-evaluation. Phase~1 uses temperature-calibrated DistilBERT predictions and a calibration-set risk--coverage criterion to route low-confidence inputs. In a three-seed comparison on the same 12,000 Reddit test posts, DistilBERT achieved 96.88 $\pm$ 0.04\% accuracy, versus 95.36 $\pm$ 0.08\% for Mistral~7B and 95.17 $\pm$ 0.07\% for Llama~2~7B frozen-backbone linear probes. A common-A100 benchmark also showed substantially higher batched throughput and lower single-post latency for DistilBERT. The independently fixed operational DistilBERT checkpoint achieved 96.69\% accuracy and routed 171 posts (1.42\%), including 87 of its 397 errors. Phase~2 used minimally sanitized original text and either Llama~2 Chain-of-Thought or Llama~3 SELF-DISCOVER. Llama~2 introduced three more errors than it corrected, yielding 96.67\% end-to-end accuracy. Llama~3 produced 30 net corrections and 96.94\% accuracy; its paired improvement remained significant after Holm correction. On a 300-example synthetic Mixed Emotion stress test, the two configurations increased accuracy from 81.33\% to 85.33\% and 87.33\%, respectively. Thus, routing concentrated errors in a small subset, but correction gains depended on the re-evaluation configuration. Because prompt development reused the routed Reddit cases, the end-to-end findings are exploratory. The study evaluates proxy emotion labels, not clinical diagnoses; independent data and expert review are needed to establish broader validity.
+Transformer classifiers can process social-media text at scale, but confidence-based selection does not ensure that a second model will correct the selected predictions. We evaluate a two-phase framework for three-class proxy emotion classification (Depression, Neutral, and Happy) that separates routing from LLM re-evaluation. Phase~1 uses temperature-calibrated DistilBERT predictions and a calibration-set risk--coverage criterion to route low-confidence inputs. In a three-seed comparison on the same 12,000 Reddit test posts, DistilBERT achieved 96.88 $\pm$ 0.04\% accuracy, versus 95.36 $\pm$ 0.08\% for Mistral~7B and 95.17 $\pm$ 0.07\% for Llama~2~7B frozen-backbone linear probes, with higher throughput and lower latency on a common A100 benchmark. The fixed operational checkpoint achieved 96.69\% accuracy and routed 171 posts (1.42\%). Re-evaluation using minimally sanitized original text yielded 96.67\% accuracy with Llama~2 Chain-of-Thought and 96.94\% with Llama~3 SELF-DISCOVER. On a 300-example synthetic Mixed Emotion stress test, accuracy increased from 81.33\% to 85.33\% and 87.33\%, respectively. Because prompt development reused the original routed Reddit cases, we additionally evaluated the frozen pipeline on 9,000 previously unused same-source posts. Routing 137 posts (1.52\%) raised accuracy from 96.73\% to 97.03\% with Llama~3, yielding 27 net corrections and a Holm-adjusted $p=0.000131$; the Llama~2 gain was not significant. These results support configuration-dependent correction with limited re-evaluation volume beyond the prompt-development sample. They concern proxy emotion labels, not clinical diagnoses; external validation and expert review remain necessary for broader use.
 \end{abstract}
 
 \begin{keywords}
@@ -68,7 +68,7 @@
 \begin{itemize}
 \item It operationalizes selective LLM re-evaluation as an end-to-end pipeline in which an efficient classifier handles high-confidence inputs and reasoning models receive only calibration-defined low-confidence inputs. The Phase~1 comparison supports operational model selection through predictive performance and measured inference throughput, latency, and memory on a common accelerator, rather than claiming exhaustive optimization of every 7B architecture.
 \item It replaces a heuristic confidence cutoff with a reproducible calibration protocol: temperature scaling, a prespecified threshold grid, a one-sided selective-risk upper bound, and maximum-coverage selection among feasible thresholds. The selected policy is fixed before held-out and Phase~2 evaluation.
-\item It evaluates routing quality separately from re-evaluation quality. Coverage, selective risk, error capture, corrected errors, introduced errors, and net corrections reveal whether the router finds difficult inputs and whether the LLM actually improves them.
+\item It evaluates routing quality separately from re-evaluation quality. Coverage, selective risk, error capture, corrected errors, introduced errors, and net corrections reveal whether the router finds difficult inputs and whether the LLM actually improves them. An additional 9,000-post same-source holdout tests the frozen pipeline beyond the sample used during prompt development.
 \item It contributes a controlled Mixed Emotion stress test and case-level reasoning audit trail for studying posts whose surface cues and dominant emotional trajectory diverge. The stress test is explicitly separated from model training and threshold selection.
 \end{itemize}
 
@@ -436,7 +436,7 @@
 
 The experiments were implemented in Python with PyTorch, Hugging Face Transformers, Datasets, scikit-learn, SciPy, and pandas; W\&B recorded the model-selection sweeps. Phase~1 used \texttt{distilbert-base-uncased} as the operational checkpoint and prespecified 7B base checkpoints for the Mistral and Llama~2 comparison classifiers. Routed Phase~2 inference used \texttt{NousResearch/Llama-2-7b-chat-hf} and \texttt{NousResearch/Meta-Llama-3-8B-Instruct}, loaded in 4-bit form with bitsandbytes in GPU-backed Google Colab sessions. Llama~2 CoT generation used a 256-token limit with sampling temperature 0.6 and top-$p$ 0.9; the Llama~3 SELF-DISCOVER structure stages used a 1,024-token limit and the terminal answer was decoded deterministically. The repository notebooks print installed package versions at runtime and save prediction-level outputs after every routed row.
 
-The three Phase~1 comparison configurations were trained and profiled on NVIDIA A100-SXM4-40GB accelerators in the same software environment. The inference benchmark used fp16, a 256-token cap, and batch size 32 over the 12,000-post held-out split; single-post latency was measured separately on 512 posts. Three warm-up batches were discarded, and each benchmark was repeated three times. CUDA synchronization bracketed timed regions, which included host-to-device transfers. NVML sampled GPU utilization and board power at 1~Hz. Appendix Table~A1d reports the measurement protocol and stage-level resource use. Phase~2 generation time, tokens, and energy were not measured in these experiments; routing volume is reported separately from computational savings.
+The three Phase~1 comparison configurations were trained and profiled on NVIDIA A100-SXM4-40GB accelerators in the same software environment. The inference benchmark used fp16, a 256-token cap, and batch size 32 over the 12,000-post held-out split; single-post latency was measured separately on 512 posts. Three warm-up batches were discarded, and each benchmark was repeated three times. CUDA synchronization bracketed timed regions, which included host-to-device transfers. NVML sampled GPU utilization and board power at 1~Hz. Appendix Table~A1d reports the measurement protocol and stage-level resource use. The additional 9,000-post evaluation ran on an NVIDIA L4 and recorded Phase~2 generation calls, tokens, and elapsed generation time separately (Appendix~F). These measurements do not constitute an all-input LLM timing baseline or an energy measurement.
 
 \subsection{Model Selection and Calibration Protocol}
 
@@ -458,9 +458,16 @@
 
 The metrics were organized by the research question they answer rather than treated as interchangeable indicators. For RQ1a, predictive performance was evaluated with accuracy and class-wise precision, recall, and F1, with unweighted macro averages reported across Depression, Neutral, and Happy; calibration was evaluated with NLL, Brier score, ECE, and adaptive ECE. For RQ1b, matched classifier performance is reported as mean $\pm$ standard deviation across three seeds and interpreted together with model parameter scale, trainable scope, and the common-hardware inference benchmark. Timing variability is reported separately as mean $\pm$ SD across three measurement repetitions. For RQ2, routing was evaluated with coverage, routing rate, accepted-set risk and its one-sided upper bound, error capture, and routing precision. For RQ3, the primary end-to-end outcome was the example-level change in correctness after accepted Phase~1 predictions were recombined with routed Phase~2 labels. This change was summarized by full-set accuracy change and by corrected, introduced, and net errors. Calibration and routing measures are supporting diagnostics: they establish whether the confidence policy is credible and whether it concentrates difficult cases, but they do not by themselves establish that Phase~2 improves the final system.
 
-Phase~1 and end-to-end correctness are paired for each example. Two-sided exact McNemar tests use the discordant counts: corrected Phase~1 errors and newly introduced errors. Paired percentile-bootstrap 95\% confidence intervals use 50,000 row-level resamples and seed 20260813. Holm adjustment covers the four dataset--reasoner comparisons. These analyses characterize the observed prediction pairs conditional on the fitted models and selected prompts; they do not account for prompt selection, repeated LLM generation, or dependence between posts from the same author. The three-seed intervals in the classifier comparison instead describe variation across training runs on a common test set.
+Phase~1 and end-to-end correctness are paired for each example. Two-sided exact McNemar tests use the discordant counts: corrected Phase~1 errors and newly introduced errors. For the original Reddit and Mixed Emotion evaluations, paired percentile-bootstrap 95\% confidence intervals use 50,000 row-level resamples and seed 20260813, with Holm adjustment across the four dataset--reasoner comparisons. The additional holdout uses 10,000 paired correctness resamples and seed 20260911, with its two prespecified comparisons against Phase~1 treated as a separate Holm family. These analyses characterize the observed prediction pairs conditional on the fitted models and selected prompts; they do not account for prompt selection, repeated LLM generation, or dependence between posts from the same author. The three-seed intervals in the classifier comparison instead describe variation across training runs on a common test set.
 
 Residual accepted-set errors were audited separately using the fixed 12,000-example held-out protocol and the calibration-selected threshold. The audit reports accepted selective risk, reference-to-prediction transitions, confidence-band error rates, and Depression false-negative risk among accepted reference-Depression posts. For all 310 accepted errors, the stored model input was linked back to the retained Reddit title and selftext by normalized exact matching against \texttt{title\_with\_selftext\_cleaned}; all 310 rows were matched. Six high-confidence cases were then selected because the original post itself provides comparatively clear evidence for the stored proxy label and exposes a distinct error pattern. These cases illustrate failure modes rather than estimate their prevalence. Appendix Tables~A4e--A4f report the reference label, prediction, calibrated confidence, minimally sanitized original-post excerpt, and text-grounded audit rationale. Qualitative review does not overwrite any reference label or alter any reported metric.
+
+\subsection{Additional Same-Source Holdout Protocol}
+\label{sec:additional-holdout-protocol}
+
+After freezing the model-specific prompts and input policy, we formed an additional balanced holdout of 9,000 posts from unused records in the same source corpus. We excluded text matching any post in the previously used 120,000-post sample after Unicode NFKC normalization, case folding, and whitespace normalization, and removed within-pool normalized-text duplicates. Sampling used seed 20260911 and retained 3,000 posts per class. No selected text overlapped the original sample under this matching rule. This is a same-source holdout, not an author-disjoint, temporal, or external-domain test.
+
+The saved operational DistilBERT checkpoint, temperature $T^*=1.7706$, threshold $\tau^*=0.70$, and model-specific prompt and input policies were retained without fitting or selection on these 9,000 posts. Both reasoners processed the same routed IDs in separate configurations. Accepted predictions remained unchanged, and any unparseable routed output would be counted as incorrect rather than silently replaced by Phase~1. Evaluation was a single fixed run per configuration, not a new three-seed training comparison. Appendix~F records reproducibility details and resource accounting.
 
 \section{Experimental Results}
 
@@ -707,6 +714,35 @@
 \label{fig:error-structure}
 \end{figure*}
 
+\subsection{Additional Same-Source Holdout Results}
+\label{sec:additional-holdout-results}
+
+On the additional 9,000 posts, the unchanged operational classifier achieved 96.73\% accuracy and routed 137 posts (1.52\%). The routed subset contained 67 of 294 Phase~1 errors (22.79\% error capture), with a 48.91\% error rate versus 3.27\% overall. The remaining 8,863 accepted posts had a 2.56\% error rate. Thus, the fixed policy again concentrated errors in a small subset, while leaving 227 accepted errors outside the correction path.
+
+Table~\ref{tab:additional-holdout} reports the complete-set effects. Llama~3 SELF-DISCOVER corrected 36 errors and introduced nine, giving 27 net corrections and 97.03\% accuracy. Its $+0.30$ percentage-point gain remained significant after Holm adjustment ($p=0.000131$). Llama~2 CoT corrected 41 errors but introduced 26, yielding 96.90\% accuracy; its paired interval included zero. This distinction supports the evaluated Llama~3 configuration without implying that any LLM re-evaluator will improve the classifier or establishing a direct statistical difference between the two reasoners.
+
+\begin{table*}[t]
+\centering
+\caption{Additional same-source holdout: fixed-policy performance on 9,000 posts}
+\label{tab:additional-holdout}
+\footnotesize
+\setlength{\tabcolsep}{4pt}
+\renewcommand{\arraystretch}{1.16}
+\begin{tabularx}{\textwidth}{@{}Y c c c c c c c@{}}
+\toprule
+Configuration & Accuracy & Macro F1 & Corrected & Introduced & Change & 95\% CI (pp) & Holm $p$ \\
+\midrule
+Phase~1 & 96.73\% & 96.73\% & -- & -- & -- & -- & -- \\
++ Llama~2 CoT & 96.90\% & 96.90\% & 41 & 26 & $+0.17$ pp & [$-0.01$, 0.34] & 0.0864 \\
++ Llama~3 SELF-DISCOVER & 97.03\% & 97.03\% & 36 & 9 & $+0.30$ pp & [0.16, 0.46] & 0.000131 \\
+\bottomrule
+\end{tabularx}
+\vspace{2pt}
+\parbox{\textwidth}{\footnotesize\textit{Note.} Both reasoners re-evaluated the same 137 posts; all other predictions were retained. No parsing failures occurred. Changes and paired percentile-bootstrap intervals use all 9,000 posts (10,000 resamples). Holm adjustment covers the two exact McNemar comparisons against Phase~1 in this additional evaluation, separately from Table~\ref{tab:paired-statistics}.}
+\end{table*}
+
+The Llama~3 improvement corresponds to a 9.18\% reduction in total errors (294 to 267). It realizes 40.30\% of the fixed router's 67-error correction opportunity; the conditional perfect-correction ceiling is 97.48\%. This evaluation extends the evidence beyond the prompt-development sample, while remaining conditional on the same source and proxy-label construction. Class-wise results and generation measurements appear in Appendix~F.
+
 \subsection{Observed Correction Cases and Reasoning Traceability}
 
 Appendix~D reports one observed Mixed Emotion Depression correction for each Phase~2 model using the stored output columns rather than an invented illustration. It also reports an observed Reddit original-text Llama~3 correction in which Phase~1 predicted Happy for a post describing numbness and exhaustion from trying to remain positive. Llama~2 CoT retains independent assessment, Phase~1 comparison, and terminal-label fields, while Llama~3 SELF-DISCOVER retains an auditable SELECT--ADAPT--IMPLEMENT trace before the final answer.
@@ -717,13 +753,13 @@
 
 The three research questions receive distinct answers. For \textbf{RQ1a}, DistilBERT achieved strong held-out predictive performance, and temperature scaling reduced NLL, Brier score, and fixed-bin ECE on the calibration split. For \textbf{RQ1b}, the completed three-seed comparison retained DistilBERT: its mean held-out accuracy and macro F1 were both 96.88\%, compared with 95.36\% for the Mistral frozen-backbone probe and 95.17\% for the Llama~2 probe. This is evidence for the operational choice under the bounded comparison, reinforced by the measured inference advantage, not a ranking of maximally tuned architectures. For \textbf{RQ2}, the fixed confidence policy preserved 98.58\% Reddit coverage while concentrating 87 of the 397 Phase~1 errors in the 171 routed posts; routed-sample accuracy was 49.12\%, compared with 96.69\% overall. The same fixed policy identified a 44-example Mixed Emotion subset with 52.27\% Phase~1 accuracy, compared with 81.33\% overall. For \textbf{RQ3}, re-evaluation benefit was model-dependent. Llama~2 remained statistically indistinguishable from Phase~1 on Reddit, whereas Llama~3 raised routed-only accuracy to 66.67\% and produced a positive paired full-test effect. On the controlled stress test, Llama~2 and Llama~3 raised routed-only accuracy to 79.55\% and 93.18\%, respectively, with Llama~3 avoiding the six introduced errors observed for Llama~2. Confidence therefore identifies a correction opportunity, but the reasoning policy determines how much of that opportunity is realized.
 
-Only 1.42\% of Reddit posts and 14.67\% of Mixed Emotion examples underwent re-evaluation. These are fractions of inputs, not counts of individual LLM calls: CoT and SELF-DISCOVER can require multiple generation stages per routed post. The Phase~1 benchmark establishes the inference-cost advantage of the selected classifier configuration, whereas total cascade time and monetary savings require Phase~2 measurements. Every input still incurs Phase~1 computation, so the fraction avoiding re-evaluation is not a measured GPU-time saving. The stress-test routing rate characterizes its constructed scenario mix, not an expected deployment workload.
+Only 1.42\% of the original Reddit test posts, 14.67\% of Mixed Emotion examples, and 1.52\% of additional holdout posts underwent re-evaluation. These are fractions of inputs, not counts of individual LLM calls: CoT and SELF-DISCOVER require multiple generation stages per routed post. The Phase~1 benchmark establishes the inference advantage of the selected classifier configuration; the additional holdout separately records the cost of successful Phase~2 generation. Llama~3 used more generation time and tokens than Llama~2 despite fewer calls (Appendix~F). Every input still incurs Phase~1 computation, and no all-input LLM timing baseline was run, so the fraction avoiding re-evaluation is not a measured GPU-time saving. The stress-test routing rate characterizes its constructed scenario mix, not an expected deployment workload.
 
 A conditional routing oracle further clarifies the scale of the observed gains without introducing a new tuned baseline. If every routed Phase~1 error were corrected and no correct routed prediction were changed, Reddit accuracy could rise from 96.69\% to at most 97.42\% under the fixed 171-post routing policy; the 30 Llama~3 net corrections realize 34.5\% of the 87-error correction opportunity. On Mixed Emotion, the corresponding ceiling is 88.33\%, and the 18 Llama~3 net corrections realize 85.7\% of the 21-error opportunity. These ceilings are conditional on the already selected router and are not claims about an attainable model. Their purpose is to distinguish a limited routing opportunity from a failure to exploit routed cases. Definitions and the complete decomposition are reported in Appendix Table~A4g.
 
 In addition to selective computation, the framework adds an inspectable rationale channel for routed cases. Unlike a single-stage classifier that outputs only a class label or probability score, the Phase 2 record retains the generated rationale and terminal label. This improves case-level traceability, but the rationale is not treated as proof of faithfulness, clinical validity, or prediction correctness.
 
-Taken together, the experiments support the Phase~1 and routing components of the framework for the constructed proxy-label task. They also show that Phase~2 value is model- and input-dependent: Llama~3 produced positive paired effects on both the natural Reddit test set and the Mixed Emotion stress test, whereas Llama~2 did not improve Reddit end-to-end accuracy. The synthetic stress test provides focused evidence about emotionally ambiguous inputs, while the Reddit rerun reports performance with the original-text input policy. Neither result constitutes broad evidence of real-world mental-health generalization.
+Taken together, the experiments support the Phase~1 and routing components of the framework for the constructed proxy-label task. Llama~3 produced positive paired effects on the original Reddit test, the Mixed Emotion stress test, and the additional same-source holdout. Llama~2 showed no statistically detectable gain on either Reddit evaluation. The additional holdout strengthens evidence for the frozen Llama~3 configuration beyond the prompt-development sample, but does not establish broad real-world mental-health generalization or isolate the effect of the reasoning method from the underlying model.
 
 \section{Limitations and Future Work}
 
@@ -733,15 +769,15 @@
 
 \textit{Comparison scope.} The Phase~1 study compares three practical configurations under a limited search budget, not equivalently adapted or maximally optimized architectures. Three seeds characterize a narrow range of training variability. In Phase~2, model family and prompting method vary together, so the observed advantage of Llama~3 SELF-DISCOVER cannot be attributed to SELF-DISCOVER alone. A crossed model--prompt evaluation would be needed to separate these effects. Similarly, the final original-text rerun supports the reported input policy but does not establish a general causal effect of preserving any particular textual feature.
 
-\textit{Prompt development and statistical scope.} Prompt variants were examined on the routed Reddit cases under the earlier cleaned-input condition. Although prompts were frozen before the original-text rerun, the same IDs and labels were reused. The end-to-end comparisons are therefore exploratory, not independent confirmatory tests. Bootstrap intervals and Holm-adjusted McNemar tests quantify the final paired outcomes but do not remove this development dependence. The next validation should freeze prompts and input handling on a separate development subset, then evaluate once on independent data. This issue is distinct from the calibration-partition fitting of $T^*$ and $\tau^*$.
-
-\textit{Computational and explanatory scope.} The controlled Phase~1 benchmark compares our implementations on one accelerator model and serving configuration, not optimized inference systems across hardware platforms. Three timing repetitions characterize within-session variation; they do not capture all variation across Colab sessions. GPU-board energy estimates exclude CPU and system-level consumption and have limited temporal resolution for short timed regions. Phase~2 generation was not instrumented, so the results do not establish total cascade runtime, energy, or monetary savings. Our per-post SELF-DISCOVER adaptation does not reuse a task-level plan, so efficiency results from the original method cannot be transferred to this implementation. Generated rationales remain unvalidated explanations, and multistage inference may have different overhead across configurations. Phase~2 resource measurements and independent review of both labels and rationales are the next steps.
+\textit{Prompt development and statistical scope.} Prompt variants were examined on the original routed Reddit cases under the earlier cleaned-input condition. Freezing prompts before the original-text rerun did not remove that development dependence, so those original end-to-end comparisons remain exploratory. The additional 9,000-post holdout evaluates the frozen pipeline on previously unused same-source text and provides separate evidence of improvement, without retroactively changing the status of the original analyses. Normalized exact-text exclusion does not establish author independence or remove all near-duplicates, and the paired intervals do not quantify repeated-generation variability. External or author-disjoint evaluation and expert-reviewed labels remain important extensions. This issue is distinct from calibration-partition fitting of $T^*$ and $\tau^*$.
+
+\textit{Computational and explanatory scope.} The controlled Phase~1 benchmark compares our implementations on one accelerator model and serving configuration, not optimized inference systems across hardware platforms. Three timing repetitions characterize within-session variation; they do not capture all variation across Colab sessions. GPU-board energy estimates exclude CPU and system-level consumption and have limited temporal resolution for short timed regions. The additional L4 run records successful Phase~2 generation calls, tokens, and time, but excludes loading and retried work and does not measure energy or an all-input LLM baseline. It therefore does not establish total cascade or monetary savings. Our per-post SELF-DISCOVER adaptation does not reuse a task-level plan, so efficiency results from the original method cannot be transferred to this implementation. Generated rationales still require independent assessment of faithfulness and label support.
 
 \section{Conclusion}
 
 This study evaluated selective LLM re-evaluation by separating classifier performance, calibration, error concentration, and net correction. The three-seed comparison and common-hardware inference benchmark supported retaining DistilBERT within the tested configurations: it combined the highest mean predictive scores with substantially higher batched throughput, lower single-post latency, and lower inference memory. Its fixed operational checkpoint routed 1.42\% of Reddit posts, capturing 87 of 397 Phase~1 errors. Llama~3 SELF-DISCOVER corrected 42 errors and introduced 12, raising full-set accuracy from 96.69\% to 96.94\%; Llama~2 CoT produced no detectable improvement. On the synthetic stress test, the corresponding gains were 6.00 and 4.00 percentage points.
 
-The central finding is that selecting an error-prone subset and correcting it are separate requirements. High routing concentration alone did not ensure a positive end-to-end effect, and accepted errors remained outside the correction path. The reported gains describe fixed prediction pairs under an exploratory prompt-development protocol. Independent evaluation with frozen prompts and expert-reviewed labels is needed to establish generalization. The framework remains a proxy emotion-classification pipeline, not a clinical diagnostic system.
+The central finding is that selecting an error-prone subset and correcting it are separate requirements. On an additional 9,000-post same-source holdout, the frozen pipeline re-evaluated 1.52\% of posts and improved accuracy from 96.73\% to 97.03\% with Llama~3, with a significant paired gain. This strengthens support beyond the original prompt-development sample while leaving accepted errors outside the correction path. External validation and expert-reviewed labels are needed to establish broader applicability. The framework remains a proxy emotion-classification pipeline, not a clinical diagnostic system.
 
 \balance
 \begin{thebibliography}{46}
@@ -1366,6 +1402,55 @@
 For each example, write one realistic post between 60 and 90 words. Do not include explicit diagnosis claims, medication names, therapy claims, suicide, self-harm, crisis language, usernames, URLs, subreddit names, hashtags, or personally identifying information. Avoid duplicated phrasing and avoid making the label obvious through a single keyword. Return structured records with: example identifier, target label, scenario type, primary context, dominant emotional trajectory, text, and brief label rationale.
 \end{minipage}
 
+\par\medskip
+\clearpage
+\section{Additional Holdout Reproducibility and Resource Use}
+\label{app:additional-holdout}
+
+The additional evaluation used 3,000 previously unused posts per class, selected with seed 20260911 after normalized exact-text exclusion against the original 120,000-post sample and removal of duplicate candidate text. Saved example IDs aligned inputs, reference labels, Phase~1 outputs, and both re-evaluators. The operational checkpoint and $T^*=1.7705598383050807$, $\tau^*=0.70$, maximum Phase~1 length 256, and batch size 32 were fixed. No training, temperature fitting, threshold search, or prompt selection used this holdout.
+
+\par\smallskip
+Generation used the model-specific protocols in Appendices~B--C, with 4-bit model loading and per-post SELF-DISCOVER structure generation. Each model--example seed was derived from the first 32 bits of SHA-256 of \texttt{42:model\_key:example\_id}, making its assignment independent of resume order. Model repository revisions were resolved once and recorded for this run: Llama~2 \texttt{351844e75ed0bcbbe3f10671b3c808d2b83894ee} and Llama~3 \texttt{53346005fb0ef11d3b6a83b12c895cca40156b6c}. Historical LLM commit hashes were unavailable, so byte-identical revisions across historical and additional runs are not established. The runtime used NVIDIA L4, CUDA 12.8, Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 5.14.1, Accelerate 1.14.0, and bitsandbytes 0.50.2.
+
+\par\medskip
+\Needspace{13\baselineskip}
+\noindent\textsc{Appendix Table A6. Additional holdout: class-wise F1 and routed accuracy}\par
+\smallskip
+\small
+\setlength{\tabcolsep}{5pt}
+\renewcommand{\arraystretch}{1.16}
+\noindent\begin{tabularx}{\textwidth}{@{}Y c c c c@{}}
+\toprule
+Configuration & Depression F1 & Neutral F1 & Happy F1 & Routed accuracy \\
+\midrule
+Phase~1 & 95.87\% & 98.82\% & 95.52\% & 51.09\% \\
++ Llama~2 CoT & 96.23\% & 98.53\% & 95.94\% & 62.04\% \\
++ Llama~3 SELF-DISCOVER & 96.36\% & 98.67\% & 96.07\% & 70.80\% \\
+\bottomrule
+\end{tabularx}
+\par\smallskip
+\noindent\footnotesize\textit{Note.} Class F1 uses all 9,000 posts; routed accuracy uses the same 137 posts for each configuration. Both re-evaluators increased overall macro F1 but slightly reduced Neutral F1. These are fixed-run results, not means across training seeds.\normalsize
+
+\par\medskip
+\Needspace{17\baselineskip}
+\noindent\textsc{Appendix Table A6b. Recorded generation workload on the additional holdout}\par
+\smallskip
+\small
+\noindent\begin{tabularx}{\textwidth}{@{}Y r r@{}}
+\toprule
+Measurement & Llama~2 CoT & Llama~3 SELF-DISCOVER \\
+\midrule
+Re-evaluated posts & 137 & 137 \\
+Generation calls & 685 & 548 \\
+Input tokens (all calls) & 583,566 & 732,956 \\
+Generated tokens & 100,840 & 199,716 \\
+Generation time (s) & 5,502.86 & 11,372.40 \\
+Per-post elapsed time, summed (s) & 5,504.93 & 11,375.40 \\
+\bottomrule
+\end{tabularx}
+\par\smallskip
+\noindent\footnotesize\textit{Note.} Five and four generation calls were recorded per post, respectively. Input-token counts sum repeated prompt/context tokens across calls, not unique corpus tokens. CUDA synchronization bracketed generation. Times sum successfully saved calls or per-post processing loops, excluding model loading, downloads, and retried work. Phase~1 recorded 23.60 seconds of forward computation and 743,332 non-padding input tokens over 9,000 posts; this is not complete application latency. The 98.48\% of posts avoiding Phase~2 is not a measured time, energy, or monetary saving. These L4 measurements are separate from the A100 classifier benchmark.\normalsize
+
 % REQUIRED BEFORE SUBMISSION:
 % 1. Add the final Acknowledgment text, including funding and the IEEE-required
 %    disclosure of any AI-generated manuscript text and the affected sections.
```
