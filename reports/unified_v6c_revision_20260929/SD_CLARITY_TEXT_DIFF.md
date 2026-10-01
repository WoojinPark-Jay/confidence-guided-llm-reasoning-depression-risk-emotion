# SELF-DISCOVER 설명 및 본문 도식 보강: LaTeX 수정 전후

설명 보강 직전 원고와 본문 Figure 2를 추가한 최종 원고의 차이. 그림 내부 변경은 SD_CONTRIBUTION_CLARITY_KO.md 참조.

```diff
--- before/main.tex
+++ after/main.tex
@@ -76,5 +76,5 @@
 \item It replaces a heuristic confidence cutoff with a reproducible calibration protocol: temperature scaling, a prespecified threshold grid, a one-sided selective-risk upper bound, and maximum-coverage selection among feasible thresholds. The selected policy is fixed before held-out and Phase~2 evaluation.
 \item It evaluates routing quality separately from re-evaluation quality. Coverage, selective risk, error capture, corrected errors, introduced errors, and net corrections reveal whether the router finds difficult inputs and whether the LLM actually improves them. The final comparison crosses two language models with three re-evaluation protocols on the same routed examples within each dataset.
-\item It contributes a controlled Mixed Emotion stress test and an output-recording protocol for inspecting posts whose surface cues and dominant emotional trajectory diverge. The stress test is explicitly separated from model training and threshold selection.
+\item It develops a task-specific SELF-DISCOVER adaptation \cite{ref18} combining an emotion-oriented module bank, a reusable model-generated procedure, and fixed checks for emotional subject, temporal context, competing label evidence, and alternative interpretations. Its empirical evaluation includes a controlled Mixed Emotion stress test and recorded outputs for inspecting ambiguous or shifting affect. The stress test is explicitly separated from model training and threshold selection.
 \end{itemize}

@@ -396,17 +396,26 @@
 The routing mechanism is entirely confidence-based during inference. It does not directly detect blended emotion or sentiment shift. Instead, the Mixed Emotion Dataset serves a complementary evaluation role by stress-testing whether the Phase 1 classifier and the Phase 2 reasoning stage behave more reliably on examples where the one-label-per-input assumption is more difficult. This separation keeps the routing rule reproducible while allowing the evaluation to probe emotionally complex cases.

+\begin{figure*}[t]
+\centering
+\includegraphics[width=\textwidth]{figures/sd_task_level_overview.pdf}
+\caption{Task-specific SELF-DISCOVER design within selective re-evaluation. (a) Before post classification, the selected LLM constructs a model-specific procedure from researcher-authored task materials; the code saves it for reuse. (b) Each routed original post is evaluated anew using that procedure and fixed subject, time-frame, evidence, and alternative-interpretation checks. The dashed arrow denotes procedure reuse, not answer reuse. Discovery is skipped when a retained plan exists. Appendix~\ref{app:sd-templates} provides the detailed workflow and exact prompts; label parsing and Phase~1 fallback follow Algorithm~\ref{alg:sd-final}.}
+\label{fig:sd-task-level}
+\end{figure*}
+
 \subsection{Phase 2: Factorial Model and Protocol Comparison}
 Direct prompting requests one final label without a rationale and includes the Phase~1 prediction. CoT first requests an independent assessment, subsequently discloses the Phase~1 label for comparison, and finally requests a terminal label. The implemented CoT dialogue contains five generation calls, including the initial acknowledgment and text-response turns. Final SELF-DISCOVER withholds the Phase~1 label throughout. These are complete prompting protocols: their comparison does not isolate rationale length alone, because label disclosure and call structure also differ.

 \subsection{Task-Level SELF-DISCOVER}
-SELF-DISCOVER separates \emph{constructing a procedure} from \emph{applying it}. Each model generates a plan for the three-class task, stores it, and reuses it across routed posts. This \emph{cached plan} is a saved procedure, not stored post-level answers. Every post still requires a new model response.
-
-Three inputs have distinct roles: researcher-authored policy, modules, and execution instructions define the task; the model generates a plan through SELECT, ADAPT, and IMPLEMENT \cite{ref18}; and the original post supplies the evidence. Thus, model-generated planning operates within researcher-authored guidance, rather than discovering the classification policy itself.
-
-\textit{Plan construction.} SELECT chooses from 18 emotion-oriented modules; ADAPT specializes them; IMPLEMENT requests at most three one-line JSON steps, each returning to the original text and deferring label selection until the end. This format is requested, not enforced. Llama~2 and Llama~3 retain separate plans; each model reuses its plan across Reddit and Mixed Emotion.
+SELF-DISCOVER separates \emph{constructing a procedure} from \emph{applying it}, as summarized in Figure~\ref{fig:sd-task-level}. Each model generates a plan for the three-class task, stores it, and reuses it across routed posts. This \emph{cached plan} is a saved procedure, not stored post-level answers. Every post still requires a new model response.
+
+\textit{Task-specific design.} We retain the SELECT--ADAPT--IMPLEMENT framework of SELF-DISCOVER \cite{ref18} and specialize its inputs and execution guidance to proxy emotion classification. Researchers supply the class policy, 18 emotion-oriented reasoning modules, and fixed evidence-checking instructions; the LLM generates the procedure from those materials. The adaptation addresses distinctions that isolated affect words cannot resolve: whose emotion is expressed, whether distress is current or retrospectively resolved, and whether positive wording describes an actual state rather than a wish. The contribution is this task-specific specification and its integration with selective re-evaluation, not the invention of task-level planning or caching.
+
+\textit{Who constructs the plan, and when.} Before classifying a routed post, the code checks for a retained plan for the selected LLM. If none exists, that LLM receives only the researcher-authored task description, policy, and module bank. It does not receive the first post, any other evaluation post, a reference label, or a Phase~1 prediction during discovery. SELECT chooses relevant modules; ADAPT specializes them to the task; IMPLEMENT generates the reusable procedure, which the code stores. If a plan already exists, all three discovery calls are skipped. Llama~2 and Llama~3 generate separate plans; each model reuses its own plan across Reddit and Mixed Emotion.
+
+IMPLEMENT requests at most three one-line JSON steps, each returning to the original text and deferring label selection until the end. This format is requested, not enforced; the realized plans are reproduced in Appendix~\ref{app:sd-templates}. Access to the post is an instruction for subsequent execution, not an input to plan construction.

 \textit{Per-post execution.} The prompt combines the plan, policy, original post, and fixed checks. It asks whose emotion is expressed, whether it is current or retrospectively resolved, and which quotations support or oppose each label. A counterfactual check asks what evidence would favor the second-best label and whether it is present. The model emits a final classification line. SD never receives the Phase~1 label; the evaluator retains it as a fallback if parsing fails.

-The design defers commitment while comparing interpretations; its generated explanations do not establish faithful internal reasoning. Appendix~\ref{app:sd-templates} provides the workflow, templates, and actual plans; Appendix~\ref{app:sd-variants} compares the earlier per-post and final task-level designs without attributing their differences to any single component.
+This division keeps the procedure shared while requiring fresh, post-specific evidence and a new decision for every routed input. The evidence ledger and counterfactual check are researcher-authored execution instructions, not automatically discovered properties of the cached plan. Generated explanations do not establish faithful internal reasoning. Appendix~\ref{app:sd-templates} provides the workflow, templates, and actual plans; Appendix~\ref{app:sd-variants} compares the earlier per-post and final task-level designs without attributing their differences to any single component.

 \begin{algorithm}[t]
@@ -414,8 +423,12 @@
 \label{alg:sd-final}\footnotesize
 \begin{algorithmic}[1]
-\State \textbf{Input:} model $g$, policy, domain modules, routed posts
-\State SELECT relevant modules for the task
-\State ADAPT selected modules to the classification policy
-\State IMPLEMENT and store a model-specific task plan (or load its existing cache)
+\State \textbf{Input:} model $g$; researcher-authored task, policy, modules, execution guidance; routed posts
+\If{a retained plan for $g$ exists}
+\State Load that plan; skip discovery
+\Else
+\State Give $g$ task materials only; no post text or labels
+\State $g$: SELECT relevant modules and ADAPT them to the task
+\State $g$: IMPLEMENT the procedure; store it as the plan
+\EndIf
 \For{each routed post $x$}
 \State Execute the plan on the original post, without the Phase~1 label
@@ -610,4 +623,6 @@
 \subsection{What the Model--Protocol Comparison Shows}
 Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 on both sets. However, the 0.2833-point Reddit gain uses a 1.82\% routing rate, whereas the 9.00-point Mixed Emotion gain uses 28.67\%. These are different correction opportunities, not directly comparable effect sizes. Likewise, one introduced stress-test error versus 38 Reddit errors warrants dataset-specific interpretation rather than a universal reliability claim.
+
+The methodological value of the adaptation is to specify what evidence a selected post should be checked against, rather than merely requesting a longer explanation. Emotional subject and time-frame checks address attribution and trajectory, while the three-label ledger and counterfactual check make competing interpretations explicit. The observed gains support this complete configuration within the evaluated settings; they do not identify which instruction caused the gain. Appendix~\ref{app:sd-variants} shows that the final design improves on the earlier independent-countercheck variant for Llama~3 on both sets, but not for every model--dataset combination.

 The final model-by-protocol comparison is more informative than contrasting different models with different prompts only. Nevertheless, different context windows, generation budgets, and exposure to the Phase~1 label remain part of the evaluated configurations. Llama~3's advantage is therefore a configuration-level finding, not an isolated causal estimate of model generation or parameter count. Likewise, Direct is a label-only re-evaluation baseline: its benefit does not prove that an explicit reasoning trace is necessary.
@@ -891,5 +906,7 @@
 This appendix documents the final task-level SELF-DISCOVER protocol (artifact identifier v6c). Read it in two parts: \emph{discovery} generates and stores a model-specific procedure, whereas \emph{execution} applies that procedure to each routed original post. Figure~C1 separates these scopes. The source templates below are reproduced verbatim; the explanatory paragraphs describe how they connect, rather than modifying the prompts.

-The stored plan is a reusable text artifact, not a set of cached predictions. Discovery receives the task definition and module descriptions; individual post text enters during execution. A model uses its own plan for both datasets. Researcher-authored policy and checking instructions remain fixed alongside the model-generated plan. Thus, reusing a plan does not mean reusing another post's answer, updating model weights, or bypassing per-post inference.
+The chronology and authorship are distinct. Researchers first write the task description, class policy, module bank, and execution guidance. When no retained plan exists, the selected LLM performs SELECT, ADAPT, and IMPLEMENT using the task materials only; the code saves its generated procedure before classifying any post. No individual post or reference label enters these discovery calls. Only then does the LLM receive a routed original post together with the saved procedure and fixed execution guidance. A retained plan bypasses all three discovery calls.
+
+The stored plan is a reusable text artifact, not a set of cached predictions. Each model uses its own plan for both datasets. Reusing it does not mean learning a procedure from the first evaluation post, reusing that post's answer, updating model weights, or bypassing per-post inference.

 \Needspace{24\baselineskip}
@@ -897,5 +914,5 @@
 \includegraphics[width=0.96\textwidth]{figures/appendix_sd_workflow.pdf}
 \end{center}
-\noindent\footnotesize\textbf{Appendix Figure C1.} Final task-level SELF-DISCOVER workflow. Gray inputs are researcher-authored or supplied text; blue stages produce model outputs. Discovery creates a model-specific plan when no retained plan is available. The dashed path reuses that saved procedure across posts and datasets; solid arrows carry stage inputs or outputs. Each post receives a fresh execution response. Parsing and fallback are performed by the evaluation code, not an additional LLM call.\normalsize\par\medskip
+\noindent\footnotesize\textbf{Appendix Figure C1.} Final task-level SELF-DISCOVER workflow, separating authorship, timing, and input scope. (a) Researchers supply task materials; before post classification, the selected LLM generates a plan through SELECT, ADAPT, and IMPLEMENT if no retained plan exists. No evaluation post or reference label enters discovery. (b) The same LLM applies the saved plan and fixed researcher-authored checks to each routed original post, generating a fresh response. The dashed arrow denotes plan reuse, not reuse of an answer. Parsing and fallback are code operations, not additional LLM calls.\normalsize\par\medskip

 \Needspace{15\baselineskip}\noindent\textsc{Appendix Table C1. Final SELF-DISCOVER stages}\par\smallskip

```
