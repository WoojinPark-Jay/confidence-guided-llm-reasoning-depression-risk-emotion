"""Connect the paper's methods, evidence, and exact appendix templates."""


def replace_once(source, old, new):
    assert source.count(old) == 1, old[:120]
    return source.replace(old, new, 1)


def enrich_source(source):
    source = source.replace(r'\clearpage\twocolumn\raggedbottom' + '\n' + r'\section{Discussion}',
                            r'\section{Discussion}')
    source = source.replace('Appendix Table A2.', 'Appendix Table B1.')
    source = source.replace('Appendix Table A3.', 'Appendix Table C1.')
    source = source.replace('Appendix Table A5.', 'Appendix Table F1.')
    source = source.replace('Appendix Table~A5', 'Appendix Table~F1')
    source = source.replace(
        'Final full-set outcomes. Accuracy and macro F1 are percentages; change is relative to Phase 1 in percentage points.',
        'Final full-set outcomes on 12,000 Reddit posts and 300 Mixed Emotion examples. Accuracy and macro F1 are percentages; change is relative to Phase 1 in percentage points. $C$ counts corrected errors and $I$ newly introduced errors; $C-I$ is their net difference. Unparsed responses retain Phase 1 predictions.')
    anchor = r'\subsection{Task-Level SELF-DISCOVER}'
    start = source.index(anchor)
    end = source.index(r'\begin{algorithm}', start)
    source = source[:start] + r'''\subsection{Task-Level SELF-DISCOVER}
SELF-DISCOVER separates \emph{constructing a procedure} from \emph{applying it}, as summarized in Figure~\ref{fig:sd-task-level}. Each model generates a plan for the three-class task, stores it, and reuses it across routed posts. This \emph{cached plan} is a saved procedure, not stored post-level answers. Every post still requires a new model response.

\textit{Task-specific design.} We retain the SELECT--ADAPT--IMPLEMENT framework of SELF-DISCOVER \cite{ref18} and specialize its inputs and execution guidance to proxy emotion classification. Researchers supply the class policy, 18 emotion-oriented reasoning modules, and fixed evidence-checking instructions; the LLM generates the procedure from those materials. The adaptation addresses distinctions that isolated affect words cannot resolve: whose emotion is expressed, whether distress is current or retrospectively resolved, and whether positive wording describes an actual state rather than a wish. The contribution is this task-specific specification and its integration with selective re-evaluation, not the invention of task-level planning or caching.

\textit{Who constructs the plan, and when.} Before classifying a routed post, the code checks for a retained plan for the selected LLM. If none exists, that LLM receives only the researcher-authored task description, policy, and module bank. It does not receive the first post, any other evaluation post, a reference label, or a Phase~1 prediction during discovery. SELECT chooses relevant modules; ADAPT specializes them to the task; IMPLEMENT generates the reusable procedure, which the code stores. If a plan already exists, all three discovery calls are skipped. Llama~2 and Llama~3 generate separate plans; each model reuses its own plan across Reddit and Mixed Emotion.

IMPLEMENT requests at most three one-line JSON steps, each returning to the original text and deferring label selection until the end. This format is requested, not enforced; the realized plans are reproduced in Appendix~\ref{app:sd-templates}. Access to the post is an instruction for subsequent execution, not an input to plan construction.

\textit{Per-post execution.} The prompt combines the plan, policy, original post, and fixed checks. It asks whose emotion is expressed, whether it is current or retrospectively resolved, and which quotations support or oppose each label. A counterfactual check asks what evidence would favor the second-best label and whether it is present. The model emits a final classification line. SD never receives the Phase~1 label; the evaluator retains it as a fallback if parsing fails.

This division keeps the procedure shared while requiring fresh, post-specific evidence and a new decision for every routed input. The evidence ledger and counterfactual check are researcher-authored execution instructions, not automatically discovered properties of the cached plan. Generated explanations do not establish faithful internal reasoning. Appendix~\ref{app:sd-templates} provides the workflow, templates, and actual plans; Appendix~\ref{app:sd-variants} compares the earlier per-post and final task-level designs without attributing their differences to any single component.

''' + source[end:]
    source = replace_once(source, r'\subsection{Phase 2: Factorial Model and Protocol Comparison}', r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/sd_task_level_overview.pdf}
\caption{Task-specific SELF-DISCOVER design within selective re-evaluation. (a) Before post classification, the selected LLM constructs a model-specific procedure from researcher-authored task materials; the code saves it for reuse. (b) Each routed original post is evaluated anew using that procedure and fixed subject, time-frame, evidence, and alternative-interpretation checks. The dashed arrow denotes procedure reuse, not answer reuse. Discovery is skipped when a retained plan exists. Appendix~\ref{app:sd-templates} provides the detailed workflow and exact prompts; label parsing and Phase~1 fallback follow Algorithm~\ref{alg:sd-final}.}
\label{fig:sd-task-level}
\end{figure*}

\subsection{Phase 2: Factorial Model and Protocol Comparison}''')
    source = replace_once(source, r'''\State \textbf{Input:} model $g$, policy, domain modules, routed posts
\State SELECT relevant modules for the task
\State ADAPT selected modules to the classification policy
\State IMPLEMENT a compact task-level plan; cache the output''', r'''\State \textbf{Input:} model $g$; researcher-authored task, policy, modules, execution guidance; routed posts
\If{a retained plan for $g$ exists}
\State Load that plan; skip discovery
\Else
\State Give $g$ task materials only; no post text or labels
\State $g$: SELECT relevant modules and ADAPT them to the task
\State $g$: IMPLEMENT the procedure; store it as the plan
\EndIf''')

    source = replace_once(source,
        'It contributes a controlled Mixed Emotion stress test and case-level reasoning audit trail for studying posts whose surface cues and dominant emotional trajectory diverge.',
        r'It develops a task-specific SELF-DISCOVER adaptation \cite{ref18} combining an emotion-oriented module bank, a reusable model-generated procedure, and fixed checks for emotional subject, temporal context, competing label evidence, and alternative interpretations. Its empirical evaluation includes a controlled Mixed Emotion stress test and recorded outputs for inspecting ambiguous or shifting affect.')

    old = 'The contribution is the integration and evaluation protocol, together with a task-specific adaptation of structured re-evaluation, not a new calibration estimator or language-model architecture.'
    source = replace_once(source, old, old + r'''
Classifier performance and resource measurements support the first-stage choice; calibration and error concentration assess routing; and the model--protocol comparison measures correction gains against newly introduced errors. The task-level SELF-DISCOVER adaptation is evaluated within this comparison, not assumed superior to label-only prediction.
''')
    source = replace_once(source, r'\section{Discussion}', r'''\Needspace{6\baselineskip}
\section{Discussion}
\subsection{An Efficient First Stage and a Selective Second Stage}''')
    source = replace_once(source, 'Confidence routing identifies an error-enriched subset, but enrichment alone is insufficient.', r'''\subsection{Selection and Correction Are Different Outcomes}
Confidence routing identifies an error-enriched subset, but enrichment alone is insufficient.''')
    source = replace_once(source, 'The final model-by-protocol comparison is more informative than contrasting different models with different prompts only.', r'''\subsection{What the Model--Protocol Comparison Shows}
Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 on both sets. However, the 0.2833-point Reddit gain uses a 1.82\% routing rate, whereas the 9.00-point Mixed Emotion gain uses 28.67\%. These are different correction opportunities, not directly comparable effect sizes. Likewise, one introduced stress-test error versus 38 Reddit errors warrants dataset-specific interpretation rather than a universal reliability claim.

The methodological value of the adaptation is to specify what evidence a selected post should be checked against, rather than merely requesting a longer explanation. Emotional subject and time-frame checks address attribution and trajectory, while the three-label ledger and counterfactual check make competing interpretations explicit. The observed gains support this complete configuration within the evaluated settings; they do not identify which instruction caused the gain. Appendix~\ref{app:sd-variants} shows that the final design improves on the earlier independent-countercheck variant for Llama~3 on both sets, but not for every model--dataset combination.

The final model-by-protocol comparison is more informative than contrasting different models with different prompts only.''')
    source = replace_once(source, r'\section{Final SELF-DISCOVER Templates}', r'\section{Final SELF-DISCOVER Templates}\label{app:sd-templates}')
    old = 'This is the final compact task-level protocol (implementation identifier v6c). The identifier distinguishes execution artifacts; it is not an additional comparator in the main tables. SELECT, ADAPT, and IMPLEMENT are task-level discovery calls. Their output is cached and inserted into the per-post execution template. Each model discovers its own plan; the input policy and evidence-oriented execution instructions are shared. The requested maximum of three steps is not enforced through a validator or retry mechanism.'
    source = replace_once(source, old, r'''This appendix documents the final task-level SELF-DISCOVER protocol (artifact identifier v6c). Read it in two parts: \emph{discovery} generates and stores a model-specific procedure, whereas \emph{execution} applies that procedure to each routed original post. Figure~C1 separates these scopes. The source templates below are reproduced verbatim; the explanatory paragraphs describe how they connect, rather than modifying the prompts.

The chronology and authorship are distinct. Researchers first write the task description, class policy, module bank, and execution guidance. When no retained plan exists, the selected LLM performs SELECT, ADAPT, and IMPLEMENT using the task materials only; the code saves its generated procedure before classifying any post. No individual post or reference label enters these discovery calls. Only then does the LLM receive a routed original post together with the saved procedure and fixed execution guidance. A retained plan bypasses all three discovery calls.

The stored plan is a reusable text artifact, not a set of cached predictions. Each model uses its own plan for both datasets. Reusing it does not mean learning a procedure from the first evaluation post, reusing that post's answer, updating model weights, or bypassing per-post inference.''')
    source = replace_once(source,
        'Final task-level SELF-DISCOVER workflow. Each model generates its own cached plan, reused for Reddit and Mixed Emotion. The compact-plan instruction is not a guaranteed output constraint. The diagram summarizes the final protocol rather than the earlier per-post discovery variant.',
        'Final task-level SELF-DISCOVER workflow, separating authorship, timing, and input scope. (a) Researchers supply task materials; before post classification, the selected LLM generates a plan through SELECT, ADAPT, and IMPLEMENT if no retained plan exists. No evaluation post or reference label enters discovery. (b) The same LLM applies the saved plan and fixed researcher-authored checks to each routed original post, generating a fresh response. The dashed arrow denotes plan reuse, not reuse of an answer. Parsing and fallback are code operations, not additional LLM calls.')
    source = replace_once(source,
        'The policy placeholder expands to the shared classification policy. The source template\'s reference to a title and body also remains in the Mixed Emotion run, whose input is the synthetic post text.',
        r'''\textit{Purpose and scope.} These are researcher-authored inputs to model-level discovery, not an individual evaluation post. The task description defines the three-label annotation problem, and the module bank supplies 18 candidate operations concerning emotional subject, time frame, trajectory, topic, and competing evidence. The policy placeholder expands to the shared classification policy in Appendix~B. The title--body wording is retained verbatim in the Mixed Emotion run, whose supplied input is the synthetic post text.''')
    source = replace_once(source,
        'The module placeholder expands to the complete bank above; its spelling is retained from the code.',
        r'''\textit{Input:} the task description and full module bank. \textit{Output:} the model's selected module descriptions, passed to ADAPT. This selection occurs during plan construction, not separately for each evaluated post. Placeholder spelling, including \texttt{resonining\_modules}, is retained from the code.''')
    source = replace_once(source, r'\subsection{ADAPT}' + '\n' + r'\lstinputlisting{prompts/sd_adapt.txt}', r'''\subsection{ADAPT}
\textit{Input:} the task description and SELECT output. \textit{Output:} task-specialized module descriptions, passed to IMPLEMENT. This step asks the model to translate the selected operations into the emotion-classification setting; it does not yet assign a post-level label.
\lstinputlisting{prompts/sd_adapt.txt}''')
    source = replace_once(source, r'\subsection{Compact IMPLEMENT}' + '\n' + r'\lstinputlisting{prompts/sd_implement.txt}', r'''\subsection{Compact IMPLEMENT}
\textit{Input:} the task description and ADAPT output. \textit{Output:} a reusable reasoning plan, retained for later execution. Here IMPLEMENT means constructing the procedure, not executing it on every post. The prompt requests at most three one-line steps and access to the full post at each step. There is no conformance validator or retry enforcing those requests; the realized plans are reported below.
\lstinputlisting{prompts/sd_implement.txt}''')
    source = replace_once(source,
        'The system message below precedes the execution user message. The structure placeholder is the model-generated cached plan, not a manually supplied answer, and the text placeholder is the routed original post.',
        r'''\textit{Input:} one routed original post, the shared policy, the model's saved plan, and the fixed execution instructions below. \textit{Output:} a newly generated response ending in a classification line. The system message precedes the user message; \texttt{structure} is replaced by the saved plan and \texttt{text} by this post. No Phase~1 label is supplied to the model.

The evidence ledger is a three-way comparison: for each label the model is asked to identify the strongest supporting and opposing quotation. The subsequent counterfactual check evaluates a competing interpretation rather than changing the post. These are explicit instructions attached to every execution call, not properties guaranteed by caching or by the discovered plan alone.''')
    anchor = r'\subsection{Realized Cached Plans}'
    source = replace_once(source, anchor, r'''\Needspace{15\baselineskip}
\subsection{Illustrative Passage Through the Execution Stage}
Consider the constructed sentence, ``Last year I felt isolated; now I enjoy meeting my friends again and feel relieved.'' This is an explanatory example, not a dataset record, model response, or additional evaluation result. The same cached plan would be inserted regardless of this sentence's eventual label. A policy-consistent reading would separate past isolation from current relief, use the quoted time markers as evidence, and examine whether unresolved current distress is actually stated. Under the stated policy, the resolved positive trajectory supports Happy. The example illustrates what the instructions request; it does not show that every generated response follows them.

The evaluator then reads the terminal label from the actual response. A permitted label replaces the Phase~1 prediction for this routed post. If no label is parsed, the evaluator retains Phase~1 instead. Unrouted posts never enter this execution stage. The scored prediction is therefore distinct from the generated explanation, and a parsing failure is not automatically a wrong classification.

''' + anchor)
    source = replace_once(source, r'\section{SELF-DISCOVER Design-Variant Comparison}', r'\section{SELF-DISCOVER Design-Variant Comparison}\label{app:sd-variants}')
    anchor = r'\par\medskip\noindent\textsc{Appendix Table D1. SELF-DISCOVER design-variant comparison}'
    structural = r'''\par\medskip\noindent\textsc{Appendix Table D1. Earlier and final SELF-DISCOVER designs}\par\smallskip
{\small\renewcommand{\arraystretch}{1.18}
\noindent\begin{tabularx}{\textwidth}{@{}p{0.20\textwidth}YY@{}}\toprule
Component & Earlier independent-countercheck & Final task-level compact plan \\\midrule
Discovery scope & A plan constructed for each post & A model-specific plan constructed for the task and reused \\
Module bank & General reasoning operations & 18 emotion-oriented operations \\
Execution guidance & Independent assessment and countercheck & Explicit subject/time checks, three-label quotation ledger, and counterfactual check \\
Phase 1 label & Withheld during SD reasoning & Withheld during SD reasoning \\
Evaluation role & Earlier design comparator in this appendix & SELF-DISCOVER condition in the main comparison \\\bottomrule
\end{tabularx}}\par\medskip

'''
    source = replace_once(source, anchor, structural + r'\par\medskip\noindent\textsc{Appendix Table D2. Full-set outcomes of the two SD designs}')
    source = source.replace(r'\noindent\textsc{Appendix Table C1.',
                            r'\Needspace{15\baselineskip}\noindent\textsc{Appendix Table C1.')
    source = source.replace(r'\section{Synthetic Mixed Emotion Dataset Generation Protocol}',
                            r'\clearpage' + '\n' + r'\section{Synthetic Mixed Emotion Dataset Generation Protocol}')
    source = replace_once(source,
        'Dataset & Model & Earlier accuracy & Final accuracy & Earlier net & Final net',
        'Dataset & Model & Earlier acc. & Final acc. & Earlier net & Final net')
    source = replace_once(source,
        'The final design improves three of these four model--dataset combinations;',
        r'''\noindent\textit{Table note.} Accuracy uses all 12,000 Reddit posts or all 300 Mixed Emotion examples, not only routed posts. Net denotes corrected minus introduced errors relative to the same final Phase~1 predictions. ``Earlier'' refers only to the independent-countercheck design, not another intermediate compact-plan version. Table~D2 reports observed outcomes, not a paired significance test between the two SD designs.

The final design improves three of these four model--dataset combinations;''')
    source = replace_once(source,
        'These calculations use the final reported corrected/introduced counts, with all unparseable SD responses left at the implemented Phase~1 fallback.',
        r'''Each row asks whether one complete re-evaluation configuration differs from Phase~1 on the same examples. It does not ask whether SD is better than Direct or CoT. ``Change'' is the full-set accuracy difference in percentage points; the interval estimates uncertainty in that difference; exact $p$ compares the numbers corrected and newly harmed; Holm $p$ adjusts the 12 displayed comparisons.

These calculations use the final reported corrected/introduced counts, with all unparseable SD responses left at the implemented Phase~1 fallback.''')
    # Keep each explanatory lead-in with the beginning of its verbatim template.
    for title in ['SELECT', 'ADAPT', 'Compact IMPLEMENT', 'Per-Post Execution',
                  'Task Description and Module Bank', 'Realized Cached Plans']:
        source = source.replace(r'\subsection{' + title + '}',
                                r'\Needspace{12\baselineskip}' + '\n' + r'\subsection{' + title + '}')
    start = source.index(r'\section{Synthetic Mixed Emotion Dataset Generation Protocol}')
    synthetic = source[start:].replace(r'\footnotesize', r'\small')
    synthetic = synthetic.replace(r'\begin{minipage}[t]{0.49\textwidth}\scriptsize',
                                  r'\begingroup\small')
    synthetic = synthetic.replace(r'\end{minipage}\hfill', r'\par\endgroup\medskip')
    synthetic = synthetic.replace(r'\end{minipage}', r'\par\endgroup')
    source = source[:start] + synthetic
    return source
