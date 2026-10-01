"""Improve manuscript clarity and emphasis without changing evidence or scope."""


EDITS = [
    (
        r'''\item It develops a task-specific SELF-DISCOVER adaptation \cite{ref18} combining an emotion-oriented module bank, a reusable model-generated procedure, and fixed checks for emotional subject, temporal context, competing label evidence, and alternative interpretations. Its empirical evaluation includes a controlled Mixed Emotion stress test, a separate 9,000-post evaluation with the final Llama~3 protocols fixed, and row-aligned records of corrections and introduced errors. The stress test is explicitly separated from model training and threshold selection.''',
        r'''\item It develops a task-specific SELF-DISCOVER adaptation \cite{ref18} combining an emotion-oriented module bank, a reusable model-generated plan, and fixed checks for emotional subject, temporal context, competing evidence, and alternative interpretations. The adaptation is evaluated on the primary Reddit test set, a controlled Mixed Emotion stress test excluded from training and threshold selection, and a separate 9,000-post evaluation with the final Llama~3 protocols fixed. Example-aligned predictions quantify corrected and introduced errors.''',
    ),
    (
        'These are methodological foundations, not themselves evidence of depression-detection performance.',
        '',
    ),
    (
        r'''Wang et al. \cite{ref43} used Reddit writings to predict questionnaire-related depression levels and generate explanations for individual questionnaire items. Lan et al.'s DORIS framework \cite{ref44} instead uses LLM-derived symptom annotations and temporal mood-course features with a gradient-boosting-tree classifier, alongside generated justifications. These are distinct ways of combining prediction and explanation, not evidence that a generated rationale necessarily corrects an initial label. Omar and Levkovich's review \cite{ref45} includes both encoder models, such as BERT and RoBERTa, and generative approaches; it identifies predictive potential while emphasizing further validation, privacy, and ethical safeguards. Its findings should not be read as established therapeutic efficacy of generative chatbots.''',
        r'''Wang et al. \cite{ref43} used Reddit writings to predict questionnaire-related depression levels and generate explanations for individual questionnaire items. Lan et al.'s DORIS framework \cite{ref44} combines LLM-derived symptom annotations and temporal mood-course features with a gradient-boosting-tree classifier and generated justifications. These studies illustrate distinct ways of integrating prediction and explanation. Omar and Levkovich's review \cite{ref45} identifies predictive potential across encoder and generative models while emphasizing validation, privacy, and ethical safeguards.''',
    ),
    (
        'Selective classification addresses a related but distinct problem: trading coverage for accepted-set risk by abstaining on selected inputs \\cite{ref16}.',
        'Selective classification addresses a related but distinct problem: trading coverage for accepted-set risk by withholding an immediate prediction on selected inputs, a strategy termed abstention \\cite{ref16}.',
    ),
    (
        'For each Reddit post, both textual content and auxiliary metadata were retained during dataset construction.',
        'For each Reddit post, the text fields and metadata needed for preprocessing, linkage, and evaluation were retained during dataset construction.',
    ),
    (
        'Phase~2 therefore used the minimally sanitized original text to retain these cues and sentence boundaries.',
        'For routed Reddit posts, Phase~2 used the linked original title and selftext, not the Phase~1-cleaned input. It decoded HTML, replaced URLs and direct usernames, and normalized whitespace while preserving wording, capitalization, punctuation, negation, sentence order, and sentence boundaries.',
    ),
    (
        'In addition to the primary Reddit dataset, we constructed a supplementary synthetic Mixed Emotion Dataset to examine model behavior under emotionally ambiguous conditions.',
        'In addition to the primary Reddit dataset, we constructed a controlled synthetic Mixed Emotion stress-test dataset to examine model behavior under emotionally ambiguous conditions.',
    ),
    (
        'This design is intended to test whether confidence-guided routing and reasoning-based re-evaluation can address cases that are plausibly more difficult for a single-stage classifier.',
        'This design tests whether confidence-guided routing and reasoning-based re-evaluation can address cases containing competing or shifting emotional evidence.',
    ),
    (
        'Here, synthetic posts serve a different purpose: supplementary stress-test evaluation, not training augmentation or clinical validation.',
        'Here, synthetic posts provide controlled stress-test evaluation, distinct from training augmentation or clinical validation.',
    ),
    (
        'These checks were not an independent expert-annotation study; results describe the final 300 synthetic examples rather than naturally occurring mixed-emotion performance.',
        'These checks establish adherence to the dataset design criteria. Independent expert annotation and naturally occurring mixed-emotion evaluation remain outside the scope of this dataset.',
    ),
    (
        'However, the routed Reddit test cases were reused during Phase~2 prompt development, so final prompt-configuration comparisons are interpreted as exploratory.',
        'The routed Reddit test cases also informed Phase~2 prompt development; within-set prompt-configuration rankings are therefore interpreted as exploratory.',
    ),
    (
        'Auxiliary metadata, such as subreddit source, author-related metadata, timestamp, and adult-content flag, was not used as a direct predictive feature in either phase.',
        'The remaining metadata, such as subreddit source, author-related metadata, timestamp, and adult-content flag, was not used as a direct predictive feature in either phase.',
    ),
    (
        'Thus, this is a comparison of practical classifier configurations, not an isolated test of architecture or maximum attainable performance.',
        'The comparison therefore evaluates practical classifier configurations under shared data partitions and a bounded search budget.',
    ),
    (
        'Auxiliary analyses compare raw MSP, temperature-scaled MSP, entropy-based certainty, and probability margin;',
        'Supporting analyses compare raw MSP, temperature-scaled MSP, entropy-based certainty, and probability margin;',
    ),
    (
        'The export notebook prints a warning and falls back to the lowest empirical risk, then highest coverage, if no candidate satisfies the constraint. That branch was not needed here: the selected calibration point satisfies the prescribed upper-bound criterion. A fallback, if invoked, would not be interpreted as a risk-feasible policy. The temperature, threshold, risk target, confidence level, and candidate grid are stored with the calibration and threshold tables. The model source and seed are recorded separately in the execution environment artifact.',
        'The selected calibration point satisfied the upper-bound criterion, so the export notebook\'s infeasibility fallback was not invoked. The stored calibration, threshold, and environment artifacts record the policy values and model and seed identifiers.',
    ),
    (
        "Thus, neither total generation cost nor token budget is claimed to be equal across all protocols. Llama~2's shorter context also constrains multi-turn inference. Template text and nominal execution budgets are disclosed in the appendices; wall-clock or monetary superiority of the final Phase~2 configurations is not inferred from accuracy.",
        "Generation budgets therefore remain part of each complete protocol, and Llama~2's shorter context also constrains multi-turn inference. The appendices disclose template text and nominal execution budgets; a matched end-to-end time and cost comparison is reserved for separate benchmarking rather than inferred from accuracy.",
    ),
    (
        r'''DistilBERT had the highest mean score and approximately 67 million parameters, compared with about seven billion for each comparator. These results support retaining it for the tested pipeline, without establishing superiority over fully adapted 7B models.''',
        r'''DistilBERT had the highest mean score and approximately 67 million parameters, compared with about seven billion for each comparator. Under the tested configurations and shared budget, its predictive performance and measured inference efficiency support its selection as the first-stage classifier.''',
    ),
    (
        r'''Table~\ref{tab:final-e2e} reports the complete final comparison. Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 among these six configurations on both datasets. The advantage over Llama~3 CoT on Reddit is six correct predictions (0.0500 percentage points). Paired tests do not establish an SD advantage over Direct or CoT there (Holm $p=0.2163$ and $0.3616$). On Mixed Emotion, SD exceeds both alternatives within each model after the stated two-comparison Holm adjustment; for Llama~3, the adjusted values are $0.0020$ versus Direct and $<0.0001$ versus CoT.''',
        r'''Table~\ref{tab:final-e2e} reports the complete final comparison. Llama~3 SELF-DISCOVER produces the largest observed Phase~1 gain on both datasets: 34 net corrections on Reddit and 27 on Mixed Emotion, reaching 97.2000\% and 92.6667\% accuracy. It also leads the six configurations in accuracy and macro F1 on each dataset. On Mixed Emotion, SD exceeds both alternatives within each model after Holm adjustment; for Llama~3, adjusted $p=0.0020$ versus Direct and $p<0.0001$ versus CoT. On Reddit, its six-prediction lead over Llama~3 CoT is not statistically significant, nor is its difference from Direct (Holm $p=0.3616$ and $0.2163$).''',
    ),
    (
        'Thus, SELF-DISCOVER does not outperform Direct within every model and dataset combination.',
        'For Llama~2, Direct, CoT, and SELF-DISCOVER yield nine, one, and three net corrections, showing a model--protocol interaction.',
    ),
    (
        'This low introduced-error count is an observed property of the 300-example stress test, not a population safety guarantee.',
        'The single introduced error indicates strong preservation of initially correct labels within this controlled 300-example stress test.',
    ),
    (
        r'''Table~\ref{tab:additional9000} reports the fixed-protocol results. Direct and CoT each yield 13 net corrections; SELF-DISCOVER yields 17 and reaches 97.7667\% accuracy. The SD gain over Phase~1 is 0.1889 percentage points (95\% CI [0.0444, 0.3333]; Holm $p=0.0412$). Its four-correct-prediction lead over either alternative is not significant (Holm $p=0.9614$ for both comparisons). The evidence therefore supports improvement over the first-stage classifier, not a statistically established ranking of the three protocols on this set. There are two Direct parsing failures and none for CoT or SD; all use the same fallback rule.''',
        r'''Table~\ref{tab:additional9000} reports the fixed-protocol results. SELF-DISCOVER yields 17 net corrections, increases accuracy from 97.5778\% to 97.7667\%, and significantly improves over Phase~1 by 0.1889 percentage points (95\% CI [0.0444, 0.3333]; Holm $p=0.0412$). Direct and CoT each yield 13 net corrections. SELF-DISCOVER retains the highest observed accuracy, although its four-correct-prediction lead over either alternative is not statistically significant (Holm $p=0.9614$ for both comparisons). There are two Direct parsing failures and none for CoT or SD; all use the same fallback rule.''',
    ),
    (
        'Confidence routing identifies an error-enriched subset, but enrichment alone is insufficient.',
        'Confidence routing creates an error-enriched correction opportunity; the re-evaluator determines the resulting net gain.',
    ),
    (
        r'''Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 on both main sets and among the three protocols in the additional evaluation. Its within-model advantage is statistically supported on Mixed Emotion, but not on either Reddit set. Improvement over Phase~1 and superiority to another re-evaluator are different claims.''',
        r'''Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 on all three evaluation sets and significantly improves final accuracy over Phase~1 under the stated Holm adjustments. Within-model protocol comparisons answer a narrower question: SD's advantage over Direct and CoT is supported on Mixed Emotion, while the smaller differences on the Reddit evaluations remain uncertain.''',
    ),
    (
        r'''A unified DistilBERT-to-LLM pipeline connects classifier selection, calibrated routing, and selective original-text re-evaluation through one final classifier. The row-verified comparison shows that routing concentrates errors but does not guarantee beneficial correction: outcomes depend on both the language model and the prompting protocol. Llama~3 SELF-DISCOVER achieves the highest observed accuracy and macro F1 in the main Reddit and Mixed Emotion comparisons and in the additional 9,000-post evaluation. Its gains over Phase~1 are 34, 27, and 17 net corrections, respectively. The additional gain remains significant after adjustment across three Phase~1 comparisons. SD also exceeds Direct and CoT on Mixed Emotion within both models, whereas the Reddit results do not establish significant between-protocol superiority. The task-specific reusable plan and evidence checks are therefore supported as a useful complete re-evaluation configuration, not as a universally superior reasoning mechanism. These findings concern proxy-label text classification and do not support clinical deployment without further validation.''',
        r'''A unified DistilBERT-to-LLM pipeline connects efficient classifier selection, calibrated routing, and selective original-text re-evaluation. Across the main Reddit test, the Mixed Emotion stress test, and the additional 9,000-post evaluation, Llama~3 SELF-DISCOVER converts the routed correction opportunity into 34, 27, and 17 net corrections, respectively, and significantly improves accuracy over Phase~1 under the stated Holm adjustments. It achieves the highest observed accuracy and macro F1 among the evaluated configurations on all three sets. Paired comparisons further show that SD exceeds Direct and CoT on Mixed Emotion within both models, while the smaller between-protocol differences on Reddit remain unresolved. The results support the task-specific reusable plan and evidence checks as an effective complete re-evaluation configuration and demonstrate that model and protocol choice determine how routing translates into final gains. These findings establish value for proxy-label text classification; clinical use requires separate validation and governance.''',
    ),
    (
        r'\section{Supplementary Methodological Details}',
        r'\section{Additional Methodological Details}',
    ),
    (
        'actual class-balanced examples from the supplementary Mixed Emotion stress test',
        'class-balanced examples from the controlled Mixed Emotion stress test',
    ),
    (
        'final 300-example supplementary dataset',
        'final 300-example controlled stress-test dataset',
    ),
    (
        'for supplementary stress-test evaluation.',
        'for controlled stress-test evaluation.',
    ),
    (
        'used only for supplementary robustness evaluation.',
        'used only for controlled stress-test evaluation.',
    ),
]


def polish_language(source):
    for old, new in EDITS:
        if source.count(old) != 1:
            raise ValueError(f'Expected one language-edit anchor: {old[:100]}')
        source = source.replace(old, new, 1)
    return source
