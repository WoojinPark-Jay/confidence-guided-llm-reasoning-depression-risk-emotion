"""Update the reviewed manuscript from the complete row-verified final results."""
from pathlib import Path
import difflib
import json
import os
import re
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import pymupdf as fitz
from update_architecture_labels_20260930 import update_architecture
from manuscript_frontmatter_20260930 import polish_frontmatter
from manuscript_language_polish_20261001 import polish_language

from manuscript_visual_style_20260929 import save, INK, BLUE, GRAY, LIGHT_BLUE, style_axes

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
BASE = REPO / 'reports/unified_v6c_revision_20260929'
AUDIT = REPO / 'reports/final_results_audit_20260930'
REPORT = REPO / 'reports/verified_manuscript_20260930'
OUT = WORK / 'Paper_20260930_verified'
PACKAGE = WORK / 'final_results_audit_20260930/received_package'
METRICS = pd.read_csv(AUDIT / 'row_verified_metrics.csv')
STATS = pd.read_csv(AUDIT / 'phase1_paired_statistics.csv')
CROSS = pd.read_csv(AUDIT / 'sd_vs_baselines_paired_statistics.csv')
assert len(METRICS) == 15 and STATS.evidence.eq('row_verified').all()
assert len(CROSS) == 10 and CROSS.status.eq('row_verified').all()
ORDER = {'Reddit':0,'Mixed Emotion':1,'Additional Reddit 9000':2}
METHOD_ORDER = {'Direct':0,'CoT':1,'SELF-DISCOVER':2}
METRICS = METRICS.assign(_d=METRICS.dataset.map(ORDER),_m=METRICS.method.map(METHOD_ORDER)).sort_values(['_d','model','_m'])
STATS = STATS.assign(_d=STATS.dataset.map(ORDER),_m=STATS.method.map(METHOD_ORDER)).sort_values(['_d','model','_m'])
CROSS = CROSS.assign(_d=CROSS.dataset.map(ORDER),_m=CROSS.comparison.map({'SD vs Direct':0,'SD vs CoT':1})).sort_values(['_d','model','_m'])


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one edit anchor: {old[:90]}')
    return source.replace(old, new, 1)


def section(source, start, end, value):
    a, b = source.index(start), source.index(end, source.index(start))
    return source[:a] + value.strip() + '\n\n' + source[b:]


def pvalue(p):
    return r'$<0.0001$' if p < .0001 else f'{p:.4f}'


def appendix_table(title, headings, rows, spec):
    return '\n'.join([
        r'\Needspace{12\baselineskip}',
        r'\par\medskip\noindent\textsc{' + title + r'}\par\smallskip',
        r'{\small\renewcommand{\arraystretch}{1.15}\noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}' + spec + r'@{}}\toprule',
        ' & '.join(headings) + r' \\\midrule',
        *[' & '.join(row) + r' \\' for row in rows],
        r'\bottomrule\end{tabular*}}\par\medskip',
    ])


def figure_matrices(name, conditions):
    fig, axes = plt.subplots(1, len(conditions), figsize=(11 if len(conditions)==2 else 13, 4.25))
    cmap = LinearSegmentedColormap.from_list('verified_blues', ['#F3F4F5', '#3E719F'])
    for k, (ax, (dataset, model, method, title)) in enumerate(zip(axes, conditions)):
        stem = f'{dataset}_{model}_{method}'.replace(' ', '_')
        matrix = pd.read_csv(AUDIT / f'{stem}_confusion_matrix.csv', index_col=0).to_numpy()
        rates = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(rates, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                color = 'white' if rates[i,j] > .55 else INK
                ax.text(j, i-.08, f'{matrix[i,j]:,}', ha='center', va='center', color=color, weight='bold', fontsize=12)
                ax.text(j, i+.18, f'{rates[i,j]:.1%}', ha='center', va='center', color=color, fontsize=10)
        ax.set_xticks(range(3), ['Depression','Neutral','Happy'])
        ax.set_yticks(range(3), ['Depression','Neutral','Happy'])
        ax.xaxis.tick_top(); ax.xaxis.set_label_position('top')
        ax.set_xlabel('Predicted label', labelpad=9)
        ax.set_ylabel('Reference label', labelpad=10)
        ax.set_title(f'({chr(97+k)})  {title}', pad=42)
        ax.tick_params(length=0, pad=8)
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1)
        ax.tick_params(which='minor', length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.subplots_adjust(left=.11, right=.98, bottom=.06, top=.76, wspace=.54 if len(conditions)==2 else .85)
    save(fig, OUT, name)


def build():
    OUT.mkdir(exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    for p in (WORK / 'Paper_20260929_unified_v6c').iterdir():
        if p.suffix in {'.cls','.sty','.fd','.map','.pfb','.tfm','.bib','.png'}:
            shutil.copy2(p, OUT / p.name)
    for folder in ['figures','prompts','evidence']:
        shutil.copytree(BASE / folder, OUT / folder, dirs_exist_ok=True)
    (OUT / 'qa').mkdir(exist_ok=True)
    for p in AUDIT.glob('*.csv'): shutil.copy2(p, OUT / 'evidence' / p.name)
    old = (BASE / 'main.tex').read_text()
    s = old
    s = replace(s, r'\title{Confidence-Guided Selective LLM Re-Evaluation for Depression-Risk-Related Emotion Classification in Social Media Text}', r'\title{Confidence-Guided Selective LLM Re-Evaluation for Emotion Classification in Social Media}')
    s = section(s, r'\begin{abstract}', r'\end{abstract}', r'''\begin{abstract}
Confidence-based routing can concentrate classifier errors, but improvement depends on how selected inputs are re-evaluated. We study a two-phase pipeline for three-class proxy emotion classification of social-media posts. A validation-selected DistilBERT model supplies calibrated predictions; high-confidence labels are retained, while low-confidence posts undergo original-text re-evaluation. We compare Direct, Chain-of-Thought (CoT), and task-adapted SELF-DISCOVER with Llama~2 and Llama~3. On 12,000 Reddit posts, Phase~1 achieves 96.9167\% accuracy and routes 218 posts (1.82\%). Llama~3 SELF-DISCOVER reaches 97.2000\%, correcting 72 errors and introducing 38. On 300 synthetic Mixed Emotion examples, it increases accuracy from 83.6667\% to 92.6667\%, with 28 corrections and one introduced error. SELF-DISCOVER outperforms Direct and CoT on the stress test for both models in paired comparisons, whereas its small lead within Llama~3 on Reddit is not statistically significant. A separate 9,000-post same-source evaluation with the final classifier, prompts, and cached plan fixed routes 111 posts (1.23\%). Llama~3 SELF-DISCOVER improves accuracy from 97.5778\% to 97.7667\% (Holm-adjusted $p=0.0412$ versus Phase~1); its lead over Direct and CoT is not significant. These results support selective re-evaluation and show that model and protocol choice affect both correction yield and introduced errors. The labels are non-clinical proxies, and synthetic stress-test gains do not establish clinical validity.''')
    s = replace(s, 'The evaluation addresses three research questions.', 'The evaluation addresses three research questions, with a separate same-source evaluation of the fixed final Llama~3 protocols.')
    s = replace(s, 'Its empirical evaluation includes a controlled Mixed Emotion stress test and recorded outputs for inspecting ambiguous or shifting affect.', 'Its empirical evaluation includes a controlled Mixed Emotion stress test, a separate 9,000-post evaluation with the final Llama~3 protocols fixed, and row-aligned records of corrections and introduced errors.')
    s = replace(s, r'\subsection{Ethical Considerations}', r'''\subsection{Additional Same-Source Evaluation}
A separate balanced evaluation set comprises 9,000 posts (3,000 per class) sampled from records in the same source corpus that were not included in the original 120,000-post sample. Selection excludes normalized exact-text matches to the original sample and duplicate normalized texts within the candidate pool using Unicode NFKC normalization, case folding, and whitespace normalization. Sampling uses a fixed random seed. These procedures remove normalized exact-text overlap under the specified rule; the set remains same-source and is not guaranteed to be disjoint by author or time.

Before evaluating this set, we fixed the final seed-42 DistilBERT model trained under the selected hyperparameters, the calibration temperature, the routing threshold, the three Llama~3 prompting protocols, and the previously generated SELF-DISCOVER plan. No training, recalibration, threshold selection, SELF-DISCOVER plan generation, or prompt revision used these 9,000 posts. Direct, CoT, and SELF-DISCOVER evaluated the same routed cases, and all end-to-end metrics were computed over the full 9,000-post set.

\subsection{Ethical Considerations}''')
    s = replace(s, 'Figure~\\ref{fig:architecture} retains the original overview layout; the final configuration details are specified in the text.', 'Figure~\\ref{fig:architecture} summarizes this shared pipeline; Figure~\\ref{fig:sd-task-level} details the SELF-DISCOVER procedure.')
    s = replace(s, 'Original artwork is retained in this review copy pending label updates; the final experiment evaluates both models with Direct, CoT, and SELF-DISCOVER as specified in the text.', 'Llama~2 and Llama~3 are each evaluated with Direct, CoT, and SELF-DISCOVER on the same routed posts. Parsed labels replace the initial predictions; parsing failures retain Phase~1 labels.')
    s = section(s, 'Appendix~\\ref{app:statistics} reports count-based paired accuracy summaries.', r'\section{Experimental Results}', r'''All 15 final conditions are aligned by example ID and checked against the stored Phase~1 predictions and reference labels. The checks cover missing or duplicate IDs, parsed outputs, fallback behavior, full-set accuracy, macro F1, class-wise metrics, and corrected/introduced counts. The additional evaluation also validates 777 saved generation calls against their run settings and request hashes. No failed response is regenerated or assigned a reference-derived label.

Two-sided exact McNemar tests compare paired correctness. Against Phase~1, discordant counts are $C$ and $I$; between two re-evaluators, they are the numbers correct only under each method. We estimate 95\% percentile-bootstrap intervals for accuracy differences with 50,000 multinomial resamples of the paired correctness-difference categories $+1,-1,0$. This yields the same distribution as resampling paired rows for that scalar difference. Holm correction uses the 12 main configuration-versus-Phase~1 tests as one family and the three additional-evaluation tests as another. Within each dataset--model combination, the two SD-versus-Direct/CoT tests form a separate family. Appendix~\ref{app:statistics} reports unadjusted and adjusted values and all comparison scopes.

These intervals characterize example-level accuracy differences conditional on the fitted models and protocols. They are not confidence intervals for macro F1 or training-seed variability, and they do not account for within-author dependence. The main prompt-development comparisons are exploratory; the additional evaluation applies the frozen final protocols without adapting them during that run.''')
    s = replace(s, 'The advantage over Llama~3 CoT on Reddit is six correct predictions (0.0500 percentage points); numerical ordering alone does not establish a significant between-method difference.', 'The advantage over Llama~3 CoT on Reddit is six correct predictions (0.0500 percentage points). Paired tests do not establish an SD advantage over Direct or CoT there (Holm $p=0.2163$ and $0.3616$). On Mixed Emotion, SD exceeds both alternatives within each model after the stated two-comparison Holm adjustment; for Llama~3, the adjusted values are $0.0020$ versus Direct and $<0.0001$ versus CoT.')
    s = replace(s, r'\subsection{Conditional Correction Opportunity}', r'''\begin{figure*}[t]\centering
\includegraphics[width=0.96\textwidth]{figures/final_sd_confusion.pdf}
\caption{Final Llama~3 SELF-DISCOVER confusion matrices, reconstructed from all end-to-end predictions. Rows are reference labels and columns are predictions; counts and within-class percentages share the Phase~1 figure's scale. Unparsed routed responses retain the initial label.}
\label{fig:final-sd-confusion}
\end{figure*}

\subsection{Additional Evaluation of the Fixed Pipeline}
On the separate 9,000-post set, Phase~1 correctly classifies 8,782 posts (97.5778\%) and routes 111 (1.23\%). The routed subset contains 48 of the 218 Phase~1 errors: its error rate is 43.24\%, compared with 2.42\% overall. Thus, the fixed threshold again selects an error-enriched subset while retaining 98.77\% coverage.

Table~\ref{tab:additional9000} reports the fixed-protocol results. Direct and CoT each yield 13 net corrections; SELF-DISCOVER yields 17 and reaches 97.7667\% accuracy. The SD gain over Phase~1 is 0.1889 percentage points (95\% CI [0.0444, 0.3333]; Holm $p=0.0412$). Its four-correct-prediction lead over either alternative is not significant (Holm $p=0.9614$ for both comparisons). The evidence therefore supports improvement over the first-stage classifier, not a statistically established ranking of the three protocols on this set. There are two Direct parsing failures and none for CoT or SD; all use the same fallback rule. Appendix~\ref{app:additional} provides the fixed settings and class-level results.

\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.14}
\caption{Additional 9,000-post evaluation with the final Llama~3 protocols fixed. Accuracy and macro F1 are percentages. Change and its 95\% interval are in percentage points relative to Phase~1; Holm $p$ adjusts the three comparisons with Phase~1.}
\label{tab:additional9000}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}\toprule
Protocol & Accuracy & Macro F1 & $C$ & $I$ & Change & Paired 95\% CI & Holm $p$ \\\midrule
Phase 1 & 97.5778 & BASE9000F1 & -- & -- & -- & -- & -- \\
Direct & 97.7222 & 97.7242 & 32 & 19 & +0.1444 & [-0.0111, 0.3000] & 0.1320 \\
CoT & 97.7222 & 97.7234 & 28 & 15 & +0.1444 & [0.0000, 0.2889] & 0.1320 \\
SELF-DISCOVER & 97.7667 & 97.7688 & 30 & 13 & +0.1889 & [0.0444, 0.3333] & 0.0412 \\\bottomrule
\end{tabular*}\end{table*}

\subsection{Conditional Correction Opportunity}''')
    p1 = json.loads((PACKAGE / 'additional_9000_20260930/phase1/phase1_summary.json').read_text())
    s = s.replace('BASE9000F1', f'{p1["phase1_macro_f1"]*100:.4f}')
    s = replace(s, 'These are descriptive ceilings under a fixed router, not achievable performance guarantees or alternative tuned baselines.', r'On the additional 9,000 posts, the corresponding ceiling is 98.1111\% and SD realizes $17/48=35.42\%$ of the net-correction opportunity. These are descriptive ceilings under a fixed router, not achievable performance guarantees or alternative tuned baselines.')
    s = replace(s, 'Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 on both sets. However,', 'Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 on both main sets and among the three protocols in the additional evaluation. Its within-model advantage is statistically supported on Mixed Emotion, but not on either Reddit set. Improvement over Phase~1 and superiority to another re-evaluator are different claims. Moreover,')
    s = replace(s, 'Prompt variants were developed and compared using these evaluation cases. Consequently, the selected prompt\'s observed ranking and count-based significance summaries remain exploratory. A frozen final configuration should be assessed on unused data to estimate generalization after selection. Earlier evaluations of a different downstream configuration are not used as independent validation of the final pipeline in this manuscript.', 'The main Reddit and Mixed Emotion cases informed prompt development, so their configuration rankings are exploratory and the reported corrections do not adjust for prompt selection. The separate 9,000-post evaluation applies the final Llama~3 protocols and retained plan without changes during evaluation, providing an additional same-source check. It does not establish author-disjoint, temporal, cross-platform, or clinical generalization, and its small between-protocol differences remain uncertain.')
    s = replace(s, 'The final SELF-DISCOVER summaries contain two such failures on Reddit and one on Mixed Emotion for Llama~3.', 'Final SELF-DISCOVER has 21 parsing failures for Llama~2 on Reddit and none on Mixed Emotion, and two and one, respectively, for Llama~3. The additional Llama~3 SD evaluation has no parsing failures. The 21 Llama~2 Reddit fallbacks are part of its measured configuration-level performance rather than excluded cases.')
    s = section(s, r'\section{Conclusion}', r'\begin{thebibliography}', r'''\section{Conclusion}
A unified DistilBERT-to-LLM pipeline connects classifier selection, calibrated routing, and selective original-text re-evaluation through one final classifier. The row-verified comparison shows that routing concentrates errors but does not guarantee beneficial correction: outcomes depend on both the language model and the prompting protocol. Llama~3 SELF-DISCOVER achieves the highest observed accuracy and macro F1 in the main Reddit and Mixed Emotion comparisons and in the additional 9,000-post evaluation. Its gains over Phase~1 are 34, 27, and 17 net corrections, respectively. The additional gain remains significant after adjustment across three Phase~1 comparisons. SD also exceeds Direct and CoT on Mixed Emotion within both models, whereas the Reddit results do not establish significant between-protocol superiority. The task-specific reusable plan and evidence checks are therefore supported as a useful complete re-evaluation configuration, not as a universally superior reasoning mechanism. These findings concern proxy-label text classification and do not support clinical deployment without further validation.''')
    s = replace(s, 'Remaining final SD parsing failures are zero for Llama~2 on both sets, two for Llama~3 Reddit, and one for Llama~3 Mixed Emotion.', 'Remaining final SD parsing failures are 21 for Llama~2 Reddit, zero for Llama~2 Mixed Emotion, two for Llama~3 Reddit, and one for Llama~3 Mixed Emotion. There are no SD parsing failures in the additional 9,000-post evaluation.')
    # Replace the provisional appendix with actual row-aligned inference tables.
    appendix = r'''\section{Paired Accuracy Comparisons}\label{app:statistics}
All rows below use ID-aligned final predictions, including the implemented Phase~1 fallback for parsing failures. Table~E1 compares each main configuration with Phase~1; Table~E2 instead compares SD with Direct or CoT on the same examples. A positive change favors the named re-evaluator in E1 and SD in E2. Intervals are 50,000-resample paired percentile-bootstrap intervals for accuracy change, in percentage points. Exact $p$ is a two-sided McNemar test. These tests do not measure macro F1 uncertainty or adjust for prompt development on the main evaluation cases.
'''
    rows = []
    for r in STATS[STATS.n != 9000].itertuples():
        rows.append([r.dataset, r.model, 'SD' if r.method=='SELF-DISCOVER' else r.method, f'{r.change_pp:+.4f}', f'[{r.ci_low_pp:.3f}, {r.ci_high_pp:.3f}]', pvalue(r.exact_p), pvalue(r.holm_p)])
    appendix += appendix_table('Appendix Table E1. Main comparisons with Phase 1', ['Dataset','Model','Protocol','Change (pp)',r'Paired 95\% CI','Exact $p$','Holm $p$'], rows, 'lllrrrr')
    appendix += '\nHolm adjustment in Table~E1 covers all 12 rows.\n'
    rows = []
    for r in CROSS.itertuples():
        rows.append(['Additional 9,000' if r.dataset.startswith('Additional') else r.dataset, r.model, r.comparison.replace('SD vs ','SD -- '), f'{r.change_pp:+.4f}', f'[{r.ci_low_pp:.3f}, {r.ci_high_pp:.3f}]', pvalue(r.exact_p), pvalue(r.holm_p_family2)])
    appendix += appendix_table('Appendix Table E2. SD versus alternative re-evaluators', ['Dataset','Model','Comparison','Change (pp)',r'Paired 95\% CI','Exact $p$','Holm $p$'], rows, 'lllrrrr')
    appendix += '\nHolm adjustment in Table~E2 is applied to the two comparisons within each dataset--model combination, not to all ten rows as one family.\n'
    rows = []
    for r in STATS[STATS.n == 9000].itertuples():
        rows.append(['SD' if r.method=='SELF-DISCOVER' else r.method, f'{r.change_pp:+.4f}', f'[{r.ci_low_pp:.4f}, {r.ci_high_pp:.4f}]', pvalue(r.exact_p), pvalue(r.holm_p)])
    appendix += appendix_table('Appendix Table E3. Additional 9,000 posts versus Phase 1', ['Protocol','Change (pp)',r'Paired 95\% CI','Exact $p$','Holm $p$'], rows, 'lrrrr')
    appendix += '\nTable~E3 treats its three comparisons as a separate Holm family. Non-significance of an SD-versus-alternative comparison is not evidence of equivalence.\n'
    s = section(s, r'\Needspace{28\baselineskip}'+'\n'+r'\section{Paired Accuracy Summaries}', r'\clearpage'+'\n'+r'\section{Synthetic Mixed Emotion Dataset Generation Protocol}', appendix)
    rows = []
    for r in METRICS.itertuples():
        stem=f'{r.dataset}_{r.model}_{r.method}'.replace(' ','_')
        cr=json.loads((AUDIT/f'{stem}_classification_report.json').read_text())
        rows.append(['Additional 9,000' if r.dataset.startswith('Additional') else r.dataset,r.model,'SD' if r.method=='SELF-DISCOVER' else r.method,*[f'{cr[label]["f1-score"]*100:.2f}' for label in ['Depression','Neutral','Happy']],str(r.parse_failures)])
    extra=r'''\clearpage
\section{Class-Level Results and Additional Evaluation}\label{app:additional}
Table~G1 reports class-wise F1 for each complete pipeline. All metrics use the full dataset, not only routed posts. Failures count routed responses without a parsed label; those records retain their Phase~1 predictions and remain in every metric. Reddit and Mixed Emotion contain 4,000 and 100 reference examples per class, respectively; the additional set contains 3,000 per class.
'''
    extra+=appendix_table('Appendix Table G1. Class-wise F1 and retained parsing failures', ['Dataset','Model','Protocol','Depression F1','Neutral F1','Happy F1','Failures'],rows,'lllrrrr')
    extra+=r'''
\subsection{Fixed Additional-Evaluation Protocol}
The additional run loads the same final seed-42 DistilBERT weights and uses $T^*=1.480195$, $\tau^*=0.70$, a 256-token limit, and fp32 inference with batch size 64. Phase~1 training, calibration fitting, and threshold search are disabled. The Llama~3 protocols use the exact templates in Appendices~B--C and the retained model-specific SD plan, with no new discovery calls. All three methods receive the same 111 routed original texts; the 6,000-character input policy, greedy decoding, 4-bit NF4 loading, and per-call limits (1,024 for Direct/CoT; 1,536 for SD) remain fixed. Execution uses an NVIDIA L4 GPU. Runtime, model revision, input hashes, and the saved plan are recorded with the run.

There are 111 Direct calls, 555 CoT calls, and 111 SD calls. These counts include the five-turn CoT dialogue but no new SD discovery. All 777 saved calls were checked against their request hashes and run configuration; final outputs were aligned by ID and recombined with accepted Phase~1 predictions. The separate example-level tests in Appendix~E use these complete predictions. Call counts alone do not establish a matched time or monetary-cost advantage.

\begin{center}\includegraphics[width=\textwidth]{figures/additional9000_confusion.pdf}\end{center}
\noindent\footnotesize\textbf{Appendix Figure G1.} End-to-end confusion matrices for Direct, CoT, and final SELF-DISCOVER on the same 9,000 posts. Rows are reference labels; columns are final predictions. Counts and within-class percentages use a common color scale.\normalsize
'''
    s = replace(s, r'\EOD', extra+'\n'+r'\EOD')
    s = replace(s, r'\clearpage'+'\n'+r'\section{Synthetic Mixed Emotion Dataset Generation Protocol}', r'\Needspace{18\baselineskip}'+'\n'+r'\section{Synthetic Mixed Emotion Dataset Generation Protocol}')
    s = polish_frontmatter(s)
    s = polish_language(s)
    (OUT/'main.tex').write_text(s)
    for font in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'): font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.titleweight':'bold','axes.titlesize':12,'axes.labelweight':'bold','pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
    figure_matrices('final_sd_confusion', [('Reddit','Llama 3','SELF-DISCOVER','Reddit - final SD'),('Mixed Emotion','Llama 3','SELF-DISCOVER','Mixed Emotion - final SD')])
    figure_matrices('additional9000_confusion', [('Additional Reddit 9000','Llama 3',m,title) for m,title in [('Direct','Direct'),('CoT','CoT'),('SELF-DISCOVER','SELF-DISCOVER')]])
    update_architecture(BASE, OUT)
    (REPORT/'MANUSCRIPT_FULL_DIFF.md').write_text('# 검증 완료 원고 변경 내역\n\n```diff\n'+''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='20260929/main.tex',tofile='20260930/main.tex'))+'```\n')
    compile_pdf()
    for folder in ['figures','prompts','evidence']:
        shutil.copytree(OUT/folder,REPORT/folder,dirs_exist_ok=True)
    shutil.copy2(OUT/'main.tex',REPORT/'main.tex')


def compile_pdf():
    tex='/Users/woojinpark/.cache/codex-runtimes/codex-texlive/small/bin/universal-darwin'
    tiny='/Users/woojinpark/Library/TinyTeX/texmf-dist'
    cache='/Users/woojinpark/Library/Caches/Tectonic/bundles/data/6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c'
    env=os.environ.copy()
    env.update(TEXINPUTS=f'.:{tiny}/tex//:{cache}//:',TFMFONTS=f'.:{tiny}/fonts/tfm//:',VFFONTS=f'.:{tiny}/fonts/vf//:',T1FONTS=f'.:{tiny}/fonts/type1//:',ENCFONTS=f'.:{tiny}/fonts/enc//:',TEXFONTMAPS=f'.:/Users/woojinpark/Library/TinyTeX/texmf-var/fonts/map/pdftex/updmap//:{tiny}/fonts/map//:')
    for i in range(3):
        p=subprocess.run([tex+'/pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=OUT,env=env,capture_output=True,text=True)
        (OUT/'qa'/f'build-{i}.log').write_text(p.stdout+p.stderr)
        if p.returncode: raise RuntimeError(p.stdout[-3500:])
    doc=fitz.open(OUT/'main.pdf')
    (OUT/'qa/extracted_text.txt').write_text('\n'.join(p.get_text() for p in doc))
    log=(OUT/'main.log').read_text()
    qa={'pages':len(doc),'overfull':[l for l in log.splitlines() if 'Overfull' in l], 'undefined_references':'undefined references' in log,'visual_review':'pending'}
    (OUT/'qa/verification.json').write_text(json.dumps(qa,indent=2))
    print(json.dumps(qa,indent=2))


if __name__=='__main__': build()
