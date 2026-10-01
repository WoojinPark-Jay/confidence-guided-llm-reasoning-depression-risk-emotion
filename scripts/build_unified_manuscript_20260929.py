"""Build a review manuscript from the final unified experiment and v6c outputs.

Preserves the previously reviewed literature and reference list. Historical
downstream results are replaced, not pooled with the final experiment.
"""
from pathlib import Path
import ast
import csv
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from manuscript_visual_style_20260929 import build_figures, restyle_source
from manuscript_exposition_20260929 import enrich_source

ROOT = Path('/Users/woojinpark/Documents/헬스케어 논문')
REPO = ROOT / 'repo_final_push_nolfs'
BASE = ROOT / 'Paper_20260912_focused'
OUT = ROOT / 'Paper_20260929_unified_v6c'
P1 = Path('/Users/woojinpark/Downloads/distilbert_phase1_final_outputs (1)')
NBROOT = next(Path('/Users/woojinpark/Downloads').glob('SELF_DISCOVER_v6c*'))
OUT.mkdir(exist_ok=True)
for sub in ['figures', 'prompts', 'evidence', 'qa']:
    (OUT / sub).mkdir(exist_ok=True)
for p in BASE.iterdir():
    if p.suffix in {'.cls', '.sty', '.fd', '.map', '.pfb', '.tfm', '.bib', '.png'}:
        shutil.copy2(p, OUT / p.name)
old = (BASE / 'main.tex').read_text()

def between(a, b):
    return old[old.index(a):old.index(b)]

def literals(path):
    result = {}
    nb = json.loads(path.read_text())
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        result[target.id] = value
    return result

sdfile = next(NBROOT.glob('*llama3_reddit_output.ipynb'))
sd = literals(sdfile)
nbdir = REPO / 'notebooks/colab/final'
directfile = next(nbdir.glob('05_2*.ipynb'))
baseprompts = literals(directfile)
cot2file = next(nbdir.glob('13_reddit*.ipynb'))
cot2 = literals(cot2file)
assert sd['CLASSIFICATION_POLICY'].strip() == baseprompts['CLASSIFICATION_POLICY'].strip()
assert [s.strip() for s in cot2['LLAMA2_COT_REQUESTS']] == [s.strip() for s in baseprompts['COT_REQUESTS']]

prompt_sources = {}
def prompt(name, text, source):
    # Values are extracted from notebook string literals, not paraphrased.
    (OUT / 'prompts' / (name + '.txt')).write_text(text.strip() + '\n')
    prompt_sources[name] = {'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}

for name, key in [('policy', 'CLASSIFICATION_POLICY'), ('sd_system', 'ANNOTATOR_SYSTEM_PROMPT'),
                  ('sd_modules', 'EMOTION_REASONING_MODULES'), ('sd_task', 'TASK_LEVEL_DESCRIPTION'),
                  ('sd_select', 'select_prompt'), ('sd_adapt', 'adapt_prompt'),
                  ('sd_implement', 'IMPLEMENT_COMPACT_PROMPT'), ('sd_execute', 'V6_TEMPLATE')]:
    prompt(name, sd[key], sdfile)
prompt('direct', baseprompts['DIRECT_TEMPLATE'], directfile)
for i, text in enumerate(baseprompts['COT_REQUESTS']):
    prompt(f'cot_{i}', text, directfile)
prompt('cot_text', "Text:\n{phase2_original_text}", directfile)
for model in ['llama2', 'llama3']:
    path = next(NBROOT.glob(f'*{model}_mixed_emotion_output.ipynb'))
    notebook = json.loads(path.read_text())
    plans = []
    for cell in notebook['cells']:
        for output in cell.get('outputs', []):
            text = ''.join(output.get('text', []))
            marker = '--- Compact reasoning structure (v6c) ---'
            if marker in text:
                plans.append(text.split(marker, 1)[1].split('--- end ---', 1)[0].strip())
    assert len(plans) == 1 and len(plans[0]) < 2000
    prompt(f'sd_plan_{model}', plans[0], path)
(OUT / 'evidence/prompt_sources.json').write_text(json.dumps(prompt_sources, indent=2))

# Aggregate counts retain the user's reviewed Direct/CoT results and the colleague's
# v6c output. They do not pretend to be a new row-level audit of v6c predictions.
rows = [
 ('Reddit','Llama 2','Direct',96.9898,45,36),
 ('Reddit','Llama 2','CoT',96.9240,48,47),
 ('Reddit','Llama 2','SELF-DISCOVER',96.9391,52,49),
 ('Reddit','Llama 3','Direct',97.1056,68,45),
 ('Reddit','Llama 3','CoT',97.1471,63,35),
 ('Reddit','Llama 3','SELF-DISCOVER',97.1990,72,38),
 ('Mixed Emotion','Llama 2','Direct',75.1627,11,33),
 ('Mixed Emotion','Llama 2','CoT',78.9298,19,33),
 ('Mixed Emotion','Llama 2','SELF-DISCOVER',89.3452,22,5),
 ('Mixed Emotion','Llama 3','Direct',89.3803,26,9),
 ('Mixed Emotion','Llama 3','CoT',84.6760,15,12),
 ('Mixed Emotion','Llama 3','SELF-DISCOVER',92.6966,28,1),
]
df = pd.DataFrame(rows, columns=['dataset','model','method','macro_f1_percent','corrected','introduced'])
df['N'] = df.dataset.map({'Reddit':12000,'Mixed Emotion':300})
df['baseline_correct'] = df.dataset.map({'Reddit':11630,'Mixed Emotion':251})
df['net'] = df.corrected - df.introduced
df['accuracy_percent'] = 100 * (df.baseline_correct + df.net) / df.N
df['change_pp'] = 100 * df.net / df.N
df['exact_p'] = [binomtest(int(c), int(c+i), .5).pvalue for c,i in zip(df.corrected, df.introduced)]
order = np.argsort(df.exact_p.to_numpy())
adj = np.minimum(1, np.maximum.accumulate(df.exact_p.to_numpy()[order] * np.arange(12,0,-1)))
df['holm_p'] = 0.0
df.loc[order, 'holm_p'] = adj
rng = np.random.default_rng(20260929)
for idx, r in df.iterrows():
    # Equivalent to resampling the paired correctness differences (+1,-1,0).
    b = rng.multinomial(int(r.N), [r.corrected/r.N, r.introduced/r.N, 1-(r.corrected+r.introduced)/r.N], size=50000)
    ci = np.percentile(100*(b[:,0]-b[:,1])/r.N, [2.5,97.5])
    df.loc[idx,'ci_low_pp'], df.loc[idx,'ci_high_pp'] = ci
df.to_csv(OUT / 'evidence/final_results_and_count_statistics.csv', index=False)

cal = pd.read_csv(P1/'advanced_confidence_threshold_analysis/calibration_metric_summary.csv')
selected = pd.read_csv(P1/'phase1_selected_threshold.csv').iloc[0]
summary = json.loads((P1/'distilbert_phase1_summary.json').read_text())
for name in ['best_hyperparameters.json','distilbert_phase1_summary.json','phase1_selected_threshold.csv']:
    shutil.copy2(P1/name, OUT/'evidence'/name)
shutil.copy2(P1/'phase1_run_environment.json', OUT/'evidence/phase1_run_environment.json')
for name in ['calibration_metric_summary.csv','threshold_provenance.json','risk_coverage_curve_calibration_points.csv',
             'calibration_reliability_raw_msp.csv','calibration_reliability_temperature_scaled_msp.csv']:
    shutil.copy2(P1/'advanced_confidence_threshold_analysis'/name, OUT/'evidence'/name)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42, 'ps.fonttype':42})
colors = ['#557A95','#AB655C','#378376']
def savefig(name):
    plt.savefig(OUT/'figures'/f'{name}.pdf', bbox_inches='tight')
    plt.savefig(OUT/'figures'/f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close()

phase1_metrics = {}
fig, axs = plt.subplots(1, 2, figsize=(9, 3.7))
for ax, dataset, filename, expected in zip(
    axs, ['Reddit', 'Mixed Emotion'], ['phase1_test_predictions.csv','phase1_mixed_emotion_predictions.csv'],
    [(12000,370,218,98),(300,49,86,28)]):
    pred = pd.read_csv(P1/filename)
    wrong = pred.label_str != pred.phase1_label
    assert (len(pred), int(wrong.sum()), int(pred.phase1_routed.sum()), int((wrong & pred.phase1_routed).sum())) == expected
    labels = ['Depression','Neutral','Happy']
    matrix = confusion_matrix(pred.label_str, pred.phase1_label, labels=labels)
    phase1_metrics[dataset] = {'accuracy':100*accuracy_score(pred.label_str,pred.phase1_label),
        'macro_f1':100*f1_score(pred.label_str,pred.phase1_label,average='macro'), 'confusion_matrix':matrix.tolist()}
    ax.imshow(matrix, cmap='Blues')
    for a in range(3):
        for b in range(3):
            ax.text(b,a,str(matrix[a,b]),ha='center',va='center',color='white' if matrix[a,b]>matrix.max()/2 else 'black')
    ax.set_xticks(range(3),labels,fontsize=9);ax.set_yticks(range(3),labels,fontsize=9)
    ax.set(xlabel='Predicted label',ylabel='Reference label',title=dataset)
fig.tight_layout();savefig('phase1_confusion')
(OUT/'evidence/phase1_verified_metrics.json').write_text(json.dumps(phase1_metrics,indent=2))

fig,axs=plt.subplots(1,2,figsize=(11,3.5))
for ax,data,base in zip(axs,['Reddit','Mixed Emotion'],[96.9166667,83.6666667]):
    part=df[df.dataset==data]
    for j,method in enumerate(['Direct','CoT','SELF-DISCOVER']):
        vals=part[part.method==method].accuracy_percent.to_numpy()
        bars=ax.bar(np.arange(2)+(j-1)*.24, vals, .23, color=colors[j],label=method)
        ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
    ax.axhline(base,color='#444444',ls='--',label='Phase 1')
    ax.set_xticks([0,1],['Llama 2','Llama 3']);ax.set_title(data)
    ax.set_ylim((96.75,97.35) if data=='Reddit' else (70,97)); ax.set_ylabel('End-to-end accuracy (%)')
    ax.spines[['top','right']].set_visible(False)
axs[0].legend(loc='lower left',fontsize=8,ncol=2)
fig.tight_layout();savefig('end_to_end')
fig,axs=plt.subplots(1,2,figsize=(11,3.6))
for ax,data in zip(axs,['Reddit','Mixed Emotion']):
    part=df[df.dataset==data];x=np.arange(6)
    ax.bar(x-.18,part.corrected,.36,label='Corrected',color='#378376')
    ax.bar(x+.18,part.introduced,.36,label='Introduced',color='#AB655C')
    ax.set_xticks(x,['L2\nDirect','L2\nCoT','L2\nSD','L3\nDirect','L3\nCoT','L3\nSD']);ax.set_title(data)
    ax.set_ylabel('Posts');ax.spines[['top','right']].set_visible(False)
axs[0].legend();fig.tight_layout();savefig('corrections')

fig,axs=plt.subplots(1,2,figsize=(10,3.6))
for fname,label,color in [('raw_msp','Raw MSP',colors[1]),('temperature_scaled_msp','Temperature-scaled MSP',colors[2])]:
    d=pd.read_csv(P1/f'advanced_confidence_threshold_analysis/calibration_reliability_{fname}.csv')
    xcol=next(c for c in d if 'confidence' in c and 'mean' in c)
    ycol=next(c for c in d if 'accuracy' in c)
    axs[0].plot(d[xcol],d[ycol],'o-',label=label,color=color,ms=4)
axs[0].plot([0,1],[0,1],ls='--',color='gray');axs[0].set(xlabel='Mean confidence',ylabel='Accuracy',title='Calibration split reliability');axs[0].legend(fontsize=8)
d=pd.read_csv(P1/'advanced_confidence_threshold_analysis/risk_coverage_curve_calibration_points.csv')
xc=next(c for c in d if c=='coverage');yc=next(c for c in d if c in ['selective_risk','risk'])
axs[1].plot(d[xc]*100,d[yc]*100,color=colors[0]);axs[1].scatter([selected.coverage*100],[selected.selective_risk*100],color=colors[1],zorder=3,label='Selected threshold 0.70');axs[1].set(xlabel='Coverage (%)',ylabel='Selective risk (%)',title='Calibration risk--coverage');axs[1].legend(fontsize=8)
fig.tight_layout();savefig('calibration')

# A replacement vector diagram avoids retaining the obsolete two-condition figure.
fig,ax=plt.subplots(figsize=(12,4.5));ax.set(xlim=(0,12),ylim=(0,4.5));ax.axis('off')
def box(x,y,w,h,text,color='#E5EEF3'):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='#4E626C',lw=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(a,b,label=None):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=12,color='#46525B'))
    if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=9)
box(.1,2.6,2.3,1.5,'Training + validation\nSelected DistilBERT\nconfiguration\nThree-seed comparison')
box(3,2.6,2.6,1.5,'Calibration split\nFit T; select threshold\nT = 1.4802\nThreshold = 0.70')
box(.1,.4,2.3,1.5,'Evaluation inputs\nReddit: 12,000\nMixed Emotion: 300')
box(3,.4,2.6,1.5,'Final DistilBERT\nCalibrated confidence\nAccept or route')
box(6.2,.4,2.8,1.5,'Routed posts\nLlama 2 or Llama 3\nDirect / CoT / SD\nSeparate configurations')
box(9.6,.4,2.3,1.5,'Recombined outputs\nAccuracy / Macro F1\nCorrected / Introduced')
box(6.2,2.6,5.7,1.5,'Final SELF-DISCOVER\nTask-level SELECT → ADAPT → compact IMPLEMENT\nCached plan → evidence-based execution per post', '#E6EFE9')
arrow((2.4,3.35),(3,3.35));arrow((2.4,2.7),(3,1.8));arrow((4.3,2.6),(4.3,1.9));arrow((2.4,1.15),(3,1.15));arrow((5.6,1.15),(6.2,1.15),'route');arrow((9,1.15),(9.6,1.15));arrow((7.6,2.6),(7.6,1.9));
ax.plot([5.4,5.8,10.75,10.75],[1.9,2.2,2.2,1.9],color='#46525B',lw=1);ax.text(8.8,2.26,'accept: retain Phase 1 label',fontsize=9,ha='center')
fig.tight_layout();savefig('architecture_unified')

def table(caption,label,header,body,widths=None):
    fmt=widths or ('l'+'r'*(len(header)-1))
    return '\n'.join([r'\begin{table*}[t]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.14}',
                      '\\caption{'+caption+'}', '\\label{'+label+'}', '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}'+fmt+'@{}}',r'\toprule',
                      ' & '.join(header)+r' \\',r'\midrule',*[' & '.join(map(str,r))+r' \\' for r in body],
                      r'\bottomrule',r'\end{tabular*}',r'\end{table*}',''])
def fig(name,caption,label):
    return '\n'+r'\begin{figure*}[t]\centering'+'\n'+r'\includegraphics[width=\textwidth]{figures/'+name+'.pdf}\n'+r'\caption{'+caption+'}\n'+r'\label{'+label+'}\n'+r'\end{figure*}'+'\n'

preamble=old[:old.index(r'\begin{abstract}')].replace(r'\usepackage{balance}',r'\usepackage{balance}'+'\n'+r'\usepackage{listings}')
preamble=preamble.replace(r'\begin{document}',r'\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible,keepspaces=true,showstringspaces=false,frame=single,rulecolor=\color{gray},aboveskip=6pt,belowskip=8pt}'+'\n'+r'\begin{document}')
preamble=preamble.replace(r'\begin{document}',r'\def\thevol{XX}\def\theyear{2026}\clubpenalty=10000\widowpenalty=10000'+'\n'+r'\begin{document}')
abstract=r'''\begin{abstract}
Confidence-based routing can concentrate classifier errors, but improvement also depends on how the selected inputs are re-evaluated. We study a two-phase pipeline for three-class proxy emotion classification of social-media posts. A sweep-selected DistilBERT configuration supplies both the final classifier and its calibrated routing outputs; high-confidence predictions are retained, and low-confidence posts are re-evaluated using their original text. We compare Direct, Chain-of-Thought (CoT), and an adapted task-level SELF-DISCOVER protocol with Llama~2 and Llama~3. On 12,000 Reddit posts, Phase~1 achieves 96.9167\% accuracy and routes 218 posts (1.82\%). Llama~3 SELF-DISCOVER achieves 97.2000\%, correcting 72 errors while introducing 38. On a balanced 300-example synthetic Mixed Emotion stress test, the same routing policy selects 86 posts; accuracy increases from 83.6667\% to 92.6667\%, with 28 corrected errors and one introduced error. Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 among the six final configurations on both sets, although method rankings differ for Llama~2. The results distinguish error selection from correction and support structured, evidence-oriented re-evaluation within the evaluated settings. Prompt-configuration comparisons are exploratory, and the synthetic stress test does not establish clinical or external-domain validity.
\end{abstract}
'''
front=between(r'\begin{keywords}',r'\section{Literature Review}')
front=front.replace('An additional 9,000-post same-source holdout tests the frozen pipeline beyond the sample used during prompt development.', 'The final comparison crosses two language models with three re-evaluation protocols on the same routed examples within each dataset.')
front=front.replace('The contribution is the integration and evaluation protocol, not a new calibration estimator, language-model architecture, or reasoning method.', 'The contribution is the integration and evaluation protocol, together with a task-specific adaptation of structured re-evaluation, not a new calibration estimator or language-model architecture.')
litdata=between(r'\section{Literature Review}',r'\section{Methodology}')
litdata=litdata.replace('as disclosed in the experimental limitations', 'so final prompt-configuration comparisons are interpreted as exploratory')

method_intro=r'''\section{Methodology}
\subsection{Unified Classifier-to-Re-Evaluation Pipeline}
The final DistilBERT configuration is selected through the Phase~1 validation-based search. A final training run under that configuration supplies the predictions, logits, calibration temperature, and routed IDs used throughout the downstream experiment. No separately configured operational classifier is substituted after the classifier comparison. Three-seed classifier summaries describe training variability; downstream results describe this single linked final run, not an average of three reasoning pipelines.

At inference, accepted predictions retain their Phase~1 labels. Every routed post is independently evaluated under each of six configurations: Llama~2-7B-Chat or Llama~3-8B-Instruct, each with Direct, CoT, or final SELF-DISCOVER prompting. These are separate alternatives, not an ensemble or a dataset-specific selection rule. All configurations within a dataset use the same Phase~1 outputs and routed IDs. Figure~\ref{fig:architecture} retains the original overview layout; the final configuration details are specified in the text.
'''
for suffix in ['pdf','drawio']:
    shutil.copy2(BASE/'figures'/f'architecture_figure_publication_final.{suffix}', OUT/'figures'/f'architecture_figure_publication_final.{suffix}')
method_intro+=fig('architecture_figure_publication_final','Confidence-guided two-phase pipeline. Accepted predictions retain the Phase 1 label, while routed posts enter re-evaluation. Original artwork is retained in this review copy pending label updates; the final experiment evaluates both models with Direct, CoT, and SELF-DISCOVER as specified in the text.','fig:architecture')
models=between(r'\subsection{Phase 1: Initial Emotion Classification}',r'\subsection{Calibrated Confidence-Based Routing')
models=models.replace('only the fixed operational DistilBERT checkpoint supplies downstream routing scores','the final DistilBERT run under the selected configuration supplies downstream routing scores')
calmethod=between(r'\subsection{Calibrated Confidence-Based Routing',r'\subsection{Phase 2: Reasoning-Based')
calmethod=calmethod.replace('11,830','11,807').replace('Phase~2 reasoner $g$ (Llama~2 CoT or Llama~3 SELF-DISCOVER)','Phase~2 configuration $g$ (model and prompting protocol)')
calmethod=calmethod.replace(r'\State \textbf{return} the parsed Phase~2 final label',r'\State If parsing fails, retain $\hat{y}^{(1)}$ and record failure'+'\n'+r'    \State \textbf{return} the parsed label or recorded fallback')
calmethod=calmethod.replace(r'if $r_i=1$; otherwise',r'if $r_i=1$ and parsing succeeds; otherwise')
calmethod=calmethod.replace(r'Generate rationale and terminal label with $g(x^{(2)},\hat{y}^{(1)})$', 'Generate a label with $g$ using its prescribed inputs')
calmethod=calmethod.replace('If none remain, report infeasibility and stop', 'If none remain, flag infeasibility; a fallback is not risk-feasible')
fallback_start = calmethod.index('The implementation records an explicit infeasibility status')
fallback_end = calmethod.index(r'\subsection{Relationship to Mixed-Emotion Evaluation}', fallback_start)
calmethod = calmethod[:fallback_start] + r'''The export notebook prints a warning and falls back to the lowest empirical risk, then highest coverage, if no candidate satisfies the constraint. That branch was not needed here: the selected calibration point satisfies the prescribed upper-bound criterion. A fallback, if invoked, would not be interpreted as a risk-feasible policy. The temperature, threshold, risk target, confidence level, and candidate grid are stored with the calibration and threshold tables. The model source and seed are recorded separately in the execution environment artifact.

''' + calmethod[fallback_end:]
reasoning=r'''\subsection{Phase 2: Factorial Model and Protocol Comparison}
Direct prompting requests one final label without a rationale and includes the Phase~1 prediction. CoT first requests an independent assessment, subsequently discloses the Phase~1 label for comparison, and finally requests a terminal label. The implemented CoT dialogue contains five generation calls, including the initial acknowledgment and text-response turns. Final SELF-DISCOVER withholds the Phase~1 label throughout. These are complete prompting protocols: their comparison does not isolate rationale length alone, because label disclosure and call structure also differ.

\subsection{Task-Level SELF-DISCOVER}
The final protocol adapts SELECT, ADAPT, and IMPLEMENT \cite{ref18} to the three-class emotion task. SELECT chooses from 18 domain-oriented reasoning modules. ADAPT specializes the selected modules to the classification policy. IMPLEMENT requests a JSON plan with at most three one-line steps, requiring each step to return to the original post and delaying label selection until the final step. The compact-plan constraint is an instruction, not a programmatic validator; generated plans can depart from it. Each model uses its own discovered plan, which is cached and reused across posts in a run.

The execution prompt combines the plan, class policy, and original post. It asks the model to identify whose emotional state is expressed and whether that state is current or retrospectively resolved; cite supporting and opposing text for all three labels; and consider what evidence would favor a second-best label. One final line supplies the classification. This procedure is intended to reduce premature commitment and unsupported reinterpretation, but generated text does not establish faithful internal reasoning or clinical validity. The full literal prompt templates are reproduced in the appendices.

\begin{algorithm}[t]
\caption{Final task-level SELF-DISCOVER re-evaluation}
\label{alg:sd-final}\footnotesize
\begin{algorithmic}[1]
\State \textbf{Input:} model $g$, policy, domain modules, routed posts
\State SELECT relevant modules for the task
\State ADAPT selected modules to the classification policy
\State IMPLEMENT a compact task-level plan; cache the output
\For{each routed post $x$}
\State Execute the plan on the original post, without the Phase~1 label
\State Request source quotes, three-label evidence, and a counterfactual check
\State Parse the final label; retain Phase~1 if parsing fails
\State Store the response, parsed status, and final prediction
\EndFor
\end{algorithmic}\end{algorithm}

\subsection{Input and Output Policy}
Reddit Phase~1 uses cleaned title--body text; Phase~2 uses the linked, minimally sanitized original title and selftext. The original-text policy preserves punctuation, negation, and discourse structure but changes the input representation as well as the prediction mechanism. Improvements are therefore attributed to the complete re-evaluation configuration, not solely to reasoning. Long posts follow the notebook's 6,000-character input policy. Mixed Emotion uses the complete synthetic text rather than a separately lemmatized version.

The three permitted outputs are Depression, Neutral, and Happy. The parser prioritizes the terminal label marker and records unparseable responses. Such responses retain the Phase~1 prediction in the end-to-end evaluation; parsing failure is not automatically counted as a classification error. The remaining final SELF-DISCOVER failures are not regenerated or assigned labels from the reference answer. Label normalization and any recovery of an explicit answer must be applied without consulting the reference label.
'''
setup=r'''\section{Experimental Setup}
\subsection{Classifier Selection and Calibration}
All classifier configurations use the common 84,000/12,000/12,000/12,000 train/validation/calibration/test partitions, a maximum length of 256 tokens, and a four-trial Bayesian search maximizing validation macro F1. The selected DistilBERT settings are learning rate $4.0368\times10^{-5}$, batch size 64, an epoch budget of 3, and weight decay $10^{-3}$. The epoch budget is a selected training hyperparameter, not a statement that every seed's best validation checkpoint occurred at that epoch. Seeds 42, 43, and 44 characterize classifier variation; the final downstream run uses seed 42. Appendix Table~A1c records the selected settings for all three classifiers.

The downstream export loads the saved reference seed-42 model from this selected configuration; it does not perform a new training run inside the export notebook. Its environment record identifies fp32 evaluation, a 256-token maximum input length, and the same selected hyperparameters. The final model is calibrated with $T^*=1.480195$ and routed at $\tau^*=0.70$. The calibration partition has 11,807 accepted and 193 routed posts, accepted-set empirical risk 1.9395\%, and one-sided upper bound 2.1615\%, below the 5\% target. These calibration counts are distinct from the 218 routed Reddit test posts. No Mixed Emotion labels are used for temperature fitting or threshold selection.

\subsection{Execution and Resource Scope}
The retained Phase~1 benchmark compares the selected classifier configurations on NVIDIA A100-SXM4-40GB hardware, with fp16 inference, a 256-token cap, and batch size 32. Three timing repetitions are distinct from the three training seeds. These classifier-stage measurements are not measurements of the complete final reasoning pipeline; Appendix Table~A1d retains their original measurement scope.

Phase~2 uses chat-tuned Llama~2~7B and instruction-tuned Llama~3~8B with quantized loading and greedy decoding. Direct and final CoT request up to 1,024 new tokens per call. Final SELF-DISCOVER requests up to 1,024 for Llama~2 and 1,536 for Llama~3 in its per-post execution call. Thus, neither total generation cost nor token budget is claimed to be equal across all protocols. Llama~2's shorter context also constrains multi-turn inference. Template text and nominal execution budgets are disclosed in the appendices; wall-clock or monetary superiority of the final Phase~2 configurations is not inferred from accuracy.

\subsection{Evaluation and Statistical Scope}
Accuracy and macro F1 use all examples, after recombining accepted and routed predictions. Corrected errors $C$ are Phase~1 mistakes made correct by re-evaluation; introduced errors $I$ are previously correct predictions made wrong. The full-set accuracy change in percentage points is $100(C-I)/N$. Wrong-to-wrong changes are not corrections. Routed accuracy, error capture, and accepted-set risk measure different aspects of the pipeline.

Appendix~\ref{app:statistics} reports count-based paired accuracy summaries. For a comparison with Phase~1, the discordant pairs are exactly $C$ and $I$, so the two-sided exact McNemar test is a binomial test with $C+I$ trials and probability one half. Paired percentile-bootstrap intervals for accuracy change are reconstructed from the three correctness-difference categories $+1,-1,0$, with counts $C,I,N-C-I$, using 50,000 multinomial resamples. This is equivalent to a paired row bootstrap for that scalar accuracy difference. It does not reconstruct class-wise metrics or paired comparisons between two re-evaluators. Holm correction is applied across the 12 final configuration-versus-Phase~1 tests. These intervals and tests describe the reported final counts; prompt development and selection make them exploratory rather than selection-adjusted confirmation of the winning prompt.
'''

phase1results=between(r'\subsection{Phase 1 Performance on Reddit Data}', 'The matched comparison and the downstream experiment serve different purposes.')
shutil.copy2(BASE/'figures/figure_e_phase1_model_comparison.pdf',OUT/'figures/figure_e_phase1_model_comparison.pdf')
phase1results+=r'''
The final linked DistilBERT run achieves 96.9167\% accuracy and 96.9166\% macro F1 on Reddit. This run uses the same selected hyperparameter configuration as the three-seed comparison. Its calibration and routing outputs, rather than those of a separately configured model, define every downstream condition.
\subsection{Calibration and Routing}
Temperature scaling reduces all four calibration diagnostics (Table~\ref{tab:calibration-final}). The fixed test router selects 218 of 12,000 Reddit posts and 86 of 300 Mixed Emotion examples. The difference in routing volume is an observed response to the stress-test inputs, not a dataset-specific threshold adjustment.
'''
calbody=[]
for _,r in cal[cal.split=='calibration'].iterrows():
    calbody.append(['Raw MSP' if r.confidence_method=='raw_msp' else 'Temperature-scaled MSP',f'{r.temperature:.4f}',f'{r.nll:.4f}',f'{r.brier_score:.4f}',f'{r.ece:.4f}',f'{r.adaptive_ece:.4f}'])
phase1results+=table('Final DistilBERT calibration diagnostics (calibration split)','tab:calibration-final',['Score','Temperature','NLL','Brier','ECE','Adaptive ECE'],calbody)
phase1results+=fig('calibration','Calibration diagnostics for the final linked DistilBERT model. Reliability and risk--coverage curves use the calibration split; the selected point is not selected on test outcomes.','fig:calibration-final')
routebody=[]
for data,N,E,n,ER in [('Reddit',12000,370,218,98),('Mixed Emotion',300,49,86,28)]:
    routebody.append([data,str(N),str(n),f'{100*n/N:.2f}',f'{100*(1-n/N):.2f}',f'{100*ER/E:.2f}',f'{100*ER/n:.2f}',f'{100*(E-ER)/(N-n):.2f}'])
phase1results+=table('Fixed-policy routing outcomes; rates are percentages','tab:routing-final',['Dataset','$N$','Routed','Route rate','Coverage','Error capture','Routed error','Accepted risk'],routebody)
phase1results+=r'''
On Reddit, the routed subset contains 98 of the 370 Phase~1 errors (26.49\% error capture). Its error rate is 44.95\%, compared with 3.08\% over the full set, an approximately 14.58-fold enrichment. The accepted subset still contains 272 errors, beyond the scope of routed-only correction. On Mixed Emotion, routing captures 28 of 49 errors (57.14\%); 21 errors remain accepted. Routed-set Phase~1 accuracy is 55.05\% on Reddit and 67.44\% on Mixed Emotion.

\subsection{End-to-End Model and Prompting Results}
Table~\ref{tab:final-e2e} reports the complete final comparison. Llama~3 SELF-DISCOVER has the highest observed accuracy and macro F1 among these six configurations on both datasets. The advantage over Llama~3 CoT on Reddit is six correct predictions (0.0500 percentage points); numerical ordering alone does not establish a significant between-method difference.
'''
resultbody=[]
last_dataset = None
for _,r in df.iterrows():
    if r.dataset != last_dataset:
        metrics = phase1_metrics[r.dataset]
        resultbody.append([r.dataset,'DistilBERT','Phase 1',f"{metrics['accuracy']:.4f}",f"{metrics['macro_f1']:.4f}",'--','--','--','--'])
        last_dataset = r.dataset
    resultbody.append([r.dataset,r.model,r.method,f'{r.accuracy_percent:.4f}',f'{r.macro_f1_percent:.4f}',str(r.corrected),str(r.introduced),f'{r.net:+d}',f'{r.change_pp:+.4f}'])
phase1results+=table('Final full-set outcomes. Accuracy and macro F1 are percentages; change is relative to Phase 1 in percentage points.','tab:final-e2e',['Dataset','Model','Protocol','Accuracy','Macro F1','$C$','$I$','$C-I$','Change'],resultbody,'lllrrrrrr')
phase1results+=fig('end_to_end','Full-set accuracy for the final six configurations. Dashed lines mark the linked Phase 1 baselines. Axes are truncated; the two panels have different scales. SD denotes the final task-level compact-plan SELF-DISCOVER protocol.','fig:final-effect')
phase1results+=r'''
\subsection{Correction and Harm}
On Reddit, Llama~3 SELF-DISCOVER corrects 72 errors and introduces 38, producing 34 net corrections. Direct and CoT yield 23 and 28 net corrections, respectively. Llama~2 yields smaller gains: Direct has nine net corrections, CoT one, and SELF-DISCOVER three. Thus, SELF-DISCOVER does not outperform Direct within every model and dataset combination.

On Mixed Emotion, Llama~3 SELF-DISCOVER corrects 28 errors and introduces one. Total errors decrease from 49 to 22, a 55.10\% relative reduction, and accuracy increases by 9.00 percentage points. This low introduced-error count is an observed property of the 300-example stress test, not a population safety guarantee. Llama~2 SELF-DISCOVER also improves the baseline (17 net corrections), whereas its Direct and CoT configurations introduce more errors than they correct. The pattern supports evaluating both correction yield and damage to initially correct predictions.
'''
phase1results+=fig('corrections','Corrected and introduced errors for each final configuration. Net improvement requires corrections to exceed newly introduced errors. The two panels use different count scales.','fig:final-errors')
phase1results+=r'''
\subsection{Conditional Correction Opportunity}
With the router held fixed, a conditional oracle corrects every routed error without changing any initially correct prediction. Its full-set ceiling is $\mathrm{Acc}_{P1}+E_R/N$, where $E_R$ is the number of routed Phase~1 errors. The ceiling is 97.7333\% on Reddit and 93.0000\% on Mixed Emotion. Final Llama~3 SELF-DISCOVER realizes $34/98=34.69\%$ and $27/28=96.43\%$ of these respective net-correction opportunities. These are descriptive ceilings under a fixed router, not achievable performance guarantees or alternative tuned baselines.
'''
discussion=r'''\section{Discussion}
The linked experiment separates three decisions: selecting a practical classifier, routing uncertain posts, and choosing a re-evaluation protocol. DistilBERT retains strong three-seed predictive performance and a measured classifier-stage inference advantage under the bounded Phase~1 comparison. The final run then supplies one consistent set of predictions and confidence scores for every Phase~2 condition. Classifier means and downstream single-run scores answer different questions without requiring different hyperparameter configurations.

Confidence routing identifies an error-enriched subset, but enrichment alone is insufficient. Llama~2 CoT is nearly neutral on Reddit and harmful on Mixed Emotion, whereas final SELF-DISCOVER with Llama~3 improves both sets. On the stress test, searching for evidence for and against all three classes and distinguishing present distress from resolved or attributed events are plausible contributors to the observed pattern. The experiment evaluates the complete prompt package; it does not establish the separate causal effect of each instruction or demonstrate that the generated rationale faithfully reveals the model's computation.

The final model-by-protocol comparison is more informative than contrasting different models with different prompts only. Nevertheless, different context windows, generation budgets, and exposure to the Phase~1 label remain part of the evaluated configurations. Llama~3's advantage is therefore a configuration-level finding, not an isolated causal estimate of model generation or parameter count. Likewise, Direct is a label-only re-evaluation baseline: its benefit does not prove that an explicit reasoning trace is necessary.

\section{Limitations and Future Work}
Subreddit-derived, sentiment-filtered labels are proxy constructs and can reward topical or community-specific shortcuts. Random post-level splits do not establish author-disjoint or temporal independence. The Mixed Emotion data are synthetic, scenario-balanced, and not independently expert-labeled; their results should not be generalized to naturally occurring emotional ambiguity or clinical depression.

Prompt variants were developed and compared using these evaluation cases. Consequently, the selected prompt's observed ranking and count-based significance summaries remain exploratory. A frozen final configuration should be assessed on unused data to estimate generalization after selection. Earlier evaluations of a different downstream configuration are not used as independent validation of the final pipeline in this manuscript.

Reddit re-evaluation uses richer original text than the cleaned Phase~1 input. This is a deliberate system design but prevents attributing the entire gain to prompting alone. Residual parsing failures retain Phase~1 predictions; their presence and the rule must be considered when comparing protocols. The final SELF-DISCOVER summaries contain two such failures on Reddit and one on Mixed Emotion for Llama~3. No unresolved failure is manually relabeled from its reference answer.

The current experiments do not quantify repeated-generation variation for all final configurations or supply a fully matched end-to-end cost benchmark. Greedy decoding does not guarantee bitwise equality across hardware and software environments. Independent expert review, broader evaluation, and generation-cost accounting are useful extensions, rather than reasons to repeat the completed Phase~1 model comparison.

\section{Conclusion}
A unified DistilBERT-to-LLM pipeline allows classifier performance, routing concentration, and correction quality to be evaluated without substituting a separately configured downstream classifier. Among the final Direct, CoT, and SELF-DISCOVER configurations for two language models, Llama~3 SELF-DISCOVER achieves the highest observed accuracy and macro F1 on both Reddit and the synthetic Mixed Emotion stress test. Its gains are 34 and 27 net corrections, respectively. The principal empirical finding is configuration-dependent correction: routing creates an opportunity, but the re-evaluation protocol determines whether the opportunity is used without introducing excessive new errors. These exploratory proxy-label results support further validation, not clinical deployment claims.
'''

refs=between(r'\begin{thebibliography}',r'\section{Supplementary Methodological Details}')
refs=refs.replace(r'\onecolumn',r'\onecolumn\raggedbottom')
app=between(r'\section{Supplementary Methodological Details}',r'\section{Chain-of-Thought Prompting Protocol}')
start=app.index('The downstream calibration and routing analyses retained')
end=app.index(r'\normalsize',start)
app=app[:start]+'The final downstream DistilBERT run uses the selected configuration in this table, with temperature 1.480195 and threshold 0.70. The single-run downstream score is not the three-seed mean.'+app[end:]
app=app.replace(r'\Needspace{39\baselineskip}',r'\Needspace{12\baselineskip}')
app=app.replace(r'\Needspace{24\baselineskip}',r'\Needspace{4\baselineskip}')
app+=r'''\section{Final Prompting Protocols}
The following templates are extracted from the executed notebook code. Placeholder fields are substituted at runtime. The shared policy is identical in the supplied final SD and Llama~3 baseline templates; the final Llama~2 CoT instruction sequence matches the Llama~3 sequence. Direct discloses the Phase~1 label immediately, CoT discloses it after an independent assessment, and final SD never discloses it. No reference label is inserted into any prompt.

\subsection{Shared Classification Policy}
\lstinputlisting{prompts/policy.txt}
\subsection{Direct Re-Evaluation}
One user message is formed by inserting the policy, original text, and Phase~1 label into the following template. No explicit rationale is requested.
\lstinputlisting{prompts/direct.txt}
\subsection{Chain-of-Thought Dialogue}
The following system message and user instructions are sent in order. Assistant responses are retained in the conversation. After the first user instruction the model generates an acknowledgment; after the text is provided it generates a text response; then independent analysis, comparison, and final-decision calls follow. There are five generation calls per routed post, not one.
\noindent\textbf{System message}
\lstinputlisting{prompts/cot_0.txt}
\noindent\textbf{Initial user instruction}
\lstinputlisting{prompts/cot_1.txt}
\noindent\textbf{Post input}
\lstinputlisting{prompts/cot_text.txt}
\noindent\textbf{Independent analysis}
\lstinputlisting{prompts/cot_2.txt}
\noindent\textbf{Comparison with Phase 1}
\lstinputlisting{prompts/cot_3.txt}
\noindent\textbf{Final decision}
\lstinputlisting{prompts/cot_4.txt}

\section{Final SELF-DISCOVER Templates}
This is the final compact task-level protocol (implementation identifier v6c). The identifier distinguishes execution artifacts; it is not an additional comparator in the main tables. SELECT, ADAPT, and IMPLEMENT are task-level discovery calls. Their output is cached and inserted into the per-post execution template. Each model discovers its own plan; the input policy and evidence-oriented execution instructions are shared. The requested maximum of three steps is not enforced through a validator or retry mechanism.
\subsection{Task Description and Module Bank}
The policy placeholder expands to the shared classification policy. The source template's reference to a title and body also remains in the Mixed Emotion run, whose input is the synthetic post text.
\lstinputlisting{prompts/sd_task.txt}
\lstinputlisting{prompts/sd_modules.txt}
\subsection{SELECT}
The module placeholder expands to the complete bank above; its spelling is retained from the code.
\lstinputlisting{prompts/sd_select.txt}
\subsection{ADAPT}
\lstinputlisting{prompts/sd_adapt.txt}
\subsection{Compact IMPLEMENT}
\lstinputlisting{prompts/sd_implement.txt}
\subsection{Per-Post Execution}
The system message below precedes the execution user message. The structure placeholder is the model-generated cached plan, not a manually supplied answer, and the text placeholder is the routed original post.
\lstinputlisting{prompts/sd_system.txt}
\lstinputlisting{prompts/sd_execute.txt}

\subsection{Realized Cached Plans}
The following plans are reproduced from the recorded notebook outputs, not rewritten to make them conform to the instructions. The Mixed Emotion runs explicitly reused the cached plans. Llama~3 produced three evidence-oriented steps plus a separate final step. Llama~2 produced four prose steps and included provisional label assignment inside early steps. These differences show why the requested plan constraints should not be reported as enforced behavior; they do not independently establish the cause of the performance difference.
\Needspace{10\baselineskip}
\noindent\textbf{Llama 2 generated plan}
\lstinputlisting{prompts/sd_plan_llama2.txt}
\Needspace{10\baselineskip}
\noindent\textbf{Llama 3 generated plan}
\lstinputlisting{prompts/sd_plan_llama3.txt}

\subsection{Generation Budget and Failure Handling}
Direct and CoT request 1,024 new tokens per generation call. Final SD execution requests 1,024 for Llama~2 and 1,536 for Llama~3. The SD discovery calls request up to 512 tokens for Llama~2 and 768 for Llama~3. The source uses a shared per-model cache for the task-level structures, allowing reuse across the Reddit and Mixed Emotion runs. The templates specify instructions, not guaranteed instruction compliance. Quantized model loading, model-specific chat formatting, context limits, and call structure are part of the evaluated implementation. A shorter plan instruction does not by itself prove lower measured cost.

Remaining final SD parsing failures are zero for Llama~2 on both sets, two for Llama~3 Reddit, and one for Llama~3 Mixed Emotion. These remain failures and use Phase~1 fallback. No additional model call is added to resolve them in the reported final SD results.
'''
app+=r'''\section{SELF-DISCOVER Design-Variant Comparison}
The earlier independent-countercheck variant generated an instance-specific plan using general reasoning modules; the final version uses a domain-oriented task-level plan and an explicit evidence ledger. Both withhold the Phase~1 label. Multiple components change jointly, so this is a design-variant comparison rather than a single-component causal ablation. The earlier variant is not used in the main six-configuration comparison.

\begin{center}\small
\begin{tabular}{llrrrr}\toprule
Dataset & Model & Earlier accuracy & Final accuracy & Earlier net & Final net \\\midrule
Reddit & Llama 2 & 97.0500\% & 96.9417\% & 16 & 3 \\
Reddit & Llama 3 & 97.1583\% & 97.2000\% & 29 & 34 \\
Mixed Emotion & Llama 2 & 87.6667\% & 89.3333\% & 12 & 17 \\
Mixed Emotion & Llama 3 & 90.3333\% & 92.6667\% & 20 & 27 \\\bottomrule
\end{tabular}\end{center}
The final design improves three of these four model--dataset combinations; it does not uniformly dominate the earlier design. Its Llama~3 configuration improves both datasets. The change in input-independent planning also changes the number and scope of discovery calls, so a cost comparison requires measured generation records rather than accuracy alone.

\section{Paired Accuracy Summaries}\label{app:statistics}
These calculations use the final reported corrected/introduced counts, with all unparseable SD responses left at the implemented Phase~1 fallback. For accuracy change, the paired correctness-difference counts are sufficient for the exact test and paired bootstrap. This does not constitute a new audit of all raw responses, nor a paired SD-versus-CoT significance test. The latter requires matched per-example predictions from both configurations. Intervals are percentile bootstrap intervals in percentage points. Holm adjustment covers all 12 rows below; it does not account for selection over previously explored prompts.

\begin{center}\footnotesize
\begin{tabular}{lllrrrr}\toprule
Dataset & Model & Protocol & Change (pp) & Paired 95\% CI & Exact $p$ & Holm $p$ \\\midrule
'''
def pval(p):
    return '$<0.0001$' if p<.0001 else f'{p:.4f}'
for _,r in df.iterrows():
    app+=' & '.join([r.dataset,r.model,r.method,f'{r.change_pp:+.4f}',f'[{r.ci_low_pp:.3f}, {r.ci_high_pp:.3f}]',pval(r.exact_p),pval(r.holm_p)])+r' \\'+'\n'
app+=r'\bottomrule\end{tabular}\end{center}'+'\n'
app+=r'''\Needspace{24\baselineskip}
\section{Final Phase 1 Confusion Matrices}
The matrices below are recomputed from the final Phase~1 prediction exports on all 12,000 Reddit and 300 Mixed Emotion examples. They describe the linked classifier, not the three-seed average. Final Phase~2 class-wise matrices are not reconstructed from aggregate corrected/introduced counts.
\begin{center}
\includegraphics[width=0.85\textwidth]{figures/phase1_confusion.pdf}
\end{center}
'''
synthetic=between(r'\section{Synthetic Mixed Emotion Dataset Generation Protocol}',r'\section{Additional Holdout Class-Wise Results}')
synthetic=synthetic.replace(r'\Needspace{19\baselineskip}','')
source=preamble+abstract+front+litdata+method_intro+models+calmethod+reasoning+setup+r'\section{Experimental Results}'+'\n'+phase1results+discussion+refs+app+synthetic+'\n'+r'\EOD'+'\n'+r'\end{document}'+'\n'
source=source.replace('two Phase~2 configurations','six Phase~2 configurations')
build_figures(OUT, P1, df, phase1_metrics, selected)
source=restyle_source(source)
source=enrich_source(source)
(OUT/'main.tex').write_text(source)

# Keep review work separate from the manuscript's empirical claims.
review='''# 통합 최종 실험 + SELF-DISCOVER v6c 논문 업데이트

## 핵심 변경 전후
| 영역 | 이전 원고 | 이번 원고 |
|---|---|---|
| Phase 1 연결 | 비교용 설정과 별도 운영 체크포인트 | 선택된 설정의 seed42 저장 모델을 불러와 끝까지 사용 |
| Temperature | 1.7706 | 1.480195 |
| Reddit | 96.69%, 171건 routing | 96.9167%, 218건 routing |
| Mixed Emotion | 81.33%, 44건 routing | 83.6667%, 86건 routing |
| 재평가 비교 | Llama2 CoT / Llama3 SD 중심 | 두 모델 각각 Direct / CoT / 최종 SD |
| SD 계획 | 게시글별 계획 생성 | 감정 모듈 기반 과제 수준 계획 생성 후 재사용 |
| 대표 최종 SD 결과 | Reddit 96.94%, Mixed 87.33% | Reddit 97.2000%, Mixed 92.6667% |
| 구형 추가평가 | 이전 모델의 9,000건 결과 포함 | 최종 모델의 증거로 섞지 않음; 원본 보존 |
| 통계 | 이전 결과의 검정 | 최신 C/I 기반 정확도 비교 재계산; 방법 간 검정은 별도 |

## 이번 반영 완료
- 비교용/운영용으로 나뉘던 DistilBERT 설명을 선택된 설정의 최종 실행에서 calibration, routing, 재평가까지 이어지는 구조로 교체.
- Phase 1 세 모델의 완료된 3시드 비교와 A100 효율성 자료 유지. Llama/Mistral 재학습은 요청하거나 실행하지 않음.
- 최종 DistilBERT 출력에서 T=1.480195, threshold=0.70, calibration 193건, Reddit test 218건, Mixed 86건으로 갱신.
- 최신 calibration 진단 표와 reliability/risk-coverage 그림 재생성.
- 모델 2개 × 방법 3개 × 데이터 2개 전체 결과 표, 정확도 그림, 수정/신규 오류 그림 작성.
- 최종 SD는 v6c만 본문에 사용. 이전에 사용자가 실행한 SD와의 비교는 Appendix 설계 변형 비교표로만 제시.
- 기존 개별 게시글 계획 생성에서 도메인 모듈 기반 공통 계획 생성/재사용으로 SD 알고리즘과 본문 설명 변경. 아키텍처 그림은 사용자 요청에 따라 기존 원본 PDF로 복원하고, 그림 내 문구 수정은 별도 가이드로 분리. 현재 그림의 구형 두 조건 표시는 아직 최종 여섯 조건으로 바뀌지 않았으므로 캡션에 검토용 원본임을 명시.
- Direct, CoT, SD 정책·모듈·SELECT·ADAPT·IMPLEMENT·실행 템플릿을 코드 문자열에서 추출해 부록에 수록.
- 노트북에 출력된 Llama2/Llama3의 실제 cached plan도 부록에 그대로 수록. 요구한 단계 수와 실제 생성 형태의 차이를 구분.
- 최종 Phase 1 행별 예측을 직접 재집계하고 두 데이터셋 혼동행렬을 다시 생성. Phase 2 혼동행렬은 원시 자료 확보 후 작성.
- SD 파싱 실패는 그대로 두고 기존 Phase 1 fallback 집계 유지. 정답을 보고 라벨을 넣지 않음.
- 최종 C/I 집계로 정확도 차이의 대응 부트스트랩과 exact McNemar를 계산. 12개 비교에 Holm 보정. 재평가 방법끼리의 paired 검정은 수행한 것으로 쓰지 않음.
- 이전 모델의 9,000건 추가평가, 이전 오류사례·혼동행렬·구형 유의성 결과는 현재 최종 실험 결과로 재사용하지 않음. 이전 파일은 삭제하지 않고 보존.
- 초록, 방법론, 실험설정, 결과, 논의, 한계, 결론을 새 결과에 맞춰 작성.

## 다음에 꼭 마무리할 것
1. 최종 v6c 원시 결과 CSV 네 개와 캐시된 계획 JSON, 실행 manifest 확보. 계획 본문은 노트북 출력에서 확보해 부록에 넣었고, 원본 JSON/manifest까지 재현 패키지에 묶어야 함. 현재 노트북 출력과 공동연구자 요약이 SD 수치 근거이며, 모든 행을 다시 확인한 상태는 아님.
2. 12조건의 최종 예측을 example_id로 정렬해 집계 수치를 한 번 더 대조. 잔여 실패는 fallback 그대로 유지. 기존 명시적 라벨 복구 내역과 v6c parser 규칙의 일관성 확인.
3. SD 대 CoT, SD 대 Direct의 같은 사례 대응 비교. 현재 통계표는 각 조건 대 Phase 1만 계산한 것임. 최종 confusion matrix와 클래스별 F1은 실제 행별 예측 확보 후 작성. 집계만으로 추정하지 않음.
4. 최종 SD discovery budget, 실제 캐시 계획, 모델 revision, context 처리, 환경을 manifest와 대조. 본문에 이미 L2 1024/L3 1536 상한 차이를 명시. 무조건 동일조건이라고 주장하지 않음.
5. seed42 실행 메타데이터에서 선택 설정과 저장 모델을 불러온 사실은 확인 완료. 모델 파일 해시까지 묶어 재현 패키지를 마무리할 것. export 노트북에서 새 학습을 했다고 쓰지 않음. 재학습이 필요한 단계가 아님.
6. 저자·소속·연락처·연구비 및 데이터/코드 공개 범위 확정. 현재 저자란은 검토용 placeholder 유지. footer의 권호 XX도 편집부 확정 전 placeholder이며, 이전 템플릿의 2023년은 2026년으로 변경.

## 추가하면 유용하지만 이번 수치 반영과 별개인 작업
- 현재 최종 모델/프롬프트를 고정한 미사용 데이터 평가. 이전 9,000건 결과를 새 모델의 독립검증처럼 가져오지 않음.
- Mixed Emotion과 대표 오류 사례에 대한 독립 연구자 검토.
- 최종 Phase 2의 실제 토큰, 호출 수, 실행시간 합산. Phase 1 효율성 실험은 이미 완료되어 유지함.

## 해석 기준
- v6c Llama3가 최종 여섯 조건 중 두 데이터에서 최고 관측 성능. Llama2 Reddit에서는 Direct가 SD보다 높음.
- Mixed Llama3 SD는 28개 수정, 1개 신규 오류, 27개 순수정. 남아 있는 파싱 실패 1개와 신규 오류 1개는 같은 개념이 아님.
- 기존 SD와 최종 SD는 여러 요소가 같이 달라진 설계 변형 비교. 압축 단계 하나만의 인과효과라고 쓰지 않음.
- 프롬프트 탐색에서 관측된 성능 순위와 외부 일반화는 구분. 탐색 결과의 p값은 프롬프트 선택까지 보정한 확증 결과가 아님.
- 구형 대표 사례는 현 모델에 해당하는지 모르는 채 다시 싣지 않음. 최종 데이터 기준 사례 확인 후 보강.

## 제공 파일
- main.pdf: 전체 업데이트 검토본. 제출 완결본이라는 뜻은 아님.
- main.tex / Overleaf ZIP: 수정 가능한 전체 원고와 벡터 그림, 프롬프트.
- MANUSCRIPT_FULL_DIFF.md: 직전 원고 대비 실제 전체 변경 내역.
- evidence/final_results_and_count_statistics.csv: 표 및 통계 재현용 집계.
- evidence/prompt_sources.json: 부록 프롬프트의 코드 출처와 파일 해시.
'''
review_source = REPO/'reports/unified_v6c_revision_20260929/UPDATE_AND_REMAINING_KO.md'
(OUT/'UPDATE_AND_REMAINING_KO.md').write_text(review_source.read_text() if review_source.exists() else review)
(OUT/'MANUSCRIPT_FULL_DIFF.md').write_text('# 원고 전체 변경 내역\n\n```diff\n'+''.join(difflib.unified_diff(old.splitlines(True),source.splitlines(True),fromfile='20260912/main.tex',tofile='20260929/main.tex'))+'```\n')

TEX='/Users/woojinpark/.cache/codex-runtimes/codex-texlive/small/bin/universal-darwin'
TINY='/Users/woojinpark/Library/TinyTeX/texmf-dist'
CACHE='/Users/woojinpark/Library/Caches/Tectonic/bundles/data/6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c'
env=os.environ.copy()
env.update(TEXINPUTS=f'.:{TINY}/tex//:{CACHE}//:',TFMFONTS=f'.:{TINY}/fonts/tfm//:',VFFONTS=f'.:{TINY}/fonts/vf//:',T1FONTS=f'.:{TINY}/fonts/type1//:',ENCFONTS=f'.:{TINY}/fonts/enc//:',TEXFONTMAPS=f'.:/Users/woojinpark/Library/TinyTeX/texmf-var/fonts/map/pdftex/updmap//:{TINY}/fonts/map//:')
for i in range(3):
    p=subprocess.run([TEX+'/pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=OUT,env=env,capture_output=True,text=True)
    (OUT/'qa'/f'build-{i}.log').write_text(p.stdout+p.stderr)
    if p.returncode:
        raise RuntimeError(p.stdout[-3500:])
from pypdf import PdfReader
pdf=PdfReader(OUT/'main.pdf')
text='\n'.join(p.extract_text() or '' for p in pdf.pages)
(OUT/'qa/extracted_text.txt').write_text(text)
log=(OUT/'main.log').read_text()
qa={'pages':len(pdf.pages),'undefined_references':'undefined references' in log,'overfull':[s for s in log.splitlines() if 'Overfull' in s],
    'old_downstream_values_remaining':[v for v in ['96.69','81.33','1.7706'] if v in source],
    'visual_review':'pending','final_result_rows':len(df),'source':'notebook aggregates + previously reviewed final baseline counts'}
(OUT/'qa/verification.json').write_text(json.dumps(qa,indent=2))
print(json.dumps(qa,indent=2))
print('Output:',OUT)
