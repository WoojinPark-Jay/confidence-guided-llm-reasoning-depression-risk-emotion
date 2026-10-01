"""Restore the reviewed manuscript's serif/blue-gray visual conventions."""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import PercentFormatter, MultipleLocator
from matplotlib.text import Text

INK = '#30363A'
BLUE = '#245C96'
LIGHT_BLUE = '#8DAAC2'
GRAY = '#969EA6'
GRID = '#E1E4E7'
PALE = '#E7EDF1'
METHODS = ['Direct', 'CoT', 'SELF-DISCOVER']
METHOD_COLORS = [GRAY, LIGHT_BLUE, BLUE]


def style_axes(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#9A9FA3')
        ax.spines[side].set_linewidth(.55)
    ax.tick_params(width=.5, length=3, color='#9A9FA3')
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.5)


def save(fig, out, name):
    # Match the original figure's type size after reduction to the paper's width.
    for text in fig.findobj(match=Text):
        text.set_fontsize(text.get_fontsize() * 1.22)
    for suffix, kwargs in [('pdf', {}), ('svg', {}), ('png', {'dpi': 450})]:
        fig.savefig(out / 'figures' / f'{name}.{suffix}', facecolor='white',
                    bbox_inches='tight', pad_inches=.12, **kwargs)
    plt.close(fig)


def build_figures(out, p1, results, phase1_metrics, selected):
    for font in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 11,
        'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
        'ytick.color': INK, 'axes.titleweight': 'bold', 'axes.titlesize': 12,
        'axes.labelweight': 'bold', 'axes.labelsize': 11,
        'legend.fontsize': 10, 'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.unicode_minus': False,
    })
    predictions = {data: pd.read_csv(p1 / name) for data, name in [
        ('Reddit', 'phase1_test_predictions.csv'),
        ('Mixed Emotion', 'phase1_mixed_emotion_predictions.csv')]}

    # The original figure used a common row-normalized scale, not independent counts.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.25))
    cmap = LinearSegmentedColormap.from_list('reviewed_blues', ['#F3F4F5', '#3E719F'])
    for i, (ax, (data, metrics)) in enumerate(zip(axes, phase1_metrics.items())):
        matrix = np.array(metrics['confusion_matrix'])
        rates = matrix / matrix.sum(axis=1, keepdims=True)
        for row in range(3):
            for col in range(3):
                ax.add_patch(Rectangle((col-.5,row-.5),1,1,
                                       facecolor=cmap(rates[row,col]),edgecolor='white',lw=.6))
        ax.set(xlim=(-.5,2.5),ylim=(2.5,-.5),aspect='equal')
        for row in range(3):
            for col in range(3):
                color = 'white' if rates[row, col] > .55 else INK
                ax.text(col, row-.08, f'{matrix[row,col]:,}', color=color,
                        ha='center', va='center', weight='bold', fontsize=12)
                ax.text(col, row+.18, f'{rates[row,col]:.1%}', color=color,
                        ha='center', va='center', fontsize=10)
        ax.set_xticks(range(3), ['Depression', 'Neutral', 'Happy'])
        ax.set_yticks(range(3), ['Depression', 'Neutral', 'Happy'])
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('Predicted label', labelpad=9)
        ax.set_ylabel('Reference label', labelpad=10)
        ax.set_title(f'({chr(97+i)})  {data} · Phase 1 · {metrics["accuracy"]:.2f}%', pad=42)
        ax.tick_params(length=0, pad=8)
        ax.set_xticks(np.arange(-.5,3,1), minor=True)
        ax.set_yticks(np.arange(-.5,3,1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1)
        ax.tick_params(which='minor', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=.11, right=.98, bottom=.06, top=.76, wspace=.54)
    save(fig, out, 'phase1_confusion')

    # Retain the original three-panel calibration figure and its metric comparison.
    cal = pd.read_csv(out / 'evidence/calibration_metric_summary.csv')
    raw = cal[(cal.split == 'calibration') & (cal.confidence_method == 'raw_msp')].iloc[0]
    scaled = cal[(cal.split == 'calibration') & (cal.confidence_method == 'temperature_scaled_msp')].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    for key, label, color in [('raw_msp', 'Raw MSP', GRAY),
                              ('temperature_scaled_msp', 'Scaled MSP', BLUE)]:
        d = pd.read_csv(out / f'evidence/calibration_reliability_{key}.csv').dropna()
        axes[0].plot(d.mean_confidence, d.accuracy, 'o-', color=color,
                     label=label, markersize=5, markeredgecolor='white', linewidth=1.6)
    axes[0].plot([.3,1],[.3,1], '--', color='#77818A', lw=1, label='Ideal calibration')
    axes[0].set(xlim=(.3,1.01), ylim=(0,1.03), xlabel='Mean confidence', ylabel='Observed accuracy')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].set_title('(a)  Reliability before and after scaling', loc='left', pad=40)
    axes[0].text(0,1.035, f'ECE: {raw.ece:.2%} → {scaled.ece:.2%}', transform=axes[0].transAxes, fontsize=10)
    keys = ['nll','brier_score','ece']
    ratios = np.array([scaled[k]/raw[k]*100 for k in keys])
    y = np.arange(3)
    axes[1].barh(y-.16, [100]*3, .28, color=GRAY, label='Raw = 100%')
    axes[1].barh(y+.16, ratios, .28, color=BLUE, label='Scaled')
    for j,k in enumerate(keys):
        axes[1].text(ratios[j]+2,j+.16,f'{ratios[j]:.1f}%',va='center',fontsize=10)
        axes[1].text(0,j-.37, f'{raw[k]:.4f} → {scaled[k]:.4f}',fontsize=10)
    axes[1].set_yticks(y,['NLL','Brier','ECE'])
    axes[1].set(xlim=(0,119),ylim=(2.6,-.75),xlabel='Relative value (raw = 100%)')
    axes[1].xaxis.set_major_formatter(PercentFormatter(100,0))
    axes[1].legend(loc='upper left',bbox_to_anchor=(-.03,1.15),ncol=2,fontsize=9)
    axes[1].set_title('(b)  Calibration metric reduction',loc='left',pad=40)
    d = pd.read_csv(out / 'evidence/risk_coverage_curve_calibration_points.csv')
    axes[2].plot(d.coverage*100,d.selective_risk*100,color=INK,lw=1.8)
    axes[2].scatter([selected.coverage*100],[selected.selective_risk*100],s=48,color=BLUE,edgecolors='white',zorder=3)
    axes[2].plot([selected.coverage*100]*2,[0,selected.selective_risk*100],color=BLUE,ls='--',lw=.8)
    axes[2].text(.04,.93,f'Threshold 0.70\n{selected.coverage:.2%} coverage',transform=axes[2].transAxes,color=BLUE,va='top')
    axes[2].set(xlabel='Coverage',ylabel='Selective risk',xlim=(20,101),ylim=(0,3.1))
    axes[2].xaxis.set_major_formatter(PercentFormatter(100,0))
    axes[2].yaxis.set_major_formatter(PercentFormatter(100,1))
    axes[2].set_title('(c)  Risk–coverage curve',loc='left',pad=40)
    for ax in axes:
        style_axes(ax, 'x' if ax is axes[1] else 'y')
    fig.subplots_adjust(left=.065,right=.99,bottom=.18,top=.73,wspace=.46)
    save(fig,out,'calibration')

    fig,axes=plt.subplots(1,3,figsize=(13.2,4.4))
    for data,color in [('Reddit',BLUE),('Mixed Emotion',LIGHT_BLUE)]:
        p=predictions[data]
        confidence=p.phase1_confidence.to_numpy()
        wrong=(p.label_str!=p.phase1_label).to_numpy()
        xs=np.sort(confidence)
        axes[0].plot(xs,100*np.arange(1,len(xs)+1)/len(xs),color=color,label=data,lw=1.8)
        order=np.argsort(confidence,kind='stable')
        budget=100*np.arange(1,len(xs)+1)/len(xs)
        capture=100*np.cumsum(wrong[order])/wrong.sum()
        axes[1].plot(budget,capture,color=color,lw=1.8,label=data)
        route=p.phase1_routed.to_numpy(dtype=bool)
        axes[1].scatter([100*route.mean()],[100*wrong[route].sum()/wrong.sum()],color=color,s=35,zorder=4,edgecolors='white')
        j=0 if data=='Reddit' else 1
        full=100*wrong.mean(); routed=100*wrong[route].mean()
        axes[2].plot([full,routed],[j,j],color=color,lw=1.7)
        axes[2].scatter([full],[j],color=GRAY,s=42,zorder=3)
        axes[2].scatter([routed],[j],color=color,s=55,zorder=3)
        axes[2].text(full,j+.18,f'{full:.2f}%',ha='center',fontsize=10)
        axes[2].text(routed,j+.18,f'{routed:.2f}%',ha='center',fontsize=10)
        axes[2].text((full+routed)/2,j-.12,f'{routed/full:.2f}×',ha='center',weight='bold',fontsize=11)
    axes[0].axvline(.7,color=GRAY,ls='--',lw=1)
    axes[0].text(.71,75,'Threshold\n0.70',fontsize=9)
    axes[0].set(xlabel='Calibrated confidence',ylabel='Cumulative share of posts (%)',xlim=(.3,1),ylim=(0,101))
    axes[0].legend(loc='upper left',fontsize=9)
    axes[1].plot([0,100],[0,100],color=GRAY,ls='--',lw=.8)
    axes[1].set(xlabel='Posts routed (%)',ylabel='Phase 1 errors captured (%)',xlim=(0,100),ylim=(0,103))
    axes[2].set_yticks([0,1],['Reddit','Mixed'])
    axes[2].set(xlabel='Error rate (%)',xlim=(0,60),ylim=(1.5,-.6))
    axes[2].legend(handles=[Line2D([],[],marker='o',color=GRAY,ls='',label='Full set'),Line2D([],[],marker='o',color=BLUE,ls='',label='Routed set')],loc='upper left',ncol=2,fontsize=9)
    for ax,title in zip(axes,['(a)  Confidence distribution','(b)  Error capture by routing budget','(c)  Error concentration in routed posts']):
        style_axes(ax);ax.set_title(title,loc='left',pad=20)
    fig.subplots_adjust(left=.065,right=.99,bottom=.19,top=.84,wspace=.5)
    save(fig,out,'routing_concentration')

    # Dot plots retain small differences without encoding them as truncated bars.
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for panel,(ax,(data,metrics)) in enumerate(zip(axes,phase1_metrics.items())):
        part=results[results.dataset==data]
        for model_index,model in enumerate(['Llama 2','Llama 3']):
            for j,method in enumerate(METHODS):
                row=part[(part.model==model)&(part.method==method)].iloc[0]
                y=model_index*4+j
                ax.plot([metrics['accuracy'],row.accuracy_percent],[y,y],color=METHOD_COLORS[j],lw=1.8)
                ax.scatter(row.accuracy_percent,y,color=METHOD_COLORS[j],s=52,edgecolors='white',zorder=3)
                ax.text(1.025,y,f'{row.accuracy_percent:.4f}%',transform=ax.get_yaxis_transform(),va='center',fontsize=10,weight='bold' if j==2 else 'normal')
        ax.axhline(3,color=GRID,lw=.6)
        ax.axvline(metrics['accuracy'],color=INK,ls='--',lw=.9)
        ax.set_yticks([0,1,2,4,5,6],['L2 · Direct','L2 · CoT','L2 · SD','L3 · Direct','L3 · CoT','L3 · SD'])
        ax.set_ylim(6.7,-.8)
        ax.set_xlim((96.89,97.23) if data=='Reddit' else (74,94))
        ax.set_xlabel('Full-set accuracy (%)')
        ax.set_title(f'({chr(97+panel)})  {data}',loc='left',pad=32)
        ax.text(0,1.04,f'Phase 1 = {metrics["accuracy"]:.4f}% (dashed)',transform=ax.transAxes,fontsize=10)
        style_axes(ax,'x')
    fig.subplots_adjust(left=.09,right=.88,top=.79,bottom=.17,wspace=.8)
    save(fig,out,'end_to_end')

    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for panel,(ax,data) in enumerate(zip(axes,['Reddit','Mixed Emotion'])):
        part=results[results.dataset==data]
        ys=[0,1,2,4,5,6]
        ax.barh(ys,-part.introduced.to_numpy(),.62,color=GRAY,label='Introduced')
        ax.barh(ys,part.corrected.to_numpy(),.62,color=BLUE,label='Corrected')
        for y,(_,row) in zip(ys,part.iterrows()):
            for val,sign in [(row.introduced,-1),(row.corrected,1)]:
                ax.text(sign*val/2 if val>=5 else sign*(val+1),y,str(val),va='center',ha='center' if val>=5 else ('right' if sign<0 else 'left'),color='white' if val>=5 else INK,weight='bold',fontsize=10)
            ax.text(1.03,y,f'{int(row.net):+d}',transform=ax.get_yaxis_transform(),va='center',ha='center',weight='bold')
        ax.text(1.03,1.035,'Net',transform=ax.transAxes,ha='center',weight='bold',fontsize=10)
        ax.axvline(0,color=INK,lw=.7)
        ax.axhline(3,color=GRID,lw=.6)
        ax.set_yticks(ys,['L2 · Direct','L2 · CoT','L2 · SD','L3 · Direct','L3 · CoT','L3 · SD'])
        ax.set_ylim(6.7,-.8)
        ax.set_xlim((-57,82) if data=='Reddit' else (-40,34))
        ax.set_xlabel('Introduced ← Posts → Corrected')
        ax.set_title(f'({chr(97+panel)})  {data}',loc='left',pad=35)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,fontsize=10)
        style_axes(ax,'x')
    fig.subplots_adjust(left=.1,right=.94,top=.78,bottom=.18,wspace=.46)
    save(fig,out,'corrections')
    build_sd_overview(out)
    build_workflow(out)


def build_sd_overview(out):
    fig, ax = plt.subplots(figsize=(11.8, 5.1))
    ax.set(xlim=(0, 12), ylim=(0, 5.4))
    ax.axis('off')

    def panel(x, y, w, h, title, body, fill=PALE):
        ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec='#687E8F', lw=.9))
        ax.text(x+w/2, y+h-.19, title, ha='center', va='top',
                fontsize=10.6, weight='bold')
        if body:
            ax.text(x+w/2, y+.43*h, body, ha='center', va='center',
                    fontsize=9.6, linespacing=1.45)

    def arrow(start, end, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                    mutation_scale=12, lw=1.05, shrinkA=2, shrinkB=2,
                    linestyle=(0, (4, 3)) if dashed else '-',
                    color=BLUE if dashed else '#56616A'))

    ax.text(.1, 5.17, '(a)  Plan construction | Before post classification',
            weight='bold', fontsize=11.5)
    panel(.1, 3.55, 2.7, 1.35, 'Researcher-authored inputs',
          'Task + class policy\n18 emotion-oriented modules', '#F3F4F5')
    panel(3.3, 3.55, 5.45, 1.35, 'LLM: construct the procedure', '')
    for x, title, detail in [(4.22, 'SELECT', 'Choose modules'),
                              (6.02, 'ADAPT', 'Specialize to task'),
                              (7.82, 'IMPLEMENT', 'Write the plan')]:
        ax.text(x, 4.14, title, ha='center', va='center', fontsize=10, weight='bold')
        ax.text(x, 3.79, detail, ha='center', va='center', fontsize=8.9)
    arrow((4.85, 4.14), (5.35, 4.14))
    arrow((6.66, 4.14), (7.08, 4.14))
    panel(9.25, 3.55, 2.55, 1.35, 'Saved model-specific plan',
          'Code stores the procedure\nNot a post or an answer', '#D8E2EA')
    arrow((2.8, 4.225), (3.3, 4.225))
    arrow((8.75, 4.225), (9.25, 4.225))

    ax.text(.1, 3.14, '(b)  Fresh execution | Every routed post',
            weight='bold', fontsize=11.5)
    ax.plot([10.525, 10.525, 6.025], [3.55, 2.86, 2.86],
            color=BLUE, lw=1.05, ls=(0, (4, 3)))
    arrow((6.025, 2.86), (6.025, 2.5), True)
    ax.text(8.2, 3.02, 'Reuse the procedure', ha='center', fontsize=9, color=BLUE)

    panel(.1, .8, 2.7, 1.7, 'Post-specific input',
          'This post: original text\nPolicy + fixed checks\nPhase 1 label withheld', '#F3F4F5')
    panel(3.3, .8, 5.45, 1.7, 'LLM: evidence-based re-evaluation',
          'Emotional subject and time frame\nQuotes for / against all three labels\nCheck the competing interpretation')
    panel(9.25, .8, 2.55, 1.7, 'New response',
          'Evidence + final label\nDepression / Neutral / Happy', '#D8E2EA')
    arrow((2.8, 1.65), (3.3, 1.65))
    arrow((8.75, 1.65), (9.25, 1.65))
    ax.text(.1, .36, 'Discovery uses no evaluation post or reference label. A saved plan bypasses all three discovery calls.',
            fontsize=9.1)
    ax.text(.1, .05, 'Each LLM retains its own plan across both datasets; every routed post receives a new execution response.',
            fontsize=9.1)
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    save(fig, out, 'sd_task_level_overview')


def build_workflow(out):
    fig,ax=plt.subplots(figsize=(11.8,7.5))
    ax.set(xlim=(0,12),ylim=(0,8.6));ax.axis('off')
    def box(x,y,w,h,title,body,fill=PALE):
        ax.add_patch(Rectangle((x,y),w,h,fc=fill,ec='#687E8F',lw=.9))
        ax.text(x+w/2,y+h-.22,title,ha='center',va='top',weight='bold',fontsize=11)
        ax.text(x+w/2,y+h/2-.18,body,ha='center',va='center',fontsize=10,linespacing=1.4)
    def arrow(start,end,dashed=False):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,lw=1.1,
                                    linestyle=(0,(4,3)) if dashed else '-',color=BLUE if dashed else '#56616A'))
    ax.text(.1,8.32,'(a)  Before post classification: construct the plan if no saved plan exists',weight='bold',fontsize=12)
    ax.text(.1,7.91,'No evaluation post or reference label is supplied during SELECT, ADAPT, or IMPLEMENT.',fontsize=10)
    ax.text(1.25,7.50,'RESEARCHER-AUTHORED',ha='center',weight='bold',fontsize=9)
    ax.text(5.9,7.50,'GENERATED BY THE SELECTED LLM',ha='center',weight='bold',fontsize=9)
    ax.text(10.65,7.50,'SAVED BY CODE',ha='center',weight='bold',fontsize=9)
    box(.1,5.95,2.3,1.35,'Task materials','Task + class policy\n18 reasoning modules','#F3F4F5')
    box(2.8,5.95,1.7,1.35,'SELECT','Choose relevant\nmodules')
    box(4.9,5.95,1.7,1.35,'ADAPT','Specialize them\nto the task')
    box(7,5.95,2,1.35,'IMPLEMENT','Write a shared\njudgment procedure','#D8E2EA')
    box(9.5,5.95,2.3,1.35,'Cached plan','Reusable procedure\nNot a post or answer','#D8E2EA')
    for a,b in [(2.4,2.8),(4.5,4.9),(6.6,7),(9,9.5)]:
        arrow((a,6.625),(b,6.625))
    ax.plot([10.65,10.65,5.9],[5.95,4.47,4.47],color=BLUE,lw=1.1,ls=(0,(4,3)))
    arrow((5.9,4.47),(5.9,4.2),True)
    ax.text(7.0,5.32,'Reuse the same plan for every post',fontsize=9.5,color=BLUE)
    ax.text(.1,4.91,'(b)  For each routed post: the LLM generates a fresh analysis and label',weight='bold',fontsize=12)
    ax.text(.1,4.65,'Post text first enters here. An existing plan bypasses all three discovery calls in (a).',fontsize=10)
    box(.1,.7,2.8,3.5,'Execution inputs',
        'This post\nOriginal text\n\nFixed researcher guidance\nPolicy + evidence checks\n\nPhase 1 label withheld','#F3F4F5')
    box(3.5,.7,4.8,3.5,'LLM: per-post execution',
        'Saved plan + this post + fixed guidance\n\nWhose emotion? Current or past?\nQuotes for / against all three labels\nCheck a competing interpretation\n\nChoose the final label only at the end')
    box(8.8,2.6,3,1.6,'Generated output','Evidence + final label\nOne execution call','#D8E2EA')
    box(8.8,.7,3,1.4,'Evaluation code','Parse label; if unsuccessful,\nretain Phase 1 prediction','#F3F4F5')
    arrow((2.9,2.45),(3.5,2.45));arrow((8.3,3.4),(8.8,3.4));arrow((10.3,2.6),(10.3,2.1))
    ax.text(.1,.2,'Llama 2 and Llama 3 each generate their own plan; each plan is reused for Reddit and Mixed Emotion.',fontsize=9.5)
    fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
    save(fig,out,'appendix_sd_workflow')


def restyle_source(source):
    source=source.replace(r'\maketitle',r'\maketitle'+'\n'+r'\raggedbottom')
    source=source.replace(r'\section{Discussion}',r'\clearpage\twocolumn\raggedbottom'+'\n'+r'\section{Discussion}')
    source=source.replace(r'\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible,keepspaces=true,showstringspaces=false,frame=single,rulecolor=\color{gray},aboveskip=6pt,belowskip=8pt}',
        r'''\definecolor{PromptRule}{HTML}{8DAAC2}
\lstset{basicstyle=\rmfamily\small,language={},keywordstyle={},commentstyle={},stringstyle={},literate={*}{{\char42}}1 {-}{{\char45}}1,breaklines=true,columns=fullflexible,keepspaces=true,showstringspaces=false,frame=tb,rulecolor=\color{PromptRule},aboveskip=7pt,belowskip=10pt,framesep=5pt,linewidth=\dimexpr\linewidth-18pt\relax,xleftmargin=6pt,xrightmargin=0pt,breakatwhitespace=true}
\setlength{\textfloatsep}{12pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{12pt plus 2pt minus 2pt}''')
    # Keep the main-text matrix with the discussion it supports.
    source=re.sub(r'\\Needspace\{24\\baselineskip\}\s*\\section\{Final Phase 1 Confusion Matrices\}.*?\\end\{center\}', '', source, flags=re.S)
    anchor=r'\subsection{Calibration and Routing}'
    matrix=r'''\subsection{Class-Level Error Structure}
Figure~\ref{fig:phase1-confusion} shows the final linked classifier's errors on both datasets. Cell counts and row-normalized percentages distinguish the balanced Reddit test from the smaller stress test. These are the final Phase~1 predictions, not pooled three-seed counts or re-evaluation outputs.
\begin{figure*}[t]
\centering
\includegraphics[width=0.96\textwidth]{figures/phase1_confusion.pdf}
\caption{Final Phase 1 confusion matrices on Reddit ($N=12{,}000$) and Mixed Emotion ($N=300$). Rows are reference labels; columns are predictions. Each cell shows a count and its percentage within the reference class. Both panels use the same row-normalized color scale.}
\label{fig:phase1-confusion}
\end{figure*}

'''
    source=source.replace(anchor,matrix+anchor)
    source=source.replace('Reliability and risk--coverage curves use the calibration split;', 'Reliability, relative metric reduction, and risk--coverage curves use the calibration split;')
    source=source.replace('Full-set accuracy for the final six configurations. Dashed lines mark the linked Phase 1 baselines. Axes are truncated; the two panels have different scales.', 'Full-set accuracy for the final six configurations. Points show observed accuracy; horizontal segments connect each point to the linked Phase 1 baseline (dashed line). The two panels use different axis ranges.')
    source=source.replace('Corrected and introduced errors for each final configuration. Net improvement requires corrections to exceed newly introduced errors. The two panels use different count scales.', 'Corrected (blue, right) and introduced (gray, left) errors for each final configuration. Introduced counts are plotted to the left for comparison, not as negative error counts. Right-hand labels show net corrections. The panels use different count scales.')
    routing=r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/routing_concentration.pdf}
\caption{Fixed-policy routing concentration. (a) Cumulative calibrated-confidence distributions. (b) Errors captured when posts are ordered from lowest to highest confidence; dots identify the fixed threshold and the diagonal is a proportional-capture reference. (c) Full-set and routed error rates, with enrichment ratios. Curves describe the evaluation sets; they are not used to select the threshold.}
\label{fig:routing-concentration}
\end{figure*}

'''
    source=source.replace(r'\subsection{End-to-End Model and Prompting Results}',routing+r'\subsection{End-to-End Model and Prompting Results}')
    # Match the original appendix's tabular summaries, followed by readable exact text.
    overview=r'''
\par\medskip
\noindent\textsc{Appendix Table A2. Final re-evaluation protocols}\par\smallskip
{\small\renewcommand{\arraystretch}{1.18}
\noindent\begin{tabularx}{\textwidth}{@{}p{0.13\textwidth}YYY@{}}\toprule
Protocol & Information supplied & Processing structure & Scored output \\\midrule
Direct & Original post, shared policy, Phase 1 label & One label-only generation call & One parsed label; otherwise Phase 1 fallback \\
CoT & Original post; Phase 1 label revealed after independent assessment & Five dialogue calls, including acknowledgment and text-response turns & Terminal label from the final response; otherwise fallback \\
SELF-DISCOVER & Original post, policy, cached model-specific plan; no Phase 1 label & Task-level discovery, followed by one execution call per routed post & Terminal label after evidence and counterfactual checks; otherwise fallback \\\bottomrule
\end{tabularx}}\par\medskip
The sections below reproduce the literal prompt text. Placeholder names remain as written in the executable templates.

'''
    source=source.replace(r'\subsection{Shared Classification Policy}',overview+r'\subsection{Shared Classification Policy}')
    workflow=r'''
\Needspace{24\baselineskip}
\begin{center}
\includegraphics[width=0.96\textwidth]{figures/appendix_sd_workflow.pdf}
\end{center}
\noindent\footnotesize\textbf{Appendix Figure C1.} Final task-level SELF-DISCOVER workflow. Each model generates its own cached plan, reused for Reddit and Mixed Emotion. The compact-plan instruction is not a guaranteed output constraint. The diagram summarizes the final protocol rather than the earlier per-post discovery variant.\normalsize\par\medskip

\noindent\textsc{Appendix Table A3. Final SELF-DISCOVER stages}\par\smallskip
{\small\renewcommand{\arraystretch}{1.18}
\noindent\begin{tabularx}{\textwidth}{@{}p{0.15\textwidth}YYY@{}}\toprule
Stage & Input & Requested operation & Retained output \\\midrule
SELECT & Task policy and 18 domain modules & Select task-relevant reasoning operations & Selected modules \\
ADAPT & Task description and selected modules & Specialize operations to the three-class task & Adapted modules \\
IMPLEMENT & Adapted modules & Request a compact plan of at most three one-line steps & Actual model-generated plan, cached without a conformance validator \\
Execution & Original post, policy, cached plan & Compare evidence for all three labels and check an alternative interpretation & Response text, parsed label, and failure status \\\bottomrule
\end{tabularx}}\par\medskip

'''
    source=source.replace(r'\subsection{Task Description and Module Bank}',workflow+r'\subsection{Task Description and Module Bank}')
    source=source.replace(r'\lstinputlisting{prompts/sd_modules.txt}',r'\lstinputlisting{prompts/sd_modules.txt}')
    # Increase the appendix tables to the original readable full-width format.
    source=source.replace(r'\begin{center}\small'+'\n'+r'\begin{tabular}{llrrrr}',r'\par\medskip\noindent\textsc{Appendix Table D1. SELF-DISCOVER design-variant comparison}\par\smallskip'+'\n'+r'{\small\renewcommand{\arraystretch}{1.18}\noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{}}')
    source=source.replace(r'\begin{center}\footnotesize'+'\n'+r'\begin{tabular}{lllrrrr}',r'\par\medskip\noindent\textsc{Appendix Table E1. Paired accuracy comparisons with Phase 1}\par\smallskip'+'\n'+r'{\small\renewcommand{\arraystretch}{1.18}\noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrr@{}}')
    source=source.replace(r'\end{tabular}\end{center}',r'\end{tabular*}}\par\medskip')
    source=source.replace(r'\subsection{Shared Classification Policy}',r'\Needspace{13\baselineskip}'+ '\n'+r'\subsection{Shared Classification Policy}')
    source=source.replace(r'\subsection{Direct Re-Evaluation}',r'\Needspace{18\baselineskip}'+'\n'+r'\subsection{Direct Re-Evaluation}')
    source=source.replace(r'\section{Paired Accuracy Summaries}',r'\Needspace{28\baselineskip}'+'\n'+r'\section{Paired Accuracy Summaries}')
    for title in ['System message','Initial user instruction','Post input','Independent analysis','Comparison with Phase 1','Final decision']:
        source=source.replace(r'\noindent\textbf{'+title+'}',r'\par\smallskip\Needspace{6\baselineskip}\noindent\textbf{'+title+'}'+r'\par')
    for model,label in [('llama2','Llama 2'),('llama3','Llama 3')]:
        block=r'\Needspace{10\baselineskip}'+'\n'+r'\noindent\textbf{'+label+' generated plan}'+'\n'+r'\lstinputlisting{prompts/sd_plan_'+model+'.txt}'
        source=source.replace(block,r'\par\medskip\noindent\begin{minipage}{\textwidth}'+'\n'+r'\textbf{'+label+' generated plan}'+'\n'+r'\lstinputlisting{prompts/sd_plan_'+model+'.txt}'+'\n'+r'\end{minipage}\par\medskip')
    return source
