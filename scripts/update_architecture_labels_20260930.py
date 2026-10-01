"""Preserve the original vector artwork; update labels and matching draw.io text."""
import html
from pathlib import Path
import xml.etree.ElementTree as ET
import pymupdf as fitz


def update_architecture(base, out):
    doc = fitz.open(base / 'figures/architecture_figure_publication_final.pdf')
    page = doc[0]
    # Rectangles are text interiors only. Redaction leaves vector lines untouched.
    edits = [
        ('DEVPKG', (423,108,594,166), (0.9569,.9647,.9686), 9,
         'Phase 1 model development', ['Train + validation: Bayesian sweep', 'Selected DistilBERT configuration', 'Final seed42 model used throughout', 'Llama 2 / Mistral comparator runs']),
        ('REASON', (687,366,842,433), (0.9569,.9647,.9686), 9.2,
         'Phase 2 re-evaluators', ['Llama 2-7B-Chat / Llama 3-8B-Instruct', 'Each: Direct, CoT, SELF-DISCOVER', 'SD: shared plan + per-post execution', 'Six separate configurations', 'No ensemble']),
        ('P2LABEL', (875,367,966,408), (.1922,.3725,.5255), 8.6,
         'Parse Phase 2 label', ['If parsing fails,', 'retain Phase 1 label']),
        ('FINAL', (1086,128,1230,188), (.8863,.9255,.9569), 9,
         'End-to-end prediction', ['Accepted: retain Phase 1 label', 'Routed: use parsed Phase 2 label', 'Parsing failure:', 'retain Phase 1 label']),
        ('BOUND', (46,402,291,449), (1,1,1), 8.7,
         'Data-use scope', ['Evaluation sets excluded from Phase 1', 'development and calibration.', 'Main evaluation cases informed', 'Phase 2 prompt development.']),
        ('AU', (1086,370,1232,449), (1,1,1), 9.1,
         'Statistical and audit artifacts', ['Routed-case output records', 'Paired bootstrap confidence intervals', 'Exact McNemar test', 'Holm correction', 'Class-wise error analysis']),
    ]
    for _, box, fill, _, _, _ in edits:
        page.add_redact_annot(fitz.Rect(box), fill=fill, cross_out=False)
    page.apply_redactions(images=0, graphics=0)
    fontdir = Path('/System/Library/Fonts/Supplemental')
    regular, bold = fontdir/'Times New Roman.ttf', fontdir/'Times New Roman Bold.ttf'
    page.insert_font(fontname='TNR', fontfile=str(regular))
    page.insert_font(fontname='TNRBold', fontfile=str(bold))
    f_regular, f_bold = fitz.Font(fontfile=str(regular)), fitz.Font(fontfile=str(bold))
    for key, box, _, size, title, body in edits:
        rect=fitz.Rect(box)
        linegap=size*1.15
        lines=[title,*body]
        size=min(size, min((rect.width-4)/f.text_length(t,fontsize=1) for t,f in [(title,f_bold),*[(t,f_regular) for t in body]]))
        linegap=size*1.15
        top=rect.y0+(rect.height-linegap*len(lines))/2+size
        for i, text in enumerate(lines):
            font=f_bold if i==0 else f_regular
            x=rect.x0+(rect.width-font.text_length(text,fontsize=size))/2
            page.insert_text((x,top+i*linegap),text,fontname='TNRBold' if i==0 else 'TNR',fontsize=size,
                             color=(1,1,1) if key=='P2LABEL' else (.149,.196,.220))
    doc.save(out/'figures/architecture_figure_publication_final.pdf',garbage=4,deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(2,2)).save(out/'figures/architecture_figure_publication_final.png')
    tree=ET.parse(base/'figures/architecture_figure_publication_final.drawio')
    cells={c.get('id'):c for c in tree.iter('mxCell')}
    for key,_,_,_,title,body in edits:
        text='<b>'+html.escape(title)+'</b><br>'+'<br>'.join(html.escape(t) for t in body)
        cells[key].set('value', '<font face="Times New Roman"'+(' color="#FFFFFF"' if key=='P2LABEL' else '')+'>'+text+'</font>')
    tree.write(out/'figures/architecture_figure_publication_final.drawio',encoding='unicode')
