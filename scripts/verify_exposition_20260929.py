"""Render the revised manuscript and check preservation and page boundaries."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

import pdfplumber
from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT = Path('/Users/woojinpark/Documents/헬스케어 논문')
REPO = ROOT / 'repo_final_push_nolfs'
OUT = ROOT / 'Paper_20260929_unified_v6c'
QA = OUT / 'qa'
source = (OUT / 'main.tex').read_text()
report = 'reports/unified_v6c_revision_20260929/'
preserved = []
for relative in ['evidence/final_results_and_count_statistics.csv',
                 'evidence/phase1_verified_metrics.json',
                 'figures/architecture_figure_publication_final.pdf'] + [
                     'prompts/' + p.name for p in sorted((OUT / 'prompts').glob('*.txt'))]:
    previous = subprocess.check_output(['git', 'show', '833b331:' + report + relative], cwd=REPO)
    assert previous == (OUT / relative).read_bytes(), relative
    preserved.append(relative)
pdf = PdfReader(OUT / 'main.pdf')
text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
assert '??' not in text
log = (OUT / 'main.log').read_text()
assert 'undefined references' not in log and 'Missing character:' not in log
for path in re.findall(r'\\(?:includegraphics(?:\[[^]]*\])?|lstinputlisting)\{([^}]+)\}', source):
    assert (OUT / path).exists(), path
overflows = []
with pdfplumber.open(OUT / 'main.pdf') as reader:
    for i, page in enumerate(reader.pages):
        for word in page.extract_words():
            if word['x0'] < 28 or word['x1'] > page.width - 28:
                overflows.append({'page': i + 1, 'text': word['text'], 'x0': word['x0'], 'x1': word['x1']})
for path in QA.glob('exposition-page-*.png'):
    path.unlink()
subprocess.run(['pdftoppm', '-r', '105', '-png', str(OUT / 'main.pdf'),
                str(QA / 'exposition-page')], check=True)
pages = sorted(QA.glob('exposition-page-*.png'))
assert len(pages) == len(pdf.pages)
for start in range(0, len(pages), 8):
    sheet = Image.new('RGB', (900, 4 * 650), '#dddddd')
    draw = ImageDraw.Draw(sheet)
    for j, path in enumerate(pages[start:start + 8]):
        thumb = Image.open(path)
        thumb.thumbnail((430, 610))
        x, y = (j % 2) * 450 + 10, (j // 2) * 650 + 28
        sheet.paste(thumb, (x, y))
        draw.text((x, y - 20), f'Page {start + j + 1}', fill='black')
    sheet.save(QA / f'exposition-contact-{start // 8}.png')
checks = {'pages': len(pdf.pages), 'preserved_files': preserved,
          'margin_overflow_words': overflows, 'undefined_references': False,
          'visual_review': 'rendered; awaiting visual inspection',
          'pdf_sha256': hashlib.sha256((OUT / 'main.pdf').read_bytes()).hexdigest()}
(QA / 'exposition_verification.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
print(json.dumps(checks, ensure_ascii=False, indent=2))
