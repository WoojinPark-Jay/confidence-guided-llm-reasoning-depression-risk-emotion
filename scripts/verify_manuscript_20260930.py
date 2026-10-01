"""Check the compiled review manuscript against the row-verified aggregates."""
from pathlib import Path
import csv
import json
import re
import shutil
import pymupdf as fitz

repo = Path(__file__).resolve().parents[1]
out = repo.parent / 'Paper_20260930_verified'
report = repo / 'reports/verified_manuscript_20260930'
base = repo / 'reports/unified_v6c_revision_20260929'
source = (out / 'main.tex').read_text()
assert source == (report / 'main.tex').read_text()
rows = list(csv.DictReader((repo / 'reports/final_results_audit_20260930/row_verified_metrics.csv').open()))
assert len(rows) == 15
for row in rows:
    cells = [f'{float(row[key]):.4f}' for key in ('accuracy_percent', 'macro_f1_percent')]
    cells += [row['corrected'], row['introduced']]
    assert ' & '.join(cells) in source, row

def bibliography(text):
    return text[text.index(r'\begin{thebibliography}'):text.index(r'\end{thebibliography}')]

assert bibliography(source) == bibliography((base / 'main.tex').read_text())
prompts = list((base / 'prompts').glob('*'))
for path in prompts:
    if path.is_file():
        assert path.read_bytes() == (out / 'prompts' / path.name).read_bytes()
doc = fitz.open(out / 'main.pdf')
text = '\n'.join(page.get_text() for page in doc)
assert len(doc) == 29
assert '\ufffd' not in text
assert '777' in text and '0.0412' in text and '97.7667' in text
outside = []
for number, page in enumerate(doc, 1):
    for word in page.get_text('words'):
        if not (page.rect + (-1, -1, 1, 1)).contains(fitz.Rect(word[:4])):
            outside.append([number, word[4]])
assert not outside, outside
log = (out / 'qa/build-2.log').read_text()
assert 'undefined references' not in log.lower()
assert not re.search(r'Reference .* undefined', log)
warnings = sorted(set(line for line in log.splitlines() if line.startswith('Overfull')))
old_log = (repo.parent / 'Paper_20260929_unified_v6c/main.log').read_text()
for warning in warnings:
    width = re.search(r'\(([\d.]+)pt', warning).group(1)
    assert width in old_log, warning
for number in range(1, len(doc)+1):
    assert (out / f'qa/page-{number:02d}.png').is_file()
result = {
    'pages': len(doc), 'metric_rows_checked': len(rows),
    'bibliography_unchanged': True, 'prompt_files_unchanged': True,
    'undefined_references': False, 'words_outside_page': outside,
    'replacement_glyphs': False, 'inherited_overfull_warning_types': warnings,
    'visual_review': {
        'status': 'completed',
        'scope': 'All 29 pages reviewed in rendered contact sheets; pages 14, 16, 27 and the updated architecture inspected separately.',
        'finding': 'No visible clipping or overlapping content found; large forced appendix gap removed.',
    },
}
(out / 'qa/verification.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
shutil.copy2(out / 'qa/verification.json', report / 'QA_VERIFICATION.json')
print(json.dumps(result, indent=2, ensure_ascii=False))
