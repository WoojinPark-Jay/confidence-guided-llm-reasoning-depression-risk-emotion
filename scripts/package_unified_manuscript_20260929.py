"""Validate and package the compiled review manuscript; no experiment is rerun."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile
from pypdf import PdfReader

ROOT = Path('/Users/woojinpark/Documents/헬스케어 논문')
OUT = ROOT/'Paper_20260929_unified_v6c'
REPO = ROOT/'repo_final_push_nolfs'
source = (OUT/'main.tex').read_text()
figure_stems = {Path(p).stem for p in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', source)}
log = (OUT/'main.log').read_text()
assert 'undefined references' not in log
assert 'Missing character:' not in log
assert all(s not in source for s in ['96.69', '81.33', '1.7706', 'v6_max_prompt'])
assert r'\lstinputlisting{prompts/sd_implement.txt}' in source
assert source.count(r'\begin{thebibliography}') == 1
for target in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',source):
    assert (OUT/target).exists(), target
for target in re.findall(r'\\lstinputlisting\{([^}]+)\}',source):
    assert (OUT/target).exists(), target
pdf = PdfReader(OUT/'main.pdf')
text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
assert '??' not in text
for s in ['97.2000','92.6667','1.480195','SELECT','Counterfactual']:
    assert s in text, s
for figure in (OUT/'figures').glob('*.pdf'):
    assert len(''.join(p.extract_text() or '' for p in PdfReader(figure).pages)) > 30, figure
readme = '''# Unified final pipeline manuscript (2026-09-29)

Open `main.tex` as the main file in Overleaf and compile with pdfLaTeX.
All referenced figures are vector PDFs with selectable text. The original
reviewed literature and reference entries are preserved. The final SD protocol
is v6c; the former independent-countercheck SD is an appendix variant only.

This is an updated review manuscript, not a declaration that every release
check is complete. Read `UPDATE_AND_REMAINING_KO.md` before submission. The
remaining work includes final row-level SD output reconciliation, aligned
between-method statistical comparisons, and author information.

The evidence folder contains aggregate results and calibration provenance;
it is not a redistribution of the Reddit post corpus. Prompt text is extracted
from notebook string literals, and recorded plans are extracted from saved
notebook output. Nominal generation budgets are disclosed separately.

The existing template produces header/footer overfull-box warnings that also
occurred before this revision. These are distinct from new body/table overflow;
page rendering is reviewed separately.
'''
report=REPO/'reports/unified_v6c_revision_20260929'
for name in ['README.md','UPDATE_AND_REMAINING_KO.md','COLLABORATOR_FILE_REQUEST_KO.md']:
    if (report/name).exists():
        shutil.copy2(report/name,OUT/name)
if not (OUT/'README.md').exists():
    (OUT/'README.md').write_text(readme)
package_readme = (OUT/'README.md').read_text().replace(
    '이 폴더는 공동 리뷰용 원고·문서·그림·프롬프트·집계 근거의 스냅샷이다. 단독 Overleaf 프로젝트가 아니며, 컴파일에는 별도 배포한 Overleaf ZIP의 IEEE Access 클래스와 지원 파일이 필요하다.',
    '이 패키지는 IEEE Access 클래스와 지원 파일을 포함한 Overleaf 프로젝트다. main.tex를 메인 파일로 지정하고 pdfLaTeX으로 컴파일한다. 공동 리뷰 문서·그림·프롬프트·집계 근거도 함께 포함한다.')
(OUT/'README.md').write_text(package_readme)
release = ROOT/'CGSLR_IEEE_Access_Unified_v6c_2026_09_29.pdf'
shutil.copy2(OUT/'main.pdf',release)
archive=ROOT/'CGSLR_IEEE_Access_Overleaf_Unified_v6c_2026_09_29.zip'
allowed={'.tex','.bib','.cls','.sty','.fd','.map','.pfb','.tfm','.png'}
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file() and (p.suffix in allowed or p.suffix == '.md'):
            z.write(p,p.relative_to(OUT))
    for sub in ['figures','prompts','evidence']:
        for p in sorted((OUT/sub).glob('*')):
            if p.is_file() and ' 2.' not in p.name and (sub != 'figures' or p.stem in figure_stems):
                z.write(p,p.relative_to(OUT))
report=REPO/'reports/unified_v6c_revision_20260929'
report.mkdir(parents=True,exist_ok=True)
for name in ['main.tex','UPDATE_AND_REMAINING_KO.md','MANUSCRIPT_FULL_DIFF.md','ARCHITECTURE_EDIT_GUIDE_KO.md','COLLABORATOR_FILE_REQUEST_KO.md','VISUAL_RESTORATION_KO.md','EXPOSITION_UPGRADE_KO.md','SD_CONTRIBUTION_CLARITY_KO.md','SD_CLARITY_TEXT_DIFF.md']:
    shutil.copy2(OUT/name,report/name)
for sub in ['figures','prompts','evidence']:
    shutil.copytree(OUT/sub,report/sub,dirs_exist_ok=True)
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [release,archive]}
(OUT/'release_sha256.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'pages':len(pdf.pages),'pdf':str(release),'overleaf':str(archive),'report':str(report)},ensure_ascii=False,indent=2))
