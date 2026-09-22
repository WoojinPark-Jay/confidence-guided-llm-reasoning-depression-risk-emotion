import ast
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

from build_final_unified_phase2_notebooks import OUT_DIR
from build_mixed_sd_countercheck_notebook import find_cell
from llama2_context_resume_patch import BUDGET, MIGRATION


def main():
    path = OUT_DIR / '12_reddit_llama2_direct_final_sd_colab.ipynb'
    nb = json.loads(path.read_text())
    run = ''.join(find_cell(nb, 'LEGACY_IMPLEMENTATION_SHA256 =')['source'])
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(run).body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ('IMPLEMENTATION_SHA256', 'LEGACY_IMPLEMENTATION_SHA256')}
    old = json.loads(subprocess.check_output(['git', 'show', '15589fa:notebooks/colab/final/' + path.name]))
    old_run = ''.join(find_cell(old, 'IMPLEMENTATION_SHA256 =')['source'])
    assert values['LEGACY_IMPLEMENTATION_SHA256'] == ast.literal_eval(ast.parse(old_run).body[0].value)
    ns = {}
    exec('def budget(model, input_ids, max_new_tokens):\n' + BUDGET +
         '    return max_new_tokens\n', ns)
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096))
    for length, expected in [(1000,1024), (3072,1024), (3500,596), (4095,1)]:
        assert ns['budget'](model, SimpleNamespace(shape=[1,length]),1024) == expected
    assert ns['budget'](model, SimpleNamespace(shape=[1,4096]),1024)['context_capacity_exceeded']
    with tempfile.TemporaryDirectory() as directory:
        manifest_path = Path(directory) / 'experiment_manifest.json'
        legacy = {'implementation_sha256': values['LEGACY_IMPLEMENTATION_SHA256'], 'input': 'fixed'}
        manifest_path.write_text(json.dumps(legacy))
        state = dict(manifest_path=manifest_path, json=json,
                     manifest={'implementation_sha256': values['IMPLEMENTATION_SHA256'], 'input':'fixed'},
                     LEGACY_IMPLEMENTATION_SHA256=values['LEGACY_IMPLEMENTATION_SHA256'],
                     RUN_METHODS=['direct'], METHOD_CONFIG={'direct':{'path':'unused'}},
                     load_results=lambda path: SimpleNamespace(empty=True))
        exec(MIGRATION,state)
        assert json.loads(manifest_path.read_text()) == state['manifest']
        assert json.loads(manifest_path.with_name('experiment_manifest.before_context_fix.json').read_text()) == legacy
    print('PASS: original manifest hash, context boundaries, migration and manifest backup.')


if __name__ == '__main__':
    main()
