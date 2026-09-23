import json
import tempfile
from pathlib import Path
from build_final_unified_phase2_notebooks import OUT_DIR
from build_mixed_sd_countercheck_notebook import find_cell


def main():
    pairs = [('05_1_final_unified_phase2_llama2_cot_matched_colab_20260914.ipynb',
              '13_reddit_llama2_cot_1024_colab.ipynb'),
             ('06_1_mixed_emotion_llama2_cot_matched_colab.ipynb',
              '14_mixed_emotion_llama2_cot_1024_colab.ipynb')]
    with tempfile.TemporaryDirectory() as d:
        for oldname, newname in pairs:
            old = json.loads((OUT_DIR / oldname).read_text())
            new = json.loads((OUT_DIR / newname).read_text())
            for cell in new['cells']:
                src = ''.join(cell['source'])
                if cell['cell_type'] == 'code' and not any(l.startswith('%') for l in src.splitlines()):
                    compile(src,newname,'exec')
            for marker in ['# Both Phase 2 notebooks', 'drive.mount("/content/drive")']:
                assert find_cell(old,marker)['source'] == find_cell(new,marker)['source']
            spaces=[]
            captures=[]
            for nb in [old,new]:
                ns={'DRIVE_OUTPUT_ROOT':Path(d), 'json':json}
                exec(''.join(find_cell(nb,'def run_llama2_cot_one(')['source']),ns)
                calls=[]
                def generate(tokenizer,model,messages,**kwargs):
                    calls.append(([dict(m) for m in messages],kwargs))
                    return dict(text='Final label: Happy',input_tokens=10,generated_tokens=4,
                                elapsed_seconds=1,hit_token_limit=False)
                ns.update(chat_generate=generate,stable_seed=lambda *a:42,parse_final_label=lambda s:'Happy')
                result=ns['run_llama2_cot_one'](None,None,dict(example_id='test',phase1_label='Neutral',phase2_original_text='test text'))
                assert len(calls)==5 and result['generated_tokens_total']==20
                spaces.append(ns);captures.append(calls)
            assert spaces[0]['LLAMA2_COT_REQUESTS']==spaces[1]['LLAMA2_COT_REQUESTS']
            assert spaces[0]['LLAMA2_CLASSIFICATION_POLICY']==spaces[1]['LLAMA2_CLASSIFICATION_POLICY']
            assert [m for m,k in captures[0]]==[m for m,k in captures[1]]
            assert all(k['max_new_tokens']==1024 and not k['do_sample'] for m,k in captures[1])
            assert spaces[0]['LLAMA2_OUTPUT_DIR']!=spaces[1]['LLAMA2_OUTPUT_DIR']
            assert len(json.loads(result['stage_logs_json']))==5
            print('PASS',newname,'unchanged inputs/prompts, five stages, 1024 cap, logs, separate output.')


if __name__=='__main__':
    main()
