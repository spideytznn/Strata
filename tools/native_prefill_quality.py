"""Teacher-forced prompt-capacity comparison, retaining original weight formats.

The reference is the accepted 4096 profile, not an unquantized model oracle.
State/logit diagnostics make this a quality run, never a performance run.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from native_quality_suite import TEXTS
from native_path_matrix import compare
from safetensors_tokenizer import SafetensorsTokenizer
from serve.frontend import ChatTemplate
from serve.server import StrataEngine, Vision, child_env, vision_env


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capacities', nargs='+', type=int, default=[4096, 6144, 8192])
    p.add_argument('--reserve-mib', type=int, default=1024)
    p.add_argument('--context-tokens', type=int, default=24000)
    p.add_argument('--positions', type=int, default=0, help='0 uses every answer token.')
    a = p.parse_args()
    assert a.capacities[0] == 4096 and all(c > 0 and c % 256 == 0 for c in a.capacities)
    assert len(set(a.capacities)) == len(a.capacities) and a.context_tokens >= max(a.capacities)
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(a.config.read_text(encoding='utf-8-sig'))
    tok = SafetensorsTokenizer.from_directory(base['tokenizer'])
    template = ChatTemplate(Path(base['chat_template']))
    docpath = ROOT / 'docs/DETAILS.md'
    docids = tok.encode(docpath.read_text(encoding='utf-8'))
    document = tok.decode((docids * ((a.context_tokens+len(docids)-1)//len(docids)))[:a.context_tokens])
    cases = []
    for category,question,answer in TEXTS:
        head = tok.encode(template.render([{'role':'user','content':'Reference document:\n'+document+'\n'+question}],enable_thinking=False),parse_special=True)
        targets = tok.encode(answer)
        if a.positions: targets = targets[:a.positions]
        for index,target in enumerate(targets):
            cases.append((f'{category}-{index}', head+targets[:index], target))
    result = dict(status='running', args=vars(a) | {'config':str(a.config),'output':str(a.output)},
                  source_doc_sha256=hashlib.sha256(docpath.read_bytes()).hexdigest(),
                  source_config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
                  note=__doc__, runs=[])
    def save():
        (a.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save()
    for capacity in a.capacities:
        active = subprocess.check_output(['powershell.exe','-NoProfile','-Command',
            "@(Get-Process -Name 'strata*' -ErrorAction SilentlyContinue).Count"],text=True)
        assert int(active.strip() or '0') == 0, 'Another Strata process is running'
        cfg = copy.deepcopy(base)
        for flag,value in [('--prefill',capacity),('--vram-reserve-mib',a.reserve_mib),('--spec',3),('--mtp-max-t',1),('--spec-min-p',0)]:
            cfg['args'][cfg['args'].index(flag)+1] = str(value)
        folder = a.output/f'capacity{capacity}'
        folder.mkdir()
        env = child_env(cfg)
        env['STRATA_DUMP_FIRST_LOGITS'] = str(folder/'logits.f32')
        row = dict(capacity=capacity,config=cfg,requests=[],exe_sha256=hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest())
        result['runs'].append(row)
        engine = vision = None
        save()
        print('START teacher capacity',capacity,flush=True)
        try:
            with (folder/'vision.log').open('w',encoding='utf-8') as vision_log:
                vision = Vision(cfg['vision'],log=vision_log,env=vision_env(cfg,env))
                engine = StrataEngine(cfg['exe'],cfg['args'],cwd=cfg['cwd'],log=str(folder/'engine.log'),env=env)
                row['info'] = dict(engine.info)
                assert row['info']['prefill_chunk'] == capacity
                for j,(name,ids,target) in enumerate(cases):
                    out = [v for v in engine.generate(ids,1,{'temperature':0},threading.Event()) if v is not None]
                    assert len(out) == 1 and engine.last['file_blobs'] == engine.last['file_mb'] == 0
                    path = folder/f'logits.f32.{j}'
                    logits = np.fromfile(path,dtype='<f4')
                    assert logits.shape == (248320,) and np.isfinite(logits).all()
                    x = logits.astype('float64')
                    record = dict(name=name,target=target,input_tokens=len(ids),
                                  input_sha256=hashlib.sha256(np.asarray(ids,dtype='<i4').tobytes()).hexdigest(),
                                  logits_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  nll=float(np.log(np.exp(x-x.max()).sum())+x.max()-x[target]),
                                  argmax=int(x.argmax()),**engine.last)
                    if capacity != a.capacities[0]:
                        ref = result['runs'][0]['requests'][j]
                        assert ref['name'] == name and ref['input_sha256'] == record['input_sha256']
                        record['comparison'] = compare(np.fromfile(a.output/f'capacity4096/logits.f32.{j}',dtype='<f4'),logits)
                    row['requests'].append(record)
                    save()
        except Exception as error:
            row.update(status='failed',error=repr(error))
            save()
            raise
        finally:
            if engine: engine.close()
            if vision: vision.shutdown()
        log = (folder/'engine.log').read_text(encoding='utf-8',errors='replace')
        assert 'post-residency expert source bytes=0' in log and 'post-residency MTP source bytes=0' in log
        summary = dict(positions=len(row['requests']),mean_nll=float(np.mean([q['nll'] for q in row['requests']])))
        if capacity != a.capacities[0]:
            summary.update(mean_nll_delta=summary['mean_nll']-result['runs'][0]['summary']['mean_nll'],
                           mean_kl=float(np.mean([q['comparison']['kl_reference_candidate'] for q in row['requests']])),
                           top1_agreement=float(np.mean([q['comparison']['argmax_equal'] for q in row['requests']])))
        row.update(status='pass',summary=summary)
        save()
        print(capacity,summary,flush=True)
    result['status'] = 'pass'
    save()


if __name__ == '__main__':
    main()
