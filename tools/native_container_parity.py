"""Sequential same-checkpoint native/GGUF first-logit and continuation gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import re
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--native-config',type=Path,required=True)
    ap.add_argument('--oracle-config',type=Path,required=True)
    ap.add_argument('--tokens',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    n=json.loads(a.native_config.read_text(encoding='utf8'))
    o=json.loads(a.oracle_config.read_text(encoding='utf8'))
    def value(flag):
        return o['args'][o['args'].index(flag)+1]
    common=['--tokens-file',str(a.tokens.resolve()),'--max-context','2048','--max-new','32',
            '--prefill','64','--spec','2','--suffix-draft','0','--expert-cache','7000',
            '--expert-profile',str(Path('data/expert-profile.bin').resolve()),'--pcie-frac','1',
            '--pcie-mode','dma','--no-prefill-borrow','--greedy','--stats','--check-logits',
            '--adapt-swaps','0','--kv','fp16']
    native=['--safetensors',n['tokenizer']]
    legacy=[]
    for flag in ['--pack','--native','--native-dense-gguf','--native-head-gguf','--embd-gguf','--ple-gguf']:
        legacy += [flag,value(flag)]
    legacy += ['--resident-experts']
    results={}
    for label,extra in [('native',native),('oracle',legacy)]:
        path=a.output/label; path.mkdir()
        env=os.environ.copy(); env.update(n['env'])
        env.update({'STRATA_VERIFY_STAGING_BLOBS':'128','STRATA_RESIDENT_PIN':'1','STRATA_STAGE_PIN':'0',
                    'STRATA_RESIDENT_HEADROOM_GIB':'4',
                    'STRATA_DUMP_VERIFY_DIAGNOSTICS':str((path/'layers.f32').resolve()),
                    'STRATA_DUMP_FIRST_LOGITS':str((path/'logits.f32').resolve())})
        cmd=[n['exe'],*extra,*common]
        (path/'command.json').write_text(json.dumps({'command':cmd,'env':{k:v for k,v in env.items() if k.startswith('STRATA_')}},indent=2),encoding='utf8')
        t=time.monotonic()
        with (path/'stdout.txt').open('w',encoding='utf8') as out,(path/'stderr.txt').open('w',encoding='utf8') as err:
            p=subprocess.run(cmd,cwd=n['cwd'],env=env,stdout=out,stderr=err,timeout=900)
        results[label]={'returncode':p.returncode,'wall_s':time.monotonic()-t}
        output=(path/'stdout.txt').read_text(encoding='utf8')
        match=re.search(r'^output\s+: (.*)$',output,re.M)
        results[label]['tokens']=list(map(int,match.group(1).split())) if match else []
        print(label,results[label],flush=True)
        if p.returncode: raise RuntimeError(f'{label} failed; inspect {path}')
    x=np.fromfile(a.output/'native/logits.f32',dtype='<f4').astype('float64')
    y=np.fromfile(a.output/'oracle/logits.f32',dtype='<f4').astype('float64')
    assert len(x)==len(y)==248320 and np.isfinite(x).all() and np.isfinite(y).all()
    results['logits']={'count':len(x),'max_abs':float(abs(x-y).max()),'rms':float(np.sqrt(np.mean((x-y)**2))),
                       'argmax':[int(x.argmax()),int(y.argmax())],'cosine':float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))}
    stride=4*2560+2560+4+10*2560+10
    dx=np.fromfile(a.output/'native/layers.f32',dtype='<f4').reshape(48,stride)
    dy=np.fromfile(a.output/'oracle/layers.f32',dtype='<f4').reshape(48,stride)
    results['layers']=[]
    for layer in range(48):
        row={'layer':layer}
        for name,lo,hi in [('residual',0,10240),('mlp_output',10240,12800),('inject',12800,12804),('expert_outputs',12804,38404)]:
            u=dx[layer,lo:hi].astype('float64');v=dy[layer,lo:hi].astype('float64')
            row[name]={'max_abs':float(abs(u-v).max()),'relative_l2':float(np.linalg.norm(u-v)/max(np.linalg.norm(v),1e-30))}
        row['router_ids_equal']=bool(np.array_equal(dx[layer,-10:].view('int32'),dy[layer,-10:].view('int32')))
        results['layers'].append(row)
    (a.output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    print(json.dumps(results['logits']),flush=True)
    assert results['logits']['max_abs']<0.02 and x.argmax()==y.argmax(), 'container forward differs'
    assert results['native']['tokens']==results['oracle']['tokens'], 'continuation differs'


if __name__=='__main__': main()
