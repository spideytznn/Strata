"""Native MTP/cache/stop acceptance on private engines, loaded sequentially."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from serve.server import StrataEngine,Vision,child_env,vision_env,sampling_defaults_from_config,EOS_IDS
from serve.frontend import ChatTemplate
from safetensors_tokenizer import SafetensorsTokenizer
from conversation_cache_parity import state_hashes


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--variants',default='0,1,2,4')
    ap.add_argument('--operator-cache',action='store_true',help='Preserve automatic cache sizing and load configured vision first.')
    ap.add_argument('--spec-min-p',type=float,default=0)
    ap.add_argument('--official-new',type=int,default=0,help='Also compare seeded official thinking outputs at this limit.')
    a=ap.parse_args()
    assert 0 <= a.spec_min_p <= 1 and a.official_new >= 0
    variants=list(map(int,a.variants.split(',')))
    assert variants and variants[0]==0 and len(variants)==len(set(variants)) and all(0<=k<=7 for k in variants)
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text(encoding='utf8'))
    tok=SafetensorsTokenizer.from_directory(cfg['tokenizer'])
    template=ChatTemplate(Path(cfg['chat_template']))
    def prompt(text):
        return tok.encode(template.render([{'role':'user','content':text}],enable_thinking=False),parse_special=True)
    texts={
        'english':'Explain why the sky is blue in two sentences.',
        'chinese':'用中文解释二分查找的前提和时间复杂度，给出一个简短例子。',
        'code':'Write a Python function returning the first index of a target in a sorted list, or -1.',
        'math':'Calculate (17 * 23) - 19. Give only the answer.',
        'eos':'Reply with exactly the single word Blue and stop.',
        'A':'Remember this exact code: ZEBRA-417. Say OK.',
        'B':'Remember this different code: ORBIT-926. Say OK.'}
    prompts={name:prompt(text) for name,text in texts.items()}
    results={'config':cfg,'config_path':str(a.config),'operator_cache':a.operator_cache,
             'spec_min_p':a.spec_min_p,'official_new':a.official_new,'variants':{}}
    def save(): (a.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    baseline={}
    for k in variants:
        assert 0 <= k <= 7
        active=subprocess.check_output(['powershell.exe','-NoProfile','-Command',
            "@(Get-Process -Name 'strata*' -ErrorAction SilentlyContinue).Count"],text=True)
        assert int(active.strip() or '0')==0,'Another Strata process is running'
        folder=a.output/f'k{k}';folder.mkdir()
        args=list(cfg['args'])
        def setting(flag,value):
            if flag in args: args[args.index(flag)+1]=str(value)
            else: args.extend([flag,str(value)])
        setting('--spec',max(2,k+1));setting('--mtp-max-t',max(1,k+1));setting('--spec-min-p',a.spec_min_p)
        if not a.operator_cache:
            setting('--expert-cache',1024);setting('--adapt-swaps',0)
            setting('--conversation-cache-mib',2048);setting('--conversation-cache-slots',4)
            setting('--prompt-cache',6)
        if k: args.extend(['--mtp','native'])
        env=child_env(cfg);env['STRATA_STATE_HASH']='1'
        if '--mtp-draft-vocab' not in args: env['STRATA_MTP_FULL_HEAD']='1'
        else: env.pop('STRATA_MTP_FULL_HEAD',None)
        if not a.operator_cache: env['STRATA_NATIVE_PREFILL_FIRST']='0'
        log=folder/'engine.log'
        record={'requests':[], 'exe_sha256':hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest()}
        results['variants'][str(k)]=record;save()
        (folder/'command.json').write_text(json.dumps({'args':args,'env':{x:y for x,y in env.items() if x.startswith('STRATA_')}},indent=2),encoding='utf8')
        vision=None
        vision_log=(folder/'vision.log').open('w',encoding='utf8')
        try:
            if a.operator_cache and cfg.get('vision'):
                vision=Vision(cfg['vision'],log=vision_log,env=vision_env(cfg,env))
            engine=StrataEngine(cfg['exe'],args,cwd=cfg['cwd'],log=str(log.resolve()),env=env)
        except BaseException:
            if vision: vision.shutdown()
            vision_log.close()
            raise
        record['info']=dict(engine.info)
        def generate(name,ids,limit=48,sampling=None):
            t=time.monotonic()
            out=[v for v in engine.generate(ids,limit,sampling or {'temperature':0},threading.Event()) if v is not None]
            row={'name':name,'ids':out,'text':tok.decode(out),'wall_s':time.monotonic()-t,**engine.last}
            record['requests'].append(row);save()
            assert 0<len(out)<=limit and row['generated']==len(out),f'{name}: output boundary'
            assert row['file_blobs']==0 and row['file_mb']==0,f'{name}: expert file read'
            if k==0: baseline[name]=out
            else: assert out==baseline[name],f'k{k} {name}: tokens differ from k0'
            print(f'k{k} {name}: {len(out)} tokens, reused={row["reused"]}, drafts={row["drafts_accepted"]}/{row["drafts_offered"]}',flush=True)
            return out
        try:
            assert record['info']['weight_source']=='safetensors'
            assert record['info']['expert_ram_bytes']==record['info']['expert_cuda_pinned_bytes']>0
            if a.operator_cache:
                assert record['info']['prefill_chunk']==int(args[args.index('--prefill')+1])
                assert record['info']['vram_free_mib']>=512
            for name in ('english','chinese','code','math','eos'):
                out=generate(name,prompts[name])
                if name=='eos': assert len(out)<48 and out[-1] in EOS_IDS,'EOS did not terminate before the cap'
            generate('sampled',prompts['english'],32,{'temperature':0.8,'top_p':0.9,'seed':42})
            if a.official_new:
                for category in ('english','chinese','code'):
                    ids=tok.encode(template.render([{'role':'user','content':texts[category]}],enable_thinking=True),parse_special=True)
                    generate(f'official-{category}',ids,a.official_new,{**sampling_defaults_from_config(cfg),'seed':9950})
            for limit in (1,2,3,7,9): generate(f'limit{limit}',prompts['english'],limit)
            head=generate('A',prompts['A'],1)
            suffix=tok.encode('<|im_end|>\n<|im_start|>user\nWhat code did I give you?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',parse_special=True)
            continuation=prompts['A']+head+suffix
            generate('B',prompts['B'],1)
            generate('A+',continuation,16)
            snapshot=folder/'session.bin'
            record['saved']=engine.session_file('save',str(snapshot.resolve()));save()
            generate('B-again',prompts['B'],1)
            record['restored']=engine.session_file('restore',str(snapshot.resolve()));save()
            generate('A+-restored',continuation,16)
            assert record['requests'][-1]['reused']>0,'disk snapshot was not reused'
            assert record['requests'][-1]['ids']==record['requests'][-3]['ids'],'own restored continuation differs'
            cancel=threading.Event();got=[]
            stream=engine.generate(prompts['code'],128,{'temperature':0},cancel)
            try:
                for v in stream:
                    if v is not None: got.append(v)
                    if len(got)==5: cancel.set();break
            finally: stream.close()
            record['cancel_visible']=got;save()
            assert got==baseline['code'][:5],'cancelled prefix differs'
            generate('after-cancel',prompts['code']+got,16)
            assert record['requests'][-1]['ids']==baseline['code'][5:21],'cancel/resume continuation differs'
        finally:
            engine.close()
            if vision: vision.shutdown()
            vision_log.close()
        text=log.read_text(encoding='utf8')
        record['state_hashes']=state_hashes(text)
        assert len(record['state_hashes'])==len(record['requests'])+1,'missing completed/cancelled state fingerprints'
        indices={r['name']:i for i,r in enumerate(record['requests'])}
        assert record['state_hashes'][indices['A+']]==record['state_hashes'][indices['A+-restored']], 'own restored main state differs'
        # The cancellation request can already have verified a window when STOP
        # reaches the engine. Compare the completed requests, then its resumed
        # continuation; omit only that deliberately interrupted request's state.
        complete=record['state_hashes'][:-2]+record['state_hashes'][-1:]
        if k==0: baseline['completed_states']=complete
        else:
            record['main_state_equal']=complete==baseline['completed_states']
            save()
            assert record['main_state_equal'],f'k{k}: completed main-model state differs'
        assert 'post-residency expert source bytes=0' in text,'missing source I/O barrier evidence'
        assert 'post-residency MTP source bytes=0' in text,'missing MTP source I/O barrier evidence'
        record['status']='pass';save()
    results['status']='pass';save()
    print('PASS: native MTP, greedy/seeded sampling, output caps, EOS, A/B/A, disk restore, cancel/resume, zero expert file reads',flush=True)


if __name__=='__main__':main()
