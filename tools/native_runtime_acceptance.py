"""Native MTP/cache/stop acceptance on private engines, loaded sequentially."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from serve.server import StrataEngine,child_env
from serve.frontend import ChatTemplate
from safetensors_tokenizer import SafetensorsTokenizer
from conversation_cache_parity import state_hashes


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--variants',default='0,1,2,4')
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text(encoding='utf8'))
    tok=SafetensorsTokenizer.from_directory(cfg['tokenizer'])
    template=ChatTemplate(Path(cfg['chat_template']))
    def prompt(text):
        return tok.encode(template.render([{'role':'user','content':text}],enable_thinking=False),parse_special=True)
    prompts={
        'english':prompt('Explain why the sky is blue in two sentences.'),
        'chinese':prompt('用中文解释二分查找的前提和时间复杂度，给出一个简短例子。'),
        'code':prompt('Write a Python function returning the first index of a target in a sorted list, or -1.'),
        'math':prompt('Calculate (17 * 23) - 19. Give only the answer.'),
        'eos':prompt('Reply with exactly the single word Blue and stop.'),
        'A':prompt('Remember this exact code: ZEBRA-417. Say OK.'),
        'B':prompt('Remember this different code: ORBIT-926. Say OK.')}
    results={'config':cfg,'variants':{}}
    def save(): (a.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    baseline={}
    for k in map(int,a.variants.split(',')):
        folder=a.output/f'k{k}';folder.mkdir()
        args=list(cfg['args'])
        def setting(flag,value):
            if flag in args: args[args.index(flag)+1]=str(value)
            else: args.extend([flag,str(value)])
        setting('--spec',max(2,k+1));setting('--mtp-max-t',max(1,k+1));setting('--spec-min-p',0)
        setting('--expert-cache',1024);setting('--adapt-swaps',0)
        setting('--conversation-cache-mib',2048);setting('--conversation-cache-slots',4)
        setting('--prompt-cache',6)
        if k: args.extend(['--mtp','native'])
        env=child_env(cfg);env['STRATA_STATE_HASH']='1';env['STRATA_MTP_FULL_HEAD']='1'
        log=folder/'engine.log'
        record={'requests':[]};results['variants'][str(k)]=record;save()
        (folder/'command.json').write_text(json.dumps({'args':args,'env':{x:y for x,y in env.items() if x.startswith('STRATA_')}},indent=2),encoding='utf8')
        engine=StrataEngine(cfg['exe'],args,cwd=cfg['cwd'],log=str(log.resolve()),env=env)
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
            for name in ('english','chinese','code','math','eos'): generate(name,prompts[name])
            generate('sampled',prompts['english'],32,{'temperature':0.8,'top_p':0.9,'seed':42})
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
            cancel=threading.Event();got=[]
            stream=engine.generate(prompts['code'],128,{'temperature':0},cancel)
            try:
                for v in stream:
                    if v is not None: got.append(v)
                    if len(got)==5: cancel.set();break
            finally: stream.close()
            record['cancel_visible']=got;save()
            generate('after-cancel',prompts['code']+got,16)
        finally:
            engine.close()
        text=log.read_text(encoding='utf8')
        record['state_hashes']=state_hashes(text)
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
        record['status']='pass';save()
    results['status']='pass';save()
    print('PASS: native MTP, greedy/seeded sampling, output caps, EOS, A/B/A, disk restore, cancel/resume, zero expert file reads',flush=True)


if __name__=='__main__':main()
