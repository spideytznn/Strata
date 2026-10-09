"""End-to-end native path matrix. One resident engine at a time; preserve raw evidence."""
import argparse,hashlib,json,os,sys,time,threading,platform,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from serve.server import StrataEngine,child_env
from serve.frontend import ChatTemplate
from safetensors_tokenizer import SafetensorsTokenizer
from native_telemetry import Telemetry

CASES={
 'reference':{},'host16':{'STRATA_NATIVE_REGISTER_GIB':'16'},
 'host24':{'STRATA_NATIVE_REGISTER_GIB':'24'},'host32':{'STRATA_NATIVE_REGISTER_GIB':'32'},
 'host40':{'STRATA_NATIVE_REGISTER_GIB':'40'},
 'tc1':{'STRATA_NVFP4_TC':'1'},'tc2':{'STRATA_NVFP4_TC':'2'},'tc3':{'STRATA_NVFP4_TC':'3'},
 # Percentages concern cold experts only; hot experts remain on the GPU.
 'cpu50':{'pcie':0.5},'cpu100':{'pcie':0},
 'adaptive':{'adapt':True},
 'mtp1':{'mtp':1},'mtp2':{'mtp':2},'mtp4':{'mtp':4},
 'prefill-int8':{'STRATA_PREFILL_NVFP4':'w4a8'},
 'prefill-fp4x2':{'STRATA_PREFILL_NVFP4':'w4a4x2'},
 'prefill-fp4':{'STRATA_PREFILL_NVFP4':'w4a4'},
}
def compare(x,y):
 x=np.asarray(x,dtype='float64');y=np.asarray(y,dtype='float64')
 assert x.shape==y.shape==(248320,) and np.isfinite(x).all() and np.isfinite(y).all()
 lp=x-x.max();lp-=np.log(np.exp(lp).sum());lq=y-y.max();lq-=np.log(np.exp(lq).sum())
 return {'max_abs':float(abs(x-y).max()),'rms':float(np.sqrt(np.mean((x-y)**2))),
         'kl_reference_candidate':float(np.exp(lp)@(lp-lq)), 'argmax_equal':bool(x.argmax()==y.argmax())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--cases',default='reference,tc1,tc2,tc3,cpu50,cpu100,adaptive,mtp1,mtp2,mtp4,prefill-fp4x2')
 p.add_argument('--rounds',type=int,default=3);p.add_argument('--new',type=int,default=128)
 p.add_argument('--long',action='store_true');p.add_argument('--cache-slots',type=int,default=4500)
 p.add_argument('--reference-output',type=Path)
 p.add_argument('--http',action='store_true')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cfg=json.loads(a.config.read_text(encoding='utf8'));tok=SafetensorsTokenizer.from_directory(cfg['tokenizer'])
 template=ChatTemplate(Path(cfg['chat_template']))
 def prompt(s):return tok.encode(template.render([{'role':'user','content':s}],enable_thinking=False),parse_special=True)
 tasks=['Explain binary search with a complete Python implementation and three worked examples.',
        '详细解释归并排序，给出 Python 实现和三个测试例子，说明每一步的结果。',
        'Write a detailed tutorial on testing an LRU cache, with Python code and examples.']
 requests=[]
 for i in range(a.rounds):
  ids=prompt(tasks[i%len(tasks)]+f' Test case {i+1}.')
  for mode in ('cold','warm'):requests.append((f'short-{i}-{mode}',ids,a.new))
 if a.long:
  head=tok.encode('<|im_start|>user\n',parse_special=True)
  tail=tok.encode('\nSummarize this text in detail.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',parse_special=True)
  unit=tok.encode('Binary search repeatedly halves a sorted search interval. Compare the middle element and retain the relevant half. ')
  # The server checkpoints before the final assistant header. Make that
  # boundary exactly 8192, so the completed batched segment is really 8192.
  turn=tok.encode('<|im_start|>',parse_special=True)[0]
  tail_turn=max(i for i,v in enumerate(tail) if v==turn)
  count=8192-len(head)-tail_turn
  ids=head+(unit*((count+len(unit)-1)//len(unit)))[:count]+tail
  assert max(i for i,v in enumerate(ids) if v==turn)==8192
  requests.extend((f'long8192chunk-{mode}',ids,a.new) for mode in ('cold','warm'))
 (a.output/'requests.json').write_text(json.dumps(requests,ensure_ascii=False,indent=2),encoding='utf8')
 result={'hardware':{'platform':platform.platform(),'processor':platform.processor()},'config':cfg,
   'binary_sha256':hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),'started_unix_s':time.time(),
   'notes':{'cpu_cases':'pcie-frac splits cold experts; GPU cache hits still run on GPU. Actual CPU entries are recorded.',
            'rounds':'Each round is a distinct short prompt with one cold and one warm run; the long prompt has one cold/warm pair.'},'cases':{}}
 def save():(a.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 for case in a.cases.split(','):
  opts=CASES[case];folder=a.output/case;folder.mkdir();args=list(cfg['args'])
  def setting(flag,v):
   if flag in args:args[args.index(flag)+1]=str(v)
   else:args.extend([flag,str(v)])
  setting('--max-context',16384 if a.long else 4096);setting('--prefill',8192 if a.long else 256)
  setting('--expert-cache',a.cache_slots);setting('--adapt-swaps',96 if opts.get('adapt') else 0)
  setting('--adapt-every',4 if opts.get('adapt') else 0);setting('--pcie-frac',opts.get('pcie',1))
  setting('--conversation-cache-mib',2048);setting('--conversation-cache-slots',2);setting('--prompt-cache',4)
  k=opts.get('mtp',0);setting('--spec',max(2,k+1));setting('--mtp-max-t',max(1,k+1));setting('--spec-min-p',0)
  if k:args.extend(['--mtp','native'])
  env=child_env(cfg);env.update({k:str(v) for k,v in opts.items() if k.startswith('STRATA_')})
  env.update({'STRATA_DUMP_FIRST_LOGITS':str((folder/'logits.f32').resolve()),'STRATA_MTP_FULL_HEAD':'1','STRATA_PREFILL_TRACE':'1'})
  (folder/'command.json').write_text(json.dumps({'command':[cfg['exe'],*args],'env':{k:v for k,v in env.items() if k.startswith('STRATA_')}},indent=2),encoding='utf8')
  row={'requests':[]};result['cases'][case]=row;save();t=time.monotonic()
  telemetry=Telemetry(folder/'telemetry.jsonl')
  try:engine=StrataEngine(cfg['exe'],args,cwd=cfg['cwd'],log=str((folder/'engine.log').resolve()),env=env)
  except Exception as e:
   row.update(status='failed',error=str(e),telemetry=telemetry.close());result['status']='failed';save();raise
  telemetry.pid=engine.proc.pid
  row['startup_s']=time.monotonic()-t;row['info']=dict(engine.info);save()
  try:
   for j,(name,ids,limit) in enumerate(requests):
    start=time.monotonic();out=[];ttft=None;last_token_s=None
    for v in engine.generate(ids,limit,{'temperature':0},threading.Event()):
     if v is not None:
      if ttft is None:ttft=time.monotonic()-start
      out.append(v);last_token_s=time.monotonic()-start
    r={'name':name,'input_tokens':len(ids),'ids':out,'text':tok.decode(out),'ttft_s':ttft,'wall_s':time.monotonic()-start,**engine.last}
    assert r['generated']==len(out) and 0<len(out)<=limit
    assert r['file_blobs']==0 and r['file_mb']==0,'expert source I/O after residency'
    r['after_first_token_s']=last_token_s-ttft
    r['after_first_token_per_s']=(len(out)-1)/r['after_first_token_s'] if len(out)>1 and r['after_first_token_s'] else None
    r['decode_generated_per_s']=len(out)*1000/r['decode_ms'] if r['decode_ms'] else None
    r['cpu_expert_entries']=r['lookups']-r['hits']
    r['routed_expert_entries']=r['lookups']+r['offloaded']
    ref=(a.reference_output or a.output)/'reference'/f'logits.f32.{j}'
    if case!='reference' and ref.exists():r['logits']=compare(np.fromfile(ref,dtype='<f4'),np.fromfile(folder/f'logits.f32.{j}',dtype='<f4'))
    ref_results=(a.reference_output or a.output)/'results.json'
    if case!='reference' and ref_results.exists():
     reference=json.loads(ref_results.read_text(encoding='utf8')).get('cases',{}).get('reference',{}).get('requests',[])
     if j<len(reference):r['tokens_equal_reference']=out==reference[j]['ids']
    if (case.startswith('host') or case=='adaptive') and 'logits' in r:
     assert r['logits']['max_abs']==0 and r.get('tokens_equal_reference'),f'{case}: data movement changed arithmetic'
    row['requests'].append(r);save()
    print(case,name,{k:r[k] for k in ('generated','prompt_read','prompt_ms','decode_ms','ttft_s','cpu_expert_entries','file_blobs')},flush=True)
   if a.http:
    from native_http_acceptance import validate_http
    row['http']=validate_http(engine,tok,template,folder/'http.json');save()
  finally:
   row['telemetry']=telemetry.close();save();engine.close()
  log=(folder/'engine.log').read_text(encoding='utf8')
  assert 'post-residency expert source bytes=0' in log
  if a.long:assert 'strata prefill executed: tokens=8192 max_chunk=8192' in log,'8192 was allocated but not executed'
  if case.startswith('tc'):assert '[nvfp4-tensor] SM120 native FP4 MMA enabled' in log
  row['status']='pass';save()
 result['status']='pass';save()
if __name__=='__main__':main()
