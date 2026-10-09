"""Teacher-forced native logits, long-context recall and real HTTP acceptance."""
import argparse,hashlib,json,sys,time,threading
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from serve.server import StrataEngine,child_env
from serve.frontend import ChatTemplate
from safetensors_tokenizer import SafetensorsTokenizer
from native_path_matrix import CASES,compare
from native_telemetry import Telemetry
from native_http_acceptance import validate_http

TEXTS=[('english','Explain how to implement a least recently used cache.',
 'A least recently used cache combines a hash table with a doubly linked list. The table maps keys to nodes, and the list records their order of use. Reading or updating an entry moves it to the front. If insertion exceeds capacity, remove the node at the back and delete its table entry. Both lookup and eviction take constant expected time.'),
 ('chinese','解释二分查找，并说明边界处理。',
 '二分查找要求数据有序。维护左闭右开的区间 [left, right)，每轮取中点。若中点值小于目标，就把左端点设为 mid + 1；否则把右端点设为 mid。区间为空时，left 是第一个不小于目标的位置。最后还要检查下标是否越界、对应元素是否等于目标。时间复杂度为 O(log n)。'),
 ('code','Implement a binary search returning the first matching index or -1.',
 'def binary_search(values, target):\n    left, right = 0, len(values)\n    while left < right:\n        mid = (left + right) // 2\n        if values[mid] < target:\n            left = mid + 1\n        else:\n            right = mid\n    return left if left < len(values) and values[left] == target else -1\n')]

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--cases',default='reference,tc3');p.add_argument('--reference-output',type=Path)
 p.add_argument('--oracle-config',type=Path,help='read-only fidelity GGUF config for the fidelity case')
 p.add_argument('--teacher',action='store_true');p.add_argument('--lengths',default='');p.add_argument('--http',action='store_true')
 p.add_argument('--expert-cache',type=int,default=4500);p.add_argument('--bare-model',action='store_true');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);cfg=json.loads(a.config.read_text(encoding='utf8'))
 tok=SafetensorsTokenizer.from_directory(cfg['tokenizer']);template=ChatTemplate(Path(cfg['chat_template']))
 def prompt(s):return tok.encode(template.render([{'role':'user','content':s}],enable_thinking=False),parse_special=True)
 result={'config':cfg,'cases':{}}
 def save():(a.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 for case in a.cases.split(','):
  folder=a.output/case;folder.mkdir();opts={} if case=='fidelity' else CASES[case];args=list(cfg['args'])
  run_cfg=cfg
  if case=='fidelity':
   if not a.oracle_config:raise ValueError('fidelity requires --oracle-config')
   run_cfg=json.loads(a.oracle_config.read_text(encoding='utf8'))
   for flag in ('--safetensors','--model'):
    if flag in args:
     i=args.index(flag);del args[i:i+2]
   for flag in ('--pack','--native','--native-dense-gguf','--native-head-gguf','--embd-gguf','--ple-gguf'):
    if flag in run_cfg['args']:args.extend([flag,run_cfg['args'][run_cfg['args'].index(flag)+1]])
   args.append('--resident-experts')
  def setting(flag,v):
   if flag in args:args[args.index(flag)+1]=str(v)
   else:args.extend([flag,str(v)])
  lengths=[int(x) for x in a.lengths.split(',') if x]
  context=((max(lengths)+256+8191)//8192)*8192 if lengths else 4096
  setting('--max-context',context);setting('--prefill',8192 if lengths else 256)
  setting('--expert-cache',a.expert_cache);setting('--adapt-every',0);setting('--adapt-swaps',0)
  setting('--pcie-frac',opts.get('pcie',1));setting('--prompt-cache',6);setting('--conversation-cache-mib',2048)
  k=opts.get('mtp',0);setting('--spec',max(2,k+1));setting('--mtp-max-t',max(1,k+1));setting('--spec-min-p',0)
  if k:args+=['--mtp','native']
  if a.bare_model and case!='fidelity':
   if '--expert-profile' in args:
    i=args.index('--expert-profile');del args[i:i+2]
   if '--safetensors' in args:args[args.index('--safetensors')]='--model'
  env=child_env({'lib_dirs':run_cfg.get('lib_dirs',[]),'env':cfg.get('env',{})});env.update({k:str(v) for k,v in opts.items() if k.startswith('STRATA_')})
  if case=='fidelity':env.update({'STRATA_RESIDENT_PIN':'1','STRATA_PARTIAL_PIN':'0','STRATA_RESIDENT_HEADROOM_GIB':'4','STRATA_STAGE_PIN':'0'})
  env.update({'STRATA_DUMP_FIRST_LOGITS':str((folder/'logits.f32').resolve()),'STRATA_PREFILL_TRACE':'1','STRATA_MTP_FULL_HEAD':'1'})
  (folder/'command.json').write_text(json.dumps({'command':[run_cfg['exe'],*args],'env':{k:v for k,v in env.items() if k.startswith('STRATA_')}},indent=2),encoding='utf8')
  row={'requests':[],'binary_sha256':hashlib.sha256(Path(run_cfg['exe']).read_bytes()).hexdigest(),'started_unix_s':time.time()};result['cases'][case]=row;save();tel=Telemetry(folder/'telemetry.jsonl');start=time.monotonic()
  try:engine=StrataEngine(run_cfg['exe'],args,cwd=cfg['cwd'],log=str((folder/'engine.log').resolve()),env=env)
  except Exception as e:
   row.update(status='failed',error=str(e),telemetry=tel.close());result['status']='failed';save();raise
  tel.pid=engine.proc.pid
  row['startup_s']=time.monotonic()-start;row['info']=dict(engine.info)
  def generate(name,ids,limit,target=None):
   start=time.monotonic();out=[];first=None
   for v in engine.generate(ids,limit,{'temperature':0},threading.Event()):
    if v is not None:
     if first is None:first=time.monotonic()-start
     out.append(v)
   input_hash=hashlib.sha256(np.asarray(ids,dtype='<i4').tobytes()).hexdigest()
   j=len(row['requests']);r={'name':name,'input_tokens':len(ids),'input_sha256':input_hash,'ids':out,'text':tok.decode(out),'ttft_s':first,'wall_s':time.monotonic()-start,**engine.last}
   assert r['generated']==len(out) and 0<len(out)<=limit
   assert r['file_blobs']==r['file_mb']==0
   if case=='fidelity':assert r['lookups']==r['hits'],'CPU Q8 fallback would confound the fidelity comparison'
   x=np.fromfile(folder/f'logits.f32.{j}',dtype='<f4').astype('float64');assert x.shape==(248320,) and np.isfinite(x).all()
   if target is not None:
    r['target']=target;r['nll']=float(np.log(np.exp(x-x.max()).sum())+x.max()-x[target]);r['argmax']=int(x.argmax())
   reference_root=a.reference_output or a.output
   ref=reference_root/'reference'/f'logits.f32.{j}'
   if case!='reference' and ref.exists():
    reference=json.loads((reference_root/'results.json').read_text(encoding='utf8'))['cases']['reference']['requests'][j]
    assert reference['name']==name,'reference requests do not align'
    r['same_input_as_reference']=reference['input_sha256']==input_hash
    if r['same_input_as_reference']:r['logits']=compare(np.fromfile(ref,dtype='<f4'),x)
   row['requests'].append(r);save();return r
  try:
   if a.teacher:
    for category,question,answer in TEXTS:
     base=prompt(question);targets=tok.encode(answer)
     for j,target in enumerate(targets):generate(f'teacher-{category}-{j}',base+targets[:j],1,target)
     print(case,category,len(targets),'positions',flush=True)
    measured=[r for r in row['requests'] if 'nll' in r]
    row['teacher_summary']={'positions':len(measured),'mean_nll':float(np.mean([r['nll'] for r in measured])),
      'by_category':{category:{'positions':len(items),'mean_nll':float(np.mean([r['nll'] for r in items]))}
        for category in ('english','chinese','code') if (items:=[r for r in measured if r['name'].startswith('teacher-'+category+'-')])}}
    comparisons=[r['logits'] for r in measured if 'logits' in r]
    if comparisons:
     baseline=json.loads(((a.reference_output or a.output)/'results.json').read_text(encoding='utf8'))['cases']['reference']['teacher_summary']
     row['teacher_summary'].update(mean_kl=float(np.mean([r['kl_reference_candidate'] for r in comparisons])),
       top1_agreement=float(np.mean([r['argmax_equal'] for r in comparisons])),
       mean_nll_delta=row['teacher_summary']['mean_nll']-baseline['mean_nll'])
    save()
   for size in lengths:
    for depth in (0.1,0.9):
     code=f'ZEBRA-{size}-D{int(depth*100)}'
     head=tok.encode('<|im_start|>user\nRead the following archive.\n',parse_special=True)
     needle=tok.encode(f'\nThe secret access code for Alice is {code}.\n')
     tail=tok.encode('\nWhat is Alice\'s secret access code? Reply only with that exact code.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',parse_special=True)
     unit=tok.encode('Archive entry: binary search repeatedly halves an ordered interval. A queue processes records in order. Trees contain nodes and edges.\n')
     count=size-len(head)-len(needle)-len(tail);filler=(unit*((count+len(unit)-1)//len(unit)))[:count];at=int(count*depth)
     ids=head+filler[:at]+needle+filler[at:]+tail;assert len(ids)==size
     r=generate(f'needle-{size}-{depth}',ids,32);r['expected']=code;r['correct']=code in r['text'];save()
     print(case,r['name'],'correct=',r['correct'],'prefill_ms=',r['prompt_ms'],flush=True)
     suffix=tok.encode('<|im_start|>user\nRepeat the same access code only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',parse_special=True)
     r2=generate(f'needle-{size}-{depth}-cached',ids+r['ids']+suffix,32);r2['expected']=code;r2['correct']=code in r2['text'];save()
     assert r2['reused']>=size-256,'long prefix was not reused'
   if a.http:row['http']=validate_http(engine,tok,template,folder/'http.json');save()
  finally:
   row['telemetry']=tel.close();save();engine.close()
  log=(folder/'engine.log').read_text(encoding='utf8')
  if case!='fidelity':assert 'post-residency expert source bytes=0' in log
  row['status']='completed';save()
 result['status']='completed';save()
if __name__=='__main__':main()
