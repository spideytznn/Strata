"""Create a local native server config without model conversion or deployment edits."""
import argparse,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--context',type=int,default=32768);p.add_argument('--prefill',type=int,default=8192)
 p.add_argument('--kv',choices=('fp16','int8'),default='fp16')
 p.add_argument('--prefill-mode',choices=('fp16','w4a8','w4a4x2','w4a4'),default='fp16')
 p.add_argument('--dedicated-prefill',action='store_true',help='reserve prompt workspace separately from the expert cache')
 p.add_argument('--parallel',type=int,choices=range(1,9),default=1,help='opt-in concurrent text slots; validated desktop profile uses 2')
 p.add_argument('--exe',type=Path,help='engine executable, including a staged build')
 p.add_argument('--load-batch',type=int,default=1,help='startup expert batch size, 1..128; 1 keeps the reference loader')
 p.add_argument('--load-workers',type=int,default=1,help='startup byte-packing workers, 1..16')
 p.add_argument('--mtp',type=int,choices=range(8),default=2);p.add_argument('--expert-cache',type=int,default=0)
 p.add_argument('--spec-min-p',type=float,help='opt-in: MTP confidence gate, independent of target sampling min_p')
 p.add_argument('--spec-min-drafts',type=int,default=0,help='opt-in: keep these first guesses before confidence gating')
 p.add_argument('--mtp-projections',choices=('bf16','q8_0'),default='bf16',help='MTP only: optional Q8_0 projection quantization at load')
 p.add_argument('--mtp-batch-bf16',action='store_true',help='opt-in: batch original BF16 MTP KV with FP32 activation splitting')
 p.add_argument('--conversation-cache-mib',type=int,default=4096)
 p.add_argument('--adapt-every',type=int,default=4);p.add_argument('--adapt-swaps',type=int,default=96)
 p.add_argument('--pcie-frac',type=float,choices=(0,0.5,1),default=1,
                help='fraction of cold experts executed on GPU; hot cache hits always use GPU')
 p.add_argument('--balanced-experts',action='store_true',help='use balanced initial slots instead of the bundled profile')
 p.add_argument('--staging',action='store_true',help='use locked RAM plus host staging instead of full CUDA mapped RAM')
 p.add_argument('--cuda-bin',type=Path);a=p.parse_args()
 if a.context<256 or a.prefill not in (256,512,1024,2048,4096,6144,8192):
  p.error('context must be at least 256; prefill must be 256/512/1024/2048/4096/6144/8192')
 if min(a.expert_cache,a.conversation_cache_mib,a.adapt_every,a.adapt_swaps)<0:p.error('cache and adaptation values must be nonnegative')
 if not 1<=a.load_batch<=128 or not 1<=a.load_workers<=16:p.error('load batch must be 1..128; load workers must be 1..16')
 if not a.mtp and a.mtp_projections!='bf16':p.error('Q8 MTP projections require enabled MTP')
 if a.mtp_batch_bf16 and (not a.mtp or a.mtp_projections!='bf16'):p.error('--mtp-batch-bf16 requires enabled BF16 MTP')
 if a.spec_min_p is not None and (not a.mtp or not 0<=a.spec_min_p<=1):p.error('--spec-min-p requires enabled MTP and a probability in 0..1')
 if not 0<=a.spec_min_drafts<=a.mtp:p.error('--spec-min-drafts must be between 0 and the MTP draft count')
 if a.parallel>1 and a.spec_min_drafts:p.error('--spec-min-drafts is a serial MTP option; omit it with --parallel')
 model=a.model.resolve(strict=True);output=a.output.resolve()
 if output==model or model in output.parents:raise ValueError('output must be outside the original model directory')
 if output.exists():raise FileExistsError(output)
 for name in ('config.json','hf_quant_config.json','model.safetensors.index.json','tokenizer.json'):
  if not (model/name).is_file():raise FileNotFoundError(model/name)
 args=['--model',str(model),'--max-context',str(a.context),'--prefill',str(a.prefill),
       '--expert-cache',str(a.expert_cache) if a.expert_cache else 'auto','--vram-reserve-mib','2048','--spec',str(max(2,a.mtp+1)),
       '--suffix-draft','0','--pcie-frac',str(a.pcie_frac),'--pcie-mode','dma','--kv',a.kv,
       '--adapt-every',str(a.adapt_every),'--adapt-swaps',str(a.adapt_swaps if a.adapt_every else 0),
       '--conversation-cache-mib',str(a.conversation_cache_mib),'--conversation-cache-slots','4','--prompt-cache','6','--stats','--check-logits']
 if not a.balanced_experts:args+=['--expert-profile',str(ROOT/'data/expert-profile.bin')]
 if a.dedicated_prefill:args+=['--no-prefill-borrow']
 if a.mtp:args+=['--mtp','native','--mtp-max-t',str(a.mtp+1),'--spec-min-p','0' if a.spec_min_p is None else str(a.spec_min_p)]
 if a.spec_min_drafts:args+=['--spec-min-drafts',str(a.spec_min_drafts)]
 cuda=Path(os.environ.get('CUDA_PATH',r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'))
 candidates=[a.cuda_bin] if a.cuda_bin else [cuda/'bin'/'x64',cuda/'bin']
 libs=[str(x.resolve()) for x in candidates if x and x.is_dir()]
 config={'exe':str(a.exe.resolve() if a.exe else ROOT/'build-native-engine'/'strata.exe'),'args':args,'cwd':str(ROOT),
         'tokenizer':str(model),'tokenizer_format':'safetensors','chat_template':str(ROOT/'config/native/froggeric-v22.5.jinja'),
         'model_name':'Qwen3.8-Flash-Next-NVFP4-Native','lib_dirs':libs,'allowed_hosts':['*'],
         'env':{'STRATA_NVFP4_F32':'1','STRATA_NVFP4_TC':'0','STRATA_PREFILL_NVFP4':a.prefill_mode,
                'STRATA_PREFILL_CPU_SHARE':'0','STRATA_PREFILL_BF16X2':'1','STRATA_SPEC_STOP_BOUNDARY':'1',
                'STRATA_NATIVE_ALLOC_PINNED':'0' if a.staging else '1',
                'STRATA_NATIVE_LOAD_BATCH':str(a.load_batch),'STRATA_NATIVE_LOAD_WORKERS':str(a.load_workers)}}
 if a.mtp_projections!='bf16':config['env']['STRATA_MTP_NATIVE_PROJECTIONS']=a.mtp_projections
 if a.parallel>1:config['parallel']=a.parallel
 if a.mtp_batch_bf16:config['env']['STRATA_MTP_BATCH_BF16']='1'
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x',encoding='utf8') as f:json.dump(config,f,ensure_ascii=False,indent=2)
 print(output)
if __name__=='__main__':main()
