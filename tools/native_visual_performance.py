"""Paired visual desktop benchmark over private stdin, never HTTP.

Load the encoder before expert sizing, as the desktop server does. Keep image
preparation, engine prompt reading, decode and total first-token latency separate.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import struct
import subprocess
import sys
import threading
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from native_telemetry import Telemetry
from safetensors_tokenizer import SafetensorsTokenizer
from serve.frontend import ChatTemplate
from serve.server import Service, StrataEngine, Vision, child_env, sampling_defaults_from_config, vision_env


def png(path):
    def chunk(kind, data):
        return struct.pack('>I', len(data))+kind+data+struct.pack('>I', zlib.crc32(kind+data))
    size = 1024
    raw = (b'\0'+bytes((255, 0, 0))*size)*size
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>IIBBBBB', size,size,8,2,0,0,0))+
                     chunk(b'IDAT', zlib.compress(raw))+chunk(b'IEND', b''))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--new', type=int, default=128)
    p.add_argument('--prefill-first', action='store_true',
                   help='Compare reserve-only 3072/3584 against preallocated 4096 with a 3072 reserve.')
    a = p.parse_args()
    assert a.rounds > 0 and a.new > 0
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(a.config.read_text(encoding='utf-8-sig'))
    assert base['args'][base['args'].index('--prefill')+1] == '4096'
    assert '--vision' in base['args'] and base['vision']['gpu']
    if a.prefill_first:
        profiles = {name:copy.deepcopy(base) for name in ['vision3072','vision3584','vision-priority3072']}
        for name,cfg in profiles.items():
            cfg['args'][cfg['args'].index('--vram-reserve-mib')+1] = '3584' if name=='vision3584' else '3072'
            cfg.setdefault('env',{})['STRATA_NATIVE_PREFILL_FIRST'] = '1' if name=='vision-priority3072' else '0'
    else:
        assert base.get('env',{}).get('STRATA_NATIVE_PREFILL_FIRST','0') != '1', 'Use --prefill-first with the new desktop profile'
        profiles = {'vision2048': copy.deepcopy(base), 'vision3072': copy.deepcopy(base), 'text3072': copy.deepcopy(base)}
        profiles['vision2048']['args'][base['args'].index('--vram-reserve-mib')+1] = '2048'
        profiles['text3072']['args'].remove('--vision')
        del profiles['text3072']['vision']
    tok = SafetensorsTokenizer.from_directory(base['tokenizer'])
    template = ChatTemplate(Path(base['chat_template']))
    docpath = ROOT/'docs/DETAILS.md'
    doc = docpath.read_text(encoding='utf-8')
    docids = tok.encode(doc)
    text = lambda n: tok.decode((docids*((n+len(docids)-1)//len(docids)))[:n])
    ask = '\nAnalyze the document in detail. Explain the design and performance tradeoffs, then give a Python LRU cache implementation with three worked examples and tests. Write a comprehensive answer.'
    fixtures = {
        'doc8k': [{'role':'user', 'content':'Document A for analysis:\n'+text(8000)+ask}],
        'doc24k': [{'role':'user', 'content':'Document B for analysis:\n'+text(24000)+ask}],
    }
    image = a.output/'red-1024.png'
    png(image)
    fixtures['image8k'] = [{'role':'user','content':[
        {'type':'image','source':str(image)},
        {'type':'text','text':'Document C. Identify the image color first, then analyze this document:\n'+text(7000)+ask}]}]
    fixture_info = {'source':str(docpath),'source_sha256':hashlib.sha256(docpath.read_bytes()).hexdigest(),
                    'messages':fixtures, 'thinking':True, 'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest()}
    (a.output/'fixtures.json').write_text(json.dumps(fixture_info,ensure_ascii=False,indent=2),encoding='utf-8')
    result = {'status':'running','rounds':a.rounds,'new':a.new,'seed':9950,'platform':platform.platform(),
              'source_config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
              'profiles':profiles,'runs':[], 'profile_mode':'prefill-first' if a.prefill_first else 'visual-reserves',
              'note':'Same binary, 4096 cap, official thinking sampling and fixed seed. Encoder loaded first. Cold/warm, image preparation and startup distinguished; no HTTP or state hashing.'}
    def save():
        (a.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save()
    try:
        for r in range(a.rounds):
            order = list(profiles)
            if r%3 == 1: order.reverse()
            elif r%3 == 2: order = order[1:]+order[:1]
            for name in order:
                active = subprocess.check_output(['powershell.exe','-NoProfile','-Command',
                    "@(Get-Process -Name 'strata*' -ErrorAction SilentlyContinue).Count"],text=True)
                assert int(active.strip() or '0') == 0, 'Another Strata process is running'
                folder = a.output/f'r{r+1}-{name}'
                folder.mkdir()
                cfg = profiles[name]
                env = child_env(cfg)
                assert not any(k in env for k in ['STRATA_STATE_HASH','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_NVFP4_TIMING'])
                sampling = {**sampling_defaults_from_config(cfg), 'seed':9950}
                row = {'round':r+1,'profile':name,'exe_sha256':hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),'requests':[]}
                result['runs'].append(row)
                save()
                tel = Telemetry(folder/'telemetry.jsonl')
                engine = vision = svc = None
                print(f'START r{r+1} {name}',flush=True)
                try:
                    start = time.perf_counter()
                    with (folder/'vision.log').open('w',encoding='utf-8') as vision_log:
                        if cfg.get('vision'): vision = Vision(cfg['vision'],log=vision_log,env=vision_env(cfg,env))
                        engine = StrataEngine(cfg['exe'],cfg['args'],cwd=cfg['cwd'],log=str(folder/'engine.log'),env=env)
                        tel.pid = engine.proc.pid
                        row.update(startup_s=time.perf_counter()-start,info=dict(engine.info))
                        assert row['info']['weight_source'] == 'safetensors'
                        startup = (folder/'engine.log').read_text(encoding='utf-8',errors='replace')
                        free = re.findall(r'strata serve: (\d+) MiB of VRAM free with everything loaded',startup)
                        row['startup_free_mib'] = int(free[-1]) if free else None
                        svc = Service(engine,tok,template,vision=vision,sampling_defaults=sampling)
                        def generate(case, messages, mode):
                            begin = time.perf_counter()
                            ids,thinking,limit = svc.prepare(messages,[],{'enable_thinking':True},max_new=a.new)
                            assert thinking
                            prepare_s = time.perf_counter()-begin
                            embeddings = getattr(svc.embeddings,'path',None)
                            tokens,first = [],None
                            try:
                                for token in engine.generate(ids,limit,sampling,threading.Event(),embeddings=embeddings):
                                    if token is not None:
                                        if first is None: first = time.perf_counter()-begin
                                        tokens.append(token)
                            finally: svc.drop_embeddings()
                            record = dict(case=case,mode=mode,ids=tokens,text=tok.decode(tokens),
                                input_sha256=hashlib.sha256(' '.join(map(str,ids)).encode('ascii')).hexdigest(),
                                image_tokens=ids.count(248056),prepare_s=prepare_s,ttft_s=first,
                                wall_s=time.perf_counter()-begin,**engine.last)
                            fresh = record['prompt_tokens']-record['reused']
                            record['fresh_tokens'] = fresh
                            record['prefill_tok_s'] = fresh*1000/record['prompt_ms']
                            record['decode_tok_s'] = len(tokens)*1000/record['decode_ms']
                            row['requests'].append(record)
                            save()
                            assert 0 < len(tokens) <= a.new and record['generated'] == len(tokens)
                            assert record['file_blobs'] == record['file_mb'] == 0
                            print(f"DONE r{r+1} {name} {case}-{mode}: fresh={fresh} reused={record['reused']} pp={record['prefill_tok_s']:.1f} decode={record['decode_tok_s']:.1f} prep={prepare_s:.3f}s ttft={first:.3f}s",flush=True)
                            return record
                        for case in ['doc8k','doc24k','image8k']:
                            if case == 'image8k' and not vision: continue
                            messages = fixtures[case]
                            cold = generate(case,messages,'cold')
                            warm = generate(case,messages,'warm')
                            warm['same_output_as_cold'] = warm['ids'] == cold['ids']
                            assert warm['reused'] > 0
                            if case == 'doc8k':
                                follow = messages+[
                                    {'role':'assistant','reasoning_content':cold['text'],'content':'The analysis is in progress.'},
                                    {'role':'user','content':'Now give concrete cache eviction examples and explain the edge cases in detail.'}]
                                generate('doc8k-followup',follow,'continuation')
                        row['status']='pass'
                finally:
                    if svc: svc.drop_embeddings()
                    if engine: engine.close()
                    if vision:
                        assert vision.dir.resolve().name.startswith('strata-vision-')
                        vision.shutdown()
                    row['telemetry'] = tel.close()
                    save()
        summary = {}
        for name in profiles:
            groups = {}
            for run in result['runs']:
                if run['profile'] != name: continue
                for req in run['requests']:
                    groups.setdefault(req['case']+'-'+req['mode'],[]).append(req)
            summary[name] = {key:{'samples':len(rows),**{
                metric:statistics.median(x[metric] for x in rows)
                for metric in ['prefill_tok_s','decode_tok_s','prepare_s','ttft_s','fresh_tokens','reused']},
                'generated_counts':[x['generated'] for x in rows]} for key,rows in groups.items()}
        result.update(status='pass',summary=summary)
        save()
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    except Exception as error:
        result.update(status='fail',error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
