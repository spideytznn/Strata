"""Stream the original BF16 ngram table into Q8_0 GGUF. Never requantizes IQ4.

Pins the source revision, verifies hash geometry, uses ggml's reference quantizer,
and checkpoints each bounded chunk with hashes for safe restart.
"""
import argparse, concurrent.futures as cf, ctypes, hashlib, json, os, pathlib
import re, struct, threading, time
import numpy as np
import requests

REPO = 'Qwen/Qwen3.8-Flash-Next'
PREFIX = 'model.language_model.layers.1.ple.ple_embedding.'
ROWS = 320001536
MULT = [23703573157769, 20109073645365, 8052911324071]
VOCAB = [20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171]
LOCAL = threading.local()
SOURCE_DIR = None

def session():
    if not hasattr(LOCAL, 'session'): LOCAL.session = requests.Session()
    return LOCAL.session

def get_range(url, first, size):
    if SOURCE_DIR is not None:
        path = SOURCE_DIR / url.rsplit('/',1)[-1]
        with path.open('rb') as f:
            f.seek(first); data=f.read(size)
        if len(data)!=size: raise ValueError(f'short local source: {path}')
        return data
    for attempt in range(6):
        try:
            start = time.monotonic()
            with session().get(url, headers={'Range':f'bytes={first}-{first+size-1}'}, stream=True, timeout=(15,30)) as r:
                r.raise_for_status()
                if r.status_code != 206 or not r.headers.get('Content-Range','').startswith(f'bytes {first}-{first+size-1}/'):
                    raise ValueError('server did not return the exact requested range')
                data = bytearray()
                for piece in r.iter_content(1024*1024):
                    data.extend(piece)
                    if len(data)>size or time.monotonic()-start>180: raise TimeoutError('range exceeded size/deadline')
                if len(data)!=size: raise ValueError('short range')
                return data
        except (requests.RequestException, ValueError, TimeoutError):
            if attempt==5: raise
            time.sleep(min(2**attempt,16))

def save_json(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(tmp,path)

def source_plan(path):
    if path.exists(): return json.loads(path.read_text())
    if SOURCE_DIR is not None:
        raise ValueError('Local conversion needs the previously verified source.json in --work; no network is used.')
    info=session().get(f'https://huggingface.co/api/models/{REPO}',timeout=30).json()
    rev=info['sha']; base=f'https://huggingface.co/{REPO}/resolve/{rev}/'
    index=session().get(base+'model.safetensors.index.json',timeout=30).json()['weight_map']
    names=sorted((n for n in index if n.startswith(PREFIX+'ngram_embedding.shard_')),key=lambda n:int(re.search(r'shard_(\d+)',n)[1]))
    assert len(names)==128
    assert [int(re.search(r'shard_(\d+)',n)[1]) for n in names]==list(range(128))
    constants=[PREFIX+k for k in ('layer_multipliers','ngram_heads_offsets','ngram_heads_vocab_sizes')]
    def header(filename):
        url=base+filename
        length=struct.unpack('<Q',get_range(url,0,8))[0]
        assert 0<length<16*1024*1024
        return filename,(8+length,json.loads(get_range(url,8,length)))
    filenames=sorted({index[n] for n in names+constants})
    with cf.ThreadPoolExecutor(max_workers=6) as pool: headers=dict(pool.map(header,filenames))
    rows=0; parts=[]
    for name in names:
        filename=index[name]; offset,h=headers[filename]; t=h[name]
        assert t['dtype']=='BF16' and len(t['shape'])==2 and t['shape'][1]==160
        n=t['shape'][0]; a,b=t['data_offsets']; assert b-a==n*320
        parts.append(dict(name=name,url=base+filename,offset=offset+a,rows=n,first_row=rows))
        rows+=n
    assert rows==ROWS
    got={}
    for name in constants:
        filename=index[name]; offset,h=headers[filename]; t=h[name]; a,b=t['data_offsets']
        assert t['dtype']=='I64'
        got[name.split('.')[-1]]=np.frombuffer(get_range(base+filename,offset+a,b-a),dtype='<i8').tolist()
    offsets=np.cumsum([0]+VOCAB[:-1]).tolist()
    assert got==dict(layer_multipliers=MULT,ngram_heads_offsets=offsets,ngram_heads_vocab_sizes=VOCAB),got
    cfg=session().get(base+'config.json',timeout=30).json()
    text=cfg.get('text_config',cfg)
    assert text['split_ngram_parts']==128 and text['ngram_size']==3 and text['heads_per_ngram']==8
    plan=dict(repo=REPO,revision=rev,rows=rows,parts=parts,constants=got,config=text)
    save_json(path,plan); return plan

def string(s):
    b=s.encode(); return struct.pack('<Q',len(b))+b

def gguf_header(plan):
    fields={'general.architecture':'qwen4exp','general.name':'Qwen3.8-Flash-Next original ngram Q8_0',
            'strata.ngram.source':REPO,'strata.ngram.revision':plan['revision']}
    h=struct.pack('<IIQQ',0x46554747,3,1,len(fields))
    for key,value in fields.items(): h+=string(key)+struct.pack('<I',8)+string(value)
    h+=string('per_layer_token_embd.weight')+struct.pack('<IQQIQ',2,160,ROWS,8,0)
    return h+b'\0'*((-len(h))%32)

def main():
    global SOURCE_DIR
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True)
    ap.add_argument('--work',type=pathlib.Path,required=True);ap.add_argument('--quant-dll',required=True)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--chunk-rows',type=int,default=4096)
    ap.add_argument('--source-dir',type=pathlib.Path,help='Read the pinned BF16 safetensors shards locally, without network access')
    ap.add_argument('--plan-only',action='store_true');args=ap.parse_args()
    assert 1<=args.workers<=8 and 0<args.chunk_rows<=131072
    SOURCE_DIR=args.source_dir
    args.work.mkdir(parents=True,exist_ok=True); plan=source_plan(args.work/'source.json')
    if SOURCE_DIR is not None:
        # Validate every local tensor's geometry/offset before touching the output.
        for url in sorted({p['url'] for p in plan['parts']}):
            length=struct.unpack('<Q',get_range(url,0,8))[0]
            assert 0<length<16*1024*1024
            header=json.loads(get_range(url,8,length))
            for p in (p for p in plan['parts'] if p['url']==url):
                t=header[p['name']]
                assert t['dtype']=='BF16' and t['shape']==[p['rows'],160]
                assert 8+length+t['data_offsets'][0]==p['offset']
                assert t['data_offsets'][1]-t['data_offsets'][0]==p['rows']*320
                assert (SOURCE_DIR/url.rsplit('/',1)[-1]).stat().st_size>=p['offset']+p['rows']*320
            for suffix,want in plan['constants'].items():
                name=PREFIX+suffix
                if name in header:
                    t=header[name];a,b=t['data_offsets'];assert t['dtype']=='I64'
                    assert np.frombuffer(get_range(url,8+length+a,b-a),dtype='<i8').tolist()==want
    print('Source validated:',plan['revision'],len(plan['parts']),'parts;',ROWS,'rows',flush=True)
    if args.plan_only:return
    dll=ctypes.CDLL(args.quant_dll);dll.q8.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64];dll.q8.restype=None
    header=gguf_header(plan); partial=args.output.with_suffix('.gguf.partial');journal=args.work/'chunks.json'
    assert not args.output.exists(),'final file already exists'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    state=json.loads(journal.read_text()) if journal.exists() else dict(revision=plan['revision'],chunk_rows=args.chunk_rows,output=str(args.output.resolve()),chunks={})
    assert state['revision']==plan['revision'] and state['chunk_rows']==args.chunk_rows and state['output']==str(args.output.resolve())
    if partial.exists():
        with partial.open('rb') as f: assert f.read(len(header))==header
    else:
        assert not state['chunks']
        with partial.open('wb') as f:f.write(header)
    jobs=[]
    with partial.open('rb') as f:
        for part in plan['parts']:
            for row in range(0,part['rows'],args.chunk_rows):
                n=min(args.chunk_rows,part['rows']-row);first=part['first_row']+row;key=str(first)
                if key in state['chunks']:
                    f.seek(len(header)+first*170);data=f.read(n*170)
                    assert hashlib.sha256(data).hexdigest()==state['chunks'][key]['sha256'],'partial-file corruption'
                else: jobs.append((part,row,n,first))
    def convert(job):
        part,row,n,first=job
        raw=get_range(part['url'],part['offset']+row*320,n*320)
        x=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
        assert np.isfinite(x).all(),'nonfinite source'
        q=np.empty(n*170,dtype=np.uint8);dll.q8(x.ctypes.data,q.ctypes.data,x.size)
        record=dict(rows=n,sha256=hashlib.sha256(q).hexdigest(),source_sha256=hashlib.sha256(raw).hexdigest())
        if row==0:
            np.savez(args.work/f"sample-{part['first_row']}.npz",source=x[:160*16].copy(),quant=q[:170*16].copy())
        return first,q,record
    start=time.monotonic();done=0;last=start
    # Bounded future set: no executor.map eager submission or accumulating converted chunks.
    with partial.open('r+b',buffering=0) as f, cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        it=iter(jobs);pending={pool.submit(convert,j) for j in [next(it,None) for _ in range(args.workers)] if j is not None}
        while pending:
            ready,pending=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
            for future in ready:
                first,q,record=future.result();f.seek(len(header)+first*170);assert f.write(q)==len(q)
                state['chunks'][str(first)]=record;done+=record['rows'];now=time.monotonic()
                if now-last>=15:
                    f.flush();os.fsync(f.fileno());save_json(journal,state);last=now
                    total=sum(v['rows'] for v in state['chunks'].values())
                    print(f'{total}/{ROWS} rows ({100*total/ROWS:.2f}%), source {done*320/1e6/(now-start):.1f} MB/s',flush=True)
                job=next(it,None)
                if job is not None:pending.add(pool.submit(convert,job))
        f.flush();os.fsync(f.fileno());save_json(journal,state)
    assert sum(v['rows'] for v in state['chunks'].values())==ROWS
    assert partial.stat().st_size==len(header)+ROWS*170
    os.replace(partial,args.output)
    save_json(args.work/'complete.json',dict(output=str(args.output),bytes=args.output.stat().st_size,revision=plan['revision'],chunks=len(state['chunks'])))
    print('Complete:',args.output,flush=True)

if __name__=='__main__':main()
