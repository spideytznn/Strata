"""Lossless BF16 sidecar for the float matrices served by --native-dense-gguf / --native-head-gguf.

Metadata is copied verbatim (including 64-bit ngram constants). Rejects every value
that is not exactly BF16; the main GGUF and the expert weights are never rewritten.
"""
import argparse
import hashlib
import json
import pathlib
import struct
import numpy as np
from iq_pack import Model, fidelity_native, tensor_bytes

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gguf',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args(); m=Model(a.gguf)
    if len(m.paths)!=1: ap.error('currently requires a single input GGUF')
    g=m.files[0]
    with a.gguf.open('rb') as f:
        f.seek(16); nkv=struct.unpack('<Q',f.read(8))[0]
        for _ in range(nkv): g._str(f); g._value(f)
        end=f.tell(); f.seek(24); metadata=f.read(end-24)
    rows=[]; off=0
    for name,(gf,t,mm,p) in m.where.items():
        if not fidelity_native(name): continue
        if t.type_name!='F32' or len(t.shape)!=2: raise ValueError(f'{name}: expected 2D F32')
        rows.append((name,gf,t,mm,p,off)); off+=(t.elements*2+31)//32*32
    def string(s):
        b=s.encode(); return struct.pack('<Q',len(b))+b
    h=b'GGUF'+struct.pack('<IQQ',3,len(rows),nkv)+metadata
    for name,gf,t,mm,p,off in rows:
        h+=string(name)+struct.pack('<I',2)+struct.pack('<2Q',*t.shape)+struct.pack('<IQ',30,off)
    h+=b'\0'*((-len(h))%32)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.out.with_suffix('.gguf.part'); records=[]
    with tmp.open('wb') as f:
        f.write(h)
        for name,gf,t,mm,p,off in rows:
            raw=tensor_bytes(mm,gf,t).view('<u4')
            if np.any(raw&65535) or not np.isfinite(raw.view('<f4')).all():
                raise ValueError(f'{name}: values are not finite exact BF16; refusing rounding')
            data=(raw>>16).astype('<u2').tobytes()
            assert f.tell()==len(h)+off
            f.write(data); f.write(b'\0'*((-len(data))%32))
            records.append({'name':name,'elements':t.elements,'exact':True,'bf16_sha256':hashlib.sha256(data).hexdigest()})
            print(name,flush=True)
    tmp.replace(a.out)
    # Read all output payloads back, compare against source bits.
    out=Model(a.out)
    for name,*_ in rows:
        expected=tensor_bytes(m.where[name][2],m.where[name][0],m.where[name][1]).view('<u4')>>16
        if not np.array_equal(out.bytes(name).view('<u2'),expected): raise ValueError(name+': readback mismatch')
    a.out.with_suffix('.validation.json').write_text(json.dumps({'status':'passed','weight_conversion':'exact F32->BF16 only','source':str(a.gguf),'tensors':records},indent=2),encoding='utf-8')
    print(f'PASS: {len(rows)} exact BF16 tensors, {a.out.stat().st_size} bytes',flush=True)

if __name__=='__main__': main()
