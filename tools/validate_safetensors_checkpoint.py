"""Read-only real-checkpoint validation; writes only small test blobs/reports.

Independent NumPy FP64 oracle and optional byte comparison against the previous
pack. The pack is a test oracle only, never a dependency of the native reader.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True,type=Path)
    p.add_argument("--exe",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--oracle-pack",type=Path)
    p.add_argument("--gpu-parity",type=Path)
    p.add_argument("--cuda-bin",type=Path)
    args=p.parse_args()
    root=args.model.resolve()
    output=args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError("output must be outside original model directory")
    output.mkdir(parents=True,exist_ok=False)
    files={}
    descriptors={}
    manifest={}
    start=time.perf_counter()
    for path in sorted(root.glob("*.safetensors")):
        f=path.open("rb")
        files[path.name]=f
        prefix=f.read(8)
        n=struct.unpack("<Q",prefix)[0]
        raw=f.read(n)
        header=json.loads(raw)
        stat=path.stat()
        manifest[path.name]={"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,
                            "header_sha256":hashlib.sha256(prefix+raw).hexdigest(),
                            "header_bytes":n,"tensor_count":len(header)-("__metadata__" in header)}
        for name,d in header.items():
            if name!="__metadata__":
                assert name not in descriptors
                descriptors[name]=(path.name,n+8,d)
    def read(name):
        file,base,d=descriptors[name]
        f=files[file]
        lo,hi=d["data_offsets"]
        f.seek(base+lo)
        b=f.read(hi-lo)
        assert len(b)==hi-lo
        return b,d
    print(f"Python independent inventory: {len(descriptors)} tensors, {len(files)} files",flush=True)
    env=os.environ.copy()
    env["STRATA_SAFETENSORS_TRACE"]="1"
    with (output/"native-validation.json").open("w",encoding="utf-8") as stdout:
        subprocess.run([str(args.exe.resolve()),"verify-set",str(root),str(output)],env=env,stdout=stdout,check=True)
    native=json.loads((output/"native-validation.json").read_text(encoding="utf-8"))
    assert native["tensor_count"]==len(descriptors)==299545
    assert len(native["files"])==len(files)==11
    assert native["model"]["nvfp4_projections"]==73728
    assert native["model"]["ngram_first_absolute_offset"]==416370
    for entry in native["files"]:
        reference=manifest[entry["name"]]
        assert all(entry[key]==reference[key] for key in ("header_bytes","tensor_count"))
        assert entry["file_bytes"]==reference["bytes"]
    for sample in native["ngram_row_samples"]:
        # Compute address independently by checkpoint segment index and row.
        segment,row=divmod(sample["row"],2500012)
        name=f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{segment}.weight"
        file,base,d=descriptors[name]
        offset=base+d["data_offsets"][0]+row*160
        assert sample["tensor"]==name and sample["absolute_offset"]==offset
        files[file].seek(offset)
        assert files[file].read(160).hex()==sample["hex"]
    fp4=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=np.float64)
    fp8=np.empty(127,dtype=np.float64)
    for b in range(127):
        e,m=(b>>3)&15,b&7
        fp8[b]=np.ldexp(1+m/8,e-7) if e else np.ldexp(float(m),-9)
    rng=np.random.default_rng(5090)
    results=[]
    oracle=args.oracle_pack.open("rb") if args.oracle_pack else None
    for item in native["verification"]:
        l,e=item["layer"],item["expert"]
        path=output/f"expert-{l}-{e}.bin"
        blob=path.read_bytes()
        assert len(blob)==2764816
        if oracle:
            oracle.seek((l*512+e)*len(blob))
            assert oracle.read(len(blob))==blob, f"legacy pack mismatch at {l}/{e}"
        source_matrices=[]
        packed_matrices=[]
        for i,proj in enumerate(["gate_proj","up_proj","down_proj"]):
            prefix=f"model.language_model.layers.{l}.mlp.experts.{e}.{proj}"
            wb,d=read(prefix+".weight")
            rows,half=d["shape"]
            raw=np.frombuffer(wb,dtype=np.uint8).reshape(rows,half)
            codes=np.empty((rows,half*2),dtype=np.uint8)
            codes[:,0::2]=raw&15
            codes[:,1::2]=raw>>4
            sb,sd=read(prefix+".weight_scale")
            scales=np.frombuffer(sb,dtype=np.uint8).reshape(sd["shape"])
            assert np.all(scales<=126)
            global_bytes,_=read(prefix+".weight_scale_2")
            input_bytes,_=read(prefix+".input_scale")
            g=struct.unpack("<f",global_bytes)[0]
            assert global_bytes==blob[-16+i*4:-12+i*4]
            assert struct.pack("<f",item["input_scales"][i])==input_bytes
            src=fp4[codes]*np.repeat(fp8[scales],16,axis=1)*g
            blocks=np.frombuffer(blob,dtype=np.uint8,count=921600,offset=i*921600).reshape(rows,-1,36)
            q=blocks[:,:,4:].reshape(rows,-1,4,8)
            decoded=np.concatenate([q&15,q>>4],axis=-1).reshape(rows,half*2)
            ds=blocks[:,:,:4].reshape(rows,-1)
            dst=fp4[decoded]*np.repeat(fp8[ds],16,axis=1)*g
            assert np.array_equal(src,dst) and np.array_equal(np.signbit(src),np.signbit(dst))
            source_matrices.append(src)
            packed_matrices.append(dst)
        x=rng.normal(0,.25,(3,2560))
        x[0]=0
        def swiglu(m):
            g=x@m[0].T
            u=x@m[1].T
            return (g/(1+np.exp(-g))*u)@m[2].T
        a,b=swiglu(source_matrices),swiglu(packed_matrices)
        assert np.isfinite(a).all() and np.array_equal(a,b)
        result={"layer":l,"expert":e,"blob_sha256":hashlib.sha256(blob).hexdigest(),
                "legacy_pack_byte_exact":True if oracle else None,"decoded_weight_max_abs_error":0,
                "fp64_swiglu_max_abs_error":float(np.max(np.abs(a-b))),"activation_rows":len(x)}
        if args.gpu_parity:
            gpu_env=os.environ.copy()
            gpu_env["STRATA_NVFP4_F32"]="1"
            gpu_env["STRATA_NVFP4_TC"]="0"
            if args.cuda_bin:
                gpu_env["PATH"]=str(args.cuda_bin)+os.pathsep+gpu_env["PATH"]
            gpu=subprocess.run([str(args.gpu_parity.resolve()),str(path),"0","0"],
                               env=gpu_env,capture_output=True,text=True)
            result["existing_gpu_reference_stdout"]=gpu.stdout
            result["existing_gpu_reference_stderr"]=gpu.stderr
            assert gpu.returncode==0, f"GPU failed: {gpu.returncode}\n{gpu.stdout}\n{gpu.stderr}"
        results.append(result)
        print(f"Expert {l}/{e}: lossless bytes, FP64 SwiGLU"+(" and existing CUDA reference passed" if args.gpu_parity else " passed"),flush=True)
    for name,f in files.items():
        f.close()
        now=(root/name).stat()
        assert now.st_size==manifest[name]["bytes"] and now.st_mtime_ns==manifest[name]["mtime_ns"]
    if oracle:
        oracle.close()
    report={"scope":"P1 metadata and six expert adapters; not end-to-end model quality or throughput",
            "source":str(root),"source_headers":manifest,"experts":results,
            "native_report":"native-validation.json","seconds":time.perf_counter()-start,
            "ngram_boundary_rows_verified":len(native["ngram_row_samples"]),
            "original_size_and_mtime_unchanged":True,"full_source_hash_verified":False,
            "gpu_test_executable":str(args.gpu_parity) if args.gpu_parity else None}
    (output/"checkpoint-validation.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("Validation report:",output/"checkpoint-validation.json",flush=True)


if __name__=="__main__":
    main()
