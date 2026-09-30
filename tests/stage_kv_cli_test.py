"""Invalid combinations must fail before loading weights (no inference)."""
import subprocess
import sys

exe=sys.argv[1]
cases=[
    (['--stage-kv'], '--stage-kv requires --stage-weights'),
    (['--stage-kv','--stage-weights'], '--stage-weights requires'),
    (['--stage-kv','--stage-weights','--serve','--native','/nonexistent','--spec','4'], '--stage-weights requires'),
]
for args,expected in cases:
    r=subprocess.run([exe,*args],capture_output=True,text=True,timeout=20)
    assert r.returncode==2,(args,r.returncode,r.stderr)
    assert expected in r.stderr,(args,r.stderr)
print(f'PASS: {len(cases)} invalid stage-KV CLI combinations rejected before weight loading')
