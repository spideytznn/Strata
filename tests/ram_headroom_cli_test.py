"""Exercise resource option parsing before CUDA initialization or model loading."""
import subprocess
import sys

binary=sys.argv[1]
for value in ('0','-1','1023','65537','4096junk','4.5','99999999999999999999',''):
    r=subprocess.run([binary,'--resident-ram-headroom-mib',value],capture_output=True,text=True)
    assert r.returncode==2 and 'requires an integer from 1024 to 65536' in r.stderr,(value,r)
for value in ('1024','4096','65536'):
    r=subprocess.run([binary,'--resident-ram-headroom-mib',value,'--help'],capture_output=True,text=True)
    assert r.returncode==0 and '--resident-ram-headroom-mib' in r.stderr+r.stdout,(value,r)
r=subprocess.run([binary,'--resident-ram-headroom-mib','4096'],capture_output=True,text=True)
assert r.returncode==2 and 'requires --resident-cpu-experts' in r.stderr,r
print('RAM HEADROOM CLI PASS: 8 invalid values, 3 valid boundaries/settings, required-mode guard')
