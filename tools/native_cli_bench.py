"""Offline native-engine benchmark. Does not start HTTP or stdin serving.

Use the same pretokenized fixture and executable for paired runs. Startup and
inference timings are recorded separately; raw stdout/stderr remain the authority.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default='config/native/rtx5090-262k-mtp2.json')
    p.add_argument('--exe', required=True)
    p.add_argument('--fixture', default='bench/results/2026-10-10-safetensors-runtime/p5-three-rounds/requests.json')
    p.add_argument('--case', default='long8192chunk-cold')
    p.add_argument('--out', required=True)
    p.add_argument('--max-new', type=int, default=128)
    p.add_argument('--env', action='append', default=[])
    p.add_argument('--arg', action='append', default=[])
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    tokens = next(x[1] for x in json.loads(Path(a.fixture).read_text()) if x[0] == a.case)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('Output directory already contains a run')
    # Do not compete with a user-started engine or alter its process.
    active = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command',
        "@(Get-Process -Name 'strata*' -ErrorAction SilentlyContinue).Count"], text=True)
    if int(active.strip() or '0'):
        raise RuntimeError('A Strata engine is already running')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
                                  '--format=csv,noheader,nounits'], text=True).strip()
    free, util = [int(x.strip()) for x in gpu.splitlines()[0].split(',')]
    if free < 24000 or util > 15:
        raise RuntimeError(f'GPU is busy: {gpu}')
    token_file = out / 'tokens.txt'
    token_file.write_text(' '.join(map(str, tokens)), encoding='ascii')
    env = os.environ.copy()
    env.update(cfg.get('env', {}))
    for item in a.env:
        k, v = item.split('=', 1)
        env[k] = v
    env['PATH'] = os.pathsep.join(cfg.get('lib_dirs', []) + [env['PATH']])
    exe = Path(a.exe).resolve()
    cmd = [str(exe)] + cfg['args'] + ['--tokens-file', str(token_file), '--max-new', str(a.max_new)] + a.arg
    if '--serve' in cmd:
        raise RuntimeError('Serving is forbidden in this offline harness')
    manifest = dict(command=cmd, env={k: v for k, v in env.items() if k.startswith('STRATA_')},
                    exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                    tokens_sha256=hashlib.sha256(token_file.read_bytes()).hexdigest(),
                    prompt_tokens=len(tokens), gpu_before=gpu, case=a.case)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    start = time.perf_counter()
    with (out / 'stdout.txt').open('wb') as stdout, (out / 'stderr.txt').open('wb') as stderr:
        result = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr, cwd=cfg['cwd'])
        with (out / 'gpu.csv').open('w', encoding='utf-8') as samples:
            while result.poll() is None:
                sample = subprocess.run(['nvidia-smi',
                    '--query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True)
                samples.write(f'{time.perf_counter()-start:.3f}, {sample.stdout.strip()}\n')
                samples.flush()
                try:
                    result.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
    manifest.update(exit_code=result.returncode, process_seconds=time.perf_counter() - start,
                    gpu_columns=['elapsed_s', 'timestamp', 'memory_used_mib', 'gpu_util_percent',
                                 'sm_mhz', 'mem_mhz', 'power_w', 'temp_c'])
    stdout = (out / 'stdout.txt').read_text(encoding='utf-8', errors='replace')
    for phase in ('decode', 'prefill'):
        m = re.search(rf'^{phase}\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s', stdout, re.M)
        if m:
            manifest[phase] = dict(tokens=int(m[1]), ms=float(m[2]), tokens_per_second=float(m[3]))
    m = re.search(r'^output\s*:(.*)$', stdout, re.M)
    if m:
        manifest['output_tokens'] = list(map(int, m[1].split()))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k in ('exit_code', 'process_seconds', 'decode', 'prefill')}), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
