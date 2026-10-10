"""Check real expert exports against FP64 and across CPU/GPU providers.

Exports are optional diagnostic blobs from validate_safetensors_checkpoint.py,
not a runtime model format. No server or full-model allocation is started.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

PICKS = ('0-0', '0-7', '12-100', '24-300', '40-5', '47-511')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--exports', type=Path, required=True)
    p.add_argument('--bin-dir', type=Path, required=True)
    p.add_argument('--cuda-bin', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(STRATA_NVFP4_F32='1', STRATA_NVFP4_TC='0',
               STRATA_NVFP4_F32_CANONICAL='1', STRATA_NVFP4_F32_UNROLL='1',
               STRATA_NVFP4_F32_REUSE='1', STRATA_NVFP4_F32_REUSE_TILE='4',
               STRATA_PARITY_SEED='9950')
    env['PATH'] = os.pathsep.join(a.cuda_bin + [env['PATH']])
    result = dict(env={k: v for k, v in env.items() if k.startswith('STRATA_')},
                  binaries={}, commands=[], experts=[])

    def save():
        (a.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf8')

    def run(name, *args, log, dump=None):
        exe = (a.bin_dir / (name + '.exe')).resolve()
        result['binaries'][name] = hashlib.sha256(exe.read_bytes()).hexdigest()
        child = env.copy()
        child.pop('STRATA_PARITY_DUMP', None)
        if dump:
            child['STRATA_PARITY_DUMP'] = str(dump.resolve())
        cmd = [str(exe), *map(str, args)]
        result['commands'].append(dict(argv=cmd, dump=child.get('STRATA_PARITY_DUMP')))
        save()
        with log.open('wb') as output:
            subprocess.run(cmd, env=child, stdout=output, stderr=subprocess.STDOUT, check=True)

    try:
        for key in PICKS:
            blob = (a.exports / f'expert-{key}.bin').resolve()
            gpu = a.output / f'gpu-{key}.bin'
            cpu = a.output / f'cpu-{key}.bin'
            run('nvfp4_expert_gpu_parity', blob, 0, 0, dump=gpu, log=a.output / f'gpu-{key}.txt')
            run('nvfp4_cpu_f32_test', blob, cpu.resolve(), log=a.output / f'cpu-{key}.txt')
            g = np.fromfile(gpu, dtype='<f4')
            c = np.fromfile(cpu, dtype='<f4')
            assert g.size == 8*(640+2560) and c.size == 4*(640+2560)
            # Both executables use MSVC's same normal RNG with seed 9950.
            gh = g[:8*640].reshape(8, 640)[:4].ravel()
            gy = g[8*640:].reshape(8, 2560)[:4].ravel()
            ch, cy = c[:4*640], c[4*640:]
            row = dict(expert=key, blob_sha256=hashlib.sha256(blob.read_bytes()).hexdigest(),
                       hidden_bits_equal=int((gh.view('i4') == ch.view('i4')).sum()), hidden_values=len(ch),
                       output_bits_equal=int((gy.view('i4') == cy.view('i4')).sum()), output_values=len(cy))
            result['experts'].append(row)
            save()
            assert np.isfinite(g).all() and np.isfinite(c).all()
            assert gh.tobytes() == ch.tobytes() and gy.tobytes() == cy.tobytes(), key
        run('nvfp4_gpu_reuse_bench', a.exports.resolve(), log=a.output / 'reuse.jsonl')
        rows = [json.loads(line) for line in (a.output / 'reuse.jsonl').read_text().splitlines()]
        assert len(rows) == 24 and all(r['bit_equal'] and r['empty_uneven_strided'] for r in rows)
        result.update(status='pass', reuse_cases=len(rows))
    except Exception as error:
        result.update(status='failed', error=str(error))
        raise
    finally:
        save()
    print('PASS: six FP64 experts, CPU/GPU hidden/output bits, 24 GPU layouts')


if __name__ == '__main__':
    main()
