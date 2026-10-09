"""Same-day, sequential native / fidelity GGUF / Q4XL session measurements.

Production configs are read only. Only weight paths are inherited; every run
gets the same context, FP16 KV, expert slots, prefill and request token IDs.
Q4XL is a different quantization/checkpoint, not a numerical quality oracle.
"""
import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.server import StrataEngine, child_env
from safetensors_tokenizer import SafetensorsTokenizer
from native_telemetry import Telemetry


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native-config', type=Path, required=True)
    p.add_argument('--fidelity-config', type=Path, required=True)
    p.add_argument('--q4xl-config', type=Path, required=True)
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', default='native,fidelity,q4xl')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--cache-slots', type=int, default=0, help='0 sizes each quantization to the same VRAM reserve')
    p.add_argument('--context', type=int, default=32768)
    p.add_argument('--reserve-mib', type=int, default=2048)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    configs = {name: json.loads(path.read_text(encoding='utf8')) for name, path in
               [('native', a.native_config), ('fidelity', a.fidelity_config), ('q4xl', a.q4xl_config)]}
    requests = json.loads(a.requests.read_text(encoding='utf8'))
    # One fresh process per round: a repeated prompt must not silently become
    # a warm measurement. Each process still tests cold/warm pairs in order.
    tok = SafetensorsTokenizer.from_directory(configs['native']['tokenizer'])
    result = {'requests_sha256': hashlib.sha256(a.requests.read_bytes()).hexdigest(),
              'conditions': {'context': a.context, 'prefill': 8192, 'kv': 'fp16',
                             'expert_slots_requested': a.cache_slots or 'auto', 'reserve_mib': a.reserve_mib, 'mtp': False,
                             'prefill_borrow': True}, 'cases': {}}
    def save():
        (a.output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    for case in a.cases.split(','):
        cfg = configs[case]
        if case != 'native':
            # All IDs sent to a legacy model must name the same vocabulary
            # entries. This also checks the many added/control tokens.
            vocab = json.loads((Path(cfg['tokenizer']) / 'vocab.json').read_text(encoding='utf8'))
            original = json.loads((Path(configs['native']['tokenizer']) / 'tokenizer.json').read_text(encoding='utf8'))
            expected = dict(original['model']['vocab'])
            expected.update({v['content']: v['id'] for v in original.get('added_tokens', [])})
            used = {v for _, ids, _ in requests for v in ids}
            reverse = {v: k for k, v in expected.items()}
            assert all(vocab.get(reverse[v]) == v for v in used), 'baseline input vocabulary differs'
        rows = result['cases'][case] = []
        for repeat in range(a.rounds):
            folder = a.output / f'{case}-{repeat}'
            folder.mkdir()
            args = []
            if case == 'native':
                args += ['--model', cfg['tokenizer']]
            else:
                for flag in ('--pack', '--native', '--native-dense-gguf', '--native-head-gguf', '--embd-gguf', '--ple-gguf'):
                    if flag in cfg['args']:
                        args += [flag, cfg['args'][cfg['args'].index(flag) + 1]]
                args += ['--resident-experts']
            args += ['--max-context', str(a.context), '--prefill', '8192', '--kv', 'fp16',
                     '--expert-cache', str(a.cache_slots) if a.cache_slots else 'auto', '--vram-reserve-mib', str(a.reserve_mib),
                     '--expert-profile', str(ROOT / 'data/expert-profile.bin'),
                     '--pcie-frac', '1', '--pcie-mode', 'dma', '--spec', '2', '--suffix-draft', '0',
                     '--adapt-every', '0', '--adapt-swaps', '0', '--conversation-cache-mib', '2048',
                     '--conversation-cache-slots', '2', '--prompt-cache', '4', '--greedy', '--stats', '--check-logits']
            local = {'lib_dirs': cfg.get('lib_dirs', []), 'env': configs['native'].get('env', {})}
            env = child_env(local)
            env.update({'STRATA_RESIDENT_PIN': '1', 'STRATA_RESIDENT_HEADROOM_GIB': '4',
                        'STRATA_PARTIAL_PIN': '0', 'STRATA_STAGE_PIN': '0',
                        'STRATA_VERIFY_STAGING_BLOBS': '128', 'STRATA_PREFILL_TRACE': '1'})
            command = [cfg['exe'], *args]
            (folder / 'command.json').write_text(json.dumps({'command': command, 'env': {k: v for k, v in env.items() if k.startswith('STRATA_')}}, indent=2), encoding='utf8')
            row = {'round': repeat, 'requests': [], 'binary_sha256': hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest()}
            rows.append(row)
            save()
            tel = Telemetry(folder / 'telemetry.jsonl')
            engine = None
            try:
                start = time.monotonic()
                engine = StrataEngine(cfg['exe'], args, cwd=str(ROOT), log=str((folder / 'engine.log').resolve()), env=env)
                tel.pid = engine.proc.pid
                row['startup_s'] = time.monotonic() - start
                row['info'] = dict(engine.info)
                for name, ids, limit in requests:
                    start = time.monotonic()
                    out = []
                    first = last = None
                    for v in engine.generate(ids, limit, {'temperature': 0}, threading.Event()):
                        if v is not None:
                            last = time.monotonic() - start
                            if first is None:
                                first = last
                            out.append(v)
                    r = {'name': name, 'input_tokens': len(ids), 'ids': out, 'text': tok.decode(out),
                         'ttft_s': first, 'wall_s': time.monotonic() - start, **engine.last}
                    assert r['generated'] == len(out) and 0 < len(out) <= limit
                    r['after_first_token_per_s'] = (len(out) - 1) / (last - first) if len(out) > 1 else None
                    r['cpu_expert_entries'] = r['lookups'] - r['hits']
                    r['routed_expert_entries'] = r['lookups'] + r['offloaded']
                    assert r['file_blobs'] == r['file_mb'] == 0, 'resident baseline unexpectedly read experts from files'
                    row['requests'].append(r)
                    save()
                    print(case, repeat, name, {k: r[k] for k in ('prompt_ms', 'decode_ms', 'ttft_s', 'cpu_expert_entries')}, flush=True)
                row['status'] = 'pass'
            finally:
                row['telemetry'] = tel.close()
                if engine is not None:
                    engine.close()
                save()
    result['status'] = 'pass'
    save()


if __name__ == '__main__':
    main()
