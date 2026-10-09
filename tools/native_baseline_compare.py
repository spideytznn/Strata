"""Same-day, sequential native / fidelity GGUF / Q4XL session measurements.

Production configs are read only. Only weight paths are inherited; every run
gets the same context, FP16 KV, VRAM reserve, prefill and request token IDs.
Auto cache sizing can produce different slot counts for different formats.
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
from serve.frontend import ChatTemplate
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
    p.add_argument('--adapt-every', type=int, default=0)
    p.add_argument('--adapt-swaps', type=int, default=96)
    p.add_argument('--native-mtp', type=int, choices=(0,1,2,4), default=0,
                   help='native-only matched follow-up; common cross-backend comparison keeps MTP off')
    p.add_argument('--reference-output', type=Path, help='native no-MTP run for exact continuation checks')
    a = p.parse_args()
    if a.native_mtp and a.cases != 'native':
        p.error('--native-mtp requires --cases native; do not mix MTP settings in the cross-backend comparison')
    a.output.mkdir(parents=True, exist_ok=False)
    configs = {name: json.loads(path.read_text(encoding='utf8')) for name, path in
               [('native', a.native_config), ('fidelity', a.fidelity_config), ('q4xl', a.q4xl_config)]}
    requests = json.loads(a.requests.read_text(encoding='utf8'))
    (a.output/'requests.json').write_bytes(a.requests.read_bytes())
    # One fresh process per round: a repeated prompt must not silently become
    # a warm measurement. Each process still tests cold/warm pairs in order.
    tok = SafetensorsTokenizer.from_directory(configs['native']['tokenizer'])
    template = ChatTemplate(Path(configs['native']['chat_template']))
    warmup = tok.encode(template.render([{'role':'user','content':
        'Explain in detail how CPU caches and main memory interact. Include several concrete examples.'}],
        enable_thinking=False), parse_special=True)
    result = {'requests_sha256': hashlib.sha256(a.requests.read_bytes()).hexdigest(),
              'conditions': {'context': a.context, 'prefill': 8192, 'kv': 'fp16',
                             'expert_slots_requested': a.cache_slots or 'auto', 'reserve_mib': a.reserve_mib, 'mtp': a.native_mtp,
                             'prefill_borrow': True, 'adapt_every': a.adapt_every,
                             'adapt_swaps': a.adapt_swaps if a.adapt_every else 0}, 'cases': {}}
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
            assert all(vocab.get(token) == i for token, i in expected.items()), 'baseline vocabulary differs'
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
                     '--pcie-frac', '1', '--pcie-mode', 'dma', '--spec', str(max(2,a.native_mtp+1)), '--suffix-draft', '0',
                     '--adapt-every', str(a.adapt_every), '--adapt-swaps', str(a.adapt_swaps if a.adapt_every else 0),
                     '--conversation-cache-mib', '2048',
                     '--conversation-cache-slots', '2', '--prompt-cache', '4', '--greedy', '--stats', '--check-logits']
            if a.native_mtp:
                args += ['--mtp','native','--mtp-max-t',str(a.native_mtp+1),'--spec-min-p','0']
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
                begin = time.monotonic()
                warm_ids = [v for v in engine.generate(warmup, 128, {'temperature':0}, threading.Event()) if v is not None]
                row['unmeasured_warmup'] = {'input_ids':warmup, 'output_ids':warm_ids,
                                            'wall_s':time.monotonic()-begin, **engine.last}
                assert len(warm_ids) >= 64, 'warmup unexpectedly ended before sustained generation'
                assert engine.last['file_blobs'] == engine.last['file_mb'] == 0
                save()
                for name, ids, limit in requests:
                    start = time.monotonic()
                    started_unix = time.time()
                    out = []
                    first = last = None
                    for v in engine.generate(ids, limit, {'temperature': 0}, threading.Event()):
                        if v is not None:
                            last = time.monotonic() - start
                            if first is None:
                                first = last
                            out.append(v)
                    r = {'name': name, 'input_tokens': len(ids), 'ids': out, 'text': tok.decode(out), 'started_unix_s':started_unix,
                         'ttft_s': first, 'wall_s': time.monotonic() - start, **engine.last}
                    assert r['generated'] == len(out) == limit, 'fixed-length comparison ended early; inspect EOS separately'
                    r['after_first_token_per_s'] = (len(out) - 1) / (last - first) if len(out) > 1 else None
                    r['cpu_expert_entries'] = r['lookups'] - r['hits']
                    r['routed_expert_entries'] = r['lookups'] + r['offloaded']
                    assert r['file_blobs'] == r['file_mb'] == 0, 'resident baseline unexpectedly read experts from files'
                    if a.reference_output and case == 'native':
                        baseline = json.loads((a.reference_output/'results.json').read_text(encoding='utf8'))['cases']['native'][repeat]
                        expected = next(x for x in baseline['requests'] if x['name'] == name)
                        r['tokens_equal_reference'] = out == expected['ids']
                        assert r['tokens_equal_reference'], 'MTP changed the matched greedy continuation'
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
