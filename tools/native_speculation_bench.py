"""Screen MTP window/confidence choices on the real visual desktop profile.

Private stdin only; retain full token sequences, actual cache reuse and counters.
Confidence gates are request-local, so they share the same weights/workspaces.
Timing runs do not enable state hashes. Cross-window output differences are
reported, never silently treated as equivalent quality.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from native_telemetry import Telemetry
from native_visual_performance import png
from safetensors_tokenizer import SafetensorsTokenizer
from serve.frontend import ChatTemplate
from serve.server import Service, StrataEngine, Vision, child_env, sampling_defaults_from_config, vision_env


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--windows', nargs='+', type=int, default=[3, 5, 7])
    p.add_argument('--gates', nargs='+', type=float, default=[0, .5, .7])
    p.add_argument('--new', type=int, default=256)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--temperature', type=float, default=1)
    p.add_argument('--cases', default='english,chinese,code,doc24k,image8k')
    p.add_argument('--pcie-fracs', nargs='+', type=float, default=[.25])
    p.add_argument('--reserve-mib', type=int, help='Explicit test override; leave the desktop config untouched.')
    a = p.parse_args()
    assert a.new > 0 and a.rounds > 0 and all(2 <= w <= 8 for w in a.windows)
    assert all(0 <= g <= 1 for g in a.gates + a.pcie_fracs)
    assert len(set(a.windows)) == len(a.windows) and len(set(a.gates)) == len(a.gates)
    assert 0 in a.gates, 'Include a zero-gate reference'
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(a.config.read_text(encoding='utf-8-sig'))
    if a.reserve_mib is not None:
        assert a.reserve_mib >= 0
        base['args'][base['args'].index('--vram-reserve-mib') + 1] = str(a.reserve_mib)
    assert '--vision' in base['args'] and base['vision']['gpu']
    tok = SafetensorsTokenizer.from_directory(base['tokenizer'])
    template = ChatTemplate(Path(base['chat_template']))
    docpath = ROOT / 'docs/DETAILS.md'
    docids = tok.encode(docpath.read_text(encoding='utf-8'))
    def document(n):
        return tok.decode((docids * ((n + len(docids) - 1) // len(docids)))[:n])
    image = a.output / 'red-1024.png'
    png(image)
    fixtures = {
        'english': 'Explain how to implement an LRU cache in Python. Include complete code, three worked examples and eviction tests.',
        'chinese': '详细解释归并排序，给出完整 Python 实现和三个测试例子，说明每一步的结果和边界处理。',
        'code': 'Write a detailed Python tutorial on binary search. Include lower_bound, upper_bound, duplicate handling and unit tests.',
        'doc24k': 'Analyze this document in detail, explain the performance tradeoffs, and give concrete implementation examples.\n' + document(24000),
        'image8k': [{'type': 'image', 'source': str(image)}, {'type': 'text', 'text': 'Identify the image color, then analyze this document in detail:\n' + document(7000)}],
    }
    cases = a.cases.split(',')
    assert cases and len(set(cases)) == len(cases) and set(cases) <= fixtures.keys()
    result = dict(status='running', args=vars(a) | {'config': str(a.config), 'output': str(a.output)},
                  source_config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
                  source_doc_sha256=hashlib.sha256(docpath.read_bytes()).hexdigest(),
                  fixtures=fixtures, runs=[], started_unix_s=time.time(),
                  note='Vision loaded first; official thinking sampling except requested temperature; seed 9950; window includes target token. Startup excluded. First/repeat pairs must match exactly; cross-window differences are reported.')
    def save():
        (a.output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save()
    references = {}
    try:
        for r in range(a.rounds):
            windows = list(a.windows) if r % 2 == 0 else list(reversed(a.windows))
            for window in windows:
                active = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command',
                    "@(Get-Process -Name 'strata*' -ErrorAction SilentlyContinue).Count"], text=True)
                assert int(active.strip() or '0') == 0, 'Another Strata process is running'
                cfg = copy.deepcopy(base)
                for flag in ('--spec', '--mtp-max-t'):
                    cfg['args'][cfg['args'].index(flag) + 1] = str(window)
                cfg['args'][cfg['args'].index('--spec-min-p') + 1] = '0'
                folder = a.output / f'r{r+1}-t{window}'
                folder.mkdir()
                env = child_env(cfg)
                assert not any(k in env for k in ('STRATA_STATE_HASH', 'STRATA_DUMP_FIRST_LOGITS', 'STRATA_PREFILL_NVFP4_TIMING'))
                row = dict(round=r+1, window=window, config=cfg, requests=[],
                           exe_sha256=hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest())
                result['runs'].append(row)
                tel = Telemetry(folder / 'telemetry.jsonl')
                engine = vision = svc = None
                print(f'START round={r+1} window={window}', flush=True)
                try:
                    start = time.perf_counter()
                    with (folder / 'vision.log').open('w', encoding='utf-8') as vision_log:
                        vision = Vision(cfg['vision'], log=vision_log, env=vision_env(cfg, env))
                        engine = StrataEngine(cfg['exe'], cfg['args'], cwd=cfg['cwd'], log=str(folder / 'engine.log'), env=env)
                        tel.pid = engine.proc.pid
                        row.update(startup_s=time.perf_counter()-start, info=dict(engine.info))
                        assert row['info']['weight_source'] == 'safetensors'
                        assert row['info']['expert_ram_bytes'] == row['info']['expert_cuda_pinned_bytes'] > 0
                        svc = Service(engine, tok, template, vision=vision)
                        for case_index, case in enumerate(cases):
                            gates = a.gates[case_index % len(a.gates):] + a.gates[:case_index % len(a.gates)]
                            for gate in gates:
                                for fraction in a.pcie_fracs:
                                    previous = None
                                    for mode in ('first', 'repeat'):
                                        begin = time.perf_counter()
                                        ids, thinking, limit = svc.prepare([{'role': 'user', 'content': fixtures[case]}], [], {'enable_thinking': True}, max_new=a.new)
                                        assert thinking
                                        prepare_s = time.perf_counter()-begin
                                        sampling = {**sampling_defaults_from_config(cfg), 'temperature': a.temperature, 'seed': 9950,
                                                    'strata_tune': {'spec_min_p': gate, 'pcie_frac': fraction}}
                                        tokens, first = [], None
                                        try:
                                            for token in engine.generate(ids, limit, sampling, threading.Event(), embeddings=getattr(svc.embeddings, 'path', None)):
                                                if token is not None:
                                                    if first is None: first = time.perf_counter()-begin
                                                    tokens.append(token)
                                        finally:
                                            svc.drop_embeddings()
                                        record = dict(case=case, gate=gate, pcie_frac=fraction, mode=mode, ids=tokens, text=tok.decode(tokens),
                                                      input_sha256=hashlib.sha256(' '.join(map(str, ids)).encode('ascii')).hexdigest(),
                                                      prepare_s=prepare_s, ttft_s=first, wall_s=time.perf_counter()-begin, **engine.last)
                                        row['requests'].append(record)
                                        save()
                                        assert 0 < len(tokens) <= a.new and record['generated'] == len(tokens)
                                        assert record['file_blobs'] == record['file_mb'] == 0
                                        if mode == 'repeat':
                                            assert tokens == previous, 'Warm cache changed output'
                                            assert record['reused'] > 0
                                        key = (case, mode)
                                        reference = references.setdefault(key, tokens)
                                        record['same_output_as_first_seen'] = tokens == reference
                                        record['first_seen_difference'] = next((i for i, (x,y) in enumerate(zip(tokens, reference)) if x != y),
                                                                          min(len(tokens),len(reference)) if len(tokens) != len(reference) else None)
                                        record['decode_tok_s'] = len(tokens)*1000/record['decode_ms']
                                        record['fresh_tokens'] = record['prompt_tokens']-record['reused']
                                        record['prefill_tok_s'] = record['fresh_tokens']*1000/record['prompt_ms']
                                        previous = tokens
                                        save()
                                        print(f"DONE r{r+1} t{window} p{gate} gpu{fraction} {case}-{mode}: {record['decode_tok_s']:.2f} tok/s, accepted={record.get('drafts_accepted')}/{record.get('drafts_offered')}, first_seen_exact={record['same_output_as_first_seen']}", flush=True)
                        row['status'] = 'pass'
                finally:
                    if svc: svc.drop_embeddings()
                    if engine: engine.close()
                    if vision: vision.shutdown()
                    row['telemetry'] = tel.close()
                    save()
                logtext = (folder / 'engine.log').read_text(encoding='utf-8', errors='replace')
                assert 'post-residency expert source bytes=0' in logtext
                assert 'post-residency MTP source bytes=0' in logtext
        zero_refs = {(run['round'],q['case'],q['mode']):q for run in result['runs'] if run['window']==a.windows[0]
                     for q in run['requests'] if q['gate']==0 and q['pcie_frac']==a.pcie_fracs[0]}
        for run in result['runs']:
            for q in run['requests']:
                ref = zero_refs[(run['round'],q['case'],q['mode'])]
                assert q['input_sha256'] == ref['input_sha256']
                q['same_output_as_reference'] = q['ids'] == ref['ids']
                q['first_difference'] = next((i for i,(x,y) in enumerate(zip(q['ids'],ref['ids'])) if x!=y),
                                             min(len(q['ids']),len(ref['ids'])) if len(q['ids'])!=len(ref['ids']) else None)
        summary = []
        for window in a.windows:
            for gate in a.gates:
                for fraction in a.pcie_fracs:
                    rows = [q for run in result['runs'] if run['window'] == window for q in run['requests']
                            if q['gate'] == gate and q['pcie_frac'] == fraction]
                    summary.append(dict(window=window, gate=gate, pcie_frac=fraction, samples=len(rows),
                                        decode_median_tok_s=statistics.median(q['decode_tok_s'] for q in rows),
                                        decode_aggregate_tok_s=sum(q['generated'] for q in rows)*1000/sum(q['decode_ms'] for q in rows),
                                        all_outputs_exact=all(q['same_output_as_reference'] for q in rows)))
        result.update(status='pass', summary=summary)
        save()
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        result.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
