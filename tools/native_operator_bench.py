"""Measure unchanged operator configs on private stdin engines, never HTTP.

Keep cold/warm requests and full output sequences, including draft acceptance.
Different configurations see the same requests in the same order. This is not
a state-hash run: timings include normal inference and cache work only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.server import StrataEngine, child_env
from serve.frontend import ChatTemplate
from safetensors_tokenizer import SafetensorsTokenizer
from native_telemetry import Telemetry

TASKS = (
    ('english', 'Explain how to implement an LRU cache in Python. Include the complete code, three worked examples and how to test eviction.'),
    ('chinese', '详细解释归并排序，给出完整 Python 实现和三个测试例子，说明每一步的结果和边界处理。'),
    ('code', 'Write a detailed Python tutorial on binary search. Include lower_bound, upper_bound, duplicate handling and unit tests.'),
)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--configs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--new', type=int, default=512)
    p.add_argument('--output-comparison', choices=('exact', 'report'), default='exact',
                   help='report records cross-config differences; warm reuse must always be exact')
    a = p.parse_args()
    assert a.rounds > 0 and a.new > 0
    a.output.mkdir(parents=True, exist_ok=False)
    results = {'runs': [], 'output_comparison': a.output_comparison,
               'started_unix_s': time.time(), 'platform': platform.platform(),
               'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip(),
               'timing_note': 'Private stdin; unchanged config; first prompt includes graph capture; no state hashes or logit dumps.'}
    def save():
        (a.output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf8')
    for index, path in enumerate(a.configs):
        active = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command',
            "@(Get-Process -Name 'strata*' -ErrorAction SilentlyContinue).Count"], text=True)
        if int(active.strip() or '0'):
            raise RuntimeError('Another Strata engine is running')
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True).strip()
        free, util = map(int, gpu.splitlines()[0].split(','))
        if free < 24000 or util > 15:
            raise RuntimeError(f'GPU is busy: {gpu}')
        cfg = json.loads(path.read_text(encoding='utf-8-sig'))
        tok = SafetensorsTokenizer.from_directory(cfg['tokenizer'])
        template = ChatTemplate(Path(cfg['chat_template']))
        folder = a.output / str(index)
        folder.mkdir()
        log = folder / 'engine.log'
        row = dict(config=cfg, config_path=str(path), requests=[], gpu_before=gpu,
                   exe_sha256=hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest())
        results['runs'].append(row)
        save()
        tel = Telemetry(folder / 'telemetry.jsonl')
        engine = None
        try:
            start = time.monotonic()
            env = child_env(cfg)
            if 'STRATA_STATE_HASH' in env or 'STRATA_DUMP_FIRST_LOGITS' in env:
                raise RuntimeError('State hashes/logit dumps would confound performance timing')
            engine = StrataEngine(cfg['exe'], cfg['args'], cwd=cfg['cwd'], log=str(log.resolve()), env=env)
            row['startup_s'] = time.monotonic() - start
            row['info'] = dict(engine.info)
            assert row['info']['weight_source'] == 'safetensors'
            assert row['info']['expert_ram_bytes'] == row['info']['expert_cuda_pinned_bytes'] > 0
            tel.pid = engine.proc.pid
            for r in range(a.rounds):
                for category, text in TASKS:
                    text += f' Test case {r+1}.'
                    ids = tok.encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False), parse_special=True)
                    previous = None
                    for mode in ('cold', 'warm'):
                        start = time.monotonic()
                        first = None
                        out = []
                        for token in engine.generate(ids, a.new, {'temperature': 0}, threading.Event()):
                            if token is not None:
                                if first is None:
                                    first = time.monotonic() - start
                                out.append(token)
                        record = dict(name=f'{category}-{r}-{mode}', ids=out,
                            input_sha256=hashlib.sha256(' '.join(map(str, ids)).encode('ascii')).hexdigest(),
                            wall_s=time.monotonic()-start, ttft_s=first, **engine.last)
                        row['requests'].append(record)
                        save()
                        assert 0 < len(out) <= a.new and record['generated'] == len(out)
                        assert record['file_blobs'] == record['file_mb'] == 0
                        if mode == 'warm':
                            assert out == previous, 'Warm prefix changed output'
                            assert record['reused'] > 0, 'Warm prefix was not reused'
                        if index:
                            ref = results['runs'][0]['requests'][len(row['requests'])-1]
                            assert record['name'] == ref['name'] and record['input_sha256'] == ref['input_sha256']
                            record['same_output_as_reference'] = out == ref['ids']
                            if a.output_comparison == 'exact':
                                assert record['same_output_as_reference'], 'Config changed greedy output'
                        previous = out
                        record['decode_tok_s'] = len(out)*1000/record['decode_ms']
                        offered = record.get('drafts_offered', 0)
                        record['draft_acceptance'] = record.get('drafts_accepted', 0)/offered if offered else None
                        save()
                        print(index, record['name'], len(out), f"{record['decode_tok_s']:.2f} tok/s", 'acceptance', record['draft_acceptance'], flush=True)
            row['decode_median_tok_s'] = statistics.median(x['decode_tok_s'] for x in row['requests'])
            row['status'] = 'pass'
        except Exception as error:
            row.update(status='failed', error=str(error))
            raise
        finally:
            row['telemetry'] = tel.close()
            if engine:
                engine.close()
            save()
        text = log.read_text(encoding='utf8')
        assert 'post-residency expert source bytes=0' in text
        assert 'post-residency MTP source bytes=0' in text
    results['status'] = 'pass'
    save()
    print(f'PASS: operator configs, {a.output_comparison} cross-config comparison, exact cold/warm outputs, zero expert/MTP source reads', flush=True)


if __name__ == '__main__':
    main()
