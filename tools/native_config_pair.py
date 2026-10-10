"""Compare complete operator configs on private stdin engines, without HTTP.

Keeps context, workspace, adaptive cache and MTP settings from each config.
State hashing is deliberately a correctness run, not a throughput measurement.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.server import StrataEngine, child_env
from conversation_cache_parity import state_hashes
from native_telemetry import Telemetry
from safetensors_tokenizer import SafetensorsTokenizer
from serve.frontend import ChatTemplate


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--configs', nargs='+', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--cases', default='details-8199,details-24583')
    p.add_argument('--reference', type=Path, help='previous passing result; compare against its last config')
    p.add_argument('--state-comparison', choices=('exact', 'report'), default='exact',
                   help='report records state differences when CPU/GPU assignments intentionally change')
    p.add_argument('--long-restore', action='store_true',
                   help='save the last long session, then restore it after the short A/B/A sequence')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    fixtures = {name: ids for name, ids, _ in json.loads(a.fixture.read_text())}
    results = {'runs': [], 'state_comparison': a.state_comparison,
               'note': 'Private stdin engines only; state hashing adds overhead.'}
    reference = None
    if a.reference:
        previous = json.loads(a.reference.read_text(encoding='utf8'))
        assert previous['status'] == 'pass', 'reference did not pass'
        reference = previous['runs'][-1]
        results['reference'] = {'path': str(a.reference), 'sha256': hashlib.sha256(a.reference.read_bytes()).hexdigest()}
    def save():
        (a.output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf8')
    for index, path in enumerate(a.configs):
        active = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command',
            "@(Get-Process -Name 'strata*' -ErrorAction SilentlyContinue).Count"], text=True)
        if int(active.strip() or '0'):
            raise RuntimeError('Another Strata engine is running')
        cfg = json.loads(path.read_text(encoding='utf-8-sig'))
        tok = SafetensorsTokenizer.from_directory(cfg['tokenizer'])
        template = ChatTemplate(Path(cfg['chat_template']))
        def prompt(text):
            return tok.encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False), parse_special=True)
        folder = a.output / str(index)
        folder.mkdir()
        log = folder / 'engine.log'
        env = child_env(cfg)
        env['STRATA_STATE_HASH'] = '1'
        row = {'config_path': str(path), 'config': cfg, 'requests': [],
               'exe_sha256': hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest()}
        results['runs'].append(row)
        save()
        tel = Telemetry(folder / 'telemetry.jsonl')
        engine = None
        try:
            engine = StrataEngine(cfg['exe'], cfg['args'], cwd=cfg['cwd'], log=str(log.resolve()), env=env)
            tel.pid = engine.proc.pid
            row['info'] = dict(engine.info)
            assert row['info']['weight_source'] == 'safetensors'
            assert row['info']['expert_ram_bytes'] == row['info']['expert_cuda_pinned_bytes'] > 0
            def generate(name, ids, limit):
                start = time.monotonic()
                out = [v for v in engine.generate(ids, limit, {'temperature': 0}, threading.Event()) if v is not None]
                record = {'name': name, 'ids': out, 'wall_s': time.monotonic()-start, **engine.last}
                row['requests'].append(record)
                save()
                assert 0 < len(out) <= limit and record['generated'] == len(out), name
                assert record['file_blobs'] == record['file_mb'] == 0, name
                before = results['runs'][0] if index else reference
                if before:
                    expected = before['requests'][len(row['requests'])-1]
                    assert name == expected['name'] and out == expected['ids'], f'{name}: tokens differ'
                print(index, name, len(out), 'reused', record['reused'], flush=True)
                return out
            for case in a.cases.split(','):
                ids = fixtures[case]
                initial = generate(case+'-initial', ids, 128)
                warm = generate(case+'-warm', ids, 128)
                assert warm == initial, f'{case}: prefix reuse changed output'
                chunk = int(cfg['args'][cfg['args'].index('--prefill')+1])
                # Prompt checkpoints follow chunk boundaries; the small tail
                # after the last full chunk is legitimately read again.
                checkpoint = ((len(ids)-1)//chunk)*chunk
                assert row['requests'][-1]['reused'] >= checkpoint > 0, f'{case}: warm prefix not reused'
            if a.long_restore:
                long_ids, long_output, long_case = ids, initial, case
                long_snapshot = folder / 'long-session.bin'
                row['long_saved'] = engine.session_file('save', str(long_snapshot.resolve()))
            pa = prompt('Remember this exact code: ZEBRA-417. Say OK.')
            pb = prompt('Remember this different code: ORBIT-926. Say OK.')
            first = generate('A', pa, 1)
            suffix = tok.encode('<|im_end|>\n<|im_start|>user\nWhat code did I give you?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', parse_special=True)
            continuation = pa + first + suffix
            generate('B', pb, 1)
            generate('A+', continuation, 16)
            assert row['requests'][-1]['reused'] >= len(pa), 'A/B/A did not reuse A'
            snapshot = folder / 'session.bin'
            row['saved'] = engine.session_file('save', str(snapshot.resolve()))
            generate('B-again', pb, 1)
            row['restored'] = engine.session_file('restore', str(snapshot.resolve()))
            generate('A+-restored', continuation, 16)
            assert row['requests'][-1]['reused'] > 0, 'snapshot restore did not reuse'
            assert row['requests'][-1]['ids'] == row['requests'][-3]['ids'], 'snapshot continuation differs'
            if a.long_restore:
                row['long_restored'] = engine.session_file('restore', str(long_snapshot.resolve()))
                restored = generate(long_case+'-disk-restored-after-short', long_ids, 128)
                assert restored == long_output, 'long disk restore changed output'
                assert row['requests'][-1]['reused'] >= ((len(long_ids)-1)//chunk)*chunk, 'long restore did not reuse'
        except Exception as error:
            row.update(status='failed', error=str(error))
            raise
        finally:
            row['telemetry'] = tel.close()
            if engine:
                engine.close()
            save()
        text = log.read_text(encoding='utf8')
        row['states'] = state_hashes(text)
        assert len(row['states']) == len(row['requests']), 'missing state fingerprints'
        before = results['runs'][0] if index else reference
        if before:
            row['committed_state_equal'] = row['states'] == before['states']
            if a.state_comparison == 'exact':
                assert row['committed_state_equal'], 'committed state differs'
        assert 'post-residency expert source bytes=0' in text
        assert 'post-residency MTP source bytes=0' in text
        assert 'tokens=8192 max_chunk=8192' in text, '8192 prefill was not executed'
        row['status'] = 'pass'
        save()
    results['status'] = 'pass'
    save()
    state_note = 'exact committed states' if a.state_comparison == 'exact' else 'state differences recorded, not gated'
    print(f'PASS: complete config tokens, {state_note}, long prefill, prefix reuse, A/B/A, disk restore, zero expert/MTP reads', flush=True)


if __name__ == '__main__':
    main()
