"""Focused cold/warm operator-state diagnostic, private engines and no HTTP.

State/logit transfers add overhead: these are correctness runs, not timings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.server import StrataEngine, child_env, SessionRefused
from serve.frontend import ChatTemplate
from safetensors_tokenizer import SafetensorsTokenizer
from conversation_cache_parity import state_hashes
from native_operator_bench import TASKS


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--configs', type=Path, nargs='+', required=True)
    p.add_argument('--category', choices=[name for name, _ in TASKS], default='code')
    p.add_argument('--new', type=int, default=1)
    p.add_argument('--reject-session', type=Path,
                   help='require config-fingerprint rejection before generating any tokens')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert a.new > 0
    a.output.mkdir(parents=True, exist_ok=False)
    results = {'runs': [], 'note': __doc__}
    for index, path in enumerate(a.configs):
        active = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command',
            "@(Get-Process -Name 'strata*' -ErrorAction SilentlyContinue).Count"], text=True)
        assert int(active.strip() or '0') == 0, 'Another Strata engine is running'
        cfg = json.loads(path.read_text(encoding='utf-8-sig'))
        tok = SafetensorsTokenizer.from_directory(cfg['tokenizer'])
        template = ChatTemplate(Path(cfg['chat_template']))
        text = dict(TASKS)[a.category] + ' Test case 1.'
        ids = tok.encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False), parse_special=True)
        folder = a.output / str(index)
        folder.mkdir()
        log = folder / 'engine.log'
        env = child_env(cfg)
        env.update(STRATA_STATE_HASH='1', STRATA_STATE_HASH_GDN='1',
                   STRATA_DUMP_FIRST_LOGITS=str((folder / 'first-logits').resolve()))
        row = dict(config=cfg, config_path=str(path), input_ids=ids, requests=[],
                   exe_sha256=hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest())
        results['runs'].append(row)
        engine = None
        try:
            engine = StrataEngine(cfg['exe'], cfg['args'], cwd=cfg['cwd'], log=str(log.resolve()), env=env)
            if a.reject_session:
                try:
                    engine.session_file('restore', str(a.reject_session.resolve(strict=True)))
                except SessionRefused as error:
                    assert error.kind == 'invalid' and not error.published
                    assert 'config fingerprint differs' in str(error), str(error)
                    row['incompatible_session_refused'] = dict(kind=error.kind, reason=str(error),
                        file_sha256=hashlib.sha256(a.reject_session.read_bytes()).hexdigest())
                    print(index, 'incompatible session rejected before payload restore', flush=True)
                else:
                    raise AssertionError('Incompatible session was accepted')
            for mode in ('cold', 'warm'):
                out = [v for v in engine.generate(ids, a.new, {'temperature': 0}, threading.Event()) if v is not None]
                row['requests'].append(dict(mode=mode, ids=out, **engine.last))
                assert 0 < len(out) <= a.new
                assert engine.last['file_blobs'] == engine.last['file_mb'] == 0
                print(index, mode, len(out), 'reused', engine.last['reused'], flush=True)
        finally:
            if engine:
                engine.close()
            row['states'] = state_hashes(log.read_text(encoding='utf8'))
            if len(row['states']) == 2:
                row['own_state_equal'] = row['states'][0] == row['states'][1]
                row['own_output_equal'] = row['requests'][0]['ids'] == row['requests'][1]['ids']
                row['state_differences'] = [k for k, v in row['states'][0].items() if v != row['states'][1][k]]
                print(index, 'state differences', row['state_differences'], flush=True)
            (a.output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
