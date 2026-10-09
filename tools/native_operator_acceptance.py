"""Exercise a generated operator config unchanged, including auto VRAM sizing."""
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
from native_http_acceptance import validate_http
from native_telemetry import Telemetry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--http', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text(encoding='utf8'))
    requests = json.loads((a.reference / 'requests.json').read_text(encoding='utf8'))
    reference = json.loads((a.reference / 'results.json').read_text(encoding='utf8'))['cases']['reference']['requests']
    tok = SafetensorsTokenizer.from_directory(cfg['tokenizer'])
    env = child_env(cfg)
    env['STRATA_PREFILL_TRACE'] = '1'
    command = [cfg['exe'], *cfg['args']]
    (a.output / 'command.json').write_text(json.dumps({'command': command,
        'env': {k:v for k,v in env.items() if k.startswith('STRATA_')}}, indent=2), encoding='utf8')
    row = {'config': cfg, 'binary_sha256': hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),
           'requests': [], 'started_unix_s': time.time()}
    def save():
        (a.output / 'results.json').write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf8')
    save()
    tel = Telemetry(a.output / 'telemetry.jsonl')
    engine = None
    try:
        start = time.monotonic()
        engine = StrataEngine(cfg['exe'], cfg['args'], cwd=cfg['cwd'],
                              log=str((a.output / 'engine.log').resolve()), env=env)
        tel.pid = engine.proc.pid
        row['startup_s'] = time.monotonic() - start
        row['info'] = dict(engine.info)
        assert engine.info['weight_source'] == 'safetensors'
        assert engine.info['expert_ram_bytes'] == engine.info['expert_cuda_pinned_bytes'] > 0
        for (name, ids, limit), expected in zip(requests, reference, strict=True):
            start = time.monotonic()
            out = []
            first = None
            for token in engine.generate(ids, limit, {'temperature':0}, threading.Event()):
                if token is not None:
                    if first is None: first = time.monotonic() - start
                    out.append(token)
            r = {'name':name, 'ids':out, 'text':tok.decode(out), 'ttft_s':first,
                 'wall_s':time.monotonic()-start, **engine.last}
            r['tokens_equal_reference'] = out == expected['ids']
            row['requests'].append(r)
            save()
            assert r['generated'] == len(out) == limit
            assert r['file_blobs'] == r['file_mb'] == 0
            assert r['tokens_equal_reference'], name
            print(name, r['ttft_s'], r['decode_ms'], flush=True)
        if a.http:
            row['http'] = validate_http(engine, tok, ChatTemplate(Path(cfg['chat_template'])), a.output / 'http.json')
    except Exception as error:
        row.update(status='failed', error=str(error))
        raise
    finally:
        row['telemetry'] = tel.close()
        if engine: engine.close()
        save()
    log = (a.output / 'engine.log').read_text(encoding='utf8')
    try:
        assert 'tokens=8192 max_chunk=8192' in log, 'actual 8192-token prefill was not executed'
        assert 'post-residency expert source bytes=0' in log
        row['status'] = 'pass'
    except Exception as error:
        row.update(status='failed', error=str(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
