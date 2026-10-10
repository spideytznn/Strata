"""Check the configured image path using private stdin, without HTTP or a benchmark.

Two generated solid-color PNGs exercise image embedding replacement, warm reuse,
image changes and return to text. This is a functional check, not a quality suite.
"""
import argparse
import hashlib
import json
import struct
import sys
import threading
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from safetensors_tokenizer import SafetensorsTokenizer
from serve.frontend import ChatTemplate
from serve.server import Service, StrataEngine, Vision, child_env, sampling_defaults_from_config, vision_env


def solid_png(path, rgb):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    size = 224
    raw = (b'\0' + bytes(rgb) * size) * size
    path.write_bytes(b'\x89PNG\r\n\x1a\n' +
                     chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)) +
                     chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    assert '--vision' in cfg['args'] and cfg['vision']['gpu']
    defaults = sampling_defaults_from_config(cfg)
    assert defaults == dict(temperature=1.0, top_p=0.95, top_k=20,
                            min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)
    tok = SafetensorsTokenizer.from_directory(cfg['tokenizer'])
    template = ChatTemplate(Path(cfg['chat_template']))
    result = dict(config=cfg, config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  exe_sha256=hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),
                  note='Functional private-stdin image check only; no HTTP or speed comparison.', requests=[])
    engine = vision = svc = None
    log = (args.output / 'engine.log').open('w', encoding='utf-8')
    def save():
        (args.output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    try:
        env = child_env(cfg)
        vision = Vision(cfg['vision'], log=log, env=vision_env(cfg, env))
        # Match server startup: reserve the encoder's GPU memory before expert sizing.
        engine = StrataEngine(cfg['exe'], cfg['args'], cwd=cfg['cwd'],
                              log=str((args.output / 'engine.log').resolve()), env=env)
        result['info'] = dict(engine.info)
        assert result['info']['weight_source'] == 'safetensors'
        svc = Service(engine, tok, template, vision=vision, sampling_defaults=defaults)
        def generate(name, messages, expected, sampling, thinking):
            ids, actual_thinking, limit = svc.prepare(messages, [], {'enable_thinking': thinking}, max_new=32)
            assert actual_thinking == thinking
            embeddings = getattr(svc.embeddings, 'path', None)
            try:
                out = [t for t in engine.generate(ids, limit, sampling, threading.Event(), embeddings=embeddings) if t is not None]
            finally:
                svc.drop_embeddings()
            text = tok.decode(out)
            row = dict(name=name, ids=out, text=text, input_tokens=len(ids),
                       image_tokens=ids.count(248056), reused=engine.last['reused'],
                       file_blobs=engine.last['file_blobs'], file_mb=engine.last['file_mb'])
            result['requests'].append(row)
            save()
            assert out and row['file_blobs'] == row['file_mb'] == 0
            if expected:
                assert expected.lower() in text.lower(), (name, text)
            return row
        previous = None
        for color, rgb in [('red', (255,0,0)), ('blue', (0,0,255)), ('red', (255,0,0))]:
            image = (args.output / (color+'.png')).resolve()
            solid_png(image, rgb)
            messages = [{'role':'user','content':[{'type':'image','source':str(image)},
                        {'type':'text','text':'What is the solid background color? Answer with one English color word only.'}]}]
            first = generate(color+'-cold-or-return', messages, color, {'temperature':0}, False)
            assert first['image_tokens'] > 0
            if color == 'red' and previous is not None:
                assert first['ids'] == previous
            if color == 'red': previous = first['ids']
            warm = generate(color+'-warm', messages, color, {'temperature':0}, False)
            assert warm['ids'] == first['ids'] and warm['reused'] > 0
        generate('text-after-images', [{'role':'user','content':'Reply with exactly ZEBRA-417.'}],
                 'ZEBRA-417', {'temperature':0}, False)
        # Exercise thinking + every official sampling field through the real image protocol.
        generate('image-official-thinking-sampling', messages, None, {**defaults, 'seed':9950}, True)
        result['status'] = 'pass'
        save()
        print('PASS: image colors, warm reuse, A/B/A, text reset and official thinking sampling')
    except Exception as error:
        result.update(status='fail', error=repr(error))
        save()
        raise
    finally:
        if svc: svc.drop_embeddings()
        if engine: engine.close()
        if vision: vision.close()
        log.close()


if __name__ == '__main__':
    main()
