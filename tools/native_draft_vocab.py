"""Build a draft-only ID list from the original HF tokenizer, retaining CJK and UTF-8 fragments.

No weights are read or quantized. The engine gathers unchanged BF16 head rows;
the target model always retains its full vocabulary and verifies every proposal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import regex
from safetensors_tokenizer import SafetensorsTokenizer


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists() and not a.output.with_suffix('.json').exists()
    tok = SafetensorsTokenizer.from_directory(a.tokenizer)
    raw = a.base.read_bytes()
    assert raw and len(raw) % 4 == 0
    ids = [x[0] for x in struct.iter_unpack('<i',raw)]
    assert len(ids) == len(set(ids)) and all(i >= 0 for i in ids)
    # The older pack's ID list includes the head's padding rows. Only actual
    # IDs in the original HF tokenizer are candidates for the native subset.
    dropped = [i for i in ids if i >= len(tok.tokens)]
    ids = [i for i in ids if i < len(tok.tokens)]
    selected = set(ids)
    cjk = regex.compile(r'\p{Script=Han}|\p{Script=Hiragana}|\p{Script=Katakana}|\p{Script=Hangul}|[\u3000-\u303f\uff00-\uffef]')
    required, cjk_count, fragment_count = set(), 0, 0
    for i in range(len(tok.tokens)):
        text = tok.decode([i])
        has_cjk = bool(cjk.search(text))
        fragment = '\ufffd' in text
        cjk_count += has_cjk
        fragment_count += fragment
        if has_cjk or fragment or i < 256 or tok.token_types[i] != 1:
            required.add(i)
    extra = sorted(required-selected)
    ids.extend(extra)
    assert required <= set(ids) and len(ids)==len(set(ids))
    payload = b''.join(struct.pack('<i',i) for i in ids)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_bytes(payload)
    manifest = dict(tokenizer=str(a.tokenizer),tokenizer_sha256=hashlib.sha256((a.tokenizer/'tokenizer.json').read_bytes()).hexdigest(),
                    base=str(a.base),base_sha256=hashlib.sha256(raw).hexdigest(),base_tokens=len(selected),
                    base_ids_absent_from_original_tokenizer=dropped,
                    full_vocabulary=len(tok.tokens),draft_tokens=len(ids),added_tokens=len(extra),
                    cjk_token_rows=cjk_count,utf8_fragment_rows=fragment_count,
                    output=str(a.output),output_sha256=hashlib.sha256(payload).hexdigest(),
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    note='Draft-only token IDs; full target vocabulary unchanged. All decoded CJK rows, UTF-8 fragments, byte fallback and added tokens retained. No model weights read or quantized.')
    a.output.with_suffix('.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
