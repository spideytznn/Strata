"""Reconstruct the capacity fixture; frozen IDs are authoritative across newlines."""
import hashlib
import json
from pathlib import Path
import sys

sys.path[:0] = ['.', 'tools']  # run from the repository root
from safetensors_tokenizer import SafetensorsTokenizer

tok = SafetensorsTokenizer.from_directory(r'D:\迅雷下载\Qwen3.8-flash-next-nvfp4')
source = Path('docs/DETAILS.md').read_bytes()
target = 262000
head = tok.encode('<|im_start|>user\nRead the following archive.\n', parse_special=True)
tail = tok.encode(f'\nThe exact access code is ZEBRA-{target}. Reply only with that exact code.'
                  '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', parse_special=True)
unit = tok.encode(source.decode('utf8'))
count = target - len(head) - len(tail)
ids = head + (unit * ((count + len(unit) - 1) // len(unit)))[:count] + tail
assert len(ids) == target
out = Path('logs/efficiency/capacity-recreated.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps([[f'capacity-{target}', ids, 128]]), encoding='utf8')
print('source SHA256:', hashlib.sha256(source).hexdigest(), 'tokens:', len(ids), 'output:', out)
