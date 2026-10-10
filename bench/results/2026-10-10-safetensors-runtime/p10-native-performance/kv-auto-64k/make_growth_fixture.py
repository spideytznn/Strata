import hashlib,json,sys
from pathlib import Path
sys.path[:0]=['.','tools']
from safetensors_tokenizer import SafetensorsTokenizer
tok=SafetensorsTokenizer.from_directory(r'D:\迅雷下载\Qwen3.8-flash-next-nvfp4')
source=Path('docs/DETAILS.md').read_bytes()
head=tok.encode('<|im_start|>user\nRead the following archive.\n',parse_special=True)
tail=tok.encode('\nThe exact access code is ZEBRA-65543. Reply only with that exact code.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',parse_special=True)
unit=tok.encode(source.decode('utf8'))
count=65543-len(head)-len(tail)
ids=head+(unit*((count+len(unit)-1)//len(unit)))[:count]+tail
assert len(ids)==65543
folder=Path('logs/efficiency')
(folder/'growth-fixture.json').write_text(json.dumps([['growth-65543',ids,128]]),encoding='utf8')
(folder/'growth-fixture-provenance.json').write_text(json.dumps(dict(source='docs/DETAILS.md',source_sha256=hashlib.sha256(source).hexdigest(),tokens=65543,expected='ZEBRA-65543',note='Capacity/grow/trim/restore fixture; repeated documentation is not a throughput benchmark.'),indent=2),encoding='utf8')
