"""Summarize completed alternating offline CLI pairs; do not infer quality from speed."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prefix', default='final-pair')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = []
    reference = None
    for round_id in range(1, a.rounds+1):
        for kind in ('old', 'new'):
            folder = a.root / f'{a.prefix}-r{round_id}-{kind}'
            m = json.loads((folder / 'manifest.json').read_text(encoding='utf8'))
            assert m['exit_code'] == 0 and m['decode']['tokens'] == len(m['output_tokens']) == 128
            assert m['prefill']['tokens'] == 8198 and m['prompt_tokens'] == 8199
            stderr = (folder / 'stderr.txt').read_text(encoding='utf8')
            stdout = (folder / 'stdout.txt').read_text(encoding='utf8')
            assert 'post-residency expert source bytes=0' in stderr
            assert 'post-residency MTP source bytes=0' in stderr
            assert 'STRATA_STATE_HASH' not in m['env'] and 'STRATA_DUMP_FIRST_LOGITS' not in m['env']
            assert m['env'].get('STRATA_PREFILL_TIMING', '0') == '0'
            if reference is None:
                reference = m
            assert m['tokens_sha256'] == reference['tokens_sha256']
            assert m['output_tokens'] == reference['output_tokens'], 'Paired greedy outputs differ'
            cpu = [json.loads(line).get('cpu_percent') for line in (folder / 'cpu.jsonl').read_text().splitlines()]
            experts = re.search(r'CPU experts ([\d.]+) distinct / ([\d.]+) routed per layer', stdout)
            row = dict(round=round_id, kind=kind, manifest=str(folder / 'manifest.json'),
                       manifest_sha256=hashlib.sha256((folder / 'manifest.json').read_bytes()).hexdigest(),
                       exe_sha256=m['exe_sha256'], prefill=m['prefill'], decode=m['decode'],
                       peak_process_cpu_percent=max((v for v in cpu if v is not None), default=None),
                       cpu_experts_per_layer=list(map(float, experts.groups())) if experts else None)
            rows.append(row)
    medians = {kind: {phase: statistics.median(r[phase]['tokens_per_second'] for r in rows if r['kind'] == kind)
                      for phase in ('prefill', 'decode')} for kind in ('old', 'new')}
    result = dict(status='pass', prompt_sha256=reference['tokens_sha256'], rounds=a.rounds,
                  output_tokens=reference['output_tokens'], all_outputs_equal=True, rows=rows,
                  median_tok_s=medians,
                  median_ratio={phase: medians['new'][phase]/medians['old'][phase] for phase in ('prefill', 'decode')},
                  note='Same varied 8199-token input, 128 outputs; fresh sequential engines, round2 reversed. '
                       'Engine phase timings exclude startup and first graph capture. CPU peak includes all process phases, '
                       'normalized across all logical CPUs; not a decode average. No state hash or GPU timing events.')
    a.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({k: result[k] for k in ('status', 'median_tok_s', 'median_ratio')}, indent=2))


if __name__ == '__main__':
    main()
