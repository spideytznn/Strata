"""Summarize repeated baseline requests without mixing cold and reused prefixes."""
import argparse
import json
import statistics
from pathlib import Path


def distribution(values):
    return {'min': min(values), 'median': statistics.median(values), 'max': max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads((args.run / 'results.json').read_text(encoding='utf8'))
    assert raw['status'] == 'pass'
    summary = {'conditions':raw['conditions'], 'cases':{},
               'notes':['Each named request is summarized across fresh processes; cold/warm are separate.',
                        'Decode rate excludes first token; tokens are measured by the engine, not text characters.',
                        'Q4XL uses different quantized weights and is not a fidelity oracle.',
                        'GPU memory and clocks include the desktop; Windows faults do not measure physical SSD reads.']}
    for case, repeats in raw['cases'].items():
        assert len(repeats) >= 3 and all(r['status'] == 'pass' for r in repeats)
        names = [r['name'] for r in repeats[0]['requests']]
        assert all([r['name'] for r in rep['requests']] == names for rep in repeats)
        row = {'repeats':len(repeats), 'requests':{},
               'expert_slots':[r['info']['expert_slots'] for r in repeats],
               'startup_s':distribution([r['startup_s'] for r in repeats]),
               'peak_working_set_bytes':max(r['telemetry']['peak_working_set_bytes'] for r in repeats)}
        for name in names:
            measured = [next(r for r in rep['requests'] if r['name'] == name) for rep in repeats]
            assert len({r['generated'] for r in measured}) == 1
            assert all(r['file_blobs'] == r['file_mb'] == 0 for r in measured)
            row['requests'][name] = {key:distribution([r[key] for r in measured]) for key in
                ('generated','ttft_s','prompt_ms','after_first_token_per_s','wall_s','reused','cpu_expert_entries')}
        samples = [json.loads(line) for folder in args.run.glob(case+'-*') if folder.is_dir()
                   for line in (folder/'telemetry.jsonl').read_text(encoding='utf8').splitlines()]
        intervals = [(r['started_unix_s'],r['started_unix_s']+r['wall_s'])
                     for rep in repeats for r in rep['requests']]
        all_gpu = [s['gpu_csv'].split(',') for s in samples if s.get('gpu_csv') and '\n' not in s['gpu_csv']]
        row['whole_process_peak_gpu_used_mib'] = max(float(g[3]) for g in all_gpu)
        gpu = [s['gpu_csv'].split(',') for s in samples if s.get('gpu_csv') and '\n' not in s['gpu_csv']
               and any(start <= s['unix_s'] <= end for start,end in intervals)]
        row['gpu_sampling_note'] = 'Distributions below use measured request intervals; peak above includes startup.'
        row['gpu_samples'] = len(gpu)
        for key, index in [('used_mib',3),('sm_mhz',6),('memory_mhz',7),('watts',8),('celsius',9)]:
            values = []
            for g in gpu:
                try: values.append(float(g[index].strip()))
                except (ValueError,IndexError): pass
            if values: row['gpu_'+key] = distribution(values)
        summary['cases'][case] = row
    with args.output.open('x', encoding='utf8') as output:
        json.dump(summary, output, indent=2)
    print(args.output)


if __name__ == '__main__':
    main()
