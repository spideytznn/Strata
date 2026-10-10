"""Audit complete speculation screens and rank only output-identical choices."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def audit(path, reference_path=None):
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['status'] == 'pass'
    args = data['args']
    windows, gates, fractions = args['windows'], args['gates'], args['pcie_fracs']
    cases = args['cases'].split(',')
    assert 0 in gates, 'A zero-gate reference is required'
    assert len(data['runs']) == args['rounds'] * len(windows)
    assert len({run['exe_sha256'] for run in data['runs']}) == 1
    seen, references, groups = set(), {}, {}
    for run in data['runs']:
        identity = (run['round'], run['window'])
        assert identity not in seen
        seen.add(identity)
        assert run['status'] == 'pass'
        info = run['info']
        assert info['weight_source'] == 'safetensors' and info['spec'] == run['window']
        assert info['expert_ram_bytes'] == info['expert_cuda_pinned_bytes'] > 0
        assert info['prefill_chunk'] == int(run['config']['args'][run['config']['args'].index('--prefill')+1])
        assert info['vram_free_mib'] >= 512
        assert len(run['requests']) == len(cases) * len(gates) * len(fractions) * 2
        requests = {}
        for q in run['requests']:
            key = (q['case'], q['gate'], q['pcie_frac'], q['mode'])
            assert key not in requests
            requests[key] = q
            assert q['case'] in cases and q['gate'] in gates and q['pcie_frac'] in fractions
            assert q['file_blobs'] == q['file_mb'] == 0
            assert 0 < q['generated'] == len(q['ids']) <= args['new']
            assert q['decode_ms'] > 0 and q['prompt_ms'] > 0
            if run['window'] == windows[0] and q['gate'] == 0 and q['pcie_frac'] == fractions[0]:
                references[(run['round'],q['case'],q['mode'])] = q
            groups.setdefault((run['window'],q['gate'],q['pcie_frac']),[]).append((run['round'],q))
        for case in cases:
            for gate in gates:
                for fraction in fractions:
                    first, repeat = (requests[(case,gate,fraction,mode)] for mode in ('first','repeat'))
                    assert first['ids'] == repeat['ids'], 'Warm cache changed output'
                    assert repeat['reused'] > 0
    external = None
    if reference_path:
        previous = json.loads(reference_path.read_text(encoding='utf-8'))
        assert previous['status']=='pass'
        prior_args=previous['args']
        external={(run['round'],q['case'],q['mode']):q for run in previous['runs'] if run['window']==prior_args['windows'][0]
                  for q in run['requests'] if q['gate']==0 and q['pcie_frac']==prior_args['pcie_fracs'][0]}
    summary = []
    for (window,gate,fraction), rows in groups.items():
        exact = []
        for r,q in rows:
            ref = references[(r,q['case'],q['mode'])]
            assert q['input_sha256'] == ref['input_sha256']
            exact.append(q['ids'] == ref['ids'])
            if external:
                ref=external[(r,q['case'],q['mode'])]
                assert q['input_sha256']==ref['input_sha256']
                assert q['ids']==ref['ids'], 'Candidate changed accepted reference output'
        assert all(exact), 'Speculation changed the zero-gate reference output'
        count = sum(q['generated'] for _,q in rows)
        elapsed = sum(q['decode_ms'] for _,q in rows)
        warm = [q for _,q in rows if q['mode']=='repeat']
        summary.append(dict(window=window, draft_tokens=window-1, gate=gate, pcie_frac=fraction,
                            samples=len(rows), all_outputs_exact=all(exact), exact_outputs=sum(exact),
                            decode_aggregate_tok_s=count*1000/elapsed,
                            warm_decode_aggregate_tok_s=sum(q['generated'] for q in warm)*1000/sum(q['decode_ms'] for q in warm),
                            by_case={case:dict(
                                decode_median_tok_s=statistics.median(q['generated']*1000/q['decode_ms'] for _,q in rows if q['case']==case),
                                accepted=sum(q.get('drafts_accepted',0) for _,q in rows if q['case']==case),
                                offered=sum(q.get('drafts_offered',0) for _,q in rows if q['case']==case)) for case in cases}))
    baseline = next(q for q in summary if q['window']==windows[0] and q['gate']==0 and q['pcie_frac']==fractions[0])
    for q in summary:
        q['decode_ratio_to_reference'] = q['decode_aggregate_tok_s']/baseline['decode_aggregate_tok_s']
        q['warm_decode_ratio_to_reference'] = q['warm_decode_aggregate_tok_s']/baseline['warm_decode_aggregate_tok_s']
    summary.sort(key=lambda q:q['warm_decode_aggregate_tok_s'],reverse=True)
    return dict(status='pass', requests=sum(len(run['requests']) for run in data['runs']),
                external_reference=None if reference_path is None else dict(path=str(reference_path),sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),all_outputs_exact=True),
                generated_tokens=sum(q['generated'] for run in data['runs'] for q in run['requests']),
                reference=dict(window=windows[0],gate=0,pcie_frac=fractions[0]),summary=summary,
                limitation='Finite capped-output screen; eligibility requires exact reference tokens. Long outputs, stop boundaries, main-state parity and long-context restore need separate acceptance.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('results',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,help='Also require exact tokens against a prior full-head screen.')
    a = p.parse_args()
    result = audit(a.results,a.reference)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for row in result['summary']:
        print(f"T{row['window']} p={row['gate']} gpu={row['pcie_frac']}: warm {row['warm_decode_aggregate_tok_s']:.2f} tok/s ({row['warm_decode_ratio_to_reference']:.3f}x), all {row['decode_aggregate_tok_s']:.2f}, exact {row['exact_outputs']}/{row['samples']}")
