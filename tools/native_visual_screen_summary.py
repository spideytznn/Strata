"""Audit capacity screens or complete visual profile pairs; report geometry differences."""
import argparse
import json
from pathlib import Path
import statistics


def audit(path, min_rounds):
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['status'] == 'pass' and data['profile_mode'] in ('capacity-screen','profile-pair')
    rounds, profiles = data['rounds'], data['profiles']
    assert rounds >= min_rounds and len(data['runs']) == rounds*len(profiles)
    if data['profile_mode']=='capacity-screen':
        assert len({r['exe_sha256'] for r in data['runs']}) == 1
    expected = {(r,name) for r in range(1,rounds+1) for name in profiles}
    actual = {(run['round'],run['profile']) for run in data['runs']}
    assert expected == actual and len(actual)==len(data['runs'])
    summary = dict(status='pass',rounds=rounds,total_requests=0,profiles={},controlled_inputs={},
                   note='Fresh prompt tokens / prompt_ms. Official thinking sampling, fixed seed; startup excluded. Cross-capacity sampled outputs may differ. Finite capped outputs do not replace teacher/long-output/cache-state acceptance.')
    for name,cfg in profiles.items():
        runs = [run for run in data['runs'] if run['profile']==name]
        assert len({run['exe_sha256'] for run in runs}) == 1, 'Binary changed between repeats of a profile'
        groups = {}
        for run in runs:
            assert run['status']=='pass' and run['startup_free_mib']>=512
            info = run['info']
            capacity = int(cfg['args'][cfg['args'].index('--prefill')+1])
            assert info['prefill_chunk']==capacity and info['weight_source']=='safetensors'
            assert info['expert_ram_bytes']==info['expert_cuda_pinned_bytes']>0
            rows = {(q['case'],q['mode']):q for q in run['requests']}
            assert len(rows)==len(run['requests'])==7
            assert set(rows)=={('doc8k','cold'),('doc8k','warm'),('doc8k-followup','continuation'),
                               ('doc24k','cold'),('doc24k','warm'),('image8k','cold'),('image8k','warm')}
            log = path.parent/f"r{run['round']}-{name}"/'engine.log'
            text = log.read_text(encoding='utf-8',errors='replace')
            assert 'post-residency expert source bytes=0' in text
            assert 'post-residency MTP source bytes=0' in text
            for q in rows.values():
                assert q['generated']==len(q['ids'])==data['new']
                assert q['file_blobs']==q['file_mb']==0
                if q['mode']=='warm':
                    assert q['ids']==rows[q['case'],'cold']['ids'] and q['reused']>0
                if q['case']=='image8k': assert q['image_tokens']==1024
                key = q['case']+'-'+q['mode']
                if q['case']!='doc8k-followup':
                    old = summary['controlled_inputs'].setdefault(key,q['input_sha256'])
                    assert old==q['input_sha256']
                groups.setdefault(key,[]).append(q)
        entry = dict(capacity=capacity,exe_sha256=runs[0]['exe_sha256'],expert_slots=[r['info']['expert_slots'] for r in runs],
                     startup_free_mib=[r['startup_free_mib'] for r in runs],cases={})
        for key,rows in groups.items():
            assert all(q['ids']==rows[0]['ids'] for q in rows), 'Repeat output differs within profile'
            entry['cases'][key] = {metric:dict(median=statistics.median(q[metric] for q in rows),
                                             min=min(q[metric] for q in rows),max=max(q[metric] for q in rows))
                                  for metric in ('prefill_tok_s','decode_tok_s','ttft_s','wall_s','prepare_s','prompt_ms','decode_ms','fresh_tokens','reused')}
        summary['profiles'][name]=entry
        summary['total_requests']+=sum(len(run['requests']) for run in runs)
    summary['cross_profile_outputs_exact'] = all(
        a['ids']==b['ids'] for r in range(1,rounds+1) for name in list(profiles)[1:]
        for a,b in zip(next(x for x in data['runs'] if x['round']==r and x['profile']==list(profiles)[0])['requests'],
                       next(x for x in data['runs'] if x['round']==r and x['profile']==name)['requests'],strict=True))
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('results',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--min-rounds',type=int,default=1)
    a=p.parse_args()
    result=audit(a.results,a.min_rounds)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS',result['total_requests'],'requests; cross-profile exact:',result['cross_profile_outputs_exact'])
    for name,row in result['profiles'].items():
        print(name,'capacity',row['capacity'],'slots',row['expert_slots'],'free',row['startup_free_mib'])
        for case in ('doc8k-cold','doc24k-cold','image8k-cold'):
            q=row['cases'][case]
            print(case,'pp',round(q['prefill_tok_s']['median'],2),'decode',round(q['decode_tok_s']['median'],2))
