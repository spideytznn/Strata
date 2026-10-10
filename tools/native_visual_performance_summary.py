"""Audit and summarize native_visual_performance.py results without rerunning inference."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.results.read_text(encoding='utf-8'))
    assert data['status'] == 'pass'
    profiles = data['profiles']
    rounds = data['rounds']
    assert rounds >= 3 and len(data['runs']) == rounds*3
    runs = {(x['round'],x['profile']):x for x in data['runs']}
    assert len(runs) == len(data['runs'])
    base = profiles['vision3072']
    priority_mode = data.get('profile_mode') == 'prefill-first'
    before = json.loads(json.dumps(base))
    if priority_mode:
        assert base['env']['STRATA_NATIVE_PREFILL_FIRST']=='0'
        before['args'][before['args'].index('--vram-reserve-mib')+1] = '3584'
        assert before == profiles['vision3584']
        priority = json.loads(json.dumps(base))
        priority['env']['STRATA_NATIVE_PREFILL_FIRST']='1'
        assert priority == profiles['vision-priority3072']
    else:
        before['args'][before['args'].index('--vram-reserve-mib')+1] = '2048'
        assert before == profiles['vision2048']
        text = json.loads(json.dumps(base))
        text['args'].remove('--vision')
        del text['vision']
        assert text == profiles['text3072']
    assert len({x['exe_sha256'] for x in runs.values()}) == 1
    summary = {'status':'pass','rounds':rounds,'profiles':{},'controlled_input_sha256':{},
               'note':'Official thinking sampling. Prefill is fresh tokens / engine prompt time. Image preparation is separate. Follow-up depends on prior sampled output; not a controlled cross-profile input. Chunk changes can change sampled outputs; no strict same-output decode speedup claim.'}
    for name in profiles:
        selected = [runs[r,name] for r in range(1,rounds+1)]
        assert all(x['status']=='pass' for x in selected)
        flexible = 'vision3072' if priority_mode else 'vision2048'
        if name != flexible: assert all(int(x['info']['prefill_chunk'])==4096 for x in selected)
        if priority_mode and name=='vision-priority3072':
            assert all(x['startup_free_mib']>=512 for x in selected)
        groups = {}
        for run in selected:
            rows = {(x['case'],x['mode']):x for x in run['requests']}
            expected = 7 if profiles[name].get('vision') else 5
            assert len(rows) == expected
            for req in rows.values():
                assert req['generated'] == len(req['ids']) == data['new']
                assert req['file_blobs'] == req['file_mb'] == 0
                key = req['case']+'-'+req['mode']
                groups.setdefault(key,[]).append(req)
                if req['mode']=='warm':
                    assert req['ids']==rows[req['case'],'cold']['ids'] and req['reused']>0
                if req['case'] != 'doc8k-followup':
                    old = summary['controlled_input_sha256'].setdefault(key,req['input_sha256'])
                    assert old == req['input_sha256']
                if req['case']=='image8k': assert req['image_tokens']==1024
        entry = {'actual_capacities':[x['info']['prefill_chunk'] for x in selected],
                 'startup_free_mib':[x['startup_free_mib'] for x in selected],'cases':{}}
        for key, rows in groups.items():
            # Repeatability and own-cache equality are correctness gates; comparisons
            # across different chunk geometries are reported rather than equated.
            assert all(x['ids']==rows[0]['ids'] for x in rows), (name,key,'repeat output differs')
            fields = ['prefill_tok_s','decode_tok_s','prompt_ms','decode_ms','prepare_s','ttft_s','fresh_tokens','reused']
            entry['cases'][key] = {field:{'median':statistics.median(x[field] for x in rows),
                'min':min(x[field] for x in rows),'max':max(x[field] for x in rows),
                'samples':[x[field] for x in rows]} for field in fields}
            entry['cases'][key]['draft_acceptance_samples'] = [x['drafts_accepted']/x['drafts_offered'] if x['drafts_offered'] else None for x in rows]
        summary['profiles'][name]=entry
    for r in range(1,rounds+1):
        reference = 'vision3584' if priority_mode else 'vision3072'
        comparison = 'vision-priority3072' if priority_mode else 'text3072'
        visual = {(x['case'],x['mode']):x for x in runs[r,reference]['requests']}
        for req in runs[r,comparison]['requests']:
            if req['case']=='doc8k-followup' and not priority_mode: continue
            assert req['ids']==visual[req['case'],req['mode']]['ids'], ('4096 control output differs',r,req['case'])
    summary['checks'] = {'total_requests':sum(len(x['requests']) for x in runs.values()),
        'all_output_caps_met':True,'all_expert_file_reads_zero':True,
        'warm_output_exact':True,'repeat_outputs_exact_within_profile':True,
        ('allocation_priority_output_exact_at_4096' if priority_mode else 'text_output_exact_when_toggling_vision_at_4096'):True,
        'cross_chunk_output_exact':all(
            runs[r,'vision3072' if priority_mode else 'vision2048']['requests'][i]['ids']==
                runs[r,'vision-priority3072' if priority_mode else 'vision3072']['requests'][i]['ids']
            for r in range(1,rounds+1) for i in range(7))}
    a.out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary['checks'],indent=2))
    for name, entry in summary['profiles'].items():
        print(name,'capacities',entry['actual_capacities'],'free MiB',entry['startup_free_mib'])
        for case in ['doc8k-cold','doc24k-cold','image8k-cold']:
            if case in entry['cases']:
                row=entry['cases'][case]
                print(case,'prefill',round(row['prefill_tok_s']['median'],2),'decode',round(row['decode_tok_s']['median'],2))


if __name__=='__main__':
    main()
