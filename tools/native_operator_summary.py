"""Audit complete desktop decode runs and derive rates from matched token outputs."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(path, reference_path=None):
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['status']=='pass' and data['output_comparison']=='exact'
    assert data['desktop_vision'] and data['thinking'] and data['sampling']['official_config']
    assert data['runs'] and data['rounds']>0
    before = data
    external = data.get('reference')
    if external:
        refpath=reference_path or Path(external['path'])
        assert hashlib.sha256(refpath.read_bytes()).hexdigest()==external['sha256']
        before=json.loads(refpath.read_text(encoding='utf-8'))
        assert before['status']=='pass' and before['output_comparison']=='exact'
    for key in ('requested_new_tokens','categories','rounds','sampling','thinking','desktop_vision'):
        assert data[key]==before[key]
    reference=before['runs'][0]
    refrows={q['name']:q for q in reference['requests']}
    expected={f'{case}-{r}-{mode}' for case in data['categories'] for r in range(data['rounds']) for mode in ('cold','warm')}
    assert set(refrows)==expected and len(refrows)==len(reference['requests'])
    basewarm=[q for q in reference['requests'] if q['name'].endswith('-warm')]
    base_rate=sum(q['generated'] for q in basewarm)*1000/sum(q['decode_ms'] for q in basewarm)
    result=dict(status='pass',requests=0,generated_tokens=0,reference=external,
                reference_warm_aggregate_tok_s=base_rate,profiles=[],
                limitation='Finite matched-output runs; startup excluded. A single fresh engine per profile is a screen, not a repeated estimate.')
    for index,run in enumerate(data['runs']):
        assert run['status']=='pass'
        cfg,info=run['config'],run['info']
        assert info['weight_source']=='safetensors' and info['expert_ram_bytes']==info['expert_cuda_pinned_bytes']>0
        assert info['prefill_chunk']==int(cfg['args'][cfg['args'].index('--prefill')+1])
        assert info['vram_free_mib']>=512
        assert cfg['sampling']==reference['config']['sampling'] and cfg['vision']==reference['config']['vision']
        assert not any(key in cfg.get('env',{}) for key in ('STRATA_STATE_HASH','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_NVFP4_TIMING'))
        rows={q['name']:q for q in run['requests']}
        assert set(rows)==expected and len(rows)==len(run['requests'])
        text=(path.parent/str(index)/'engine.log').read_text(encoding='utf-8',errors='replace')
        assert 'post-residency expert source bytes=0' in text and 'post-residency MTP source bytes=0' in text
        for name,q in rows.items():
            ref=refrows[name]
            assert q['input_sha256']==ref['input_sha256'] and q['ids']==ref['ids']
            assert 0<len(q['ids'])==q['generated']<=data['requested_new_tokens']
            assert q['file_blobs']==q['file_mb']==0 and q['decode_ms']>0 and q['prompt_ms']>0
            if name.endswith('-warm'):
                assert q['ids']==rows[name.removesuffix('-warm')+'-cold']['ids'] and q['reused']>0
        warm=[q for q in rows.values() if q['name'].endswith('-warm')]
        rate=sum(q['generated'] for q in warm)*1000/sum(q['decode_ms'] for q in warm)
        result['profiles'].append(dict(config=run['config_path'],exe_sha256=run['exe_sha256'],
            expert_slots=info['expert_slots'],vram_free_mib=info['vram_free_mib'],prefill=info['prefill_chunk'],
            warm_aggregate_tok_s=rate,ratio_to_reference=rate/base_rate,
            all_aggregate_tok_s=sum(q['generated'] for q in rows.values())*1000/sum(q['decode_ms'] for q in rows.values()),
            warm_by_case={q['name']:q['generated']*1000/q['decode_ms'] for q in warm},all_outputs_exact=True))
        result['requests']+=len(rows)
        result['generated_tokens']+=sum(q['generated'] for q in rows.values())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('results',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,help='Location of the archived external reference; its recorded hash must match.')
    a=p.parse_args()
    result=audit(a.results,a.reference)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS',result['requests'],'requests,',result['generated_tokens'],'matched tokens')
    for row in result['profiles']:
        print(Path(row['config']).stem,round(row['warm_aggregate_tok_s'],2),f"{row['ratio_to_reference']:.3f}x")
