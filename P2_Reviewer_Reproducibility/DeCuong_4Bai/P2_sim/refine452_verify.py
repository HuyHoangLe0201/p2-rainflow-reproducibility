"""Verify the conservative and paired joint RF margins at two actual horizons."""
import sys
from pathlib import Path
from refine452_screen import e, OLD, BASE
import equal45_verify as native
from equal45_audit import independent, gate_check

HORIZONS=(4097.,8193.)
METRICS=('closed_source_excess','observer_total_source_excess')
PHASES=('on','tight','reduced')
EPSILONS=(1e-8,1e-7,1e-6)
J=8

def collect(row, old=None):
    results={h:{} for h in HORIZONS}
    tau=e.row_params(row)[0]
    zero_row=e.zero_row(tau)
    if old:
        manifest=e.json.loads((OLD/'final_manifest.json').read_text())
        hashes=manifest['artifacts'] | manifest['inherited_artifact_references']
    else:
        native.S=e.S
    for name in PHASES:
        opts=dict(rtol=2e-9,atol=2e-11,coast_s=8192.,integrals=False)
        if name!='reduced':
            opts['native']=True
        if name=='tight':
            opts.update(rtol=2e-11,atol=2e-13,density=96,sample_factor=4)
        data=[]
        for r,key in ((row,name),(zero_row,{'on':'zero','tight':'zero_tight','reduced':'zero_reduced'}[name])):
            if old:
                path=old['paths'][key]
                artifact=e.WORK/path
                assert e.digest(artifact)==hashes[path]==old['artifacts'][path]
                data.append(e.restore(artifact,r))
            else:
                data.append(native.phase(r,name,refined=name=='tight',**opts))
        on,zero=data
        e._assert_pref(on,zero)
        if name=='tight':
            for r,d in ((row,on),(zero_row,zero)):
                proof,packet,_=native.certificate(r)
                native.check_admission(r,d,native.actual_score(r,d),proof,packet)
        for h in HORIZONS:
            assert h in on['time_s'] and h in zero['time_s']
            a,b=e.crop(on,h),e.crop(zero,h)
            for epsilon in ((0.,)+EPSILONS if name=='tight' else (0.,)):
                m,_=e.score(row,a,b,precise=epsilon==0,epsilon=epsilon)
                if name=='tight' and epsilon==0:
                    independent(a,m,True,True)
                    independent(b,m,False,True)
                if epsilon==0:
                    scores=[m['precise_by_chip'][J][key] for key in METRICS]
                else:
                    scores=[repr(m[key][J]) for key in ('delta_closed_by_chip','delta_window_by_chip')]
                results[h][f'{name}:{epsilon}']=dict(scores=scores,
                    counts=[m['material_full_on'][J],m['material_full_zero'][J]])
        del data,on,zero,a,b
    return results

def compare(candidate,reference):
    outputs={}
    for h in HORIZONS:
        a,b=candidate[h],reference[h]
        assert a.keys()==b.keys()
        counts=[v['counts'] for v in list(a.values())+list(b.values())]
        equal=all(c==[2,2] for c in counts)
        metrics={}
        with e.mp.workdps(70):
            for i,key in enumerate(METRICS):
                c={k:e.mp.mpf(v['scores'][i]) for k,v in a.items()}
                r={k:e.mp.mpf(v['scores'][i]) for k,v in b.items()}
                active='tight:0.0'
                gain=c[active]-r[active]
                versions={k:c[k]-r[k] for k in c}
                distance=lambda arm,keys:max(abs(arm[k]-arm[active]) for k in keys)
                fixed=[k for k in c if k.endswith(':0.0')]
                all_err=distance(c,c)+distance(r,r)
                fixed_err=distance(c,fixed)+distance(r,fixed)
                paired_err=max(abs(v-gain) for v in versions.values())
                metrics[key]=dict(gain=float(gain),conservative_error=float(all_err),
                    solver_only_error=float(fixed_err),paired_error=float(paired_err),
                    conservative_ratio=float(gain/all_err),solver_only_ratio=float(gain/fixed_err),
                    paired_ratio=float(gain/paired_err),variant_gains={k:float(v) for k,v in versions.items()})
        outputs[h]=dict(counts=counts,equal_counts=equal,metrics=metrics,
            conservative_pass=equal and all(m['conservative_ratio']>10 for m in metrics.values()),
            solver_only_pass=equal and all(m['solver_only_ratio']>10 for m in metrics.values()))
    return outputs

def main(ids):
    base=BASE['row']
    manifest=OLD/'final_manifest.json'
    old_audit=OLD/'audit.json'
    artifacts=e.json.loads(manifest.read_text())['artifacts']
    assert e.digest(old_audit)==artifacts[str(old_audit.relative_to(e.WORK))]
    audited=e.json.loads(old_audit.read_text())['records']
    old=next(r for r in audited if r['id']==base['id'])
    assert old['row']==base
    ref=collect(base,old)
    for rid in ids:
        f=e.S/'screen_records'/f'{rid}.json'
        row=e.json.loads(f.read_text())['row']
        assert row['id']==rid and 2*sum(e.row_params(row)[1:3])==642
        gate_check(row)
        candidate=collect(row)
        comparison=compare(candidate,ref)
        out=dict(row=row,reference=base,comparison=comparison,observations=candidate,reference_observations=ref,
            stage29_manifest_sha256=e.digest(manifest),screen_sha256=e.digest(f),
            verifier_sha256=e.digest(Path(__file__)))
        e.save(e.S/f'{rid}_verification.json',out)
        print(rid,{h:{k:(m['gain'],m['conservative_ratio'],m['solver_only_ratio'],m['paired_ratio'])
            for k,m in d['metrics'].items()} for h,d in comparison.items()},flush=True)

if __name__=='__main__':
    main(sys.argv[1:])
