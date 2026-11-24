"""Reproduce baseline and deterministic stored-completion experiments.

python run_campaign.py --duration 530 --output campaign_530
This is a model study at a 650kW rated electrical point, not a validated PMSM.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
from core import simulate,rf_push,rf_total,losses,PATH,GATES,MODULE

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
import rainflow

def reference(values, alpha):
    if len(values)<2:return 0.
    if len(values)==2:values=list(values)+[values[-1]]
    return sum(count*span**alpha for span,mean,count,start,end in rainflow.extract_cycles(values))

def model():
    d=json.loads((ROOT/'device_model.json').read_text())
    I=np.arange(1201,dtype=float)
    lut=np.stack([np.interp(I,d['IGBT_current_A'],d['IGBT_voltage_V']),
        np.interp(I,d['diode_current_A'],d['diode_voltage_V'])]+[
        .001*.9*np.interp(I,d['energy_current_A'],d[key]) for key in ['Eon_mJ','Eoff_mJ','Erec_mJ']])
    R=np.tile(np.where(np.arange(10)[:,None]<4,.053,.081)*np.array(d['Foster_normalized_R']),(3,1))
    return lut,R,np.array(d['Foster_tau_s'])

def mission(duration,dt=.0002,late=False):
    t=np.arange(round(duration/dt))*dt
    # Synthetic phase mission, explicitly NOT reconstructed public train records.
    scale=duration/530.
    z=t/scale
    ip=np.where(z<90,462.,np.where(z<390,250.,np.where(z<470,360.,0.)))
    ip*=np.minimum(1.,z/3.)
    m=np.where(z<90,.3+.4*z/90,np.where(z<390,.95,np.where(z<470,.7,0.)))
    pf=np.where(z<390,.95,np.where(z<470,-.9,1.))
    fe=np.where(z<90,10+140*z/90,np.where(z<390,150.,np.where(z<470,150.-140*(z-390)/80,10.)))
    if late:
        # Extra acceleration-like load in late cruise, fixed predeclared stress scenario.
        pulse=(z>=330)&(z<390)
        ip[pulse]=462.;m[pulse]=.45;fe[pulse]=35.
    return np.column_stack([ip,m,pf,fe])

def provenance():
    ds=ROOT.parent/'Tailieu_P4'/'datasheet'/'CM600DX-34T.pdf'
    return dict(device='Mitsubishi CM600DX-34T, 1700 V / 600 A',
        datasheet_url='https://www.mitsubishielectric.com/semiconductors/powerdevices/datasheets/igbt/t_series/cm600dx-34t_e.pdf',
        datasheet_local=str(ds),datasheet_sha256=hashlib.sha256(ds.read_bytes()).hexdigest() if ds.exists() else '00568bc73d72de670629781b1937d5f828534b4fbb02f03d7795c7291effeec4',
        source_pages={'static_switching_anchors':4,'thermal_resistances':5,'normalized_Foster_table':11},
        digitization_source=str(ROOT.parent/'P4_code'/'sim'/'linhkien.py'),
        digitization_sha256=hashlib.sha256((ROOT.parent/'P4_code'/'sim'/'linhkien.py').read_bytes()).hexdigest() if (ROOT.parent/'P4_code'/'sim'/'linhkien.py').exists() else '08b028b1035ecbcd13fb6e3b9d3cb9b4f678f5c2e0991724830ed40fc043fd08',
        standalone_model_sha256=hashlib.sha256((ROOT/'device_model.json').read_bytes()).hexdigest(),
        electrical=dict(Vdc=1800.,L_H=.00093,R_ohm=.02,Ipk_rated=462.,m_rated=.95,
            pf_rated=.95,Ts_s=.0002,split_capacitance_each_F=.01,
            rated_formula_W=1.5*(1800*.95/np.sqrt(3))*462*.95),
        parameter_classes=dict(datasheet='module ratings, fixed-150C chip curves, energy curves, four Foster branches',
            project='Vdc, L and rated current from CoSoToanHoc.tex; not independent motor identification',
            design_assumptions='R=.02ohm; each capacitor .01F; coolant40C; shared R=.0215K/W tau2s; 10ms edge window; tracking/NP limits',
            mission='synthetic 530s phase profile; not measured railway data'),
        control_tuning='Tracking normalized by462A squared; gate-edge weight .002; neutral-point weight2.0; gate resource penalty .0001*(used/(31-used))**2. Earlier pilots hit the NP/tracking bounds; same final electrical tuning for all controllers.',
        loss_rules='150C curves fixed; voltage scaling linear 900/1000; nearest-A lookup; chip conduction only; recovery commutation map. Terminal lead resistance, dead time and parasitics excluded.',
        thermal_rules='self Foster + shared module branch; instantaneous switching-energy impulse followed by constant-conduction ZOH; envelope R+15%,tau+-20%,loss+10%; continuous thermal peak bounded by sum of branch maxima',
        switching_limit='30 gate edges per 50 samples (10ms) per IGBT; equivalent1500 cycles/s in that window. Design choice, not an asserted module absolute-frequency rating.',
        damage_rules='persistent RF of impulse-start and period-end nominal temperatures, alpha4.8, ranges normalized by40K; A and Arrhenius multiplier omitted in nominal score. Certificate assumes one fixed common Rscale<=1.15, exact tau, fixed coolant, same losses; frozen Arrhenius cap150C relative to125C.',
        reserve_scope='stored complete feasible trajectory under deterministic electrical model; full-mission replay after a schedule change; not arbitrary electrical disturbances, online FPGA implementation, or full-box damage certificate',
        excluded='calibrated PMSM mechanics, reconstructed public rail mission, physical life prediction, HIL, coupling between separate module packages')

def summarize(metrics,damage,edges,N,duration,seconds):
    n=int(metrics[12])
    factor=1.15**4.8*np.exp(7000/398.15-7000/423.15)
    return dict(completed_steps=n,expected_steps=N,completed=bool(n==N and metrics[14]==0),
        failure_code=int(metrics[14]),rms_tracking_A=float(np.sqrt(metrics[0]/max(n,1))),
        peak_tracking_A=float(metrics[1]),nominal_temperature_sample_peak_C=float(metrics[2]),
        certified_intersample_peak_C=float(metrics[3]),minimum_safe_candidates_evaluated=int(metrics[4]),
        neutral_point_peak_V=float(metrics[5]),max_window_switching_Hz=float(metrics[6]),
        mean_switching_Hz_max_device=float(metrics[13]),conduction_energy_J=float(metrics[7]),
        switching_energy_J=float(metrics[8]),electromagnetic_energy_J=float(metrics[9]),
        peak_current_A=float(metrics[10]),envelope_violation_C=float(metrics[11]),
        nominal_normalized_damage_per_device=damage.tolist(),
        nominal_normalized_damage_max=float(np.max(damage)),
        worst_device=f'phase{np.argmax(damage)//10+1}_'+['T1','T2','T3','T4','D1','D2','D3','D4','DcU','DcL'][np.argmax(damage)%10],
        certified_normalized_damage_per_device=(factor*damage).tolist(),
        certified_normalized_damage_max=float(factor*np.max(damage)),
        fixed_scale_certificate_multiplier=float(factor),
        gate_edges=edges.tolist(),wall_seconds=seconds,
        physical_pass=bool(n==N and metrics[14]==0 and metrics[3]<=150+1e-8 and metrics[6]<=1500+1e-8 and metrics[5]<=90+1e-8),
        service_pass=bool(n==N and metrics[14]==0 and np.sqrt(metrics[0]/max(n,1))<=92.4 and metrics[1]<=277.2))

def verify_core():
    # RF replay includes final endpoint closure; exercise nested and random cycles.
    rng=np.random.default_rng(20261014)
    err=0.
    for values in [[0.,10.,9.,11.],[0.,10.,0.],[2.,2.,3.,1.]]+[rng.normal(size=100).tolist() for _ in range(60)]:
        stack=np.zeros((30,4096));stack[:,0]=values[0]
        depth=np.ones(30,dtype=np.int64);prev=np.full(30,values[0]);direction=np.zeros(30,dtype=np.int64)
        settled=np.zeros(30)
        for k,x in enumerate(values):
            assert rf_push(x,0,stack,depth,prev,direction,settled,4.8)
            got=rf_total(stack,depth,prev,settled,4.8)[0]
            want=reference(values[:k+1],4.8)/40**4.8
            err=max(err,abs(got-want))
            assert abs(got-want)<1e-12
    lut,R,tau=model()
    # Printed 150C anchors and Foster normalisation checked before simulation.
    assert abs(lut[0,600]-2.45)<1e-10
    assert abs(lut[2,600]-.1174*.9)<1e-10
    assert abs(lut[3,600]-.1437*.9)<1e-10
    assert abs(R[0].sum()-.053)<1e-12
    assert abs(R[4].sum()-.081)<1e-12
    # Six conduction paths, both signs; energy is assigned to real commutating roles.
    for old,now,sg,tr,di,on in [(0,1,1,0,8,True),(1,0,1,0,8,False),
        (1,0,-1,2,4,True),(0,-1,1,1,7,False),(-1,0,1,1,7,True),
        (0,-1,-1,3,9,True),(-1,0,-1,3,9,False)]:
        cur=np.full(3,sg*600.);q=np.full(3,now,dtype=np.int64);qold=np.full(3,old,dtype=np.int64)
        cond,en=losses(q,qold,cur,cur,lut)
        assert abs(en[tr]-lut[2 if on else 3,600])<1e-12
        assert (en[di]>0)==on
        assert np.count_nonzero(cond[:10])==2
    return dict(rainflow_max_absolute_error=err,module_anchor_checks='pass',commutation_checks='pass')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--duration',type=float,default=530.)
    ap.add_argument('--output',default='campaign_530');ap.add_argument('--late',action='store_true')
    ap.add_argument('--skip-reserve',action='store_true');args=ap.parse_args()
    out=ROOT/args.output;out.mkdir(exist_ok=True)
    proof_checks=verify_core()
    origin=provenance()
    origin['simulator_sha256']=hashlib.sha256((ROOT/'core.py').read_bytes()).hexdigest()
    origin['campaign_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prof=mission(args.duration,late=args.late);lut,R,tau=model()
    np.save(out/'mission.npy',prof)
    stages=int(np.ceil(args.duration/10))
    settings={'B0':(0.,0.,0.),'B1':(0.,.08,0.),'B2':(0.,0.,10.),'B3':(.08,0.,0.)}
    results={};allweights={};cache={}
    def run(name,weights):
        fingerprint=hashlib.sha256(weights.tobytes()).hexdigest()
        if fingerprint in cache:
            prior=cache[fingerprint]
            results[name]=dict(results[prior],wall_seconds=0.,identical_schedule_cache=prior)
            allweights[name]=weights.copy()
            np.savez_compressed(out/(name+'.npz'),trace=np.load(out/(prior+'.npz'))['trace'],weights=weights)
            print(name,'identical schedule:',prior,flush=True)
            return results[name]
        start=time.perf_counter()
        metrics,dmg,trace,ed=simulate(prof,weights,lut,R,tau)
        summary=summarize(metrics,dmg,ed,len(prof),args.duration,time.perf_counter()-start)
        np.savez_compressed(out/(name+'.npz'),trace=trace,weights=weights)
        results[name]=summary;allweights[name]=weights.copy()
        cache[fingerprint]=name
        (out/'results.json').write_text(json.dumps(dict(provenance=origin,checks=proof_checks,runs=results),indent=2),encoding='utf-8')
        print(name,json.dumps({k:summary[k] for k in ['completed','rms_tracking_A','certified_intersample_peak_C','nominal_normalized_damage_max','max_window_switching_Hz','wall_seconds']}),flush=True)
        return summary
    for name,w in settings.items():run(name,np.tile(np.array(w),(stages,1)))
    admissible=[name for name,s in results.items() if s['physical_pass'] and s['service_pass']]
    reserve_log=[]
    if admissible and not args.skip_reserve:
        backup=min(admissible,key=lambda name:results[name]['certified_normalized_damage_max'])
        # Same budget for every device. Predeclared 1% margin above known feasible backup.
        budget=results[backup]['certified_normalized_damage_max']*1.01
        accepted=allweights[backup].copy()
        accepted_name=backup
        # Two candidate macro-actions at predeclared mission fractions. Entire prefix
        # schedule is preserved exactly; deterministic replay retains the same state.
        boundaries=sorted(set([0,max(1,int(stages*.17)),max(2,int(stages*.62))]))
        for boundary in boundaries:
            if boundary>=stages:continue
            for candidate in [(0.,0.,0.),(.24,0.,0.)]:
                proposal=accepted.copy()
                proposal[boundary:min(stages,boundary+3)]=candidate
                name=f'proposal_{boundary}_{candidate[0]:.2f}'
                summary=run(name,proposal)
                if boundary>0:
                    oldtrace=np.load(out/(accepted_name+'.npz'))['trace']
                    newtrace=np.load(out/(name+'.npz'))['trace']
                    common=min(len(oldtrace),len(newtrace),boundary*1000)
                    assert np.array_equal(oldtrace[:common],newtrace[:common]),'Changed a committed prefix'
                ok=(summary['physical_pass'] and summary['service_pass'] and
                    summary['certified_normalized_damage_max']<=budget+1e-10)
                reserve_log.append(dict(boundary_s=boundary*10,candidate=candidate,
                    accepted=ok,certificate=summary['certified_normalized_damage_max'],budget=budget))
                if ok:accepted=proposal;accepted_name=name
        results['B4']=dict(results[accepted_name],stored_completion=True,backup=backup,
            same_budget_per_device=budget,reserve_at_initial_state=budget/results[accepted_name]['certified_normalized_damage_max'],
            final_schedule_source=accepted_name,proposals=reserve_log,
            interpretation='offline full-horizon stored-completion experiment, not a fast-loop online controller')
        np.savez_compressed(out/'B4.npz',trace=np.load(out/(accepted_name+'.npz'))['trace'],weights=accepted)
    elif not args.skip_reserve:
        results['B4']=dict(completed=False,reason='No baseline with physical and service pass; no certified backup exists in this tested library.')
    report=dict(provenance=origin,checks=proof_checks,experiment=dict(duration_s=args.duration,late_load=args.late,
        tuning='weights predeclared heuristic; not an externally validated or optimally tuned literature baseline',
        budget_policy='1.01 times least-damage passing backup, normalized per-device common budget; design feasibility test not real maintenance threshold'),runs=results)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('RESULTS',out/'results.json',flush=True)

if __name__=='__main__':main()
