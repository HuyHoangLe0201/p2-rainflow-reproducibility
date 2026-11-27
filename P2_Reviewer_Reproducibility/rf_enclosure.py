"""Conditional RF charge enclosures for forced-replay temperature samples."""
import argparse
from fractions import Fraction as F
from functools import lru_cache
import json
from pathlib import Path
import sys
import numba
import numpy as np

SAMPLE_SCALE=2**47
BIN=2**37
RANGE_NORMALIZER=40*1024
SCORE_SCALE=10**22
OUT_SCALE=10**10

@numba.njit(cache=True)
def range_bins(samples,bins):
    """Four-point RF counting, with exact comparisons and integer half-cycle multiplicities."""
    lower=np.zeros(bins,dtype=np.int64)
    upper=np.zeros(bins,dtype=np.int64)
    stack=np.empty(len(samples),dtype=np.int64)
    stack[0]=samples[0]
    n=1
    prev=samples[0]
    trend=0
    for k in range(1,len(samples)+1):
        if k==len(samples):
            v=prev
        else:
            x=samples[k]
            if x==prev:
                continue
            direction=1 if x>prev else -1
            v=prev
            prev=x
            if not trend or direction==trend:
                trend=direction
                continue
            trend=direction
        if v==stack[n-1]:
            continue
        stack[n]=v
        n+=1
        while n>=4:
            a,b,c,d=stack[n-4:n]
            r=abs(b-c)
            if r>abs(b-a) or r>abs(d-c):
                break
            lower[r//BIN]+=2
            upper[(r+BIN-1)//BIN]+=2
            stack[n-3]=d
            n-=2
    for k in range(1,n):
        r=abs(stack[k]-stack[k-1])
        lower[r//BIN]+=1
        upper[(r+BIN-1)//BIN]+=1
    return lower,upper

@lru_cache(None)
def ceil_power(num,den,exponent=24):
    """Least q such that q/SCORE_SCALE >= (num/den)^(exponent/5)."""
    rhs=num**exponent*SCORE_SCALE**5
    denom=den**exponent
    if rhs==0:
        return 0
    hi=1<<((max(0,rhs.bit_length()-denom.bit_length())+5)//5+1)
    lo=0
    assert hi**5*denom>=rhs
    while hi-lo>1:
        mid=(hi+lo)//2
        if mid**5*denom>=rhs:
            hi=mid
        else:
            lo=mid
    assert (hi-1)**5*denom<rhs<=hi**5*denom
    return hi

def coefficient():
    """Positive rational exponential series with a geometric remainder."""
    x=F(7000)/F("398.15")-F(7000)/F("423.15")
    assert 0<x<2
    value=term=F(1)
    for n in range(1,41):
        term=term*x/n
        value+=term
    tail=(term*x/41)/(1-x/42)
    scale=ceil_power(115,100)
    return value*F(scale-1,SCORE_SCALE),(value+tail)*F(scale,SCORE_SCALE)

def decimal_bound(x, upward):
    numerator=x.numerator*OUT_SCALE
    q=-(-numerator//x.denominator) if upward else numerator//x.denominator
    whole, frac=divmod(q,OUT_SCALE)
    return f"{whole}.{frac:010d}"

def enclose(history):
    # >=32 gives a binary64 spacing of at least 2^-47. <256 avoids overflow.
    assert np.isfinite(history).all() and history.min()>=32 and history.max()<256
    ticks=(history*SAMPLE_SCALE).astype(np.int64)
    assert np.array_equal(ticks.astype(np.float64)/SAMPLE_SCALE,history)
    coeff_lo,coeff_hi=coefficient()
    values=[]
    vd=[]
    for chip in range(history.shape[1]):
        trace=np.append(ticks[:,chip],40*SAMPLE_SCALE)
        span=int(trace.max())-int(trace.min())
        bins=(span+BIN-1)//BIN+1
        lower,upper=range_bins(trace,bins)
        lo_weight=sum(int(lower[i])*max(0,ceil_power(int(i),RANGE_NORMALIZER)-1) for i in np.flatnonzero(lower))
        hi_weight=sum(int(upper[i])*ceil_power(int(i),RANGE_NORMALIZER) for i in np.flatnonzero(upper))
        values.append((coeff_lo*F(lo_weight,2*SCORE_SCALE),coeff_hi*F(hi_weight,2*SCORE_SCALE)))
        diffs=np.abs(np.diff(trace))
        assert int(diffs.max())*128<(1<<63)
        total=sum(int(diffs[k:k+128].sum()) for k in range(0,len(diffs),128))
        # For alpha 4.8, the power of the full range in the TV bound is 19/5.
        upper_power=F(ceil_power(span,40*SAMPLE_SCALE,19),SCORE_SCALE)
        vd.append(coeff_hi*upper_power*F(total,80*SAMPLE_SCALE))
    return values,vd

def run(root, output):
    sys.path.insert(0,str(root/"DeCuong_4Bai/P2_sim"))
    import mission_rf_budget as b
    from mission_rf_boundary_scan import scan_profiles
    from mission_rf_fixed_validation import regimes_for
    profile,_=scan_profiles()["brake_+200bp"]
    phases=regimes_for(profile)
    lut,resistance,tau=b.model()
    data=root/"output/research/P2_MissionRF_BoundaryScan_20261120"
    archive=json.loads((data/"results.json").read_bytes())
    budget=F(str(archive["freeze"]["budget"]))
    controllers=archive["full"]["brake_+200bp"]["controllers"]
    arrays=np.load(data/"brake_+200bp_actions.npz")
    actions={k:arrays[k] for k in ("thermal","hard","soft")}
    thermal=None
    reports={}
    cases=[(f"nominal_{k}",k,(1.,)*4,1.,1.) for k in actions]
    cases += [("all_fast","hard",(.8,)*4,1.,1.),("all_slow","hard",(1.2,)*4,1.,1.),
              ("mixed_R","hard",(.8,1.2,.8,1.2),1.15,1.),
              ("mixed_reversed_R","hard",(1.2,.8,1.2,.8),1.15,1.),
              ("sustained_low_modulation","hard",(1.,)*4,1.,.9975),
              ("sustained_high_modulation","hard",(1.,)*4,1.,1.0025)]
    for label,key,tau_f,r_f,m_f in cases:
        scenario=profile.copy()
        scenario[b.BLOCK_STEPS:2*b.BLOCK_STEPS,1]*=m_f
        rs=resistance*r_f
        ts=tau*np.array(tau_f)
        n,mask,service,limits,settled,snapshot,_,temps=b.verify_replay(scenario,phases,actions[key],lut,rs,ts,record_rf=True)
        terminal=b.terminal_check(snapshot,lut,rs)
        info=dict(actions_sha256=b.array_digest(actions[key]),local_tau_factors=tau_f,local_R_factor=r_f,
                  modulation_factor_10_to_20_s=m_f,completed=n,failure=mask,terminal=terminal["pass_terminal"])
        reports[label]=info
        if n!=b.N or mask or not terminal["pass_terminal"]:
            print(label,info,flush=True)
            continue
        scores=b.FACTOR*b.cooldown_damage(snapshot)
        plan=b.Plan((),actions[key],service,limits,b.FACTOR*settled,scores,terminal)
        if label=="nominal_thermal":
            thermal=plan
        lohi,tv=enclose(temps)
        assert all(lo<=F(float(v))<=hi for v,(lo,hi) in zip(scores,lohi))
        assert all(hi_rational>=F(float(v)) for v,hi_rational in zip(scores,tv))
        if label.startswith("nominal_"):
            np.testing.assert_array_equal(scores,controllers[{"hard":"hard_RF","soft":"soft_RF","thermal":"hard_thermal"}[key]]["RF_per_device"])
        hi=max(hi for _,hi in lohi)
        lo=max(lo for lo,_ in lohi)
        info.update(service_matched=b.common_service(plan,thermal,.005,.005),peak_bound_C=float(limits[0]),
             rms_A=plan.mse**.5,max_RF=float(scores.max()),sum_RF=float(scores.sum()),
             RF_lower=decimal_bound(lo,False),RF_upper=decimal_bound(hi,True),
             variation_upper=decimal_bound(max(tv),True),within_upper_bound=hi<=budget,above_lower_bound=lo>budget,
             per_device_RF_enclosure=[[decimal_bound(l,False),decimal_bound(u,True)] for l,u in lohi],
             temperature_samples_sha256=b.array_digest(temps))
        del temps
        print(label,info["max_RF"],info["RF_lower"],info["RF_upper"],info["variation_upper"],info["service_matched"],flush=True)
    output.write_text(json.dumps(dict(budget=str(budget),scope="Exact finite temperature samples conditional on replay, and finite parameter stress tests",cases=reports),indent=2)+"\n",encoding="utf8")

if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("root",type=Path)
    a.add_argument("output",type=Path)
    args=a.parse_args()
    run(args.root,args.output)

