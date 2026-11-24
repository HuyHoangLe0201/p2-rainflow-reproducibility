from pathlib import Path
from zipfile import ZipFile
import hashlib,json,math,sys

package,out=map(Path,sys.argv[1:])
prefix="P2_Reviewer_Reproducibility/"
with ZipFile(package) as z:
    assert z.testzip() is None
    read=lambda name:json.loads(z.read(prefix+name))
    base=read("output/research/P2_MissionRF_BoundaryScan_20261120/results.json")
    replay=read("output/research/P2_MissionRF_Midmission_20261122/results.json")
    public=read("P2_PublicRail_FCH2Rail_20261122/results.json")
case=base["full"]["brake_+200bp"]
freeze=base["freeze"]
B=freeze["budget"]
Q0=freeze["soft_MSE_reference_A2"]
D0=freeze["soft_RF_reference"]
assert B==public["parameters"]["budget"]==replay["freeze"]["budget"]
def signed(row):
    p=row["service"]
    return p["traction"]["work_J"]>0>p["regeneration"]["work_J"]
def service(row,baseline):
    a=row["service"]
    b=baseline["service"]
    return signed(row) and all(a[p]["steps"]==b[p]["steps"] and
           a[p]["rms_A"]<=min(92.4,1.005*b[p]["rms_A"]+0.05) for p in b) and all(
           abs(a[p]["work_J"]-b[p]["work_J"])<=0.005*abs(b[p]["work_J"])
           for p in ("traction","regeneration"))
def track(p):
    return (p["rms_A"],p["peak_bound_C"],tuple(p["policy"]))
def qd(p):
    return p["rms_A"]**2/Q0,p["RF_max"]/D0
def matching(data):
    cat=data["catalog"]
    assert len(cat)==27 and all(len(p["RF_per_device"])==30 and p["RF_max"]==max(p["RF_per_device"]) for p in cat)
    tb=min(filter(signed,cat),key=track)
    assert tb["policy"]==data["controllers"]["hard_thermal"]["policy"]
    match=[x for x in cat if service(x,tb)]
    yes=[x for x in match if x["RF_max"]<=B]
    return match,yes,tb
match,yes,thermal=matching(case)
h=min(yes,key=track)
soft=lambda weight:min(match,key=lambda p:(qd(p)[0]+weight*qd(p)[1],track(p)))
assert h["policy"]==case["controllers"]["hard_RF"]["policy"]
assert soft(1)["policy"]==case["controllers"]["soft_RF"]["policy"]
assert (len(match),len(yes),len({p["actions_sha256"] for p in match}),len({p["actions_sha256"] for p in yes}))==(15,9,5,3)
lower,upper=0.0,math.inf
q,d=qd(h)
for p in match:
    qi,di=qd(p)
    slope=d-di
    if slope>0:
        upper=min(upper,(qi-q)/slope)
    elif slope<0:
        lower=max(lower,(qi-q)/slope)
    else:
        assert q<=qi
assert lower < upper and soft((lower+upper)/2)["actions_sha256"]==h["actions_sha256"]
assert lower < 0.0857 < upper and soft(0.0857)["actions_sha256"] == h["actions_sha256"]
assert [(e["step"],e["accepted"],e["reason"]) for e in replay["events"]]==[
    (50000,True,"certificate_committed"),(100000,False,"planner_unavailable")]
assert replay["events"][1]["backup_before"]==replay["events"][1]["backup_after"]
assert replay["hard_actions_sha256"]==replay["executed_actions_sha256"]
assert replay["complete_steps"]==150000
m2,y2,t2=matching(public)
assert (len(m2),len(y2),len({p["actions_sha256"] for p in m2}))==(15,0,12)
pmin=min(p["RF_max"] for p in m2)
assert pmin==min(p["RF_max"] for p in public["catalog"])
payload={"archive_sha256":hashlib.sha256(package.read_bytes()).hexdigest(),
 "B":B,"base":{"thermal_policy":thermal["policy"],"soft_policy":soft(1)["policy"],"hard_policy":h["policy"],
          "max_RF":[p["RF_max"] for p in (thermal,soft(1),h)],"RMS_A":[p["rms_A"] for p in (thermal,soft(1),h)],
          "reduction_pct":100*(1-h["RF_max"]/thermal["RF_max"]),"RMS_increase_A":h["rms_A"]-thermal["rms_A"],
          "lambda_interval_interior_selects_hard_actions":[lower,upper],
          "lambda_example":(lower+upper)/2},
 "replay":{"accepted_s":replay["proposal_at_s"],"missing_s":replay["missing_at_s"],
           "first_different_action_s":replay["first_soft_hard_difference_step"]*replay["step_s"],
           "vectors":replay["complete_steps"],"different_vectors_vs_initial":replay["samples_changed_vs_initial"],
           "RF_budget_margin":replay["budget_margin"],"RF_batch_delta":replay["batch_RF_audit"]["maximum_absolute_delta"]},
 "FCH2Rail":{"matched":len(m2),"within_budget":len(y2),"min_RF":pmin,
             "excess_pct":100*(pmin/B-1),"phase_seconds":public["phase_seconds"],
             "stretch":public["source_seconds_per_model_second"],
             "rest_ref_RMS":t2["service"]["idle"]["rms_A"]}}
out.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2))
