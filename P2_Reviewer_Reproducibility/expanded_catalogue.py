"""Replay an expanded, nested schedule library under the frozen P2 comparison rules."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

def setup(root):
    sys.path.insert(0, str(Path(root) / "DeCuong_4Bai/P2_sim"))
    import mission_rf_budget as base
    from mission_rf_boundary_scan import scan_profiles
    from mission_rf_fixed_validation import load_freeze, DEFAULT_REFERENCE, regimes_for
    base.WEIGHTS = (*base.WEIGHTS, (.48, .08, 0.))
    profile, _ = scan_profiles()["brake_+200bp"]
    return base, profile, regimes_for(profile), base.model(), load_freeze(DEFAULT_REFERENCE)[0]

def worker_init(root):
    global SETUP
    SETUP = setup(root)

def candidate(policy):
    base, profile, phases, tables, _ = SETUP
    return base.simulate_plan(policy, profile, phases, *tables)

def analytic_weight(plans, chosen, freeze):
    low, high = 0., None
    for obj in plans:
        a = (obj.worst_damage-chosen.worst_damage) / freeze["soft_RF_reference"]
        b = (chosen.mse-obj.mse) / freeze["soft_MSE_reference_A2"]
        if a > 0:
            low = max(low,b/a)
        elif a < 0:
            high = b/a if high is None else min(high,b/a)
        elif b > 0:
            return dict(nonempty=False)
    return dict(nonempty=bool(high is None or low < high), lower=float(low), upper=None if high is None else float(high))

def compare(base, pool, anchor, freeze):
    candidates = [p for p in pool if base.common_service(p,anchor,.005,.005)]
    admissible = [p for p in candidates if p.feasible(freeze["budget"])]
    best = min(candidates,key=base.tracking_key)
    hard = min(admissible,key=base.tracking_key) if admissible else None
    soft = min(candidates,key=lambda p: (
        p.mse/freeze["soft_MSE_reference_A2"] + p.worst_damage/freeze["soft_RF_reference"],
        base.tracking_key(p)))
    rows = {}
    for p in sorted(candidates,key=base.tracking_key):
        rows.setdefault(base.array_digest(p.actions),p)
    frontier = [p for p in rows.values() if not any(
        q.mse<=p.mse and q.worst_damage<=p.worst_damage
        and (q.mse<p.mse or q.worst_damage<p.worst_damage) for q in rows.values())]
    desc = lambda p: dict(policy=p.policy, rms_A=p.mse**.5, max_RF=p.worst_damage,
                          sum_RF=float(p.damage.sum()), sha256=base.array_digest(p.actions))
    return dict(matched=len(candidates), unique_matched=len(rows), within=len(admissible),
        unique_within=len({base.array_digest(p.actions) for p in admissible}),
        comparator=desc(anchor),thermal=desc(best),soft=desc(soft),hard=desc(hard) if hard is not None else None,
        frontier=[desc(p) for p in frontier], tuned_soft_for_hard=analytic_weight(candidates,hard,freeze) if hard is not None else None)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    base, _, _, _, freeze=setup(args.root)
    start=perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,initargs=(args.root,)) as pool:
        plans, refused=base.collect_catalog(pool.map(candidate,product(range(4),repeat=3)))
    wall=perf_counter()-start
    anchorable=lambda p: p.service[0,2]>0>p.service[1,2]
    original = [p for p in plans if max(p.policy)<3]
    old=min(filter(anchorable,original),key=base.tracking_key)
    new=min(filter(anchorable,plans),key=base.tracking_key)
    archive = json.loads((args.root/"output/research/P2_MissionRF_BoundaryScan_20261120/results.json").read_bytes())
    refs = {tuple(p["policy"]):p for p in archive["full"]["brake_+200bp"]["catalog"]}
    assert set(refs)=={p.policy for p in original}
    for p in original:
        stored=refs[p.policy]
        assert base.array_digest(p.actions)==stored["actions_sha256"], p.policy
        assert p.worst_damage==stored["RF_max"], p.policy
    result = dict(attempted=64,completed=len(plans),failed=refused,weights=base.WEIGHTS,
        original=compare(base,original,old,freeze), fixed_original_service=compare(base,plans,old,freeze),
        recomputed_thermal_reference=compare(base,plans,new,freeze), unique_valid=len({base.array_digest(p.actions) for p in plans}),
        raw_action_storage_bytes=sum(p.actions.nbytes for p in plans),
        catalogue=[base.summarize_plan(p,freeze["budget"],old) for p in plans])
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(f"Wall time including generation, replay and worker startup: {wall:.2f} s for {args.workers} workers",flush=True)

if __name__=="__main__":
    main()

