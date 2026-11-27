"""Certify one fixed 3L-NPC continuation over two declared demand outcomes."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from mission_rf_boundary_scan import scan_profiles
from mission_rf_budget import BLOCK_STEPS, DT, N, array_digest, common_service, crosscheck_selected, model
from mission_rf_fixed_validation import DEFAULT_REFERENCE, load_freeze, regimes_for
from mission_rf_midmission_replay import CASE, replay_plan

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "output/research/P2_MissionRF_BoundaryScan_20261120"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be fresh")
    freeze, _ = load_freeze(DEFAULT_REFERENCE)
    original, _ = scan_profiles()[CASE]
    time = np.arange(N) * DT
    mask = (time >= 10) & (time < 11)
    assert np.flatnonzero(mask)[0] == BLOCK_STEPS and mask.sum() == 5000
    regimes = regimes_for(original)
    tables = model()
    record = json.loads((ARCHIVE / "results.json").read_text(encoding="utf-8"))
    stored = record["full"][CASE]["controllers"]
    with np.load(ARCHIVE / f"{CASE}_actions.npz", allow_pickle=False) as plans:
        hard, comparator = plans["hard"], plans["thermal"]
    assert array_digest(hard) == stored["hard_RF"]["actions_sha256"]
    assert array_digest(comparator) == stored["hard_thermal"]["actions_sha256"]
    ref = replay_plan(stored["hard_thermal"]["policy"], comparator, original, regimes, tables)
    nominal = replay_plan(stored["hard_RF"]["policy"], hard, original, regimes, tables)
    budget = freeze["budget"]
    assert nominal.feasible(budget) and common_service(nominal, ref, .005, .005)

    branches, costs, reports = [], [], []
    for scale in (.999, 1.001):
        profile = original.copy()
        profile[mask, 1] *= scale
        assert np.array_equal(profile[:BLOCK_STEPS], original[:BLOCK_STEPS])
        np.testing.assert_array_equal(regimes_for(profile), regimes)
        plan = replay_plan(nominal.policy, hard, profile, regimes, tables)
        # The same nominal electrical-service reference is used for both outcomes.
        assert plan.feasible(budget) and common_service(plan, ref, .005, .005)
        settled = plan.settled
        np.testing.assert_array_equal(settled[1], nominal.settled[1])
        remaining = plan.damage - settled
        assert np.all(remaining >= -2e-13)
        np.testing.assert_allclose(np.diff(settled,axis=0) + remaining[1:],
                                   remaining[:-1], rtol=2e-13, atol=2e-13)
        branches.append(plan)
        costs.append(remaining[1])
        batch,=crosscheck_selected([plan], profile, regimes, *tables)
        rms = np.sqrt(plan.service[:,1] / plan.service[:,0])
        reference_rms = np.sqrt(ref.service[:,1] / ref.service[:,0])
        reports.append(dict(modulation_scale=scale,profile_sha256=array_digest(profile),
                            RF=plan.damage.tolist(), max_RF=plan.worst_damage,
                            rms_A=float(np.sqrt(plan.mse)), phase_rms_A=rms.tolist(),
                            phase_rms_limit_A=np.minimum(92.4,1.005*reference_rms+.05).tolist(),
                            work_mismatch_percent=(100*(plan.service[:2,2]-ref.service[:2,2])/np.abs(ref.service[:2,2])).tolist(),
                            peak_bound_C=float(plan.limits[0]), terminal=plan.terminal,
                            batch_RF_delta=batch["maximum_absolute_delta"]))
    assert branches[0].actions is branches[1].actions
    branch_difference=float(np.max(np.abs(branches[0].damage-branches[1].damage)))
    assert branch_difference > 1e-8
    shared = np.max(costs,axis=0)
    margin = budget - nominal.settled[1] - shared
    assert np.min(margin) > 0
    absolute = nominal.settled[1]+shared
    np.testing.assert_allclose(absolute,np.max([p.damage for p in branches],axis=0),rtol=2e-13,atol=2e-13)
    result=dict(status="PASS",boundary_s=10,modulation_interval_s=[10,11],
                budget=budget, action_at_boundary=int(hard[BLOCK_STEPS]), common_actions_sha256=array_digest(hard),
                suffix_steps=len(hard)-BLOCK_STEPS, nominal_thermal_reference_sha256=array_digest(comparator),
                initial_settled_RF=nominal.settled[1].tolist(), V_per_device=shared.tolist(),
                robust_max_charge=float(absolute.max()), minimum_device_margin=float(margin.min()),
                branch_damage_max_abs_difference=branch_difference, outcomes=reports,
                script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.mkdir(parents=True)
    (args.output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print({k:result[k] for k in ("status","robust_max_charge","minimum_device_margin","branch_damage_max_abs_difference")})


if __name__ == "__main__":
    main()
