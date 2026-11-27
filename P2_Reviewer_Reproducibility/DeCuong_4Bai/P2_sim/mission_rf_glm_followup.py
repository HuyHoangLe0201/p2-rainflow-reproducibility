"""FCH2Rail duration sensitivity and a fixed-budget schedule replacement test."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np

from core_recorded import simulate_recorded
from mission_rf_budget import (
    BLOCKS, BLOCK_STEPS, DT, FACTOR, IDLE_VECTOR, N, REGIMES, WEIGHTS, Plan,
    array_digest, collect_catalog, common_service, crosscheck_selected, make_catalog,
    model, reference, replay, summarize_plan, tracking_key,
)
from mission_rf_boundary_scan import scan_profiles
from mission_rf_fixed_validation import DEFAULT_REFERENCE, load_freeze, regimes_for, select
from mission_rf_midmission_replay import replay_plan
from mission_rf_public_rail import SOURCE_END_S, SOURCE_NAME, make_profile
from stored_manager import Certificate, StoredManager, cooldown_damage, terminal_check

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[2]
TRAJECTORY = ROOT / "P2_PublicRail_FCH2Rail_20261122" / SOURCE_NAME


class DurationPlan(Plan):
    """Track per-sample MSE when the extension adds samples to block three."""

    @property
    def mse(self):
        return self.service[:, 1].sum() / len(self.actions)


def fch_profile(path, length):
    # The 30 s path uses the frozen mapping directly. For the extra duration,
    # evaluate the same original samples on the new clock without resampling the 30 s profile.
    segment, baseline = make_profile(path)
    if length == N:
        return baseline
    t, _, speed, power = segment.T
    source_time = (np.arange(length) * DT) * (SOURCE_END_S / (length * DT))
    wheel = np.interp(source_time, t, power)
    speed_ratio = np.interp(source_time, t, speed) / np.max(speed)
    pulling = wheel >= 0
    idle = (wheel == 0) & (speed_ratio == 0)
    amplitude = np.where(pulling, 462 * wheel / np.max(power), 360 * wheel / np.min(power))
    modulation = np.where(idle, 0, 0.30 + np.where(pulling, 0.65, 0.40) * speed_ratio)
    factor = np.where(idle, 1, np.where(pulling, 0.95, -0.90))
    return np.column_stack((amplitude, modulation, factor, 10 + 140 * speed_ratio))


def timed_candidate(item):
    policy, profile, phases, tables = item
    length = len(profile)
    weights = np.array([WEIGHTS[index] for index in policy])
    metrics, damage, _, _, _, _, recorded = simulate_recorded(
        profile, weights, *tables, trace_stride=length, lookahead=True, horizon=5)
    if int(metrics[12]) != length or metrics[14]:
        return dict(policy=policy, reason="planner_incomplete", steps=int(metrics[12]),
                    planner_failure_code=int(metrics[14]))
    actions = recorded.astype(np.uint8)
    steps, failure, service, limits, settled, snap, rf, _ = replay(
        profile, phases, actions, *tables, DT, length, False)
    if steps != length or failure:
        raise RuntimeError(f"Unmodified forced replay disagrees for {policy}: step {steps}, mask {failure}")
    np.testing.assert_allclose(rf, damage, rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose([service[:, 1].sum(), service[:, 2].sum()], metrics[[0, 9]],
                               rtol=2e-10, atol=1e-7)
    np.testing.assert_allclose(limits[[0, 1, 2, 3, 4, 5]],
                               metrics[[3, 5, 6, 10, 1, 2]] * [1, 1, .02, 1, 1, 1],
                               rtol=2e-10, atol=1e-8)
    if actions[-1] != IDLE_VECTOR:
        actions[-1] = IDLE_VECTOR
        steps, failure, service, limits, settled, snap, _, _ = replay(
            profile, phases, actions, *tables, DT, length, False)
        if steps != length or failure:
            return dict(policy=policy, reason="terminal_000_transition",
                        steps=int(steps), replay_failure_code=int(failure))
    terminal = terminal_check(snap, tables[0], tables[1])
    if not terminal["pass_terminal"] or np.sqrt(service[:, 1].sum() / length) > 92.4:
        return dict(policy=policy, reason="terminal_or_service", steps=length, terminal=terminal)
    return DurationPlan(policy, actions, service, limits, FACTOR * settled,
                        FACTOR * cooldown_damage(snap), terminal)


def fch_test(workers, freeze, tables):
    length = round(36 / DT)
    profile = fch_profile(TRAJECTORY, length)
    phases = np.zeros(length, dtype=np.int64)
    phases[profile[:, 2] < 0] = 1
    phases[(profile[:, 0] == 0) & (profile[:, 1] == 0)] = 2
    assert phases[-1] == 2 and np.all(np.bincount(phases, minlength=3) > 0)
    with ProcessPoolExecutor(workers) as pool:
        plans, refused = collect_catalog(pool.map(
            timed_candidate, ((p, profile, phases, tables) for p in product(range(len(WEIGHTS)), repeat=BLOCKS))))
    assert len(plans) + len(refused) == 27
    eligible, choices = select(plans, freeze, rms=.005, work=.005)
    b = freeze["budget"]
    available = [p for p in eligible if p.feasible(b)]
    thermal = choices.get("hard_thermal")
    hard = choices.get("hard_RF")
    assert eligible and (hard is None) == (not available)
    witnesses = [p for p in choices.values() if p is not None]
    batch = []
    for plan in {p.policy: p for p in witnesses}.values():
        n, failure, _, _, settled, snap, _, history = replay(
            profile, phases, plan.actions, *tables, DT, length, True)
        assert n == length and not failure and terminal_check(snap, *tables[:2])["pass_terminal"]
        np.testing.assert_array_equal(FACTOR * settled, plan.settled)
        values = FACTOR * np.array([
            reference(np.append(history[:, j], 40.), 4.8) / 40.**4.8 for j in range(30)
        ])
        np.testing.assert_allclose(values, plan.damage, rtol=2e-10, atol=1e-12)
        batch.append(dict(policy=plan.policy, chips=30,
                          maximum_absolute_delta=float(np.max(np.abs(values-plan.damage)))))
    results = dict(
        duration_s=length * DT, compression=SOURCE_END_S / (length * DT), step_s=DT,
        policy_change_s=[10, 20], current_crest_limits_A=[462, 360], parameters=freeze,
        phase_seconds={k: float(np.count_nonzero(phases == i) * DT) for i, k in enumerate(REGIMES)},
        schedules_attempted=27, schedules_completed=len(plans), refused=refused, service_matched=len(eligible),
        within_budget=len(available), distinct_matched=len({array_digest(p.actions) for p in eligible}),
        distinct_within_budget=len({array_digest(p.actions) for p in available}),
        minimum_matched_RF=min(p.worst_damage for p in eligible),
        minimum_catalogue_RF=min(p.worst_damage for p in plans),
        thermal_RF=thermal.worst_damage,
        decision="refuse" if hard is None else ("retain" if np.array_equal(hard.actions, thermal.actions) else "change"),
        profiles_sha256=array_digest(profile), trajectory_sha256=sha256(TRAJECTORY.read_bytes()).hexdigest(),
        catalog=[dict(service_matched=any(p is q for q in eligible), **summarize_plan(p,b,thermal)) for p in plans],
        batch_RF_checks=batch,
    )
    return results, {k: v.actions for k,v in choices.items() if v is not None}


def early_duty(brake_start):
    # Change the phase timing of the existing synthetic +2% current duty.
    # Interpolate crest and modulation in each phase separately so the power factor
    # changes on the phase boundary rather than being interpolated across it.
    original, _ = scan_profiles()["brake_+200bp"]
    old_phases = regimes_for(original)
    times = np.arange(N) * DT
    phase_limits = (0, brake_start, brake_start + 6, 30)
    profile = np.empty_like(original)
    for phase, (a,b) in enumerate(zip(phase_limits,phase_limits[1:])):
        mask = (times >= a) & (times < b)
        values = original[old_phases == phase]
        x = np.arange(len(values))
        grid = np.arange(np.count_nonzero(mask)) * len(values) / np.count_nonzero(mask)
        for col in range(values.shape[1]):
            profile[mask,col] = np.interp(grid,x,values[:,col])
    return profile, phase_limits


def replay_test(workers, freeze, tables, brake_start):
    profile, phase_limits = early_duty(brake_start)
    regimes = regimes_for(profile)
    plans, failures = make_catalog(profile, regimes, *tables, workers)
    assert len(plans) + len(failures) == 27
    matched, winners = select(plans, freeze, rms=.005, work=.005)
    b = freeze["budget"]
    eligible = sorted((p for p in matched if p.feasible(b)), key=tracking_key)
    pairs = []
    for first,second in combinations(eligible,2):
        difference = np.flatnonzero(first.actions != second.actions)
        if len(difference) and BLOCK_STEPS <= difference[0] < 2*BLOCK_STEPS:
            pairs.append((first,second,difference))
    if not pairs:
        return dict(status="NO_PAIR", phase_limits_s=phase_limits, model_feasible=len(plans),
                    matched=len(matched), budget_feasible=len(eligible), failures=failures,
                    eligible=[summarize_plan(p,b,winners["hard_thermal"]) for p in eligible]),{}
    # Choose the pair by ordered tracking keys (fixed above), then offer the better tracker.
    replacement, initial, delta = pairs[0]
    assert initial.feasible(b) and replacement.feasible(b)
    thermal = winners["hard_thermal"]
    assert all(common_service(p,thermal,.005,.005) for p in (initial,replacement))
    profile_hash = array_digest(profile)
    model_hash = sha256(json.dumps({
        name:sha256((SOURCE.parent/name).read_bytes()).hexdigest()
        for name in ("core.py","core_recorded.py","mission_rf_replay.py","mission_rf_budget.py","stored_manager.py",SOURCE.name)
    },sort_keys=True).encode()).hexdigest()

    def cert(plan,label):
        verified = replay_plan(plan.policy,plan.actions,profile,regimes,tables)
        np.testing.assert_array_equal(verified.damage,plan.damage)
        assert verified.feasible(b) and common_service(verified,thermal,.005,.005)
        return Certificate(verified.actions,profile_hash,model_hash,True,label)

    initial_cert,replacement_cert=cert(initial,"initial"),cert(replacement,"replacement")

    def execute(kind):
        manager=StoredManager(initial_cert)
        trace=np.empty(N,dtype=np.uint8)
        for k in range(N):
            if k==BLOCK_STEPS and kind!="counterfactual":
                assert manager.replace(replacement_cert)
            if k==2*BLOCK_STEPS and kind!="counterfactual":
                request=initial_cert if kind=="wrong_prefix" else None
                assert not manager.replace(request)
                reason="executed_prefix_mismatch" if request else "planner_unavailable"
                assert manager.events[-1]["reason"]==reason
                assert manager.events[-1]["backup_before"]==manager.events[-1]["backup_after"]==replacement_cert.digest()
            trace[k]=manager.next_action()
        return trace,manager.events

    rejected,events=execute("wrong_prefix")
    missing,missing_events=execute("missing")
    baseline,_=execute("counterfactual")
    np.testing.assert_array_equal(rejected,replacement.actions)
    np.testing.assert_array_equal(missing,rejected)
    np.testing.assert_array_equal(baseline,initial.actions)
    assert events[0]["accepted"] and events[0]["reason"]=="certificate_committed"
    verified=replay_plan(replacement.policy,rejected,profile,regimes,tables)
    assert verified.feasible(b) and common_service(verified,thermal,.005,.005)
    np.testing.assert_array_equal(verified.damage,replacement.damage)
    batch,=crosscheck_selected([verified],profile,regimes,*tables)
    return dict(status="PASS",phase_limits_s=phase_limits,
                model_feasible=len(plans),matched=len(matched),budget_feasible=len(eligible),
                pairs_exercisable=len(pairs), freeze=freeze, profile_sha256=profile_hash,
                model_sha256=model_hash,first_difference_step=int(delta[0]),
                first_difference_s=float(delta[0]*DT),differences_before_20=int(np.count_nonzero(delta<2*BLOCK_STEPS)),
                changed_vectors=int(len(delta)),executed_steps=N,events=events,missing_events=missing_events,
                initial=summarize_plan(initial,b,thermal),replacement=summarize_plan(replacement,b,thermal),
                executed_sha256=array_digest(rejected),missing_sha256=array_digest(missing),batch_RF=batch), {
        "initial":initial.actions,"replacement":replacement.actions,"executed":rejected}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment",choices=("fch36","handoff"))
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--brake-start",type=float,default=14)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error("output must be fresh")
    if not 10 < args.brake_start < 20: parser.error("brake start must be between the tested boundaries")
    freeze,_=load_freeze(DEFAULT_REFERENCE)
    tables=model()
    if args.experiment=="fch36":
        result,arrays=fch_test(args.workers,freeze,tables)
    else:
        result,arrays=replay_test(args.workers,freeze,tables,args.brake_start)
    args.output.mkdir(parents=True,exist_ok=False)
    result["script_sha256"]=sha256(SOURCE.read_bytes()).hexdigest()
    (args.output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if arrays: np.savez_compressed(args.output/"actions.npz",**arrays)
    print({k:v for k,v in result.items() if k in (
        "status","decision","phase_limits_s","model_feasible","matched","budget_feasible",
        "pairs_exercisable","first_difference_s","schedules_completed","within_budget","minimum_matched_RF")},flush=True)


if __name__=="__main__":
    main()
