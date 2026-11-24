"""Run the frozen P2 selectors on the published FCH2Rail first station segment."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import product
from pathlib import Path
import json

from mission_rf_budget import (
    BLOCKS, DT, N, REGIMES, WEIGHTS, array_digest, build_domain, collect_catalog,
    crosscheck_selected, model, np, run_backup, summarize_plan,
)
from mission_rf_fixed_validation import DEFAULT_REFERENCE, load_freeze, regimes_for, run_candidate, select

SOURCE_NAME = "A Coruna-Ferrol-A Coruna_simulated_trajectory.csv"
DOI = "10.5281/zenodo.6359030"
SOURCE_END_S = 194


def make_profile(path):
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    assert raw.shape[1] == 4
    segment = raw[(raw[:, 0] >= 0) & (raw[:, 0] <= SOURCE_END_S)]
    assert np.array_equal(segment[:, 0], np.arange(SOURCE_END_S + 1))
    t, distance, speed, power = segment.T
    assert distance[-1] == 2611 and np.all(speed[135:] == 0) and np.all(power[135:] == 0)
    assert np.any(power > 0) and np.any(power < 0)
    source_time = (np.arange(N) * DT) * (SOURCE_END_S / (N * DT))
    wheel = np.interp(source_time, t, power)
    v = np.interp(source_time, t, speed) / np.max(speed)
    pulling = wheel >= 0
    idle = (wheel == 0) & (v == 0)
    ipk = np.where(pulling, 462 * wheel / np.max(power), 360 * wheel / np.min(power))
    modulation = np.where(idle, 0, 0.30 + np.where(pulling, 0.65, 0.40) * v)
    pf = np.where(idle, 1, np.where(pulling, 0.95, -0.90))
    profile = np.column_stack((ipk, modulation, pf, 10 + 140 * v))
    return segment, profile


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--trajectory", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = p.parse_args()
    if args.output.exists():
        p.error("--output must be a new directory")
    segment, profile = make_profile(args.trajectory)
    regimes = regimes_for(profile)
    freeze, sources = load_freeze(args.reference)
    source_dir = Path(__file__).resolve().parent
    sources = dict(sources)
    for name in ("mission_rf_fixed_validation.py", Path(__file__).name):
        sources[name] = sha256((source_dir / name).read_bytes()).hexdigest()
    model_hash = sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    profile_hash = array_digest(profile)
    tables = model()
    policies = tuple(product(range(len(WEIGHTS)), repeat=BLOCKS))
    with ProcessPoolExecutor(args.workers) as pool:
        plans, refused = collect_catalog(pool.map(
            run_candidate, ((policy, profile, regimes, *tables) for policy in policies)
        ))
    assert len(plans) + len(refused) == len(policies) == 27
    eligible, choice = select(plans, freeze, rms=.005, work=.005)
    budget = freeze["budget"]
    available = [plan for plan in eligible if plan.feasible(budget)]
    nodes, bellman_edges = build_domain(eligible, budget)
    report = dict(
        status="PASS", source_doi=DOI, source_member=SOURCE_NAME,
        input_trajectory_sha256=sha256(args.trajectory.read_bytes()).hexdigest(),
        source_seconds=[0, SOURCE_END_S], source_stationary_seconds=[135, SOURCE_END_S],
        distance_m=float(segment[-1, 1]), wheel_power_range_W=[
            float(min(segment[:, 3])), float(max(segment[:, 3]))
        ], maximum_speed_m_s=float(max(segment[:, 2])),
        duration_s=N * DT, step_s=DT, source_seconds_per_model_second=SOURCE_END_S/(N*DT),
        current_crest_limits_A=[462, 360], phase_seconds={
            phase: float(np.count_nonzero(regimes == k) * DT) for k, phase in enumerate(REGIMES)
        }, source_files_sha256=sources, model_sha256=model_hash, profile_sha256=profile_hash,
        parameters=freeze, relative_service_tolerances=dict(rms=.005, signed_work=.005),
        schedules_attempted=len(policies), schedules_completed=len(plans), refused=refused,
        service_matched=len(eligible), within_budget=len(available),
        distinct_matched=len({array_digest(p.actions) for p in eligible}),
        distinct_within_budget=len({array_digest(p.actions) for p in available}),
        certified_nodes=len(nodes), bellman_edges=bellman_edges,
    )
    arrays = {}
    if choice:
        thermal, soft, hard = (choice[key] for key in ("hard_thermal", "soft_RF", "hard_RF"))
        report["controllers"] = {
            name: summarize_plan(plan, budget, thermal) if plan else None
            for name, plan in choice.items()
        }
        report["catalog"] = [
            dict(service_matched=any(plan is q for q in eligible), **summarize_plan(plan, budget, thermal))
            for plan in plans
        ]
        witnesses = [thermal, soft]
        arrays.update(thermal=thermal.actions, soft=soft.actions)
        if hard is not None:
            assert () in nodes
            executed, events = run_backup(nodes[()], hard, thermal, budget, profile_hash, model_hash)
            report["fallback"] = dict(events=events, executed_steps=len(executed),
                                      executed_actions_sha256=array_digest(executed))
            arrays.update(hard=hard.actions, execution=executed)
            witnesses.append(hard)
        report["batch_RF_checks"] = crosscheck_selected(witnesses, profile, regimes, *tables)
    args.output.mkdir(parents=True, exist_ok=False)
    if arrays:
        np.savez_compressed(args.output / "selected_actions.npz", **arrays)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                                               encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k in
                     {"status", "phase_seconds", "schedules_completed", "service_matched", "within_budget",
                      "certified_nodes", "bellman_edges", "fallback"}}, indent=2))


if __name__ == "__main__":
    main()
