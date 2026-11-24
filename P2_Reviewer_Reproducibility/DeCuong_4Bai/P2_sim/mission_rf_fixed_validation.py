"""Held-out deterministic NPC missions with a fixed RF budget and penalty."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import product
from pathlib import Path
import json

from mission_rf_budget import (
    BLOCKS, BLOCK_STEPS, DT, N, WEIGHTS, SOFT_WEIGHT, array_digest, build_domain,
    collect_catalog, common_service, crosscheck_selected, mission, model, np, run_backup,
    simulate_plan, summarize_plan, tracking_key,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = ROOT.parents[1] / "output/research/P2_MissionRF_3LNPC_20261120/results.json"


def load_freeze(path):
    data = path.read_bytes()
    prior = json.loads(data)
    assert prior["status"] == "PASS"
    assert prior["time_step_s"] == DT and prior["block_steps"] == BLOCK_STEPS
    assert prior["qualification"]["candidate_weights"] == [list(w) for w in WEIGHTS]
    assert prior["qualification"]["soft_primary_normalized_weight"] == SOFT_WEIGHT
    assert all(sha256((ROOT / name).read_bytes()).hexdigest() == expected
               for name, expected in prior["sources"].items())
    thermal = prior["controllers"]["hard_thermal"]
    freeze = dict(budget=prior["budget"], penalty_weight=SOFT_WEIGHT,
                  soft_MSE_reference_A2=thermal["rms_A"]**2, soft_RF_reference=thermal["RF_max"],
                  reference_profile_sha256=prior["profile_sha256"],
                  reference_report_sha256=sha256(data).hexdigest())
    return freeze, prior["sources"]


def new_profiles():
    base = mission(N * DT)
    time = np.arange(N) * DT
    pulling, braking = (base[:, 2] > 0) & (base[:, 1] > 0), base[:, 2] < 0

    mild = base.copy()
    mild[pulling, :2] *= [.92, .97]
    mild[braking, 0] *= .94

    early = base.copy()
    longer_braking = (time >= 21.) & (time < 27.3)
    source = base[braking]
    grid = np.linspace(0., len(source) - 1, int(longer_braking.sum()))
    early[longer_braking] = np.column_stack([
        np.interp(grid, np.arange(len(source)), source[:, j]) for j in range(4)
    ])
    early[longer_braking, :2] *= [1.10, .94]

    pulse = base.copy()
    pulse[:, 0] *= 1. + .15 * np.maximum(0., 1. - np.abs(time - 18.5) / 2.5)
    pulse[(time >= 20.) & pulling, 1] *= .96

    heavy = base.copy()
    heavy[pulling, 0] = np.minimum(525., 1.12 * heavy[pulling, 0])
    heavy[braking, 0] *= 1.15

    return {
        "derated": (mild, "Traction current x0.92 and modulation x0.97; braking current x0.94."),
        "long_regeneration": (early, "Brake from 21.0 to 27.3 s, brake current x1.10 and modulation x0.94."),
        "late_traction_pulse": (pulse, "Triangular current increase up to 15% at 18.5 s; late traction modulation x0.96."),
        "heavy": (heavy, "Traction current x1.12 capped at 525 A; braking current x1.15."),
    }


def regimes_for(profile):
    regimes = np.zeros(N, dtype=np.int64)
    regimes[profile[:, 2] < 0.] = 1
    regimes[(profile[:, 0] == 0.) & (profile[:, 1] == 0.)] = 2
    assert np.all(np.bincount(regimes, minlength=3) > 0) and regimes[-1] == 2
    return regimes


def select(plans, freeze, rms=.03, work=.02):
    with_signed_service = [p for p in plans if p.service[0, 2] > 0 > p.service[1, 2]]
    if not with_signed_service:
        return [], {}
    thermal = min(with_signed_service, key=tracking_key)
    eligible = [p for p in with_signed_service if common_service(p, thermal, rms, work)]
    soft = min(eligible, key=lambda p: (
        p.mse / freeze["soft_MSE_reference_A2"] +
        freeze["penalty_weight"] * p.worst_damage / freeze["soft_RF_reference"],
        tracking_key(p)))
    certified = [p for p in eligible if p.feasible(freeze["budget"])]
    hard = min(certified, key=tracking_key) if certified else None
    return eligible, {"hard_thermal": thermal, "soft_RF": soft, "hard_RF": hard}


def decision(hard, other, budget):
    if hard is None:
        return dict(outcome="no_certified_completion", first_distinct_step=None,
                    candidate_passes_budget=other.feasible(budget))
    different = np.flatnonzero(hard.actions != other.actions)
    admissible = other.feasible(budget)
    outcome = ("same_stream" if not len(different) else
               "different_budget_feasible_preference" if admissible else "budget_forces_switch")
    step = int(different[0]) if len(different) else None
    return dict(outcome=outcome, first_distinct_step=step,
                first_distinct_time_s=None if step is None else step * DT,
                candidate_passes_budget=admissible,
                candidate_margin=budget - other.worst_damage)


def evaluate(plans, refused, profile, freeze, model_hash):
    eligible, choices = select(plans, freeze)
    if not choices:
        return dict(status="no_common_service_plan", refused=refused, model_feasible=len(plans)), [], {}
    thermal, soft, hard = choices.values()
    budget = freeze["budget"]
    nodes, bellman = build_domain(eligible, budget)
    assert (hard is not None) == (() in nodes)
    arrays = dict(thermal=thermal.actions, soft=soft.actions)
    witnesses = [thermal, soft]
    events = []
    if hard is not None:
        backup = nodes[()]
        executed, events = run_backup(backup, hard, thermal, budget, array_digest(profile), model_hash)
        witnesses += [hard, backup]
        arrays.update(hard=hard.actions, initial_backup=backup.actions, execution=executed)
    compare_thermal = decision(hard, thermal, budget)
    compare_soft = decision(hard, soft, budget)
    if hard is not None and thermal.feasible(budget):
        assert compare_thermal["outcome"] == "same_stream"
    tight, tight_choices = select(plans, freeze, rms=.005, work=.005)
    report = dict(
        status="certified" if hard is not None else "no_certified_completion",
        model_feasible=len(plans), matched=len(eligible), refused=refused,
        distinct_matched_actions=len({array_digest(p.actions) for p in eligible}),
        minimum_matched_RF=min(p.worst_damage for p in eligible),
        matched_within_budget=sum(p.feasible(budget) for p in eligible),
        distinct_actions_within_budget=len({array_digest(p.actions) for p in eligible if p.feasible(budget)}),
        versus_thermal=compare_thermal, versus_soft=compare_soft,
        bellman_edges=bellman,
        certified_prefixes=[
            dict(prefix=key, witness=p.policy, settled=p.settled[len(key)].tolist(),
                 remaining=(p.damage-p.settled[len(key)]).tolist())
            for key, p in sorted(nodes.items())
        ],
        manager_events=events,
        tighter_service=dict(matched=len(tight),
                             budget_feasible=sum(p.feasible(budget) for p in tight),
                             choices={key: None if p is None else array_digest(p.actions)
                                      for key, p in tight_choices.items()}),
        controllers={key: None if p is None else summarize_plan(p, budget, thermal)
                     for key, p in choices.items()},
        catalog=[dict(eligible=common_service(p, thermal), **summarize_plan(p, budget, thermal))
                 for p in plans],
    )
    distinct = {array_digest(p.actions): p for p in witnesses}
    return report, list(distinct.values()), arrays


def run_candidate(item):
    policy, profile, regimes, lut, resistance, tau = item
    return simulate_plan(policy, profile, regimes, lut, resistance, tau)


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    freeze, sources = load_freeze(args.reference)
    profiles = new_profiles()
    hashes = {name: array_digest(row) for name, (row, _) in profiles.items()}
    assert len(set(hashes.values())) == len(hashes) and freeze["reference_profile_sha256"] not in hashes.values()
    sources[Path(__file__).name] = sha256(Path(__file__).read_bytes()).hexdigest()
    model_hash = sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    out = dict(status="RUNNING", freeze=freeze, sources=sources, model_hash=model_hash,
               candidates_per_profile=len(WEIGHTS)**BLOCKS, profile_hashes=hashes,
               profiles={name: description for name, (_, description) in profiles.items()}, results={})
    save(args.output / "results.json", out)
    lut, resistance, tau = model()
    suite = [(name, profile, regimes_for(profile)) for name, (profile, _) in profiles.items()]
    taskset = [
        (policy, profile, regimes, lut, resistance, tau)
        for _, profile, regimes in suite
        for policy in product(range(len(WEIGHTS)), repeat=BLOCKS)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        outcomes = pool.map(run_candidate, taskset)
        for name, profile, regimes in suite:
            print(name, "catalog", flush=True)
            plans, refused = collect_catalog(next(outcomes) for _ in range(len(WEIGHTS)**BLOCKS))
            result, witnesses, actions = evaluate(plans, refused, profile, freeze, model_hash)
            out["results"][name] = result
            save(args.output / "results.json", out)
            if actions:
                np.savez_compressed(args.output / f"{name}_actions.npz", **actions)
            result["RF_batch_crosschecks"] = crosscheck_selected(
                witnesses, profile, regimes, lut, resistance, tau)
            save(args.output / "results.json", out)
            print(name, result["status"], result.get("versus_thermal"),
                  result.get("versus_soft"), flush=True)
    out["status"] = "PASS"
    save(args.output / "results.json", out)


if __name__ == "__main__":
    main()
