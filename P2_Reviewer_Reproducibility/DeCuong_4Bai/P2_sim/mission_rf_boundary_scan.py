"""Screen near-boundary 3L-NPC missions and verify switches under the frozen RF budget."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import product
from pathlib import Path
import json

from mission_rf_budget import (
    BLOCKS, DT, N, WEIGHTS, array_digest, collect_catalog, crosscheck_selected, mission, model, np,
)
from mission_rf_fixed_validation import (
    DEFAULT_REFERENCE, decision, evaluate, load_freeze, regimes_for, run_candidate, save,
    select, summarize_plan,
)


SCREEN_POLICIES = ((2, 0, 0), (0, 0, 0), (0, 0, 1))
FULL_POLICIES = tuple(product(range(len(WEIGHTS)), repeat=BLOCKS))


def scan_profiles():
    base = mission(N * DT)
    braking = base[:, 2] < 0.
    times = np.arange(N) * DT
    tail = np.maximum(0., 1. - np.abs(times - 25.65) / .8) * braking
    profiles = {}
    for bp in (-200, -100, -50, -25, 25, 50, 100, 200):
        profile = base.copy()
        profile[braking, 0] *= 1. + bp / 10_000.
        profiles[f"brake_{bp:+d}bp"] = (profile, dict(
            family="whole_regeneration", current_change_bp=bp))
    for bp in (-100, -50, 50, 100):
        profile = base.copy()
        profile[:, 0] *= 1. + bp / 10_000. * tail
        profiles[f"tail_{bp:+d}bp"] = (profile, dict(
            family="triangular_late_regeneration", current_change_bp=bp,
            center_s=25.65, halfwidth_s=.8))
    return profiles


def screen(plans, refused, freeze):
    eligible, selected = select(plans, freeze, rms=.005, work=.005)
    if not selected:
        return dict(screen_hit=False, matched=0, refused=refused, rows=[])
    thermal = selected["hard_thermal"]
    hard = selected["hard_RF"]
    change = decision(hard, thermal, freeze["budget"])
    hit = change["outcome"] == "budget_forces_switch"
    margin = min(thermal.worst_damage-freeze["budget"], freeze["budget"]-hard.worst_damage) if hit else None
    return dict(screen_hit=hit, rank_margin=margin, matched=len(eligible), refused=refused,
                minimum_matched_RF=min(plan.worst_damage for plan in eligible),
                versus_thermal=change, rows=[
                    dict(matched=any(p is plan for p in eligible),
                         **summarize_plan(plan, freeze["budget"], thermal)) for plan in plans
                ])


def candidate_args(profile, regimes, tables, policies):
    return [(policy, profile, regimes, *tables) for policy in policies]


def strict_decision(plans, freeze):
    eligible, choices = select(plans, freeze, rms=.005, work=.005)
    if not choices:
        return dict(matched=0, thermal=None, hard=None, forces_switch=False)
    thermal = choices["hard_thermal"]
    hard = choices["hard_RF"]
    change = decision(hard, thermal, freeze["budget"])
    return dict(matched=len(eligible), thermal=array_digest(thermal.actions),
                hard=None if hard is None else array_digest(hard.actions),
                forces_switch=change["outcome"] == "budget_forces_switch")


def best_scan_cases(records):
    hits = [(name, value["rank_margin"]) for name, value in records.items() if value["screen_hit"]]
    return [name for name, _ in sorted(hits, key=lambda row: (-row[1], row[0]))]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--max-full-cases", type=int, default=3)
    args = parser.parse_args()
    if args.max_full_cases < 1:
        parser.error("--max-full-cases must be at least 1")
    freeze, sources = load_freeze(args.reference)
    root = Path(__file__).resolve().parent
    for name in ("mission_rf_fixed_validation.py", Path(__file__).name):
        sources[name] = sha256((root / name).read_bytes()).hexdigest()
    model_hash = sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    profiles = scan_profiles()
    hashes = {name: array_digest(p) for name, (p, _) in profiles.items()}
    assert len(set(hashes.values())) == len(hashes)
    assert freeze["reference_profile_sha256"] not in hashes.values()
    suite = {name: (p, regimes_for(p)) for name, (p, _) in profiles.items()}
    data = dict(status="RUNNING", freeze=freeze, sources=sources, model_hash=model_hash,
                profiles={name: dict(**desc, profile_sha256=hashes[name])
                          for name, (_, desc) in profiles.items()},
                screen_policies=SCREEN_POLICIES, full_policies=FULL_POLICIES,
                verified_case=None, screen={}, full={})
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "results.json"
    save(result_path, data)
    tables = model()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [task for p, r in suite.values() for task in candidate_args(p, r, tables, SCREEN_POLICIES)]
        rows = pool.map(run_candidate, tasks)
        screened = {}
        for name in suite:
            outcomes = [next(rows) for _ in SCREEN_POLICIES]
            screened[name] = {policy: row for policy, row in zip(SCREEN_POLICIES, outcomes)}
            plans, refused = collect_catalog(outcomes)
            result = screen(plans, refused, freeze)
            data["screen"][name] = result
            save(result_path, data)
            print(name, "screen", result["screen_hit"], result.get("versus_thermal"), flush=True)

        ranked = best_scan_cases(data["screen"])
        # Check additional close cases if the small catalog had no reliable hit.
        remaining = sorted((name for name in suite if name not in ranked),
                           key=lambda name: abs(data["screen"][name].get(
                               "minimum_matched_RF", float("inf")) - freeze["budget"]))
        full_cases = (ranked + remaining)[:args.max_full_cases]
        data["full_order"] = full_cases
        for name in full_cases:
            p, regimes = suite[name]
            extra = [policy for policy in FULL_POLICIES if policy not in screened[name]]
            additional = pool.map(run_candidate, candidate_args(p, regimes, tables, extra))
            outcomes = screened[name] | dict(zip(extra, additional))
            plans, refused = collect_catalog(outcomes[policy] for policy in FULL_POLICIES)
            full, witnesses, recorded = evaluate(plans, refused, p, freeze, model_hash)
            strict = strict_decision(plans, freeze)
            full["strict_budget_trigger"] = strict
            data["full"][name] = full
            save(result_path, data)
            if recorded:
                np.savez_compressed(args.output / f"{name}_actions.npz", **recorded)
            full["RF_batch_crosschecks"] = crosscheck_selected(witnesses, p, regimes, *tables)
            save(result_path, data)
            outcome = full.get("versus_thermal", {}).get("outcome")
            print(name, "full", outcome, "strict", strict["forces_switch"], flush=True)
            if outcome == "budget_forces_switch" and strict["forces_switch"]:
                data["verified_case"] = name
                break
    data["status"] = "PASS" if data["verified_case"] else "NO_VERIFIED_SWITCH_IN_FULL_CASES"
    save(result_path, data)
    print(data["status"], data["verified_case"], flush=True)


if __name__ == "__main__":
    main()
