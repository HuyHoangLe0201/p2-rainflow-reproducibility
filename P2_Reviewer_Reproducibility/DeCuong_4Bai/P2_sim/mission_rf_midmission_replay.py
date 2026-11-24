"""Replay a certified replacement at 10 s and a missing proposal at 20 s."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

from mission_rf_boundary_scan import scan_profiles
from mission_rf_budget import (
    BLOCK_STEPS, DT, FACTOR, N, REGIMES, Plan, array_digest, common_service,
    crosscheck_selected, model, np, summarize_plan, verify_replay,
)
from mission_rf_fixed_validation import load_freeze, regimes_for
from stored_manager import Certificate, StoredManager, cooldown_damage, terminal_check

SOURCE = Path(__file__).resolve().parent
RESEARCH = SOURCE.parents[1] / "output/research"
CASE = "brake_+200bp"
RMS = 0.005
WORK = 0.005


def service_counts(run, budget, rms, work):
    """Reapply the phase matching criteria to an archived full catalog."""
    reference = run["controllers"]["hard_thermal"]["service"]
    rows = run["catalog"]
    assert len(rows) == 27
    matched = []
    for row in rows:
        phase = row["service"]
        rms_pass = all(
            phase[s]["steps"] == reference[s]["steps"]
            and phase[s]["rms_A"] <= min(92.4, (1 + rms) * reference[s]["rms_A"] + 0.05)
            for s in REGIMES
        )
        work_pass = (
            phase["traction"]["work_J"] > 0 > phase["regeneration"]["work_J"]
            and all(abs(phase[s]["work_J"] - reference[s]["work_J"])
                    <= work * abs(reference[s]["work_J"]) for s in REGIMES[:2])
        )
        if rms_pass and work_pass:
            matched.append(row)
    feasible = [row for row in matched if max(row["RF_per_device"]) <= budget]
    return dict(
        matched=len(matched), within_budget=len(feasible),
        distinct_matched=len({row["actions_sha256"] for row in matched}),
        distinct_within_budget=len({row["actions_sha256"] for row in feasible}),
    )


def replay_plan(policy, actions, profile, regimes, tables):
    end, failure, service, limits, settled, snapshot, _, _ = verify_replay(
        profile, regimes, actions, *tables,
    )
    assert end == N and failure == 0
    terminal = terminal_check(snapshot, tables[0], tables[1])
    assert terminal["pass_terminal"]
    return Plan(tuple(policy), actions, service, limits, FACTOR * settled,
                FACTOR * cooldown_damage(snapshot), terminal)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a fresh directory")

    boundary = RESEARCH / "P2_MissionRF_BoundaryScan_20261120"
    report_path = boundary / "results.json"
    actions_path = boundary / f"{CASE}_actions.npz"
    variants_path = RESEARCH / "P2_MissionRF_FixedValidation_20261120/results.json"
    reference_path = RESEARCH / "P2_MissionRF_3LNPC_20261120/results.json"
    archive = json.loads(report_path.read_bytes())
    variants = json.loads(variants_path.read_bytes())
    freeze, _ = load_freeze(reference_path)
    assert archive["status"] == variants["status"] == "PASS"
    assert archive["freeze"] == variants["freeze"] == freeze
    for name, digest in archive["sources"].items():
        assert sha256((SOURCE / name).read_bytes()).hexdigest() == digest
    profile, _ = scan_profiles()[CASE]
    profile_hash = array_digest(profile)
    assert profile_hash == archive["profiles"][CASE]["profile_sha256"]
    run = archive["full"][CASE]
    budget = freeze["budget"]
    counts = {}
    for name, row in {CASE: run, **variants["results"]}.items():
        loose = service_counts(row, budget, 0.03, 0.02)
        strict = service_counts(row, budget, RMS, WORK)
        assert (loose["matched"], loose["within_budget"]) == (row["matched"], row["matched_within_budget"])
        assert (strict["matched"], strict["within_budget"]) == (
            row["tighter_service"]["matched"], row["tighter_service"]["budget_feasible"]
        )
        counts[name] = dict(original=loose, strict=strict)

    regimes = regimes_for(profile)
    tables = model()
    selected = {}
    with np.load(actions_path, allow_pickle=False) as stored:
        for key, name in (("thermal", "hard_thermal"), ("soft", "soft_RF"), ("hard", "hard_RF")):
            archived = run["controllers"][name]
            actions = stored[key]
            assert array_digest(actions) == archived["actions_sha256"]
            assert archived["actions_sha256"] == run["tighter_service"]["choices"][name]
            plan = replay_plan(archived["policy"], actions, profile, regimes, tables)
            np.testing.assert_allclose(plan.damage, archived["RF_per_device"], rtol=2e-13, atol=2e-13)
            assert plan.feasible(budget) == archived["budget_pass"]
            selected[key] = plan

    thermal, soft, hard = (selected[key] for key in ("thermal", "soft", "hard"))
    assert all(common_service(plan, thermal, RMS, WORK) for plan in selected.values())
    assert soft.feasible(budget) and hard.feasible(budget) and not thermal.feasible(budget)
    first_difference = int(np.flatnonzero(soft.actions != hard.actions)[0])
    assert first_difference >= BLOCK_STEPS
    model_hash = archive["model_hash"]
    initial = Certificate(soft.actions, profile_hash, model_hash, soft.feasible(budget), "soft")
    replacement = Certificate(hard.actions, profile_hash, model_hash, hard.feasible(budget), "hard")
    manager = StoredManager(initial)
    executed = np.empty(N, dtype=hard.actions.dtype)
    for step in range(N):
        if step == BLOCK_STEPS:
            assert manager.replace(replacement)
        if step == 2 * BLOCK_STEPS:
            assert not manager.replace(None)
        executed[step] = manager.next_action()
    accepted, missing = manager.events
    assert accepted == dict(step=BLOCK_STEPS, accepted=True, reason="certificate_committed",
                            backup_before=initial.digest(), backup_after=replacement.digest())
    assert missing == dict(step=2 * BLOCK_STEPS, accepted=False, reason="planner_unavailable",
                           backup_before=replacement.digest(), backup_after=replacement.digest())
    np.testing.assert_array_equal(executed, hard.actions)
    replayed = replay_plan(hard.policy, executed, profile, regimes, tables)
    assert common_service(replayed, thermal, RMS, WORK) and replayed.feasible(budget)
    np.testing.assert_array_equal(replayed.settled, hard.settled)
    np.testing.assert_array_equal(replayed.damage, hard.damage)
    np.testing.assert_array_equal(soft.settled[1], hard.settled[1])
    batch, = crosscheck_selected([replayed], profile, regimes, *tables)
    deltas = replayed.damage - thermal.damage
    boundary_costs = []
    for block in (1, 2):
        settled = replayed.settled[block]
        remaining = replayed.damage - settled
        assert np.all(remaining >= 0) and np.all(settled + remaining <= budget)
        boundary_costs.append(dict(step=block * BLOCK_STEPS,
                                   settled_RF=settled.tolist(), remaining_RF=remaining.tolist()))

    args.output.mkdir(parents=True, exist_ok=False)
    trace = args.output / "executed_actions.npz"
    np.savez_compressed(trace, actions=executed)
    report = dict(
        status="PASS", case=CASE, freeze=freeze, profile_sha256=profile_hash, model_sha256=model_hash,
        archive_sha256={path.name: sha256(path.read_bytes()).hexdigest()
                        for path in (report_path, actions_path)},
        variants_report_sha256=sha256(variants_path.read_bytes()).hexdigest(),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        step_s=DT, complete_steps=N, proposal_at_s=BLOCK_STEPS * DT,
        missing_at_s=2 * BLOCK_STEPS * DT, events=manager.events, service_counts=counts,
        first_soft_hard_difference_step=first_difference,
        samples_changed_vs_initial=int(np.count_nonzero(executed != soft.actions)),
        soft_actions_sha256=array_digest(soft.actions), hard_actions_sha256=array_digest(hard.actions),
        executed_actions_sha256=array_digest(executed), execution_archive_sha256=sha256(trace.read_bytes()).hexdigest(),
        executed=summarize_plan(replayed, budget, thermal), boundary_costs=boundary_costs,
        budget_margin=budget - replayed.worst_damage, batch_RF_audit=batch,
        devices_with_higher_RF_than_thermal=int(np.count_nonzero(deltas > 0)),
        maximum_device_RF_increase_vs_thermal=float(np.max(deltas)),
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "events", "complete_steps", "budget_margin", "batch_RF_audit", "service_counts",
    )}, indent=2))


if __name__ == "__main__":
    main()
