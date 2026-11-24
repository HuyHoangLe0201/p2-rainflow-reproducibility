"""Compare three selectors over the same replay-certified 3L-NPC mission catalog."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import sys
import time

# Import from the configured Python before legacy core.py adds its runtime folder.
legacy_runtime = str(Path(__file__).resolve().parent / "runtime")
if legacy_runtime in sys.path:
    sys.path.remove(legacy_runtime)
import numba
import numpy as np

from core import STATES
from core_recorded import simulate_recorded
from mission_rf_replay import replay
from run_campaign import mission, model, reference
from stored_manager import Certificate, StoredManager, cooldown_damage, terminal_check


DT = .0002
BLOCK_STEPS = 50_000
BLOCKS = 3
N = BLOCK_STEPS * BLOCKS
REGIMES = ("traction", "regeneration", "idle")
WEIGHTS = ((0., 0., 0.), (.24, 0., 0.), (0., .08, 0.))
FACTOR = 1.15**4.8 * np.exp(7000 / 398.15 - 7000 / 423.15)
SOFT_WEIGHT = 1.
SOFT_GRID = (0., .01, .03, .1, .3, 1., 3., 10., 30.)
IDLE_VECTOR = int(np.flatnonzero(np.all(STATES == 0, axis=1))[0])


@dataclass
class Plan:
    policy: tuple
    actions: np.ndarray
    service: np.ndarray
    limits: np.ndarray
    settled: np.ndarray
    damage: np.ndarray
    terminal: dict

    @property
    def mse(self):
        return self.service[:, 1].sum() / N

    @property
    def worst_damage(self):
        return float(self.damage.max())

    def feasible(self, budget):
        return bool(np.all(self.damage <= budget))


def array_digest(value):
    return sha256(value.tobytes()).hexdigest()


def common_service(plan, anchor, rms_fraction=.03, work_fraction=.02):
    """Same demand, two-sided signed work, and a shared per-phase RMS ceiling."""
    steps = anchor.service[:, 0]
    rms = np.sqrt(plan.service[:, 1] / steps)
    base_rms = np.sqrt(anchor.service[:, 1] / steps)
    work, base_work = plan.service[:2, 2], anchor.service[:2, 2]
    return bool(np.array_equal(plan.service[:, 0], steps)
            and np.all(rms <= np.minimum(92.4, (1 + rms_fraction) * base_rms + .05))
            and work[0] > 0 > work[1]
            and np.all(np.abs(work - base_work) <= work_fraction * np.abs(base_work)))


def verify_replay(profile, regimes, actions, lut, resistance, tau, record_rf=False):
    if actions.shape != (len(profile),):
        raise ValueError("Expected one recorded switch vector per sample")
    return replay(profile, regimes, actions, lut, resistance, tau, DT, BLOCK_STEPS, record_rf)


def simulate_plan(policy, profile, regimes, lut, resistance, tau):
    weights = np.array([WEIGHTS[index] for index in policy])
    met, damage, _, _, _, _, recorded = simulate_recorded(
        profile, weights, lut, resistance, tau, trace_stride=N, lookahead=True, horizon=5)
    if int(met[12]) != N or met[14]:
        return dict(policy=policy, reason="planner_incomplete", steps=int(met[12]),
                    planner_failure_code=int(met[14]))
    actions = recorded.astype(np.uint8)
    steps, failure, service, limits, settled, snap, rf, _ = verify_replay(
        profile, regimes, actions, lut, resistance, tau)
    if steps != N or failure:
        raise RuntimeError(f"Forced replay disagrees for {policy}: step {steps}, mask {failure}")
    np.testing.assert_allclose(rf, damage, rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose([service[:, 1].sum(), service[:, 2].sum()], met[[0, 9]],
                               rtol=2e-10, atol=1e-7)
    np.testing.assert_allclose(limits[[0, 1, 2, 3, 4, 5]],
                               met[[3, 5, 6, 10, 1, 2]] * [1, 1, .02, 1, 1, 1],
                               rtol=2e-10, atol=1e-8)
    # Every policy has the same terminal handoff during the zero-demand dwell.
    if actions[-1] != IDLE_VECTOR:
        actions[-1] = IDLE_VECTOR
        steps, failure, service, limits, settled, snap, rf, _ = verify_replay(
            profile, regimes, actions, lut, resistance, tau)
        if steps != N or failure:
            return dict(policy=policy, reason="terminal_000_transition",
                        steps=int(steps), replay_failure_code=int(failure))
    terminal = terminal_check(snap, lut, resistance)
    if not terminal["pass_terminal"] or np.sqrt(service[:, 1].sum() / N) > 92.4:
        return dict(policy=policy, reason="terminal_or_service", steps=N, terminal=terminal)
    terminal_damage = FACTOR * cooldown_damage(snap)
    return Plan(policy, actions, service, limits, FACTOR * settled, terminal_damage, terminal)


def collect_catalog(outcomes):
    plans, refused = [], []
    for item in outcomes:
        if isinstance(item, Plan):
            plans.append(item)
            print(item.policy, "replay pass", round(np.sqrt(item.mse), 5),
                  round(item.worst_damage, 9), flush=True)
        else:
            refused.append(item)
            print(item["policy"], item["reason"], item["steps"], flush=True)
    return plans, refused


def make_catalog(profile, regimes, lut, resistance, tau, workers):
    policies = product(range(len(WEIGHTS)), repeat=BLOCKS)
    simulate = partial(simulate_plan, profile=profile, regimes=regimes, lut=lut,
                       resistance=resistance, tau=tau)
    if workers == 1:
        return collect_catalog(map(simulate, policies))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return collect_catalog(pool.map(simulate, policies))


def tracking_key(plan):
    return (plan.mse, plan.limits[0], plan.policy)


def backup_key(plan):
    return (plan.worst_damage, plan.mse, plan.policy)


def build_domain(plans, budget):
    nodes = {}
    for plan in sorted(plans, key=backup_key):
        if not plan.feasible(budget):
            continue
        for k in range(BLOCKS + 1):
            prefix = plan.policy[:k]
            witness = nodes.setdefault(prefix, plan)
            # Equal action prefixes with identical deterministic initial conditions imply
            # equal plant, edge-window and RF states. Also check the settled ledger directly.
            assert np.array_equal(witness.actions[:k * BLOCK_STEPS], plan.actions[:k * BLOCK_STEPS])
            np.testing.assert_array_equal(witness.settled[k], plan.settled[k])
    bellman_edges = 0
    for prefix, backup in nodes.items():
        k = len(prefix)
        cost_to_go = backup.damage - backup.settled[k]
        if np.any(cost_to_go < -1e-12):
            raise AssertionError("Invalid RF ledger")
        if k == BLOCKS:
            continue
        child = nodes[backup.policy[:k+1]]
        assert child is backup  # common deterministic tie-break across the certified subtree
        increment = backup.settled[k+1] - backup.settled[k]
        assert np.all(increment >= 0)
        np.testing.assert_allclose(increment + child.damage - child.settled[k+1],
                                   cost_to_go, rtol=2e-13, atol=2e-13)
        assert np.all(backup.settled[k] + cost_to_go <= budget + 2e-13)
        bellman_edges += 1
    return nodes, bellman_edges


def run_backup(backup, choice, thermal, budget, profile_hash, model_hash):
    def certificate(plan, label):
        return Certificate(plan.actions, profile_hash, model_hash, plan.feasible(budget), label)

    manager = StoredManager(certificate(backup, "initial"))
    rejected = not manager.replace(certificate(thermal, "thermal_budget_test"))
    assert rejected == (not thermal.feasible(budget))
    if rejected:
        assert manager.events[-1]["backup_before"] == manager.events[-1]["backup_after"]
    assert manager.replace(certificate(choice, "hard_RF"))
    actions = np.empty(N, dtype=np.uint8)
    for k in range(N):
        if k == BLOCK_STEPS:
            assert not manager.replace(None)
            assert manager.events[-1]["backup_before"] == manager.events[-1]["backup_after"]
        actions[k] = manager.next_action()
    np.testing.assert_array_equal(actions, choice.actions)
    return actions, manager.events


def summarize_plan(plan, budget, baseline):
    service = {}
    for name, (steps, sse, work, reference_work), base_row in zip(REGIMES, plan.service, baseline.service):
        service[name] = dict(steps=int(steps), duration_s=float(steps * DT),
                             rms_A=float(np.sqrt(sse / steps)), work_J=float(work),
                             reference_work_J=float(reference_work),
                             work_change_vs_thermal_percent=(float(100 * (work-base_row[2]) /
                                                                    abs(base_row[2])) if name != "idle" else None))
    return dict(policy=list(plan.policy), actions_sha256=array_digest(plan.actions),
                rms_A=float(np.sqrt(plan.mse)), service=service,
                peak_bound_C=float(plan.limits[0]), nominal_peak_C=float(plan.limits[5]),
                neutral_point_peak_V=float(plan.limits[1]), max_window_edges=int(plan.limits[2]),
                peak_current_A=float(plan.limits[3]), peak_error_A=float(plan.limits[4]),
                RF_per_device=plan.damage.tolist(), RF_max=plan.worst_damage,
                limiting_device=int(np.argmax(plan.damage)) + 1,
                budget_pass=plan.feasible(budget), RF_over_budget=float(plan.worst_damage / budget),
                terminal=plan.terminal)


def compare(plans, profile, model_hash):
    anchor = min(plans, key=tracking_key)
    eligible = [plan for plan in plans if common_service(plan, anchor)]
    thermal = min(eligible, key=tracking_key)
    np.testing.assert_allclose(thermal.service[:, 3], anchor.service[:, 3], rtol=0., atol=1e-7)
    least = min(eligible, key=backup_key)
    budget = least.worst_damage + .5 * (thermal.worst_damage - least.worst_damage)
    def soft_select(weight, candidates=eligible):
        return min(candidates, key=lambda p: (
            p.mse / thermal.mse + weight * p.worst_damage / thermal.worst_damage,
            tracking_key(p)))
    curves = []
    for fraction in (None, 0., .25, .5, .75, 1., 1.25):
        bound = (float(np.nextafter(least.worst_damage, -np.inf)) if fraction is None else
                 least.worst_damage + fraction * (thermal.worst_damage - least.worst_damage))
        nodes, checked = build_domain(eligible, bound)
        curves.append(dict(fraction=fraction, budget=bound, admitted=() in nodes,
                           full_plans=sum(p.feasible(bound) for p in eligible),
                           certified_nodes=len(nodes), Bellman_checks=checked))
    hard = min((p for p in eligible if p.feasible(budget)), key=tracking_key)
    soft = soft_select(SOFT_WEIGHT)
    nodes, checked = build_domain(eligible, budget)
    executed, events = run_backup(nodes[()], hard, thermal, budget, array_digest(profile), model_hash)
    cert_rows = [
        dict(prefix=prefix, witness=plan.policy, settled=plan.settled[len(prefix)].tolist(),
             remaining=(plan.damage-plan.settled[len(prefix)]).tolist())
        for prefix, plan in sorted(nodes.items())
    ]
    strict_matches = {}
    for label, fraction in (("1pct", .01), ("0.5pct", .005)):
        matched = [p for p in eligible if common_service(p, thermal, fraction, fraction)]
        admissible = [p for p in matched if p.feasible(budget)]
        strict_matches[label] = dict(eligible=len(matched), hard_feasible=len(admissible),
                                    thermal=list(min(matched, key=tracking_key).policy),
                                    soft=list(soft_select(SOFT_WEIGHT, matched).policy),
                                    hard=list(min(admissible, key=tracking_key).policy) if admissible else None)
    results = dict(budget=budget, eligible=len(eligible), budget_sweep=curves,
                distinct_eligible_streams=len({array_digest(p.actions) for p in eligible}),
                certified_prefixes=cert_rows, Bellman_checks=checked, manager_events=events,
                soft_sweep=[dict(weight=w, **summarize_plan(soft_select(w), budget, thermal))
                            for w in SOFT_GRID],
                stricter_service=strict_matches,
                catalog=[dict(eligible=common_service(plan, thermal),
                              **summarize_plan(plan, budget, thermal)) for plan in plans],
                controllers={name: summarize_plan(p, budget, thermal) for name, p in
                             zip(("hard_thermal", "soft_RF", "hard_RF"), (thermal, soft, hard))})
    return results, (thermal, soft, hard, nodes[()]), executed


def crosscheck_selected(plans, profile, regimes, lut, resistance, tau):
    """Use the vendored batch cycle counter on all 30 chips at the chosen terminal."""
    checks = []
    for plan in {p.policy: p for p in plans}.values():
        n, fail, _, _, settled, snap, _, history = verify_replay(
            profile, regimes, plan.actions, lut, resistance, tau, record_rf=True)
        assert n == N and fail == 0
        np.testing.assert_array_equal(FACTOR * settled, plan.settled)
        scores = FACTOR * np.array([
            reference(np.append(history[:, chip], 40.), 4.8) / 40.**4.8
            for chip in range(30)
        ])
        np.testing.assert_allclose(scores, plan.damage, rtol=2e-10, atol=1e-12)
        checks.append(dict(policy=plan.policy, chips=30,
                           maximum_absolute_delta=float(np.max(np.abs(scores-plan.damage)))))
        assert terminal_check(snap, lut, resistance)["pass_terminal"]
    return checks


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory to store compact evidence")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1,
                        help="parallel full-plan replays")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    profile = mission(N * DT)
    regimes = np.zeros(N, dtype=np.int64)
    regimes[profile[:, 2] < 0.] = 1
    regimes[(profile[:, 0] == 0.) & (profile[:, 1] == 0.)] = 2
    lut, resistance, tau = model()
    root = Path(__file__).resolve().parent
    sources = {name: sha256((root / name).read_bytes()).hexdigest() for name in (
        "core.py", "core_recorded.py", "device_model.json", "stored_manager.py",
        "viability_helpers.py", "backup_helpers.py", "run_campaign.py",
        "mission_rf_replay.py", "mission_rf_budget.py")}
    model_hash = sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    plans, refusals = make_catalog(profile, regimes, lut, resistance, tau, args.workers)
    if not plans:
        (args.output / "results.json").write_text(json.dumps(dict(
            status="NO_TERMINAL_PLAN", refused=refusals, sources=sources), indent=2), encoding="utf-8")
        raise RuntimeError("No completed plan with a terminal idle certificate")
    report, chosen, executed = compare(plans, profile, model_hash)
    stored = dict(zip(("thermal", "soft", "hard", "initial_backup"), (p.actions for p in chosen)))
    np.savez_compressed(args.output / "selected_actions.npz", **stored, execution=executed)
    report.update(status="REPLAY_PASS_BATCH_RF_PENDING", scope="Offline deterministic RL/back-EMF 3L-NPC surrogate",
                  time_step_s=DT, block_steps=BLOCK_STEPS, horizon_s=N * DT,
                  replay_workers=args.workers,
                  profile_sha256=array_digest(profile), model_hash=model_hash,
                  sources=sources, versions=dict(numba=numba.__version__, numpy=np.__version__),
                  service_contract=dict(all_device_hard_limits=True, same_demand=True,
                                        phase_rms_relative_upper=.03, phase_rms_slack_A=.05,
                                        phase_signed_work_relative_band=.02),
                  qualification=dict(soft_primary_normalized_weight=SOFT_WEIGHT,
                                     budget_pick="Exploratory midpoint between best feasible RF and thermal-only RF",
                                     candidate_weights=WEIGHTS, candidates_total=len(WEIGHTS)**BLOCKS,
                                     temperature_limit_C=150., rf_terminal_cooling=True,
                                     terminal_handoff="Common forced 000 at the final zero-demand sample, replay checked",
                                     thermal_parameter_box="R upper +15%, tau +/-20%, loss upper +10%",
                                     RF_scale_scope="Fixed positive R scaling up to 1.15 at exact nominal tau, fixed coolant and loss history",
                                     RF_scale_multiplier=float(FACTOR), scope_hard="Certified stored suffixes within catalog, no plant disturbances"),
                  refused=refusals)
    (args.output / "replay_pending.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                                      encoding="utf-8")
    print("Verifying selected RF ledgers against batch rainflow", flush=True)
    checks = crosscheck_selected(chosen, profile, regimes, lut, resistance, tau)
    report.update(status="PASS", batch_RF_crosschecks=checks,
                  total_runtime_seconds=time.perf_counter() - began)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                              encoding="utf-8")
    print(json.dumps({name: {key: item[key] for key in ("policy", "rms_A", "RF_max", "budget_pass")}
                      for name, item in report["controllers"].items()}), flush=True)


if __name__ == "__main__":
    main()
