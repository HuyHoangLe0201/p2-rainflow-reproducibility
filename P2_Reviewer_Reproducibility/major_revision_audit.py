"""Checks for the P2 major revision using the archived matched-duty evidence."""
from argparse import ArgumentParser
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import sys


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def variation(samples):
    return sum(abs(b - a) for a, b in zip(samples, samples[1:]))


def foster_example():
    powers = (8, Fraction(17, 4), 8)
    histories = []
    for decay in (Fraction(1, 2), Fraction(7, 16)):
        samples = [Fraction(0)]
        for power in powers:
            samples.append(decay * samples[-1] + (1 - decay) * power)
        histories.append(samples + [Fraction(0)])  # monotone endpoint cooldown
    nominal, faster = histories
    assert nominal[0] < nominal[1] < nominal[2] < nominal[3]
    assert faster[0] < faster[2] < faster[1] < faster[3]
    small_cycle = faster[1] - faster[2]
    charges = [nominal[3] ** 2, faster[3] ** 2 + small_cycle ** 2]
    assert charges[1] > faster[3] ** 2  # matching peak ratio still misses the small cycle
    assert variation(nominal) == 2 * nominal[3]
    assert variation(faster) == 2 * (faster[3] + small_cycle)
    bound = faster[3] * max(map(variation, histories)) / 2
    assert bound >= max(charges)
    return {
        "histories": [[str(val) for val in history] for history in histories],
        "quadratic_charges": [str(val) for val in charges],
        "new_cycle_range": str(small_cycle),
        "peak_matched_charge": str(faster[3] ** 2),
        "variation_bound_for_two_histories": str(bound),
    }


def pareto_audit(root):
    data = root / "output/research/P2_MissionRF_BoundaryScan_20261120/results.json"
    report = json.loads(data.read_text(encoding="utf-8"), parse_float=Decimal)
    benchmark = report["full"]["brake_+200bp"]
    catalogue = benchmark["catalog"]
    observed = benchmark["controllers"]
    thermal = observed["hard_thermal"]
    hard = observed["hard_RF"]
    soft = observed["soft_RF"]
    budget = report["freeze"]["budget"]

    def service_matches(plan):
        observed, target = plan["service"], thermal["service"]
        return (
            observed["traction"]["work_J"] > 0 > observed["regeneration"]["work_J"]
            and all(
                observed[phase]["steps"] == target[phase]["steps"]
                and observed[phase]["rms_A"] <= min(
                    Decimal("92.4"), Decimal("1.005") * target[phase]["rms_A"] + Decimal("0.05"))
                for phase in ("traction", "regeneration", "idle"))
            and all(
                abs(observed[phase]["work_J"] - target[phase]["work_J"])
                <= Decimal("0.005") * abs(target[phase]["work_J"])
                for phase in ("traction", "regeneration"))
        )

    def tracking(plan):
        return plan["rms_A"], plan["peak_bound_C"], tuple(plan["policy"])

    matched = [p for p in catalogue if service_matches(p)]
    admitted = [p for p in matched if p["RF_max"] <= budget]
    unique = {}
    for plan in sorted(matched, key=tracking):
        same = unique.setdefault(plan["actions_sha256"], plan)
        assert all(same[key] == plan[key] for key in ("rms_A", "RF_max", "RF_per_device"))
    assert (len(catalogue), len(matched), len(admitted), len(unique)) == (27, 15, 9, 5)
    assert [min(rows, key=tracking)["actions_sha256"] for rows in (matched, admitted)] == [
        thermal["actions_sha256"], hard["actions_sha256"]]
    frontier = []
    for plan in sorted(unique.values(), key=tracking):
        if any(
            other["RF_max"] <= plan["RF_max"] and other["rms_A"] <= plan["rms_A"]
            and (other["RF_max"] < plan["RF_max"] or other["rms_A"] < plan["rms_A"])
            for other in unique.values()
        ):
            continue
        frontier.append({
            "policy": plan["policy"], "rms_A": str(plan["rms_A"]),
            "RF_max": str(plan["RF_max"]),
            "selected_by": [
                label for label, reference in (("thermal", thermal), ("hard", hard), ("soft", soft))
                if reference["actions_sha256"] == plan["actions_sha256"]
            ],
        })
    return {
        "input_sha256": digest(data),
        "matched_policy_tuples": len(matched),
        "distinct_matched_switching_sequences": len(unique),
        "budget": str(budget),
        "nondominated": frontier,
        "nearest_matched_RF_to_budget": str(min(abs(p["RF_max"] - budget) for p in matched)),
        "nearest_matched_RF_above_budget": str(min(p["RF_max"] - budget for p in matched if p["RF_max"] > budget)),
        "nearest_matched_RF_below_budget": str(min(budget - p["RF_max"] for p in admitted)),
    }


def rejection_audit(root, workers):
    sim = root / "DeCuong_4Bai/P2_sim"
    sys.path.insert(0, str(sim))
    import numpy as np
    from mission_rf_glm_followup import early_duty
    from mission_rf_budget import BLOCK_STEPS, N, make_catalog, model
    from mission_rf_fixed_validation import regimes_for
    from stored_manager import Certificate, StoredManager

    folder = root / "followup_20261126/handoff"
    anchor = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    with np.load(folder / "actions.npz") as replay:
        initial, replacement = replay["initial"], replay["replacement"]
    assert len(initial) == len(replacement) == N
    profile, _ = early_duty(14)
    plans, failures = make_catalog(profile, regimes_for(profile), *model(), workers)
    assert not failures and len(plans) == 27
    catalogue = {sha256(np.asarray(plan.actions, dtype=np.int64).tobytes()).hexdigest() for plan in plans}

    def certificate(actions, certified, label):
        return Certificate(actions, anchor["profile_sha256"], anchor["model_sha256"], certified, label)

    initial = certificate(initial, True, "initial")
    replacement = certificate(replacement, True, "replacement")
    assert {initial.digest(), replacement.digest()} <= catalogue
    manager = StoredManager(initial)
    for _ in range(BLOCK_STEPS):
        manager.next_action()
    assert manager.replace(replacement)
    assert manager.events[-1] == anchor["events"][0]
    for _ in range(BLOCK_STEPS):
        manager.next_action()

    proposal = None
    for variation_index in range(1, 28):
        candidate = replacement.actions.copy()
        for bit in range(5):
            if variation_index & (1 << bit):
                pos = len(candidate) - bit - 1
                candidate[pos] = (candidate[pos] + 1) % 27
        entry = certificate(candidate, False, "unverified proposal")
        if entry.digest() not in catalogue:
            proposal = entry
            break
    assert proposal is not None and proposal.digest() != replacement.digest()
    assert np.array_equal(proposal.actions[:2 * BLOCK_STEPS], manager.executed)
    assert not manager.replace(proposal)
    rejection = manager.events[-1]
    assert rejection["reason"] == "certificate_failed"
    assert rejection["backup_before"] == rejection["backup_after"] == replacement.digest()
    for _ in range(N - 2 * BLOCK_STEPS):
        manager.next_action()
    executed = np.array(manager.executed, dtype=np.uint8)
    assert np.array_equal(executed, replacement.actions)
    assert sha256(executed.tobytes()).hexdigest() == anchor["executed_sha256"]
    return {
        "regenerated_policy_tuples": len(plans),
        "distinct_switching_sequences": len(catalogue),
        "proposal_sha256_int64": proposal.digest(),
        "boundary_s": 20, "matching_prefix_steps": 2 * BLOCK_STEPS,
        "rejection": rejection, "executed_steps": N,
        "executed_sha256_uint8": anchor["executed_sha256"],
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = {
        "single_branch_example": foster_example(),
        "main_catalogue": pareto_audit(root),
        "uncertified_off_catalogue_proposal": rejection_audit(root, args.workers),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("PASS: exact Foster example, matched-duty Pareto frontier and off-catalogue rejection")


if __name__ == "__main__":
    main()

