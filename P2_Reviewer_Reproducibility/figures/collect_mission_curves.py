"""Replay three archived action sequences and export the 0.1 s observables for Fig. 4.

Run with the pinned environment documented in the P2 Supplementary Reproducibility
archive. Pass the extracted P2_Reviewer_Reproducibility root and an output JSON.
"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import sys


def read_verified(path, expected):
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"Unexpected SHA-256: {path.name}")
    return data


def instrument(source):
    """Add observations without editing the archived replay source on disk."""
    replacements = (
        ("@numba.njit(cache=True)", "@numba.njit(cache=False)"),
        ("block_steps, record_rf):", "block_steps, record_rf, window_steps):"),
        (
            "    voltage_delta, phase = 0., 0.",
            "    observations = np.zeros((len(profile) // window_steps, 3))\n"
            "    voltage_delta, phase = 0., 0.",
        ),
        (
            "        completed = k + 1",
            "        window = k // window_steps\n"
            "        observations[window, 0] += np.sum((predicted - reference)**2) / 3\n"
            "        observations[window, 1] = max(observations[window, 1], np.max(caps))\n"
            "        observations[window, 2] = np.max(settled)\n"
            "        completed = k + 1",
        ),
        ("settled, 4.8), history)", "settled, 4.8), history, observations)"),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"Replay instrumentation anchor differs: {old!r}")
        source = source.replace(old, new, 1)
    namespace = {}
    exec(compile(source, "<observed archived replay>", "exec"), namespace)
    return namespace["replay"]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    engine = args.archive_root / "DeCuong_4Bai/P2_sim"
    sys.path.insert(0, str(engine.resolve()))
    import numpy as np
    from mission_rf_boundary_scan import scan_profiles
    from mission_rf_budget import BLOCK_STEPS, DT, FACTOR, N, array_digest, model
    from mission_rf_fixed_validation import regimes_for
    from stored_manager import cooldown_damage, terminal_check

    case = "brake_+200bp"
    source_sha = "0edf42d76ee03746dc84fc29660f97e13d9d57223cd7e6f4234bf2baa91588e3"
    report_sha = "44d4ede2992765516c451f309f623530679eb62a31d1ae622952b7bbc615410a"
    source = read_verified(engine / "mission_rf_replay.py", source_sha)
    replay = instrument(source.decode("utf-8"))
    result_dir = args.archive_root / "output/research/P2_MissionRF_BoundaryScan_20261120"
    report = json.loads(read_verified(result_dir / "results.json", report_sha))
    profile, _ = scan_profiles()[case]
    assert array_digest(profile) == report["profiles"][case]["profile_sha256"]
    regimes = regimes_for(profile)
    tables = model()
    steps = round(0.1 / DT)
    assert N % steps == 0 and steps * DT == 0.1
    edges = np.arange(0, N + 1, steps) * DT

    changes = np.flatnonzero(np.abs(np.diff(profile[:, 0])) > 3)
    demand_indices = np.unique(np.r_[np.arange(0, N, steps), np.arange(0, 1000, 25),
                                    changes, changes + 1, N - 1])
    rows = report["full"][case]
    observed = {
        "case": case,
        "replay_source_sha256": source_sha,
        "archived_report_sha256": report_sha,
        "budget": report["freeze"]["budget"],
        "window_s": steps * DT,
        "window_edges_s": edges.tolist(),
        "phase_changes_s": ((np.flatnonzero(np.diff(regimes)) + 1) * DT).tolist(),
        "demand_time_s": np.r_[demand_indices * DT, N * DT].tolist(),
        "reference_crest_A": np.r_[profile[demand_indices, 0], profile[-1, 0]].tolist(),
        "selectors": {},
    }
    pairs = (("thermal", "hard_thermal"), ("soft", "soft_RF"), ("hard", "hard_RF"))
    with np.load(result_dir / f"{case}_actions.npz", allow_pickle=False) as actions:
        for key, name in pairs:
            expected = rows["controllers"][name]
            digest = array_digest(actions[key])
            assert digest == expected["actions_sha256"] == rows["tighter_service"]["choices"][name]
            end, failure, service, limits, ledger, snap, _, _, values = replay(
                profile, regimes, actions[key], *tables, DT, BLOCK_STEPS, False, steps)
            assert end == N and failure == 0 and terminal_check(snap, *tables[:2])["pass_terminal"]
            damage = FACTOR * cooldown_damage(snap)
            rms = np.sqrt(service[:, 1].sum() / N)
            peak = limits[0]
            np.testing.assert_allclose(rms, expected["rms_A"], rtol=2e-13)
            np.testing.assert_allclose(peak, expected["peak_bound_C"], rtol=2e-13)
            np.testing.assert_allclose(damage, expected["RF_per_device"], rtol=2e-13, atol=2e-13)
            np.testing.assert_allclose(values[:, 0].sum(), service[:, 1].sum(), rtol=2e-13)
            assert np.max(values[:, 1]) == peak
            settled = FACTOR * np.r_[0, values[:, 2]]
            np.testing.assert_allclose(settled[::BLOCK_STEPS // steps], FACTOR * ledger.max(axis=1))
            assert np.all(np.diff(settled) >= 0)
            assert bool(np.all(damage <= observed["budget"])) == expected["budget_pass"]
            observed["selectors"][name] = {
                "action_sha256": digest,
                "mission_rms_A": rms,
                "mission_peak_bound_C": peak,
                "terminal_charge": damage.max(),
                "window_rms_A": np.sqrt(values[:, 0] / steps).tolist(),
                "window_peak_bound_C": values[:, 1].tolist(),
                "settled_charge": settled.tolist(),
            }
            print(f"{name}: {end} steps; RMS {rms:.6f} A; peak {peak:.6f} C; "
                  f"settled {settled[-1]:.6f}; terminal {damage.max():.6f}", flush=True)

    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(observed, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
