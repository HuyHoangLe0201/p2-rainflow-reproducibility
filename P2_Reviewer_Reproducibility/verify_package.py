"""Check the packaged manuscript evidence in a clean Python environment."""
from argparse import ArgumentParser
from collections import Counter
import csv
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "output/research"
MODULES = ROOT / "DeCuong_4Bai/P2_sim"
ROUNDING = RESEARCH / "P2_MissionRF_Fine1e-20_20261122"
REPLAY = RESEARCH / "P2_MissionRF_Midmission_20261122"
ATTRIBUTION = ROOT / "rf_mechanism_20261119/P2_RF_attribution_20261119.json"
PINS = {"numpy": "2.2.6", "numba": "0.63.1", "llvmlite": "0.46.0", "scipy": "1.15.3"}


def read_json(path):
    return json.loads(path.read_bytes())


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_assets():
    lines = (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for line in lines:
        digest, filename = line.split("  ", 1)
        assert sha256((ROOT / filename).read_bytes()).hexdigest() == digest, filename
    return len(lines)


def verify_rounding():
    """Validate range coverage at 1e-20 and the published triplets of mechanisms."""
    doc = read_json(ROUNDING / "lattice_audit.json")
    runs = read_csv(ROUNDING / "decision_grid_runs.csv")
    levels = read_csv(ROUNDING / "binary64_levels.csv")
    thresholds = read_csv(ROUNDING / "exact_rounding_thresholds.csv")
    assert doc["status"] == "PASS"
    tags = {(r["window"], r["profile"]): r for r in levels}
    assert len(tags) == len(levels) == 515
    assert all(r["catalog_size"] == "27" for r in levels)

    def bits(x):
        return str(struct.unpack(">Q", struct.pack(">d", x))[0])

    routes = {
        "original": lambda k: 360.0 * (1.0 + (k / 10**18) / 10_000.0),
        "direct": lambda k: (360 * (10**22 + k)) / 10**22,
    }
    total, disagreement = 0, 0
    visited = set()
    for w in doc["windows"]:
        number = str(w["window"])
        spans = [r for r in runs if r["window"] == number]
        cursor, delta = w["lo"], 0
        for row in spans:
            lo, hi, size = (int(row[k]) for k in ("first_tick", "last_tick", "grid_points"))
            assert lo == cursor and size == hi - lo + 1 and size > 0
            cursor = hi + 1
            for route, project in routes.items():
                profile = row[route + "_profile"]
                assert bits(project(lo)) == bits(project(hi)) == profile
                assert tags[number, profile]["decision"] == row[route + "_decision"]
                visited.add((number, profile))
            total += size
            if row["original_decision"] != row["direct_decision"]:
                delta += size
        assert cursor == w["hi"] + 1 and delta == w["decision_disagreements"]
        disagreement += delta
    assert visited == tags.keys()
    assert total == doc["stats"]["decimal_points_per_method"] == 800_000_008
    assert disagreement == doc["stats"]["decision_disagreements"] == 73_015_367
    profile_edges, decimal_edges = {}, {}
    for route, project in routes.items():
        edges = [r for r in thresholds if r["method"] == route]
        assert Counter(e["kind"] for e in edges) == doc["stats"]["mechanism_counts"][route]
        profile_edges[route] = set()
        decimal_edges[route] = set()
        for edge in edges:
            lo, hi = int(edge["left_tick"]), int(edge["right_tick"])
            a, b = edge["left_profile"], edge["right_profile"]
            assert hi == lo + 1 and bits(project(lo)) == a and bits(project(hi)) == b
            root = Fraction(int(edge["exact_threshold_numerator"]),
                            int(edge["exact_threshold_denominator"])) * 10**20
            quotient, remainder = divmod(root.numerator, root.denominator)
            assert hi == quotient + (remainder != 0 or edge["tie_selects_profile"] == a)
            profile_edges[route].add((edge["window"], a, b))
            decimal_edges[route].add((edge["window"], lo, hi))
    assert len(thresholds) == doc["stats"]["matched_thresholds"] == 210
    assert len(profile_edges["original"] & profile_edges["direct"]) == 51
    assert not (decimal_edges["original"] & decimal_edges["direct"])
    return dict(windows=len(doc["windows"]), grid_points_per_route=total,
                decision_disagreements=disagreement, binary64_levels=len(levels),
                thresholds=len(thresholds), range_count=len(runs))


def verify_attribution():
    """Recompute the reported contrasts from the archived chip-9 cycle summaries."""
    attr = read_json(ATTRIBUTION)
    assert attr["status"] == "PASS_REPLAY" and attr["chip"] == 9
    assert attr["horizons_s"] == [4097.0, 8193.0]
    pairs_checked = 0
    with localcontext() as ctx:
        ctx.prec = 75

        def excess(arms, field):
            return Decimal(arms["on"][field]) - Decimal(arms["zero"][field])

        for name, case in attr["cases"].items():
            for horizon, row in case["curve"].items():
                ref = attr["reference"][horizon]
                arms = row["arms"]
                c = excess(arms, "closed") - excess(ref, "closed")
                t = excess(arms, "total") - excess(ref, "total")
                for actual, field in ((c, "closed_gain"), (t, "total_gain"), (t - c, "half_gain")):
                    target = Decimal(str(row[field]))
                    assert abs(actual - target) <= max(abs(target) * Decimal("1e-14"), Decimal("1e-60")), (name, horizon, field)
                for i, value in enumerate((c, t)):
                    expected = Decimal(str(row["ratio"][i]))
                    computed = value / Decimal(str(row["conservative_error"][i]))
                    assert abs(computed - expected) <= max(abs(expected) * Decimal("1e-14"), Decimal("1e-55"))
                assert all(len(arm["material"]) == 2 for arm in (*arms.values(), *ref.values()))
                pairs_checked += 1
    assert pairs_checked == 18
    anchor = attr["cases"]["anchor"]["curve"]["8193.0"]
    assert anchor["ratio"][0] > 10 > anchor["ratio"][1]
    return dict(chip=9, cases=len(attr["cases"]), comparisons=pairs_checked,
                anchor_ratios=anchor["ratio"])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must be a fresh directory")
    assert not sys.flags.optimize, "Run with Python assertions enabled"
    assert sys.version_info[:3] == (3, 10, 19), sys.version
    assert {pkg: version(pkg) for pkg in PINS} == PINS
    n_files = verify_assets()
    rounding, attribution = verify_rounding(), verify_attribution()
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, "-B", str(MODULES / "mission_rf_midmission_replay.py"),
                    "--output", str(output / "mission")], cwd=ROOT, check=True)
    actual = read_json(output / "mission/results.json")
    expected = read_json(REPLAY / "results.json")
    for field in ("events", "executed", "service_counts", "boundary_costs", "batch_RF_audit",
                  "executed_actions_sha256", "budget_margin"):
        assert actual[field] == expected[field], field
    import numpy as np
    with np.load(output / "mission/executed_actions.npz", allow_pickle=False) as a, \
            np.load(REPLAY / "executed_actions.npz", allow_pickle=False) as b:
        np.testing.assert_array_equal(a["actions"], b["actions"])
    result = dict(status="PASS", verified_files=n_files, python="3.10.19", dependencies=PINS,
                  appendix_A=rounding, appendix_B=attribution,
                  mission=dict(steps=actual["complete_steps"], events=actual["events"],
                               budget_margin=actual["budget_margin"],
                               executed_actions_sha256=actual["executed_actions_sha256"],
                               batch_RF_audit=actual["batch_RF_audit"],
                               service_counts=actual["service_counts"]))
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("PASS: 3L-NPC replay, matched catalogs, Appendix A and Appendix B checks")


if __name__ == "__main__":
    main()
