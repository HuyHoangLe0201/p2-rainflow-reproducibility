"""Recompute finite-library RF budget thresholds from an extracted P2 supplement."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

parser = ArgumentParser(description=__doc__)
parser.add_argument("supplement_root", type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
root = args.supplement_root
hashes = {}

def read_report(relative):
    data = (root / relative).read_bytes()
    hashes[relative] = sha256(data).hexdigest()
    return json.loads(data)

def tracking(plan):
    # Squaring archived RMS preserves order in these catalogues.
    return plan["rms_A"], plan["peak_bound_C"], tuple(plan["policy"])

def service_matches(plan, baseline):
    observed, target = plan["service"], baseline["service"]
    return (
        observed["traction"]["work_J"] > 0 > observed["regeneration"]["work_J"]
        and all(
            observed[phase]["steps"] == target[phase]["steps"]
            and observed[phase]["rms_A"] <= min(92.4, 1.005 * target[phase]["rms_A"] + 0.05)
            for phase in ("traction", "regeneration", "idle"))
        and all(
            abs(observed[phase]["work_J"] - target[phase]["work_J"])
            <= 0.005 * abs(target[phase]["work_J"])
            for phase in ("traction", "regeneration"))
    )

research = "output/research"
boundary = read_report(f"{research}/P2_MissionRF_BoundaryScan_20261120/results.json")
variants = read_report(f"{research}/P2_MissionRF_FixedValidation_20261120/results.json")
public = read_report("P2_PublicRail_FCH2Rail_20261122/results.json")
public_36s = read_report("followup_20261126/fch36/results.json")
budget = boundary["freeze"]["budget"]
assert budget == variants["freeze"]["budget"] == public["parameters"]["budget"]
assert public_36s["parameters"] == public["parameters"]
assert public_36s["duration_s"] == 36.0 and public_36s["schedules_completed"] == 27
reports = {
    "brake_+200bp": boundary["full"]["brake_+200bp"],
    **variants["results"],
    "FCH2Rail": public,
    "FCH2Rail_36s": public_36s,
}
expected = {
    "brake_+200bp": (15, 9, "change"),
    "derated": (3, 3, "retain"),
    "long_regeneration": (6, 0, "refuse"),
    "late_traction_pulse": (12, 6, "retain"),
    "heavy": (3, 0, "refuse"),
    "FCH2Rail": (15, 0, "refuse"),
    "FCH2Rail_36s": (10, 0, "refuse"),
}
def evaluate(name, report):
    catalogue = report["catalog"]
    assert len(catalogue) == 27 and len({tuple(p["policy"]) for p in catalogue}) == 27
    assert all(len(p["RF_per_device"]) == 30 and p["RF_max"] == max(p["RF_per_device"])
               for p in catalogue)
    signed = [p for p in catalogue
              if p["service"]["traction"]["work_J"] > 0 > p["service"]["regeneration"]["work_J"]]
    baseline = min(signed, key=tracking)
    if name == "FCH2Rail_36s":
        # The follow-up catalog records the thermal reference charge as a scalar.
        assert baseline["RF_max"] == report["thermal_RF"]
    else:
        original_baseline = report["controllers"]["hard_thermal"]
        assert baseline["policy"] == original_baseline["policy"]
        assert baseline["service"] == original_baseline["service"]
    matched = [p for p in catalogue if service_matches(p, baseline)]
    if name == "FCH2Rail_36s":
        assert {tuple(p["policy"]) for p in matched} == {
            tuple(p["policy"]) for p in catalogue if p["service_matched"]
        }
    assert min(matched, key=tracking)["policy"] == baseline["policy"]
    def decision_at(cap):
        admitted = [p for p in matched if p["RF_max"] <= cap]
        if not admitted:
            return "refuse"
        choice = min(admitted, key=tracking)
        return "retain" if choice["actions_sha256"] == baseline["actions_sha256"] else "change"

    admitted = [p for p in matched if p["RF_max"] <= budget]
    minimum, thermal = min(p["RF_max"] for p in matched), baseline["RF_max"]
    decision = decision_at(budget)
    formula = "refuse" if budget < minimum else "change" if budget < thermal else "retain"
    assert decision == formula
    assert decision_at(math.nextafter(minimum, -math.inf)) == "refuse"
    assert decision_at(minimum) == ("change" if minimum < thermal else "retain")
    assert decision_at(math.nextafter(thermal, -math.inf)) == (
        "change" if minimum < thermal else "refuse")
    assert decision_at(thermal) == "retain"
    assert (len(matched), len(admitted), decision) == expected[name]
    return {
        "model_feasible": len(catalogue),
        "service_matched": len(matched),
        "within_budget": len(admitted),
        "distinct_matched": len({p["actions_sha256"] for p in matched}),
        "distinct_within_budget": len({p["actions_sha256"] for p in admitted}),
        "minimum_matched_RF": minimum,
        "minimum_catalogue_RF": min(p["RF_max"] for p in catalogue),
        "thermal_optimum_RF": thermal,
        "decision": decision,
        "minimum_minus_budget": minimum - budget,
        "thermal_minus_budget": thermal - budget,
        "switch_interval_width": thermal - minimum,
    }

assert set(reports) == set(expected)
results = {name: evaluate(name, report) for name, report in reports.items()}
assert Counter(x["decision"] for x in results.values()) == {"retain": 2, "change": 1, "refuse": 4}
out = {"scope": "Archived catalogue postprocessing with strict 0.5% service matching",
       "budget": budget, "input_sha256": hashes, "results": results}
args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
for name, item in results.items():
    print(name, f'{item["within_budget"]}/{item["service_matched"]}',
          item["minimum_matched_RF"], item["thermal_optimum_RF"], item["decision"])
