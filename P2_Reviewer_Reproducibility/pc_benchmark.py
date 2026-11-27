"""Descriptive desktop microbenchmark for stored handoff and budget reduction."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns
import numpy as np

def measure(func,n):
    for _ in range(12):
        func()
    samples=[]
    for _ in range(n):
        start=perf_counter_ns()
        func()
        samples.append((perf_counter_ns()-start)/1000)
    return dict(repetitions=n,median_us=statistics.median(samples),
                iqr_us=statistics.quantiles(samples,n=4)[2]-statistics.quantiles(samples,n=4)[0])

def run(root):
    sys.path.insert(0,str(root/"DeCuong_4Bai/P2_sim"))
    from stored_manager import Certificate,StoredManager
    folder=root/"followup_20261126/handoff"
    data=json.loads((folder/"results.json").read_bytes())
    with np.load(folder/"actions.npz") as a:
        initial = Certificate(a["initial"],data["profile_sha256"],data["model_sha256"],True,"initial")
        replacement = Certificate(a["replacement"],data["profile_sha256"],data["model_sha256"],True,"replacement")

    manager=StoredManager(initial)
    for _ in range(50_000):
        manager.next_action()
    assert manager.replace(replacement)
    accept=measure(lambda: manager.replace(replacement),101)
    assert manager.events[-1]["reason"]=="certificate_committed"
    for _ in range(50_000):
        manager.next_action()
    assert not manager.replace(initial) and manager.events[-1]["reason"]=="executed_prefix_mismatch"
    reject=measure(lambda: manager.replace(initial),101)

    # Isolate a reduction of precomputed charges. No model solve or bound construction is timed.
    seed=np.array(data["replacement"]["RF_per_device"],dtype=np.float64)
    scans=[]
    for count,outcomes,devices in ((27,1,30),(64,1,30),(64,2,30),(64,8,30),(64,8,60)):
        matrix=np.resize(seed,(count,outcomes,devices))
        scan=measure(lambda: np.all(matrix<=6.738416227606914,axis=(1,2)),1001)
        scans.append(dict(candidates=count,outcomes=outcomes,devices=devices,charge_storage_bytes=matrix.nbytes,**scan))
    return dict(scope="Desktop measured medians; not deployed-controller timing; bound construction excluded",
        platform=platform.platform(),cpu=platform.processor(),python=platform.python_version(),
        numpy=np.__version__,acceptance_at_10_s=accept,prefix_check_at_20_s=reject,precomputed_budget_scans=scans)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    a.output.write_text(json.dumps(run(a.root),indent=2)+"\n",encoding="utf8")

