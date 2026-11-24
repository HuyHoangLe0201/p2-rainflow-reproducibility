# P2 traction-inverter reproducibility

Companion repository for *Switching Schedule Selection for Rail Traction Inverters under Per-Device Rainflow Budgets*. It contains the deterministic simulation, archived schedule comparisons, the backup-replacement test and the scripts for Figure 4.

- [Reproduction instructions](P2_Reviewer_Reproducibility/README.md) cover the locked Windows x86-64 environment, the main replay, the FCH2Rail comparator and Figure 4.
- [Online Resource 2](ESM_2.zip) is the original verified archive. The browsable directory has the same files and SHA-256 values.
- [Online Resource 1](ESM_1.pdf) is the supplementary note on electrothermal checks and numerical sensitivity.

## Run the checks

From a fresh clone in PowerShell, install the pinned CPython 3.10.19 environment as described in the reproduction instructions. Use a fresh output path for each full replay.

```powershell
Set-Location P2_Reviewer_Reproducibility
$replayPython = '.\.venv\Scripts\python.exe'
& $replayPython -I -B verify_package.py --output clean_verification
& $replayPython -I -B budget_thresholds.py . --output budget_thresholds.json
& $replayPython -I -B verify_referee_cases.py ../ESM_2.zip audit.json
```

The main replay checks all 150,000 switching vectors, including an accepted replacement at 10 s and its retention when the next proposal is missing at 20 s. At the strict inverter-level service tolerance, 9 of 15 service-matched policy combinations meet the budget on the synthetic +2% braking-current case. For the mapped FCH2Rail demand, 15 of 27 policy combinations match the service rule and none meets the illustrative rainflow budget. Both results concern the evaluated schedule libraries.

The bundled FCH2Rail trajectory is an unmodified simulated data file from Sebastian Herwartz-Polster and Florian Kühlkamp, German Aerospace Center (DLR), [Zenodo record 6359030](https://zenodo.org/records/6359030). It is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The bundled rainflow implementation retains its [vendor licence](P2_Reviewer_Reproducibility/DeCuong_4Bai/P2_sim/vendor/RAINFLOW_LICENSE.txt).

Online Resource 2 SHA-256

```text
fdb2a3ce1314190e313c9be8a874f9a7b2720f7d450bd58c689b1c9cf3ceb51d
```
