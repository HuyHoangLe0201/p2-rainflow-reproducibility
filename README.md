# P2 traction-inverter reproducibility

Companion repository for *Switching Schedule Selection for Rail Traction Inverters under Per-Device Rainflow Budgets*. The package covers the deterministic three-level neutral-point-clamped (3L-NPC) inverter simulation, the matched-service comparisons, the backup-replacement test and the four-panel result figure.

- [Online Resource 2](ESM_2.zip) is the verified archive. The [browsable copy](P2_Reviewer_Reproducibility/) contains the same files and SHA-256 values.
- [Online Resource 1](ESM_1.pdf) is the supplementary note on electrothermal checks and numerical sensitivity. The [archive README](P2_Reviewer_Reproducibility/README.md) documents the provenance and scope of those checks.

## Reproduce the results

These commands use PowerShell on Windows x86-64. The replay uses CPython **3.10.19** and a wheel lock with hashes. Start with that interpreter on your `PATH`. Anaconda's Windows x86-64 build was used for verification.

### 1. Set up the pinned replay environment

```powershell
git clone --branch p2-res-2026-11-24 https://github.com/HuyHoangLe0201/p2-rainflow-reproducibility.git
Set-Location p2-rainflow-reproducibility
python --version
python -I -m venv .venv
$replayPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
$archive = (Resolve-Path .\ESM_2.zip).Path
Set-Location P2_Reviewer_Reproducibility
& $replayPython -I -m pip --isolated install --only-binary=:all: --require-hashes -r requirements-lock-win64.txt
& $replayPython -I -m pip check
$run = Join-Path ([IO.Path]::GetTempPath()) ('p2-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $run | Out-Null
$env:NUMBA_CACHE_DIR = Join-Path $run 'numba'
```

Confirm that `python --version` prints `Python 3.10.19` before creating the environment. Keep the same PowerShell session and working directory for the commands below. Results go in the fresh directory `$run`. Numba compiles functions on the first run.

### 2. Verify the main replay and schedule comparisons

```powershell
& $replayPython -I -B verify_package.py --output (Join-Path $run 'verification')
& $replayPython -I -B budget_thresholds.py . --output (Join-Path $run 'budget_thresholds.json')
& $replayPython -I -B verify_referee_cases.py $archive (Join-Path $run 'audit.json')
```

The first command verifies the archive checksums and replays 150,000 switching vectors, including acceptance at 10 s and retention when the next proposal is absent at 20 s. It also checks the archived Appendix C and D records. The other commands recalculate library budget thresholds and check the replacement events, soft-penalty sensitivity and the archived FCH2Rail result. For the synthetic case with a 2% increase in braking-current demand, nine of the 15 service-matched policy combinations meet the budget.

### 3. Regenerate the FCH2Rail comparison

```powershell
$trajectory = 'P2_PublicRail_FCH2Rail_20261122/A Coruna-Ferrol-A Coruna_simulated_trajectory.csv'
& $replayPython -B DeCuong_4Bai/P2_sim/mission_rf_public_rail.py --trajectory $trajectory --output (Join-Path $run 'fch2rail') --workers 4
```

Expected result on the mapped demand: 27 policy combinations complete replay, 15 satisfy the inverter-level service rule, and none of those 15 meets the fixed rainflow budget. The minimum matched maximum device charge is 7.439714, against a budget of 6.738416. These counts apply to the evaluated library.

### 4. Regenerate Figure 4

Use the replay environment to regenerate the observations. The render uses a separate environment with CPython **3.12.14**, NumPy **2.2.4** and Matplotlib **3.10.1**. Make `python` resolve to CPython 3.12.14 for the rendering commands and recheck its version.

```powershell
$curves = Join-Path $run 'replay_curves.json'
& $replayPython -I -B figures/collect_mission_curves.py . $curves
```

```powershell
python --version
$plotEnv = Join-Path $run 'plot-venv'
python -I -m venv $plotEnv
$plotPython = (Resolve-Path (Join-Path $plotEnv 'Scripts/python.exe')).Path
& $plotPython -I -m pip install 'numpy==2.2.4' 'matplotlib==3.10.1'
& $plotPython -I -B figures/plot_mission_curves.py $curves (Join-Path $run 'replay_curves.pdf')
Get-FileHash -Algorithm SHA256 -LiteralPath $curves, 'figures/mission_curves.json'
```

Compare the observations with the [reference JSON](P2_Reviewer_Reproducibility/figures/mission_curves.json) and the plot with the [reference vector PDF](P2_Reviewer_Reproducibility/figures/mission_curves.pdf).

## Data sources

The bundled FCH2Rail trajectory is an unmodified simulated data file from Sebastian Herwartz-Polster and Florian Kühlkamp, German Aerospace Center (DLR), [Zenodo record 6359030](https://zenodo.org/records/6359030). It is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The bundled rainflow implementation retains its [vendor licence](P2_Reviewer_Reproducibility/DeCuong_4Bai/P2_sim/vendor/RAINFLOW_LICENSE.txt).

The SHA-256 of `ESM_2.zip` is `fdb2a3ce1314190e313c9be8a874f9a7b2720f7d450bd58c689b1c9cf3ceb51d`.
