# P2 traction-inverter reproducibility

Code and data for *Switching Schedule Selection for Rail Traction Inverters under Per-Device Rainflow Budgets* by Huy Hoang Le and Kim-Anh Nguyen. The experiments use a simulated three-level neutral-point-clamped inverter.

## Schedule catalogues

| Dataset | Contents |
| --- | --- |
| [Original catalogue](P2_Reviewer_Reproducibility/output/research/P2_MissionRF_BoundaryScan_20261120/results.json) | 27 candidate policies per synthetic duty, with tracking, temperature, work and per-device rainflow charge |
| [Expanded catalogue](P2_Reviewer_Reproducibility/expanded_results.json) | All 64 candidates for the main duty, including the original 27 |
| [30 s FCH2Rail mapping](P2_Reviewer_Reproducibility/P2_PublicRail_FCH2Rail_20261122/results.json) and [36 s mapping](P2_Reviewer_Reproducibility/followup_20261126/fch36/results.json) | Catalogue and results for each mapped demand |
| [Recorded switch vectors](P2_Reviewer_Reproducibility/output/research/P2_MissionRF_BoundaryScan_20261120/brake_+200bp_actions.npz) | Stored vectors for the three selectors in the main duty. The replay scripts regenerate the switching schedules. |

The catalogues distinguish policy combinations from unique switching sequences. See the [model, scripts and data layout](P2_Reviewer_Reproducibility/README.md) for the service rule, additional tests and archived results.

## Reproduce the results

Use Windows x86-64 and CPython **3.10.19**. We verified the Anaconda build with the pinned wheels and hashes below. Start with that interpreter on the PowerShell path and clone from a short working directory.

```powershell
git clone --branch p2-res-2026-11-27 https://github.com/HuyHoangLe0201/p2-rainflow-reproducibility.git
Set-Location p2-rainflow-reproducibility
$base = (Get-Command python -CommandType Application | Select-Object -First 1).Source
if ((& $base --version) -ne "Python 3.10.19") { throw "Select CPython 3.10.19." }
& $base -I -m venv .venv
$Python = (Resolve-Path .\.venv\Scripts\python.exe).Path
Set-Location P2_Reviewer_Reproducibility
& $Python -I -m pip --isolated install --only-binary=:all: --require-hashes -r requirements-lock-win64.txt
& $Python -I -m pip check
$run = Join-Path $env:TEMP ("p2-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $run | Out-Null
$env:NUMBA_CACHE_DIR = Join-Path $run 'numba'
```

Keep this PowerShell session for the following commands. Results and the Numba compilation cache go into the new directory above.

```powershell
& $Python -I -B verify_package.py --output (Join-Path $run 'verification')
& $Python -I -B budget_thresholds.py . --output (Join-Path $run 'thresholds.json')
& $Python -I -B verify_referee_cases.py . (Join-Path $run 'audit.json')
& $Python -I -B figures/build_color_result_figures.py . (Join-Path $run 'figures')
```

The verifier checks the file hashes and replays the 150,000 switching vectors in the main duty. The two archive checks recompute budget thresholds, verify the original backup test and check the FCH2Rail comparison. At the +2% braking-current demand, 9 of the 15 service-matched policies meet the budget. For the two FCH2Rail mappings, the corresponding counts are 0/15 and 0/10. The last command generates the two colour result figures as vector TikZ from the archived data.

To replay the 64 policies and run the charge and sensitivity checks, follow the [extended commands](P2_Reviewer_Reproducibility/README.md#extended-numerical-evidence-27-november-2026). The [additional replay commands](P2_Reviewer_Reproducibility/README.md#glm-referee-follow-up-26-november-2026) cover the common backup, prefix rejection and 36 s mapping.

## Supporting checks

- [Electrothermal package checks](P2_Reviewer_Reproducibility/DeCuong_4Bai/P2_kiemtoan/results.json), [source saturation](P2_Reviewer_Reproducibility/output/research/P2_SourceSaturation_20261115.json), [gain–budget sweep](P2_Reviewer_Reproducibility/output/research/P2_GainBudgetTradeoff_20261115.json), and [paired rainflow summaries](P2_Reviewer_Reproducibility/rf_mechanism_20261119/P2_RF_attribution_20261119.json). The source-redistribution diagnostics use a separate thermal model. The paired summary records the cycle endpoints and source-file hashes. Full-state trajectories for these diagnostics are outside the repository.
- [Binary64 input-conversion audit](P2_Reviewer_Reproducibility/output/research/P2_MissionRF_Fine1e-20_20261122/), including the input grid, switch decisions and exact rounding thresholds. Use each case's specified conversion route when reproducing it.
- [Recorded-sample charge enclosures](P2_Reviewer_Reproducibility/rf_enclosure_results.json), [thermal/modulation stress results](P2_Reviewer_Reproducibility/major_revision_results.json) and [desktop timings](P2_Reviewer_Reproducibility/pc_benchmark_results.json). These checks concern the declared model and recorded samples. Physical lifetime and deployment timing require separate validation.

## Data sources

The unmodified FCH2Rail trajectory is from Sebastian Herwartz-Polster and Florian Kühlkamp, German Aerospace Center (DLR). [Zenodo dataset](https://doi.org/10.5281/zenodo.6359030) · [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)

The bundled rainflow implementation retains its [original licence](P2_Reviewer_Reproducibility/DeCuong_4Bai/P2_sim/vendor/RAINFLOW_LICENSE.txt).
