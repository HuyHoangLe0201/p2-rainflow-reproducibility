# Replay environment and evidence

This directory contains the simulator, schedule catalogues, archives of replayed
results, and scripts for the paper *Switching Schedule Selection for Rail Traction
Inverters under Per-Device Rainflow Budgets*. Start with the
[repository README](../README.md) for the schedule and figure entry points.

## Environment and clean replay

The tested platform is Windows x86-64 with CPython 3.10.19. The wheel hashes in
requirements-lock-win64.txt pin NumPy 2.2.6, Numba 0.63.1, llvmlite 0.46.0 and SciPy 1.15.3.
Numba uses the SciPy BLAS interface during forced replay.
The independent batch rainflow counter is shipped in DeCuong_4Bai/P2_sim/vendor/
with its license.

Use a clone at a short path, such as `C:\p2`, to avoid Windows path-length
errors during environment setup. Open PowerShell in this directory
containing verify_package.py. Select a conda environment with CPython 3.10.19 for Windows x86-64.
The clean run uses the Anaconda build. Keep this PowerShell session for the
commands below so that the absolute path in `$Python` remains available.

    $ReplayBase = (Get-Command python -CommandType Application | Select-Object -First 1).Source
    if ((& $ReplayBase --version) -ne "Python 3.10.19") {
        throw "Select CPython 3.10.19 before creating .venv."
    }
    & $ReplayBase -I -m venv .venv
    $Python = (Resolve-Path -LiteralPath ".\.venv\Scripts\python.exe").Path
    & $Python --version
    & $Python -I -m pip --isolated install --only-binary=:all: --require-hashes -r requirements-lock-win64.txt
    & $Python -I -m pip check
    $env:NUMBA_CACHE_DIR = Join-Path $env:TEMP ("p2-numba-" + [guid]::NewGuid())
    & $Python -I -B verify_package.py --output clean_verification

Use a fresh output directory on each run. A successful check writes
clean_verification/verification.json and an executed trace under
clean_verification/mission/. The first run compiles Numba functions locally.
Leave Python assertions enabled.

## Evidence and scope

- **Main mission.** The verifier applies SHA-256 checks to the shipped files,
  reconstructs the +2% braking profile and replays the archived thermal, soft RF and
  hard RF switching schedules in the deterministic 3L-NPC model. It then executes the
  stored manager with a certified replacement at 10 s and a missing proposal at
  20 s, replays the executed schedule, and compares all 150,000 vectors, the event
  log, service metrics and 30-device ledger with the archived result. It also
  runs the batch RF counter on the executed trajectory.
- **Five mission catalogs.** The verifier recomputes service and budget counts
  from the 27 archived candidates per mission at both tolerances. At 0.5%,
  the counts within budget over matched are 9/15, 3/3, 0/6, 6/12 and 0/3.
- **Appendix D and the numerical sensitivity audit.** The verifier checks 874
  integer ranges, both scalar
  conversions at their endpoints, 515 input labels and 210 decision
  thresholds. The candidate metrics are archived.
  For the Appendix D cases, use the archived IEEE 754 binary64 input
  representations and keep each case's specified conversion route. Intermediate
  rounding and a single final rounding can give different schedule decisions
  near the audited transitions.
- **Appendix C and the electrothermal checks.** The rf_mechanism_20261119
  directory contains the original
  chip-9 attribution JSON and figure. The JSON preserves the full/residual cycle
  endpoints, paired source and zero-arm scores, and hashes of its source files.
  The verifier recomputes paired contrasts and ratios for nine cases at two
  horizons from the recorded summaries. The raw full-state ODE traces are
  outside this compact package. The remaining Appendix C references are supplied
  as supporting archived evidence.

For compatibility, verification.json retains its original field names.
The key appendix_A reports the Appendix D input-conversion audit. The key
appendix_B reports the Appendix C paired rainflow attribution check.

The package verifies the declared deterministic model and library. HIL,
physical lifetime validation and generalized mission optimization are outside its scope.


## FCH2Rail comparator

The included CSV is the unmodified DLR Institute of Vehicle Concepts
A Coruna–Ferrol–A Coruna simulated trajectory from the FCH2Rail dataset.
Dataset DOI https://doi.org/10.5281/zenodo.6359030
Source data licence https://creativecommons.org/licenses/by/4.0/
The simulation reads the original samples for source seconds 0 through 194.
The manuscript specifies the time compression and surrogate mapping.
Use the same locked environment from the previous section:

    & $Python -B DeCuong_4Bai/P2_sim/mission_rf_public_rail.py --trajectory "P2_PublicRail_FCH2Rail_20261122/A Coruna-Ferrol-A Coruna_simulated_trajectory.csv" --output public_case_replay --workers 4

The baseline verifier checks hashes for these added assets. The command above
regenerates the 27-schedule comparator. Expected results are 27 completions,
15 service matches and zero within-budget schedules. The minimum matched RF
is 7.439714 against the original budget of 6.738416. Input and source hashes,
device charges and batch checks are recorded in P2_PublicRail_FCH2Rail_20261122/results.json.


## Checks added for the revised manuscript

Run these standard-library scripts from this directory with the pinned Python interpreter.

    & $Python -B budget_thresholds.py . --output budget_thresholds.json
    & $Python -B verify_referee_cases.py . audit.json

The first command recalculates the decision thresholds for all seven
catalogues in the manuscript table, including the 36 s FCH2Rail mapping.
The second checks the original 10 s acceptance and 20 s missing-proposal log,
then matches the hard-budget sequence under
alternative soft-penalty weights, and checks the FCH2Rail catalogue. The supplied
referee_case_evidence.json records the expected case results. The archive hash in that record identifies the earlier verified ZIP. The directory-mode audit reports the hash of the current checksum manifest.

## Figure 4

The collector uses the replay interpreter in `$Python`. It verifies the
archived selected schedules and writes a fresh observation file. The plot uses
a separate environment with CPython 3.12.14, NumPy 2.2.4 and Matplotlib 3.10.1.

From the extracted root, replace `$PlotBase` with the absolute path to a
CPython 3.12.14 interpreter. Use fresh output names for repeated runs.

    $PlotBase = "C:\path\to\python.exe"
    if ((& $PlotBase --version) -ne "Python 3.12.14") {
        throw "Select CPython 3.12.14 before creating .venv-plot."
    }
    & $PlotBase -I -m venv .venv-plot
    $PlotPython = (Resolve-Path -LiteralPath ".\.venv-plot\Scripts\python.exe").Path
    & $PlotPython -I -m pip --isolated install --only-binary=:all: numpy==2.2.4 matplotlib==3.10.1
    & $PlotPython -I -m pip check

    Push-Location figures
    try {
        & $Python -B collect_mission_curves.py .. replay_curves.json
        if ($LASTEXITCODE -ne 0) { throw "Figure 4 replay failed." }
        & $PlotPython -B plot_mission_curves.py replay_curves.json replay_curves.pdf
    } finally {
        Pop-Location
    }

mission_curves.json and mission_curves.pdf are the reference observations
and vector figure. The replay workflow is also checked from a clean checkout.


## GLM referee follow-up, 26 November 2026

The original benchmark, figure files and original replay evidence are retained for
the comparisons in the article. The local supplement adds three experiment
records under followup_20261126. Run from the extracted reproducibility root
with the pinned environment described above and fresh output directories.

    & $Python -B DeCuong_4Bai/P2_sim/mission_rf_two_outcomes.py --output followup_two_outcomes
    & $Python -B DeCuong_4Bai/P2_sim/mission_rf_glm_followup.py handoff --brake-start 14 --workers 3 --output followup_handoff
    & $Python -B DeCuong_4Bai/P2_sim/mission_rf_glm_followup.py fch36 --workers 4 --output followup_fch36

- The two-outcome test uses one exact action sequence for both deterministic
  modulation profiles, which differ from 10 to 11 s. Its componentwise
  remaining-charge bound enumerates exactly those two outcomes.
- The handoff test uses traction/braking/dwell at 0–14/14–20/20–30 s.
  Both schedules pass service and budget from the initial state. The initial
  certificate subsequently fails the executed-prefix check at 20 s. The
  missing-proposal and no-replacement cases are separate controls.
- The FCH2Rail sensitivity uses the same input samples, amplitude mapping, model,
  sample step, budget, normalisers and policy changes at 10 and 20 s. The final
  block grows to 16 s. Compare the 36 s result with the archived 30 s mapping;
  the two simulations concern only the tested libraries and surrogate demands.

## Major revision checks, 27 November 2026

From the extracted package root with the replay environment prepared, run:

    & $Python .\major_revision_audit.py --output .\major_revision_replay.json --workers 4
    Compare-Object (Get-Content .\major_revision_results.json) (Get-Content .\major_revision_replay.json) -SyncWindow 0

An empty comparison means the records match. The script checks the sampled
single-branch counterexample in exact rational arithmetic and computes the
tracking--charge Pareto frontier from the archived matched catalogue. It then
regenerates all 27 schedules for the retimed duty and tests rejection of an
unverified proposal outside the catalogue that shares the executed prefix.
This is an admission-interface check. The replay does not validate the altered
suffix. The numerical distance from the chosen budget describes sensitivity
of the stored charges. It is not a verified interval error estimate.

## Extended numerical evidence, 27 November 2026

From the extracted root, use the checked replay environment defined above:

    & $Python .\expanded_catalogue.py . .\expanded_replay.json --workers 4
    & $Python .\rf_enclosure.py . .\rf_enclosure_replay.json
    & $Python .\verify_exact_rf.py
    & $Python .\pc_benchmark.py . .\pc_benchmark_replay.json

The first command replays the nested 64-policy catalogue. It checks equality
of the 27 original action hashes and charges. The output records comparisons
with the original and recomputed thermal service anchors, tuned soft-weight
intervals and the full candidate metrics. The second command evaluates the
three nominal selections and six stress cases of the nominal hard schedule.
It encloses RF charge only for the exact finite temperatures produced by the
replay, which are identified by hashes. It also reports total-variation
bounds on those histories. It does not enclose errors in the
electrothermal trajectory or prove coverage of untested thermal settings.
It processes one temperature history at a time in memory. These large
working arrays are not part of the distributed archive.
The third command compares the four-point counter against an independent
three-point implementation on 15,625 short histories and three near-ties.
Catalogue and enclosure result files are deterministic in the pinned
environment. The desktop microbenchmark records local wall-clock values
for fixed-size operations and its timings will vary between runs.

## Colour result figures

The manuscript embeds Figure 3 and Figure 4 as vector TikZ. Run the commands
below from this directory with the replay Python selected
above. They regenerate both vector figures directly from the checked device
charges and mission curves already in this archive. Once the manuscript source
has been extracted, the optional check compares the regenerated figure code
against the blocks in main.tex.

    & $Python .\figures\build_color_result_figures.py . .\colour_figures
    & $Python .\figures\build_color_result_figures.py . .\colour_figures --verify-manuscript 'C:\path\to\main.tex'

The PDF and Matplotlib plotting command in the earlier Figure 4 section
are the monochrome archive preview. The same replay observations are used
for both versions. The current manuscript's colour layout is generated by
the TikZ command above.
