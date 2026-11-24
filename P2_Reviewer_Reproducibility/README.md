# Railway Engineering Science Online Resource 2

Article: Switching Schedule Selection for Rail Traction Inverters under Per-Device Rainflow Budgets

Authors: [Names to be confirmed]

Corresponding author: [Name, affiliation and email to be confirmed]

This archive provides simulation inputs, archived comparator libraries and
scripts to verify the deterministic replay, budget thresholds and Figure 4.
Online Resource 1 (ESM_1.pdf) is the accompanying Supplementary Note.

## Environment and clean replay

The tested platform is Windows x86-64 with CPython 3.10.19. The wheel hashes in
requirements-lock-win64.txt pin NumPy 2.2.6, Numba 0.63.1, llvmlite 0.46.0 and SciPy 1.15.3.
Numba uses the SciPy BLAS interface during forced replay.
The independent batch rainflow counter is shipped in DeCuong_4Bai/P2_sim/vendor/
with its license.

Open PowerShell in the extracted directory containing verify_package.py. If
CPython 3.10.19 is not installed, obtain that interpreter with your conda
distribution and verify the version before continuing. The clean run uses the
Anaconda Windows x86-64 build.

    python --version
    python -I -m venv .venv
    $Python = ".\.venv\Scripts\python.exe"
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
- **Appendix D and Supplementary Note section 2.** The verifier checks 874
  integer ranges, both scalar
  conversions at their endpoints, 515 input labels and 210 decision
  thresholds. The candidate metrics are archived.
- **Appendix C and Supplementary Note section 1.** The rf_mechanism_20261119
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

Run these standard-library scripts with the pinned Python interpreter. For the first
command, run from the extracted supplementary root. For the second, supply the
path to the supplementary ZIP, for example from the extracted root when the ZIP is
one level above it.

    python budget_thresholds.py . --output budget_thresholds.json
    python verify_referee_cases.py ../ESM_2.zip audit.json

The first command recalculates the decision thresholds for the six catalogues.
The second checks the 10 s and 20 s event log, matches the hard-budget sequence under
alternative soft-penalty weights, and checks the FCH2Rail catalogue. The supplied
referee_case_evidence.json records the expected case results. Its archive_sha256
identifies the earlier verified ZIP from which the data were taken. The same data
and simulation code are carried forward in this archive.

## Figure 4

Work in the figures directory inside this extracted archive. Use the pinned
CPython 3.10.19 replay environment above for the first command. The collector
verifies the archived selected schedules and writes a fresh observation file.
Use a new output name if a previous run created replay_curves.json.

    python -B collect_mission_curves.py .. replay_curves.json

From the same directory, switch to the plotting environment. The verified
plotting versions are Python 3.12.14, NumPy 2.2.4 and Matplotlib 3.10.1.

    python plot_mission_curves.py replay_curves.json replay_curves.pdf

mission_curves.json and mission_curves.pdf are the reference observations
and vector figure. The delivered archive was checked from a new extraction.
