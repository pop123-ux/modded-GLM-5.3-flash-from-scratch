# Included evidence

These are actual small CPU execution receipts from preparing this release. Read REPORT.md for run sizes and outcomes.

- `pytest-output.txt`, `pytest-results.xml`: final correctness suite.
- `research-smoke/`: 22-condition matrix, final six-condition rerun, combined-feature training/validation and isolated attention-storage measurement.
- `cpu-smoke/ablation-matrix/`: historical v0.2 17-condition matrix.
- `PREVIOUS_REPORT_v0.2.md`: historical verification report.
- `cpu-smoke/`: other functional runs, attention profiling, validation, RL, vision and structural studies.
- `SOURCE_MANIFEST.json`: SHA-256 of final source/config/doc/notebook files; excludes itself, caches and generated receipts.

No trained weights are included. Results have not been promoted into a speedrun leaderboard. The exact upstream baseline tree contains its original reference report; those historical claims are not new measurements of this project.

- `PREVIOUS_REPORT_v0.3.md`: verification history before the v0.4 inference/notebook/plotting expansion. The existing training receipts are historical functional runs; rendering them again does not constitute a new training experiment.
