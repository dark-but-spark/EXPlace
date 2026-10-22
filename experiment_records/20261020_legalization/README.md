# DREAMPlace legality repair validation

The final superblue18 sanity run
`long_runs/legalization_hannan_fix_20261020_093027` completed both reward arms
and accepted all 10/10 physical evaluations. It verifies that the patched
local grid repair and native Hannan fallback can recover the previously failing
short-test placement.

The larger run `long_runs/legalization_matrix_20261020_100329` did **not**
finish. Its last launcher update was 2026-10-20 14:59 UTC:

- 28/48 planned tasks have `SUCCESS`;
- two superblue10 seed-7 tasks stopped after episodes 20 and 30 and retain
  stale `RUNNING` markers;
- 145 physical rows were written, of which 132 are valid (91.0%);
- superblue1: 60/60 valid;
- superblue4: 47/60 valid, with 13 `dreamplace_legality_check_failed` rows;
- superblue10: 25/25 valid among completed/partial work;
- superblue18 was not started.

Thus the repair is effective on the sanity case and the observed superblue1/10
placements, but is not a complete cross-design fix because superblue4 still
fails. The 50-episode matrix is a correctness smoke test, not a convergence or
speed benchmark. Adaptive weights only begin changing near the end, and the
incomplete/invalid pairs must not be used as a full PPA comparison.

The two subdirectories contain compact task summaries, plans, metadata, and
validity-aware derived CSVs. Models, DEFs, images, and full stdout remain only
in the ignored local run roots.
