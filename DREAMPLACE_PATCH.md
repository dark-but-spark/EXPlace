# DREAMPlace source patch

The legality repair is kept in the EXPlace repository as
`patches/dreamplace/macro_legalize_local_grid_repair.patch`. The downloaded
DREAMPlace tree is intentionally ignored by git, so the patch is applied to
the local source tree immediately before compilation by `script/make.sh`.

The patch is idempotent. `script/apply_dreamplace_patch.sh` detects the marker
in `macro_legalize.cpp`, performs a dry-run on a clean source tree, and then
applies the patch. Re-run it after replacing or re-downloading the DREAMPlace
source. A legacy patch must be removed by restoring a clean DREAMPlace tree
before applying this version.

After final grid alignment, the repair scores each conflicted macro by its
overlap count and area against other macros and fixed nodes. It tests blocker
edges/corners and a bounded site/row neighborhood, accepting only moves that
reduce that macro's score. If local repair stalls, it retries DREAMPlace's
native Hannan grid legalization and realigns the result. The full legality
check remains authoritative: unresolved layouts are rejected by EXPlace and
are never converted into valid evaluations.
