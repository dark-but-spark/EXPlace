# P1-1 pilot: 5 paired placements on bp_fe

Status: **pipeline validation only.** n = 5, one circuit, and the sampling design
severely compresses the independent variable — see "Attenuation" below. Do not
read these numbers as a P1-1 verdict.

## Setup

| field | value |
| --- | --- |
| design | `bp_fe` (OpenROAD `nangate45`, 11 macros) |
| placements | 5, seeds `3 7 11 17 23`, 600 episodes each, `config/or_batch.yaml` |
| sampler | one per seed: the `best_gp_hpwl` checkpoint (the DEF that EXPlace saved) |
| Tier-1 | DREAMPlace proxy `gp_hpwl` — OpenTimer is unavailable here (assertion, see TODO) |
| Tier-2 | OpenROAD flow with `ROUTING_LAYER_ADJUSTMENT=0.13` and 30 congestion iterations |
| flow cost | ~8–9 min per sample (GRT ≈ 3.4 min, DRT ≈ 3.4 min), 0 congestion warnings |
| batch wall clock | 48 min (5 trainings in parallel, then 5 flows sequentially) |

## Labels

| seed | Tier-1 `gp_hpwl` | post-GRT TNS / WNS (ns) | post-DRT TNS / WNS (ns) | DRT violations |
| ---: | ---: | ---: | ---: | ---: |
| 3 | **10 885 164** (best) | −75.59 / −0.27 | −103.89 / −0.29 | 748 |
| 7 | 10 972 720 | **−57.85 / −0.22** | −87.78 / −0.27 | 1320 |
| 11 | 10 984 766 | −60.59 / −0.21 | −86.83 / −0.30 | **0** |
| 17 | 11 009 845 | −64.17 / −0.27 | −83.29 / −0.28 | 28 |
| 23 | **11 059 812** (worst) | −97.49 / −0.36 | −143.57 / −0.41 | 15 |

## Correlation (quality ranks; positive = proxy is informative)

`ranks()` treats lower `gp_hpwl` as better and more negative TNS/WNS as worse.

| comparison | Spearman ρ |
| --- | ---: |
| gp_hpwl vs post-GRT TNS | **+0.40** |
| gp_hpwl vs post-GRT WNS | +0.30 |
| gp_hpwl vs post-DRT TNS | **0.00** |
| gp_hpwl vs post-DRT WNS | +0.50 |

Sign convention check: raw-value correlation reported by `batch_p1_1.sh` is
−0.40 / −0.46 / 0.00 / −0.50, i.e. the same magnitudes with the opposite sign
(raw `gp_hpwl` and raw `TNS` move in opposite directions as quality improves).

For n = 5 the two-sided 5 % critical value is |ρ| ≈ 0.90, so none of these is
significant (|ρ| = 0.5 gives p ≈ 0.39).

## Attenuation: the sampling design, not the proxy, is the main problem

| quantity | span across the 5 samples |
| --- | ---: |
| `gp_hpwl` | **1.6 %** (10.885M – 11.060M) |
| post-GRT TNS | 40.7 % |
| post-GRT WNS | 41.7 % |
| post-DRT TNS | 42.0 % |

Taking the `best_gp_hpwl` checkpoint of each seed samples only the top of the
quality distribution. With almost no spread in the independent variable, any
rank correlation is pulled toward zero by construction (classic attenuation).
This is **not** evidence that HPWL is useless; it is evidence that the sampler
must include bad placements. PPAPlace avoided this by sweeping 20 RTLMP weight
configurations per circuit, including deliberate extremes.

## Diagnostics that survive the small n

1. **One coherent sample, one counter-example.** seed 23 is worst on both HPWL
   and post-GRT timing; seed 3 has the best HPWL but 4th-best post-GRT TNS.
   That pattern is what a weak proxy looks like.
2. **DRC/routability is orthogonal to GP-HPWL.** seed 11 has the third-best
   HPWL and **0** DRT violations, while seed 3 has the best HPWL and 748
   violations, and seed 7 has 1320. The reward contains no congestion or DRC
   term, and this is the most actionable finding from the pilot.
3. **Spread is much larger downstream than upstream** (1.6 % vs ~41 %), so the
   labels are discriminative even though the proxy samples are not.

## Caveats

- Tier-2 timing depends on OpenROAD's own cell placement; the two tiers share
  only the macro placement. That is the intended macro-level comparison, but it
  adds noise.
- `Tier-1` here is a proxy, not OpenTimer timing: enabling the timer on
  `or_cases` aborts with
  `timing_cpp.cpp:162 Assertion 'net_iter != timer.nets().end()' failed`.
- DRT is not DRC-clean (0–1320 violations); safe for timing labels, not for
  signoff claims.

## Next steps (in order)

1. **Widen the quality span** before computing ρ: dump a DEF at every periodic
   evaluation (early checkpoints are genuinely worse), or add deliberately poor
   configurations. Target ≥ 20–30 % span in `gp_hpwl`.
2. **Add rWL / congestion / DRC as label dimensions** — cheap now that the flow
   runs, and finding (2) above shows they carry independent information.
3. **Grow to 10–12 samples per circuit across ≥ 2 circuits**, then report
   per-circuit Spearman with Fisher-z aggregation.
4. Only after that, revisit whether a surrogate is worth training.

Reproduce:

```bash
DESIGN=bp_fe NICK=bp_fe CFGDIR=bp_fe_top SEEDS="3 7 11 17 23" GPUS="3 4 5 6 7" \
  bash script/tier2/batch_p1_1.sh
cat experiment_records/20261117_p1_1_labels/{labels.csv,summary.txt}
```
