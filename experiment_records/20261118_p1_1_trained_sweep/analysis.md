# Trained sweep (600 episodes fixed): analysis

samples with post-GRT labels: 16

| seed | halo | cab | gp_hpwl (M) | grt_tns | grt_wns | drt_tns | rWL (um) | DRC | note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 0 | 0 | 10.898 | -68.43 | -0.26 |  | 68042 | 591 | grt_only_rc2 |
| 7 | 0 | 0 | 10.962 | -82.35 | -0.32 | -121.32 | 70717 | 5 | ok |
| 11 | 0 | 0 | 10.927 | -79.76 | -0.32 | -111.41 | 73139 | 3 | ok |
| 17 | 0 | 0 | 21.447 | -99.1 | -0.32 | -135.18 | 73973 | 72 | ok |
| 3 | 0 | 2 | 10.898 | -79.68 | -0.26 | -118.11 | 67684 | 273 | ok |
| 7 | 0 | 2 | 10.845 | -81.88 | -0.27 | -111.34 | 69110 | 370 | ok |
| 11 | 0 | 2 | 11.055 | -79.34 | -0.27 | -112.67 | 64674 | 239 | ok |
| 17 | 0 | 2 | 11.155 | -114.12 | -0.39 |  | 65041 | 315 | grt_only_rc2 |
| 3 | 2 | 0 | 10.937 | -79.01 | -0.26 | -114.04 | 64534 | 401 | ok |
| 7 | 2 | 0 | 11.023 | -69.58 | -0.23 | -98.77 | 73376 | 8 | ok |
| 11 | 2 | 0 | 10.965 | -47.64 | -0.27 | -80.26 | 68165 | 6 | ok |
| 17 | 2 | 0 | 10.968 | -90.06 | -0.31 | -121.67 | 74319 | 23 | ok |
| 3 | 2 | 2 | 10.817 | -78.64 | -0.25 | -113.77 | 68340 | 118 | ok |
| 7 | 2 | 2 | 10.864 | -102.96 | -0.34 | -135.28 | 68023 | 263 | ok |
| 11 | 2 | 2 | 11.108 | -80.08 | -0.24 | -109.3 | 65856 | 98 | ok |
| 17 | 2 | 2 | 11.090 | -105.29 | -0.36 |  | 67898 | 200 | grt_only_rc2 |

## Correlations (raw values; negative = proxy informative)
- rho(gp_hpwl, grt_tns) = **-0.359**  (n=16)
- rho(gp_hpwl, grt_wns) = -0.258  (n=16)
- rho(gp_hpwl, drt_tns) = +0.104  (n=13)
- rho(gp_hpwl, rwl_um)  = -0.071  (n=16)
- rho(gp_hpwl, drc)     = -0.250  (n=16)

## Outlier sensitivity
- gp_hpwl span: 10.817M .. 21.447M (98.3%)
- excluding placements above 15M: n=15, rho = **-0.282**
- in-band gp_hpwl span: 3.12%  while grt_tns spans -114.12 .. -47.64 (140% of |best|)
- outliers (>15M): seed 17 halo0 cab0 gp=21.45M tns=-99.1

## Per-configuration (4 seeds each)
| halo | cab | n | gp mean (M) | grt_tns mean | rho (4 seeds) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 4 | 13.559 | -82.41 | -1.000 |
| 0 | 2 | 4 | 10.988 | -88.75 | -0.200 |
| 2 | 0 | 4 | 10.973 | -71.57 | +0.000 |
| 2 | 2 | 4 | 10.970 | -91.74 | -0.400 |

## Reading
- in-band rho = -0.282 < 0.4 -> P1-1 decision gate says stop learning reward weights from this proxy
- Any pooled correlation across episode lengths is confounded: episode count correlates
  with both gp_hpwl (-0.79) and post-GRT TNS (+0.68). Compare inside one length only.
- DRT columns are empty where the flow died in 6_report (PSM-0069); post-GRT labels there
  are still valid (the GRT stage completed).
