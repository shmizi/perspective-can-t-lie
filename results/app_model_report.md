# Demo app model: evaluation

Images that pass the applicability rule: 839 of 1842 (198 real, 641 generated, after holding out 6 demo examples; training share generated = 0.76, re-based to 0.50 for display).
Features (10, geometry only): l2_rms_deg, l2_mean_deg, l2_capped_mean_deg, l2_unexplained_frac, vp_std_max_deg, ortho_err_max_deg, ortho_err_rms_deg, orthocenter_offset, atl_logf_spread, atl_frac_impossible.

## 5-fold cross-validation

* AUC 0.817
* Brier score 0.138 (at the training prior)
* At the 50 % line after re-basing: 77% of generated images called generated, 74% of real photos called real

| shown score (50/50 prior) | images | share actually generated, re-weighted to 50/50 |
|---|---|---|
| 0% to 20% | 60 | 9% |
| 20% to 40% | 139 | 27% |
| 40% to 60% | 199 | 41% |
| 60% to 80% | 409 | 82% |
| 80% to 100% | 32 | 82% |

## Real photographs vs each generator (same CV predictions)

| generator | AUC | images |
|---|---|---|
| ChatGPT | 0.589 | 36 |
| Flux | 0.771 | 207 |
| Gemini | 0.796 | 55 |
| SD 1.5 | 0.906 | 152 |
| SDXL | 0.844 | 191 |

## Leave one generator out (never seen in training)

| held-out generator | AUC vs real photos | images |
|---|---|---|
| ChatGPT | 0.585 | 36 |
| Flux | 0.640 | 207 |
| Gemini | 0.802 | 55 |
| SD 1.5 | 0.876 | 152 |
| SDXL | 0.813 | 191 |