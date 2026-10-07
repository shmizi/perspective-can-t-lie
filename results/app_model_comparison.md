# Demo app model: alternatives and shortcuts

839 images (198 real, 641 generated), 5-fold CV AUC, mean over fold seeds 0, 1, 2.

## Classifier, same 10 geometry features

| classifier | AUC |
|---|---|
| Random forest + calibration (the app) | 0.812 |
| Random forest, no calibration | 0.814 |
| Extra trees | 0.791 |
| Gradient boosting | 0.792 |
| Logistic regression | 0.733 |

## Image shape (width / height): a shortcut, not used

| input | AUC |
|---|---|
| shape alone | 0.632 |
| 10 features + shape | 0.858 |

## Each feature on its own (AUC, direction-free)

| feature | AUC |
|---|---|
| l2_rms_deg | 0.668 |
| l2_mean_deg | 0.656 |
| l2_capped_mean_deg | 0.513 |
| l2_unexplained_frac | 0.524 |
| vp_std_max_deg | 0.538 |
| ortho_err_max_deg | 0.571 |
| ortho_err_rms_deg | 0.571 |
| orthocenter_offset | 0.667 |
| atl_logf_spread | 0.627 |
| atl_frac_impossible | 0.599 |
