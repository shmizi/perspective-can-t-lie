"""Train and evaluate the demo app's "% likely AI-generated" model.

    python scripts/train_app_model.py            # features (cached) + evaluation + final model
    python scripts/train_app_model.py --refresh  # recompute features

Features come from `projgeo.explain.explain`, the same function the app runs,
so training and serving cannot drift apart.  Only images that pass the
applicability rule are used, because the app only scores those.

Evaluation, written to results/app_model_report.md:
  * 5-fold stratified CV: AUC, Brier score, calibration table
  * AUC of real vs each generator, from the same CV predictions
  * leave-one-generator-out: train without generator G, test on G - what
    happens on an AI model the app has never seen
"""

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from projgeo.appmodel import MODEL_PATH, analysis_width, feature_matrix, make_model, rebase
from projgeo.datasets.folder import image_files
from projgeo.datasets.yorkurban import YorkUrban
from projgeo.explain import FEATURES, explain

SETS = {  # name: (folder or "yorkurban", is_ai, display name)
    "yorkurban": ("yorkurban", 0, "York Urban (real)"),
    "real-commons": ("data/real/commons", 0, "Commons (real)"),
    "sd15_rich": ("data/generated/sd15_rich", 1, "SD 1.5"),
    "sd15_pilot": ("data/generated/sd15_pilot", 1, "SD 1.5"),
    "sdxl_rich": ("data/generated/sdxl_rich", 1, "SDXL"),
    "sdxl_pilot": ("data/generated/sdxl_pilot", 1, "SDXL"),
    "gemini": ("data/generated/gemini", 1, "Gemini"),
    "gptimage": ("data/generated/gptimage", 1, "ChatGPT"),
    "flux_rich": ("data/generated/flux_rich", 1, "Flux"),
}
CACHE = Path("outputs/app_features.csv")
# demo examples are held out of training so the app shows honest, unseen scores
HELD_OUT = {(e["set"], e["file"]) for e in json.loads(Path("app/examples.json").read_text(encoding="utf-8"))}


def images(src):
    if src == "yorkurban":
        for im in YorkUrban("data/real/YorkUrbanDB"):
            yield im.name, im.image
        return
    for p in image_files(src):
        yield p.name, cv2.imread(str(p))


def compute_features():
    rows = []
    for name, (src, is_ai, disp) in SETS.items():
        items = list(images(src))
        for fname, img in tqdm(items, desc=name):
            ex = explain(img, match_width=analysis_width(img))
            row = {"set": name, "path": fname, "is_ai": is_ai, "model": disp,
                   "admitted": ex["admitted"], "vert_dist_h": ex["vert_dist_h"]}
            if ex["features"]:
                row.update(ex["features"])
            rows.append(row)
    df = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def cv_predict(df, seed=0):
    X, y = feature_matrix(df), df.is_ai.values
    p = np.zeros(len(df))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = make_model(seed).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def leave_one_generator_out(df, seed=0):
    out = {}
    real = df[df.is_ai == 0].reset_index(drop=True)
    for gen in sorted(df[df.is_ai == 1].model.unique()):
        held = df[(df.is_ai == 1) & (df.model == gen)]
        others = df[(df.is_ai == 1) & (df.model != gen)]
        p_real = np.zeros(len(real))
        p_gen = np.zeros(len(held))
        folds = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in folds.split(real, real.set):          # stratify real by source set
            train = pd.concat([real.iloc[tr], others])
            m = make_model(seed).fit(feature_matrix(train), train.is_ai.values)
            p_real[te] = m.predict_proba(feature_matrix(real.iloc[te]))[:, 1]
            p_gen += m.predict_proba(feature_matrix(held))[:, 1] / folds.get_n_splits()
        y = np.r_[np.zeros(len(real)), np.ones(len(held))]
        out[gen] = (roc_auc_score(y, np.r_[p_real, p_gen]), len(held))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    df_all = compute_features() if args.refresh or not CACHE.exists() else pd.read_csv(CACHE)
    held = df_all.apply(lambda r: (r.set, r.path) in HELD_OUT, axis=1)
    df = df_all[df_all.admitted & ~held].reset_index(drop=True)
    share = float(df.is_ai.mean())

    p_cv = cv_predict(df)
    y = df.is_ai.values
    p50 = rebase(p_cv, share)
    lines = ["# Demo app model: evaluation", "",
             f"Images that pass the applicability rule: {len(df)} of {len(df_all)} "
             f"({(df.is_ai == 0).sum()} real, {(df.is_ai == 1).sum()} generated, after holding out "
             f"{int((held & df_all.admitted).sum())} demo examples; "
             f"training share generated = {share:.2f}, re-based to 0.50 for display).",
             f"Features ({len(FEATURES)}, geometry only): {', '.join(FEATURES)}.", "",
             "## 5-fold cross-validation", "",
             f"* AUC {roc_auc_score(y, p_cv):.3f}",
             f"* Brier score {brier_score_loss(y, p_cv):.3f} (at the training prior)",
             f"* At the 50 % line after re-basing: {np.mean(p50[y == 1] >= 0.5):.0%} of generated "
             f"images called generated, {np.mean(p50[y == 0] < 0.5):.0%} of real photos called real", "",
             "| shown score (50/50 prior) | images | share actually generated, re-weighted to 50/50 |",
             "|---|---|---|"]
    w = np.where(y == 1, 0.5 / share, 0.5 / (1 - share))        # re-weight to the 50/50 prior
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        m = (p50 >= lo) & (p50 < hi)
        if m.sum():
            obs = np.sum(w[m] * y[m]) / np.sum(w[m])
            lines.append(f"| {lo:.0%} to {min(hi, 1):.0%} | {m.sum()} | {obs:.0%} |")
    lines += ["", "## Real photographs vs each generator (same CV predictions)", "",
              "| generator | AUC | images |", "|---|---|---|"]
    real_mask = y == 0
    for gen in sorted(df[df.is_ai == 1].model.unique()):
        g = (df.model == gen).values
        lines.append(f"| {gen} | {roc_auc_score(np.r_[np.zeros(real_mask.sum()), np.ones(g.sum())], np.r_[p_cv[real_mask], p_cv[g]]):.3f} | {g.sum()} |")
    lines += ["", "## Leave one generator out (never seen in training)", "",
              "| held-out generator | AUC vs real photos | images |", "|---|---|---|"]
    for gen, (auc, n) in leave_one_generator_out(df).items():
        lines.append(f"| {gen} | {auc:.3f} | {n} |")

    final = make_model().fit(feature_matrix(df), y)
    bundle = {"model": final, "features": FEATURES, "train_ai_share": share,
              "n_real": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
              "cv_auc": float(roc_auc_score(y, p_cv)),
              "reference": {grp: {f: df[df.is_ai == lab][f].tolist() for f in FEATURES}
                            for grp, lab in (("real", 0), ("ai", 1))},
              "report": "\n".join(lines)}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    Path("results/app_model_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("saved", MODEL_PATH, f"{MODEL_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
