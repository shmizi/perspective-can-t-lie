"""One Camera or Not? - interactive demo of the geometric-consistency check.

    streamlit run app/streamlit_app.py

Every number on screen comes from `projgeo.explain.explain`, the same code the
report's results come from, and the score from the model trained by
`scripts/train_app_model.py` on that code's output.
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from projgeo import appmodel                                       # noqa: E402
from projgeo.explain import (LEVEL_CAMERA_H, directions, disagreement_pct,  # noqa: E402
                             draw, explain)
from projgeo.selection import looks_uncropped                       # noqa: E402

st.set_page_config(page_title="One Camera or Not?", page_icon="📐", layout="wide")

import json                                                          # noqa: E402

EXAMPLES = {e["label"]: e for e in json.loads((ROOT / "app" / "examples.json").read_text(encoding="utf-8"))}

REASON_TEXT = [
    ("fewer than three", "It needs three directions of straight lines (for example vertical edges plus "
                         "two sets of horizontal edges), and this image does not show enough of them."),
    ("weakest family", "One of the line directions is too faint: {} of the total line length."),
    ("localised to", "A vanishing point cannot be pinned down precisely enough (within {} degrees)."),
    ("separated by only", "Two line directions are too similar to tell apart ({} degrees apart)."),
    ("separation only", "Two line directions are too close compared with how uncertain they are."),
    ("spans only", "One set of lines is bunched into a small part of the picture."),
]


# ---------------------------------------------------------------- cached work
@st.cache_resource
def model_bundle():
    try:
        return appmodel.load()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False, max_entries=200)
def analyse(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    ex = explain(img, match_width=appmodel.analysis_width(img))
    ex["orig_size"] = (w, h)
    ex["uncropped"] = looks_uncropped(w, h)
    return ex


def read_example(path: str) -> bytes | None:
    p = ROOT / path
    return p.read_bytes() if p.exists() else None


# ---------------------------------------------------------------- presentation helpers
def plain_reason(r: str) -> str:
    for key, text in REASON_TEXT:
        if key in r:
            nums = [t for t in r.replace("%", "% ").split() if any(c.isdigit() for c in t)]
            return text.format(nums[0] if nums else "") if "{}" in text else text
    return r


def verdict(p: float) -> tuple[str, str]:
    if p >= 0.7:
        return "Geometry looks generated", "error"
    if p <= 0.3:
        return "Geometry fits one real camera", "success"
    return "Unclear: geometry is in between", "warning"


def figure_for(ex):
    fig, ax = plt.subplots(figsize=(6.4, 6.4 * ex["height"] / ex["width"]))
    if ex["atlanta"]:
        draw(ax, ex, legend=True)
    else:
        ax.imshow(cv2.cvtColor(ex["img"], cv2.COLOR_BGR2RGB))
        for x1, y1, x2, y2 in ex["segs"].xy:
            ax.plot([x1, x2], [y1, y2], color="#ffd400", lw=0.8, alpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, ex["width"])
        ax.set_ylim(ex["height"], 0)
    fig.tight_layout(pad=0.2)
    return fig


def score(ex, bundle):
    if not ex or not ex["admitted"] or bundle is None:
        return None
    return appmodel.ai_probability(bundle, ex["features"])


def summary_row(name, ex, p):
    row = {"image": name, "result": "", "likely AI (%)": None, "focal disagreement": ""}
    if ex is None:
        row["result"] = "could not read file"
    elif not ex["admitted"]:
        row["result"] = "cannot measure"
    else:
        lf = ex["atlanta"]["logf_spread"]
        row["result"] = verdict(p)[0] if p is not None else "measured"
        row["likely AI (%)"] = round(100 * p) if p is not None else None
        row["focal disagreement"] = f"{disagreement_pct(lf):.0f}%" if np.isfinite(lf) else "not measurable"
    return row


def why_this_score(ex, bundle):
    with st.expander("Why this score: each measurement against real photos"):
        rows = []
        for key, label in [("atl_logf_spread", "Focal disagreement (log spread)"),
                           ("orthocenter_offset", "Lens-centre offset (orthocentre)"),
                           ("ortho_err_max_deg", "Worst right-angle error (deg)"),
                           ("l2_capped_mean_deg", "Line concurrency error (deg)")]:
            v = ex["features"][key]
            real = np.asarray(bundle["reference"]["real"][key], float)
            ai = np.asarray(bundle["reference"]["ai"][key], float)
            pct = appmodel.percentile_vs(bundle, key, v)
            rows.append({"measurement": label,
                         "this image": f"{v:.3f}" if np.isfinite(v) else "not measurable",
                         "typical real photo": f"{np.nanmedian(real):.3f}",
                         "typical AI image": f"{np.nanmedian(ai):.3f}",
                         "higher than this share of real photos": f"{pct:.0f}%" if np.isfinite(pct) else ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Higher means less consistent. The score combines ten measurements like these; "
                   "no single one decides it.")


def show_image(name, ex, p, bundle, credit=None):
    left, right = st.columns([1.15, 1])
    with left:
        st.pyplot(figure_for(ex), clear_figure=True, width="stretch")
        w, h = ex["orig_size"]
        cap = f"{w} x {h} px, analysed at {ex['width']} px wide, {ex['n_segments']} line segments found."
        st.caption(cap + (f" {credit}" if credit else ""))
    with right:
        if not ex["admitted"]:
            st.warning("**Cannot measure this image.**")
            for r in ex["reasons"]:
                st.write("- " + plain_reason(r))
            st.caption("This is not a verdict either way. The check needs buildings, rooms or streets with "
                       "lots of straight edges in three directions. Landscapes, portraits and close-ups "
                       "usually cannot be measured.")
            return
        if p is not None:
            label, kind = verdict(p)
            getattr(st, kind)(f"**{label}**")
            st.metric("Likely AI-generated", f"{p:.0%}")
            st.progress(min(max(p, 0.0), 1.0))
        else:
            st.info("Model file not found; run `python scripts/train_app_model.py`. Measurements below.")

        st.markdown("**The camera check: one focal length?**")
        dirs = directions(ex)
        lf = ex["atlanta"]["logf_spread"]
        if len(dirs) < 2:
            st.write("Only one horizontal direction was found, so the focal check cannot run here. "
                     "The score uses the other measurements.")
        else:
            st.dataframe(pd.DataFrame([{"direction": f"{chr(65 + k)}",
                                        "focal length it implies": f"{d['f_px']:.0f} px" if d["f_px"]
                                        else "no camera possible"} for k, d in enumerate(dirs)]),
                         hide_index=True, width="stretch")
            if np.isfinite(lf):
                pct = appmodel.percentile_vs(bundle, "atl_logf_spread", lf) if bundle else float("nan")
                line = f"These directions disagree by **{disagreement_pct(lf):.0f}%**."
                if np.isfinite(pct):
                    line += f" That is more than {pct:.0f}% of the real photos we tested (typical real: about 15%)."
                st.write(line)
            if any(d["f_px"] is None for d in dirs):
                st.write("At least one pair of directions cannot come from any camera. "
                         "(This also happens in 28 to 44% of real photos, so on its own it proves little.)")

        warn = []
        if np.isfinite(ex["vert_dist_h"]) and ex["vert_dist_h"] > LEVEL_CAMERA_H:
            warn.append("The camera is almost perfectly level, so the focal lengths above are poorly "
                        "determined. Read the focal check with caution.")
        if not ex["uncropped"]:
            warn.append("The image shape is not a standard camera shape, so it may be cropped. Cropping "
                        "moves the picture centre away from the lens centre, which this check assumes.")
        for wtext in warn:
            st.warning(wtext)

    if bundle and ex["admitted"]:
        why_this_score(ex, bundle)


# ---------------------------------------------------------------- page
bundle = model_bundle()
st.title("📐 One Camera or Not?")
st.write("A real photo is taken by **one camera**, so every part of it must agree on one focal length and "
         "one lens centre. AI image generators have no camera inside them. This app measures whether an "
         "image's straight lines are consistent with a single camera, and says how likely the geometry is "
         "to be AI-generated.")

tab_check, tab_how = st.tabs(["Check images", "How it works and how far to trust it"])

with tab_check:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        uploads = st.file_uploader("Upload one or more images (PNG, JPG or WebP)",
                                   type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    with c2:
        available = [k for k, e in EXAMPLES.items() if (ROOT / e["path"]).exists()]
        # ?examples=all preloads every example (handy as a bookmark for a live demo)
        preload = available if st.query_params.get("examples") == "all" else []
        picks = st.multiselect("...or try built-in examples", available, default=preload,
                               help="These images were held out of training, so their scores are honest.")

    items = []
    for f in uploads or []:
        items.append((f.name, f.getvalue(), None))
    for k in picks:
        e = EXAMPLES[k]
        data = read_example(e["path"])
        if data:
            note = f"Actually: {e['truth']}." + (f" {e['credit']}" if e["credit"] else "")
            items.append((k, data, note))

    if not items:
        st.info("Upload an image or pick an example to start. Indoor scenes and buildings work best.")
    else:
        results = []
        prog = st.progress(0.0, text="Measuring...")
        for i, (name, data, credit) in enumerate(items):
            ex = analyse(data)
            results.append((name, ex, score(ex, bundle), credit))
            prog.progress((i + 1) / len(items), text=f"Measured {i + 1} of {len(items)}")
        prog.empty()

        if len(results) > 1:
            st.subheader("Summary")
            df = pd.DataFrame([summary_row(n, e, p) for n, e, p, _ in results])
            df = df.sort_values("likely AI (%)", ascending=False, na_position="last")
            st.dataframe(df, hide_index=True, width="stretch",
                         column_config={"likely AI (%)": st.column_config.ProgressColumn(
                             "likely AI", min_value=0, max_value=100, format="%d%%")})

        for k, (name, ex, p, credit) in enumerate(results):
            with st.expander(name, expanded=(k == 0)):
                if ex is None:
                    st.error("Could not read this file as an image.")
                else:
                    show_image(name, ex, p, bundle, credit)

with tab_how:
    st.markdown("""
**What it measures.** Parallel lines in the real world (corridor edges, window rows) meet at a
*vanishing point* in a photo. The vertical direction is at a right angle to every horizontal one, so
pairing the vertical vanishing point with each horizontal one gives the camera's focal length
(f² = -(v₁ - p)·(v₂ - p)). In a real photo every pair gives the same answer. In AI images they often
disagree.

**Steps.** Resize to 640 px wide, detect line segments (LSD), find vanishing points with RANSAC (never
assuming right angles), check the image can be measured, then compute ten geometric measurements. A
calibrated random forest turns those into the score.

**What the score means.** "70% likely AI" means: if the image were equally likely to be real or AI
before we looked, the geometry makes AI about 70% likely. It is calibrated on the images below, so it
means "how much this looks like the AI images we tested, compared with the real photos we tested."
""")
    if bundle:
        st.markdown(f"**Trained on** {bundle['n_real']} real photos (York Urban, Wikimedia Commons) and "
                    f"{bundle['n_ai']} AI images (Stable Diffusion 1.5, SDXL, Gemini, ChatGPT) that "
                    f"pass the measurability check. Cross-validated AUC {bundle['cv_auc']:.2f} "
                    "(0.5 = guessing, 1.0 = perfect).")
        with st.expander("Full evaluation"):
            # demote the report's headings so they sit inside the expander
            st.markdown("\n".join("###" + ln if ln.startswith("#") else ln
                                  for ln in bundle["report"].splitlines()))
    st.markdown("""
**How far to trust it**
- It is a geometry check, not proof. It will sometimes be confidently wrong.
- It only works on images with lots of straight lines in three directions.
- An AI model it has never seen, or an unusual real camera, may fool it (see "Leave one generator out"
  in the evaluation).
- Cropped or edited photos can look inconsistent, because cropping moves the picture centre.
- Sloped structures (stairs, ramps, pitched roofs) confuse the focal check.
- For a camera held perfectly level, the focal lengths are poorly determined.
""")
