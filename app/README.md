# One Camera or Not? (demo app)

Checks whether an image's straight lines are consistent with a single real camera, and gives a
calibrated "% likely AI-generated" score with the reasons behind it.

## Run it

From the project root:

```
python -m streamlit run app/streamlit_app.py
```

Then open http://localhost:8501. To open with every built-in example already loaded (useful as a
bookmark for the live demo): http://localhost:8501/?examples=all

Needs the project's Python environment and `models/app_model.joblib` (committed). The built-in
examples read from `data/`, which is not in git, so they only appear on a machine that has the data.

## Retrain the score

```
python scripts/train_app_model.py            # uses cached features in outputs/app_features.csv
python scripts/train_app_model.py --refresh  # recompute features from data/ (about 20 minutes)
```

The evaluation is written to `results/app_model_report.md` and shown in the app's second tab.
The seven built-in examples (`app/examples.json`) are held out of training, so their scores are honest.

## Suggested demo order (about 3 minutes)

1. **Real photo: York Urban corridor.** Two directions give 649 and 713 px; the true calibrated
   focal length is 675 px. Low score.
2. **Real photo: angled street corner.** Its directions are far from right angles (a 70 degree
   error), yet the score stays low, because every horizontal direction still agrees on one focal
   length. This is why the main measure works on angled streets.
3. **AI image (Gemini): classroom.** Directions disagree by 94 per cent. High score.
4. **AI image (ChatGPT): lecture hall.** Two of its three direction pairs imply a focal length no
   camera could have, yet it only lands in "unclear" (about 60 per cent). ChatGPT is the hardest
   generator for this method (AUC about 0.6).
5. **AI image (Gemini) that fools the check.** Its directions agree within 1 per cent. Show this on
   purpose: it is a geometry check, not proof.
6. **Image the check cannot measure.** Explain the applicability rule: no verdict rather than a guess.
7. **Upload an image from the audience** (a corridor or building works best).

## What to say about the percentage

"If the image were equally likely to be real or AI before we looked, the geometry makes AI this
likely." It is calibrated on the images we tested, so an unfamiliar generator or an unusual
camera can fool it. Images are measured at their own width up to 1024 px. Overall cross-validated
AUC is about 0.82; on a generator it never saw in training, 0.59 to 0.88 (ChatGPT and Flux are the
hardest).
