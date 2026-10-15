import joblib
import pandas as pd

from src.predictor.model import FEATURES
from src.predictor.train import MODEL_PATH


def predict_match(home, away, state, match_date):
    """Return {'H': p, 'D': p, 'A': p} for a match on match_date."""
    saved = joblib.load(MODEL_PATH)
    model, medians = saved["model"], saved["medians"]

    h, a = state[home], state[away]
    row = {
        "home_elo": h["elo"], "away_elo": a["elo"],
        "home_form": h["form"], "away_form": a["form"],
        "home_scored": h["scored"], "home_conceded": h["conceded"],
        "away_scored": a["scored"], "away_conceded": a["conceded"],
        "home_rest": (match_date - h["last_date"]).days,
        "away_rest": (match_date - a["last_date"]).days,
    }

    X = pd.DataFrame([row])[FEATURES].fillna(medians)
    probs = model.predict_proba(X)[0]

    return {c: float(p) for c, p in zip(model.classes_, probs.round(3))}