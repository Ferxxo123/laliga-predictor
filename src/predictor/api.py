import pandas as pd
from fastapi import FastAPI, HTTPException

from src.predictor.config import SEASONS
from src.predictor.data import load_matches
from src.predictor.predict import predict_match
from src.predictor.state import build_team_state

app = FastAPI(title="La Liga predictor")

STATE = build_team_state(load_matches())
LAST_DATE = max(team["last_date"] for team in STATE.values())

# only teams from the latest season go in the dropdown
latest = load_matches([SEASONS[-1]])
TEAMS = sorted(set(latest["HomeTeam"]) | set(latest["AwayTeam"]))


@app.get("/teams")
def get_teams():
    return TEAMS


@app.get("/predict")
def get_prediction(home: str, away: str):
    if home not in TEAMS or away not in TEAMS:
        raise HTTPException(status_code=404, detail="Unknown team")
    if home == away:
        raise HTTPException(status_code=400, detail="Pick two different teams")

    match_date = LAST_DATE + pd.Timedelta(days=7)
    probs = predict_match(home, away, STATE, match_date)

    return {"home": home, "away": away, "probabilities": probs}