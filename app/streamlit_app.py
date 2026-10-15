import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


import pandas as pd
import streamlit as st

from src.predictor.config import DISPLAY_NAMES, SEASONS
from src.predictor.data import load_matches
from src.predictor.predict import predict_match
from src.predictor.state import build_team_state


def show(team):
    """The name to display for a team (falls back to the dataset name)."""
    return DISPLAY_NAMES.get(team, team)


@st.cache_data
def load_state():
    """Build the team state once and reuse it between clicks."""
    state = build_team_state(load_matches())
    latest = load_matches([SEASONS[-1]])
    teams = sorted(set(latest["HomeTeam"]) | set(latest["AwayTeam"]))
    return state, teams


st.title("La Liga match predictor")

state, teams = load_state()

col1, col2 = st.columns(2)
home = col1.selectbox("Home team", teams, format_func=show)
away = col2.selectbox("Away team", teams, index=1, format_func=show)

if home == away:
    st.warning("Pick two different teams")
else:
    last_date = max(team["last_date"] for team in state.values())
    probs = predict_match(home, away, state, last_date + pd.Timedelta(days=7))

    chart = pd.DataFrame(
        {
            "outcome": [f"{show(home)} win", "Draw", f"{show(away)} win"],
            "probability": [probs["H"], probs["D"], probs["A"]],
        }
    ).set_index("outcome")

    st.subheader("Probabilities")
    st.bar_chart(chart)

    st.subheader("Why?")

    rows = {}
    for team in (home, away):
        info = state[team]
        rows[show(team)] = {
            "Elo rating": round(info["elo"]),
            "Points per match (last 5)": round(info["form"], 2),
            "Goals scored per match": round(info["scored"], 2),
            "Goals conceded per match": round(info["conceded"], 2),
        }

    st.table(pd.DataFrame(rows))