import pandas as pd

from src.predictor.data import load_matches
from src.predictor.predict import predict_match
from src.predictor.state import build_team_state


def test_team_state_has_every_team():
    state = build_team_state(load_matches(["2324"]))
    assert len(state) == 20


def test_probabilities_add_up_to_one():
    state = build_team_state(load_matches(["2324"]))
    probs = predict_match("Barcelona", "Getafe", state, pd.Timestamp("2024-08-24"))
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_stronger_team_is_favourite():
    state = build_team_state(load_matches(["2324"]))
    probs = predict_match("Real Madrid", "Granada", state, pd.Timestamp("2024-08-24"))
    assert probs["H"] > probs["A"]