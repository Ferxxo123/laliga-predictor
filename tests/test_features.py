from src.predictor.data import load_matches
from src.predictor.features import START_RATING, add_elo, add_goal_form, add_form, build_features, add_rest_days

import pandas as pd




def test_first_match_uses_start_rating():
    df = add_elo(load_matches(["2324"]))
    assert df.loc[0, "home_elo"] == START_RATING
    assert df.loc[0, "away_elo"] == START_RATING


def test_add_elo_keeps_all_rows():
    matches = load_matches(["2324"])
    df = add_elo(matches)
    assert len(df) == len(matches)


def test_add_elo_does_not_modify_the_input():
    matches = load_matches(["2324"])
    add_elo(matches)
    assert "home_elo" not in matches.columns


def test_first_match_has_no_form():
    df = add_form(load_matches(["2324"]))
    assert pd.isna(df.loc[0, "home_form"])


def test_form_uses_only_previous_matches():
    df = add_form(load_matches(["2324"]))
    barca = df[(df["HomeTeam"] == "Barcelona") | (df["AwayTeam"] == "Barcelona")]
    # points Barcelona earned in each match, in date order
    points = [
        3 if (row.HomeTeam == "Barcelona" and row.FTR == "H")
        or (row.AwayTeam == "Barcelona" and row.FTR == "A")
        else 1 if row.FTR == "D"
        else 0
        for row in barca.itertuples()
    ]
    expected = sum(points[-6:-1]) / 5
    assert barca.iloc[-1]["away_form"] == expected

def test_first_match_has_no_goal_form():
    df = add_goal_form(load_matches(["2324"]))
    assert pd.isna(df.loc[0, "home_scored"])
    assert pd.isna(df.loc[0, "away_conceded"])


def test_goal_form_does_not_use_the_current_match():
    matches = load_matches(["2324"])
    changed = matches.copy()
    changed.loc[changed.index[-1], "FTHG"] = 9
    changed.loc[changed.index[-1], "FTAG"] = 9

    original = add_goal_form(matches)
    modified = add_goal_form(changed)

    cols = ["home_scored", "home_conceded", "away_scored", "away_conceded"]
    assert original[cols].equals(modified[cols])


def test_build_features_keeps_all_rows_and_adds_columns():
    matches = load_matches(["2324"])
    df = build_features(matches)
    assert len(df) == len(matches)
    for col in ["home_elo", "away_elo", "home_form", "away_form", "home_scored", "home_conceded", "away_scored", "away_conceded"]:
        assert col in df.columns


def test_rest_days_are_positive():
    df = add_rest_days(load_matches(["2324"]))
    assert pd.isna(df.loc[0, "home_rest"])
    assert (df["home_rest"].dropna() > 0).all()


TRAIN_SEASONS = ["1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122"]
VALID_SEASONS = ["2223"]
TEST_SEASONS = ["2324", "2425"]


def split_by_season(df):
    """Return (train, valid, test) tables, split by season."""
    train = df[df["season"].isin(TRAIN_SEASONS)]
    valid = df[df["season"].isin(VALID_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]

    return train, valid, test