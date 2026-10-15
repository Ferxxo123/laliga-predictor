from src.predictor.data import load_matches, load_season


def test_load_season_has_380_matches():
    df = load_season("2324")
    assert len(df) == 380


def test_load_season_has_no_missing_values():
    df = load_season("2324")
    assert df.isna().sum().sum() == 0


def test_load_season_results_are_valid():
    df = load_season("2324")
    assert set(df["FTR"]) == {"H", "A", "D"}

def test_load_matches_is_sorted_by_date():
    df = load_matches(["2324"])
    assert df["Date"].is_monotonic_increasing 