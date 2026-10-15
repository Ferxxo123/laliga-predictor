FEATURES = [
    "home_elo", "away_elo",
    "home_form", "away_form",
    "home_scored", "home_conceded",
    "away_scored", "away_conceded",
    "home_rest", "away_rest",
]


def get_xy(df, medians):
    """Return the feature table X (NaN filled with medians) and the results y."""
    X = df[FEATURES].fillna(medians)
    y = df["FTR"]
    return X, y


from lightgbm import LGBMClassifier


def make_lgbm():
    """A small, regularized gradient boosting model."""
    return LGBMClassifier(
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=8,
        min_child_samples=40,
        verbose=-1,
    )


from sklearn.ensemble import RandomForestClassifier


def make_forest():
    """A random forest with leaves large enough not to memorize matches."""
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=20,
        random_state=42,
    )