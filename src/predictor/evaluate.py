import numpy as np 
import pandas as pd

TRAIN_SEASONS = ["1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122"]
VALID_SEASONS = ["2223"]
TEST_SEASONS = ["2324", "2425"]


def split_by_season(df):
    """Return (train, valid, test) tables, split by season."""
    train = df[df["season"].isin(TRAIN_SEASONS)]
    valid = df[df["season"].isin(VALID_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]

    return train, valid, test



def log_loss(probs, results):
    """Average log loss. probs has columns pH, pD, pA; results are 'H', 'D' or 'A'."""
    chosen = [row[{"H": 0, "D": 1, "A": 2}[r]] for row, r in zip(probs, results)]
    return -np.mean(np.log(chosen))


def bookmaker_probs(df):
    """Bookmaker probabilities with the margin removed, as rows of [pH, pD, pA]."""
    raw = 1 / df[["B365H", "B365D", "B365A"]]
    return raw.div(raw.sum(axis=1), axis=0).values


def brier_score(probs, results):
    """Average Brier score. Same inputs as log_loss."""
    actual = np.array([[r == "H", r == "D", r == "A"] for r in results], dtype=float)
    return np.mean(np.sum((np.array(probs) - actual) ** 2, axis=1))


def accuracy(probs, results):
    """Share of matches where the most likely outcome was the real one."""
    picks = np.array(["H", "D", "A"])[np.argmax(probs, axis=1)]
    return np.mean(picks == np.array(results))


def calibration_table(probs, results, outcome="H", bins=5):
    """Compare predicted probability with the real rate, in groups of equal size."""
    column = {"H": 0, "D": 1, "A": 2}[outcome]
    p = np.array(probs)[:, column]
    actual = np.array(results) == outcome

    df = pd.DataFrame({"p": p, "actual": actual})
    df["group"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")

    return df.groupby("group").agg(
        predicted=("p", "mean"),
        actual=("actual", "mean"),
        matches=("actual", "size"),
    ).round(3)