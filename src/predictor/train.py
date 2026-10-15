import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.predictor.config import ROOT_DIR
from src.predictor.data import load_matches
from src.predictor.features import build_features
from src.predictor.model import FEATURES, get_xy

MODEL_PATH = ROOT_DIR / "models" / "model.joblib"


def train_final_model():
    """Fit the model on every season and save it with its fill values."""
    df = build_features(load_matches())

    medians = df[FEATURES].median()
    X, y = get_xy(df, medians)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "medians": medians}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_final_model()