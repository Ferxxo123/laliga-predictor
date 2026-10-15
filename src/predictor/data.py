import pandas as pd
import requests

from src.predictor.config import BASE_URL, RAW_DIR, SEASONS

KEEP_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]
MATCHES_PER_SEASON = 380


def download_season(season):
    """Download one La Liga season CSV into data/raw (skips it if already there)."""
    file_path = RAW_DIR / (season + ".csv")

    # Don't download again if we already have the file
    if file_path.exists():
        print(f"{season} already downloaded")
        return

    url = BASE_URL.format(season=season)
    response = requests.get(url, timeout=30)

    # Stop with a clear error if the download failed (404, 500...)
    response.raise_for_status()

    file_path.write_bytes(response.content)
    print(f"{season} downloaded")


def download_all():
    """Download every season listed in config.SEASONS."""
    for season in SEASONS:
        download_season(season)


def load_season(season):
    """Read one season's CSV and return a clean DataFrame."""
    file_path = RAW_DIR / (season + ".csv")

    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found. Run download_all() first.")

    # Some older files are not UTF-8, so fall back to latin-1 if reading fails
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")

    # Make sure the file has every column we need
    missing = [col for col in KEEP_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Season {season} is missing columns: {missing}")

    df = df[KEEP_COLUMNS].copy()

    # Drop empty rows: a real match needs a date, both teams and a result
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

    df["season"] = season

    # dayfirst + mixed handles both 23/08/2014 and 23/08/14 in the same file
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")

    # Goals come in as decimals when a column has empty cells, so make them whole numbers
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    return df


def load_matches(seasons=SEASONS):
    """Load several seasons into one DataFrame, sorted by date."""
    frames = []

    for season in seasons:
        df = load_season(season)

        if len(df) != MATCHES_PER_SEASON:
            print(f"Warning: season {season} has {len(df)} matches, expected {MATCHES_PER_SEASON}")

        frames.append(df)

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values("Date").reset_index(drop=True)

    return matches


if __name__ == "__main__":
    download_all()