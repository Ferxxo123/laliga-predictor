K = 20
FORM_MATCHES = 5
POINTS = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}
START_RATING = 1500
HOME_ADVANTAGE = 60


def update_rating(rating, expected, actual):
    """Return the new rating after a match."""
    return rating + K * (actual - expected)


def result_to_score(ftr):
    """Convert a result letter into the home team's actual score."""
    scores = {"H": 1, "D": 0.5, "A": 0}
    return scores[ftr]


def expected_score(rating_a, rating_b):
    """Chance (0 to 1) that team A does well against team B, based on their ratings."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_match(home_rating, away_rating, ftr):
    """Return the new (home, away) ratings after one match."""
    home_expected = expected_score(home_rating + HOME_ADVANTAGE, away_rating)
    home_actual = result_to_score(ftr)

    new_home = update_rating(home_rating, home_expected, home_actual)
    new_away = update_rating(away_rating, 1 - home_expected, 1 - home_actual)

    return new_home, new_away


def compute_elo(matches):
    """Return a dict with each team's final rating after all the matches."""
    ratings = {}

    for row in matches.itertuples():
        home_rating = ratings.get(row.HomeTeam, START_RATING)
        away_rating = ratings.get(row.AwayTeam, START_RATING)

        new_home, new_away = update_match(home_rating, away_rating, row.FTR)

        ratings[row.HomeTeam] = new_home
        ratings[row.AwayTeam] = new_away

    return ratings


def add_elo(matches):
    """Return a copy of matches with each team's Elo before the match."""
    ratings = {}
    home_elos = []
    away_elos = []

    for row in matches.itertuples():
        home_rating = ratings.get(row.HomeTeam, START_RATING)
        away_rating = ratings.get(row.AwayTeam, START_RATING)

        home_elos.append(home_rating)
        away_elos.append(away_rating)

        new_home, new_away = update_match(home_rating, away_rating, row.FTR)
        ratings[row.HomeTeam] = new_home
        ratings[row.AwayTeam] = new_away

    matches = matches.copy()
    matches["home_elo"] = home_elos
    matches["away_elo"] = away_elos

    return matches

def recent_average(values, n=FORM_MATCHES):
    """Average of the last n values, or NaN if there are none yet."""
    last = values[-n:]

    if len(last) == 0:
        return float("nan")

    return sum(last) / len(last)

def add_form(matches):
    """Return a copy of matches with each team's recent average points, before the match."""
    history = {}
    home_form = []
    away_form = []

    for row in matches.itertuples():
        home_history = history.setdefault(row.HomeTeam, [])
        away_history = history.setdefault(row.AwayTeam, [])

        home_form.append(recent_average(home_history))
        away_form.append(recent_average(away_history))

        home_points, away_points = POINTS[row.FTR]
        home_history.append(home_points)
        away_history.append(away_points)

    matches = matches.copy()
    matches["home_form"] = home_form
    matches["away_form"] = away_form

    return matches

def add_goal_form(matches):
    """Return a copy of matches with recent goals scored and conceded, before the match."""
    scored = {}
    conceded = {}
    home_scored, home_conceded = [], []
    away_scored, away_conceded = [], []

    for row in matches.itertuples():
        h_for = scored.setdefault(row.HomeTeam, [])
        h_against = conceded.setdefault(row.HomeTeam, [])
        a_for = scored.setdefault(row.AwayTeam, [])
        a_against = conceded.setdefault(row.AwayTeam, [])

        home_scored.append(recent_average(h_for))
        home_conceded.append(recent_average(h_against))
        away_scored.append(recent_average(a_for))
        away_conceded.append(recent_average(a_against))

        h_for.append(row.FTHG)
        h_against.append(row.FTAG)
        a_for.append(row.FTAG)
        a_against.append(row.FTHG)

    matches = matches.copy()
    matches["home_scored"] = home_scored
    matches["home_conceded"] = home_conceded
    matches["away_scored"] = away_scored
    matches["away_conceded"] = away_conceded

    return matches


def build_features(matches):
    """Return matches with every feature column added."""
    df = add_elo(matches)
    df = add_form(df)
    df = add_goal_form(df)
    df = add_rest_days(df)

    return df


def add_rest_days(matches):
    """Return a copy of matches with days since each team's previous match."""
    last_played = {}
    home_rest = []
    away_rest = []

    for row in matches.itertuples():
        home_last = last_played.get(row.HomeTeam)
        away_last = last_played.get(row.AwayTeam)

        home_rest.append((row.Date - home_last).days if home_last is not None else float("nan"))
        away_rest.append((row.Date - away_last).days if away_last is not None else float("nan"))

        last_played[row.HomeTeam] = row.Date
        last_played[row.AwayTeam] = row.Date

    matches = matches.copy()
    matches["home_rest"] = home_rest
    matches["away_rest"] = away_rest

    return matches



