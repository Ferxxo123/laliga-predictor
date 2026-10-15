from src.predictor.features import POINTS, compute_elo, recent_average


def build_team_state(matches):
    """Return {team: {elo, form, scored, conceded, last_date}} after all matches."""
    elo = compute_elo(matches)
    points, scored, conceded, last_date = {}, {}, {}, {}

    for row in matches.itertuples():
        home_pts, away_pts = POINTS[row.FTR]
        points.setdefault(row.HomeTeam, []).append(home_pts)
        points.setdefault(row.AwayTeam, []).append(away_pts)
        scored.setdefault(row.HomeTeam, []).append(row.FTHG)
        scored.setdefault(row.AwayTeam, []).append(row.FTAG)
        conceded.setdefault(row.HomeTeam, []).append(row.FTAG)
        conceded.setdefault(row.AwayTeam, []).append(row.FTHG)
        last_date[row.HomeTeam] = row.Date
        last_date[row.AwayTeam] = row.Date

    return {
        team: {
            "elo": elo[team],
            "form": recent_average(points[team]),
            "scored": recent_average(scored[team]),
            "conceded": recent_average(conceded[team]),
            "last_date": last_date[team],
        }
        for team in elo
    }