# La Liga match predictor

Predicts the probability of a home win, a draw or an away win for any La Liga match, using only past results. It is compared against the bookmaker's odds, the hardest baseline in football prediction.

**Demo:** https://laliga-predictor-zoibach3tyc8bcunadipdm.streamlit.app/#probabilities

## Results

Scored on the test set (2023-24 and 2024-25, 760 matches), which was used once, after the model was chosen on validation.

| Predictor | Log loss | Brier | Accuracy |
|---|---|---|---|
| Frequency baseline (always the training rates) | 1.073 | 0.649 | 0.442 |
| Bookmaker (Bet365, margin removed) | **0.950** | **0.564** | **0.549** |
| Logistic regression | 0.974 | 0.579 | 0.539 |

Lower is better for log loss and Brier. The model beats the baseline by about 0.10 in log loss and lands 0.024 behind the bookmaker. A gap is expected: the bookmaker also knows about injuries, line-ups, transfers and news, and this model only sees results.

## What I found

- **Home advantage is real and stable.** Over 4,180 matches (2014-2025), 45.4% ended in a home win, 26.0% in a draw and 28.6% in an away win. Home teams score about 0.36 more goals per match (1.49 vs 1.13), and the model builds this in with a 60-point Elo bonus for the home side.
- **Three very different models tie.** Logistic regression, LightGBM and a random forest all scored about 1.00 log loss on validation, so the limit is the information in the features, not the model type. I kept logistic regression because it is the simplest and the easiest to explain.
- **Probabilities are mostly well calibrated.** When the model says 45-70%, those outcomes happen at about that rate. The exception is the lowest home-win group, where it was too pessimistic on one validation season (a small sample, so not conclusive).
- **Draws are close to unpredictable from results alone.** The model can say that evenly matched teams draw slightly more often, but its draw probability barely separates draw-prone matches from the rest. The bookmaker never picks a draw either.

## How it works

1. **Data:** 11 seasons of La Liga results and Bet365 odds from [football-data.co.uk](https://www.football-data.co.uk/) (4,180 matches).
2. **Features, with no leakage:** every feature for a match uses only matches played before it. Tests check this.
   - Elo rating of both teams (K = 20, home advantage 60)
   - Average points and goals scored/conceded in the last 5 matches
   - Days of rest since the previous match
3. **Evaluation:** a time-based split (train 2014-15 to 2021-22, validation 2022-23, test 2023-24 and 2024-25), scored with log loss, Brier score and accuracy.
4. **Final model:** logistic regression refit on all 11 seasons for the app.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m pytest                              # 16 tests
python -m streamlit run app/streamlit_app.py  # the web app
uvicorn src.predictor.api:app --reload        # the API (/docs for a test page)
python -m src.predictor.train                 # retrain and save the model
```

## Project structure

```
src/predictor/
├── config.py      settings (seasons, paths, display names)
├── data.py        download and load the CSVs
├── features.py    Elo, form, goals, rest days
├── evaluate.py    time split, log loss, Brier, accuracy, calibration
├── model.py       feature list and model builders
├── train.py       fit and save the final model
├── state.py       each team's current Elo and form
├── predict.py     predict one match
└── api.py         FastAPI app
app/streamlit_app.py   web app
notebooks/             data exploration and model comparison
tests/                 pytest tests
```

## Limitations

- The data stops at the end of 2024-25. The app predicts "the next round", one week after the latest match in the data.
- No injuries, line-ups or transfers, which is the main reason the bookmaker is still ahead.
- Promoted teams start at an Elo of 1500, and a team returning after years away keeps its old rating and form.
- The reported test scores come from the model trained on 2014-2022 data. The deployed model is refit on all seasons, so it has no separate held-out score.
- Model choice used a single validation season (380 matches), so small differences between models are within noise.

## Ideas for later

- Add newer seasons by changing `LAST_YEAR` in `config.py` and rerunning the download.
- Try Pinnacle odds as a tougher benchmark.
- Tune the Elo settings (K and the home advantage) on validation.
