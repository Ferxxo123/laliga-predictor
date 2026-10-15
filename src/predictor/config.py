from pathlib import Path #helps with folder locations 

ROOT_DIR = Path(__file__).resolve().parent.parent.parent #__file__ is location of this file,
#resolve for full path and parent goes up one folder 

DATA_DIR = ROOT_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

BASE_URL = 'https://www.football-data.co.uk/mmz4281/{season}/SP1.csv'

SEASONS = []

for year in range(2014, 2025):
    season = str(year)[2:] + str(year+1)[2:]
    SEASONS.append(season)


DISPLAY_NAMES = {
    "Alaves": "Alavés",
    "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atlético de Madrid",
    "Betis": "Real Betis",
    "Celta": "Celta de Vigo",
    "Espanol": "Espanyol",
    "Leganes": "Leganés",
    "Sociedad": "Real Sociedad",
    "Valladolid": "Real Valladolid",
    "Vallecano": "Rayo Vallecano",
}