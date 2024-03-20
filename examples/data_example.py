import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from paths import mkl_data
import requests # API library

my_data_path = "/zhome/25/e/155273/Desktop/42186_moml/data/df_matches.csv"
df = pd.read_csv(my_data_path, sep=',')
# Drop rows with missing values
df = df.dropna(subset=['match_date','match_conceded', 'coach1_CR', 'coach2_CR'])
relevant = ["match_id", "division_name", "match_date", "match_time", "match_conceded", "team1_coach_id", "team1_race_name", "team2_coach_id", "team2_race_name", "team1_score", "team2_score","team1_win","team2_win","mirror_match", "coach1_ranking","coach2_ranking","coach1_CR","coach2_CR"]
df = df[relevant]
print(df.head(5))
