import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from paths import mkl_data
import requests # API library

my_data_path = "/zhome/25/e/155273/Desktop/42186_moml/data/df_matches.csv"
df = pd.read_csv(my_data_path, sep=',')
# Drop rows with missing values
df = df.dropna(subset=['match_date','match_conceded', 'coach1_CR', 'coach2_CR'])
relevant = ["match_id", "division_name", "match_date", "match_time", "match_conceded", 
            "team1_coach_id", "team1_race_name", "team2_coach_id", "team2_race_name", 
            "team1_score", "team2_score","team1_win","team2_win","mirror_match", 
            "coach1_ranking","coach2_ranking","coach1_CR","coach2_CR"]
df_matches = df[relevant]


df_matches['match_date'] = pd.to_datetime(df_matches.match_date) # Datetime object
# calculate match score difference
df_matches['team1_win'] = np.sign(df_matches['team1_score'] - df_matches['team2_score'])
df_matches['team2_win'] = np.sign(df_matches['team2_score'] - df_matches['team1_score'])


print(df_matches['team1_win'])


#For Later
#One-hot encoding
#Remove id columns

#Now
#Total Fights
# total_fights = df.shape[0]


#Fordeling over winrates (antallet af spillere på y-aksen, x-aksen er winrate, )
#The players
# players = pd.concat([df["team1_coach_id"], df["team2_coach_id"]])
# players_unique = players.unique()
# matches = df[]

# df_t1w = df["team1_win"].astype(int)   
# df_t1c = df["team1_coach_id"].unique()
# coach_wins = df.groupby('team1_coach_id')['team1_win'].sum().reset_index()
# coach_wins.columns = ['coach_id', 'total_wins']

# df_wins1 = df[df['team1_win'] > 0]
# df_coach1 = df_wins1.groupby('team1_coach_id')['team1_win'].sum()
# print(df_coach1)

# df_wins2 = df[df['team2_win'] > 0]
# df_coach2 = df_wins2.groupby('team2_coach_id')['team2_win'].sum()




# df.groupby('team1_coach_id')['team1_win'].mean().plot()
# df_t2w = df["team2_win"].astype(int)
# df_t2c = df["team2_coach_id"].unique()



# obs_unique = df[(df['team1_coach_id'] == unique_coach_id) | (df['team2_coach_id'] == unique_coach_id)]

# print(df["team1_coach_id"][players_unique[0]])


# print(type(players_unique))
# print(np.sort(players_unique), len(players_unique))

#Calculate winrate
#Plot winrate and players

#Winrate over races (racer x-aksen, winrate y-aksen)
