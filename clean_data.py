import pandas as pd
import numpy as np
import os
from isoweek import Week


def clean_data(path='data'):

    # Check if the datafolder exists, if not create it
    if not os.path.exists(path):
        assert False, "Data folder does not exist"
    
    my_data_path = os.path.join(path, 'df_matches.csv')

    df = pd.read_csv(my_data_path, sep=',')
    # Drop rows with missing values
    # df = df.dropna(subset=['match_date','match_conceded', 'coach1_CR', 'coach2_CR'])

    relevant = ["match_id", "division_name", "match_date", "match_time", "match_conceded",
                "team1_coach_id", "team1_race_name", "team2_coach_id", "team2_race_name",
                "team1_score", "team2_score","team1_win","team2_win","mirror_match",
                "coach1_ranking","coach2_ranking","coach1_CR","coach2_CR", "team1_roster_id", "team2_roster_id"]
    #irrelevant_columns = set(df.columns) - set(relevant)

    irrelevant_columns = {
        'tournament_name', 'week_year', 'cr_diff2', 'team1_cas_bh',
        'group_name', 'team2_foul', 'team2_id', 'coach2_CR_bin',
        'team1_id', 'year', 'week_date', 'week_number', 'replay_id',
        'team1_cas_rip', 'tv_diff', 'team2_block', 'coach1_CR_bin',
        'has_sp', 'team2_value', 'tournament_type', 'team1_value',
        'team2_comp', 'team1_rush', 'team1_foul', 'team2_inducements',
        'tv_diff2', 'tournament_start', 'team2_cas_rip', 'team1_inducements',
        'tv_bin', 'scheduler', 'team1_comp', 'cr_diff2_bin', 'group_id',
        'division_id', 'tournament_id', 'Unnamed: 0', 'team2_cas_si',
        'current_ruleset', 'team2_cas', 'team2_rush', 'team1_block',
        'tournament_end', 'team2_pass', 'team1_cas',
        'CR_diff', 'team1_cas_si', 'team2_cas_bh', 'team1_pass', 'tv_match',
        'tv_bin2', 'tournament_progression'
    }


    #first load from the competitve league, then filter out the relevant columns
    df_matches_competitive = df[df['division_name'] == 'Competitive']
    df_matches = df_matches_competitive[relevant]

    #Fix types in pandas

    # # convert object dtype columns to proper pandas dtypes datetime and numeric
    df_matches['match_date'] = pd.to_datetime(df_matches.match_date) # Datetime object

    # calculate match score difference
    df_matches['team1_win'] = np.sign(df_matches['team1_score'] - df_matches['team2_score'])
    df_matches['team2_win'] = np.sign(df_matches['team2_score'] - df_matches['team1_score'])

    # mirror match
    df_matches['mirror_match'] = 0
    df_matches.loc[df_matches['team1_race_name'] == df_matches['team2_race_name'], 'mirror_match'] = 1


    # mirror matches
    df_matches = df_matches.dropna(subset=['match_date'])

    df_matches['week_number'] = df_matches['match_date'].dt.isocalendar().week

    # cannot serialize numpy int OR Int64 when writing HDF5 file, so we go for plain int as all NAs are gone now
    df_matches['week_number'] = df_matches['week_number'].fillna(0).astype(int)

    # add year based on match ISO week
    df_matches['year'] = df_matches['match_date'].dt.isocalendar().year.astype(int)

    df_matches['week_year'] = df_matches['year'].astype(str) + '-' + df_matches['week_number'].astype(str)

    # use a lambda function since isoweek.Week is not vectorized 
    df_matches['week_date'] = pd.to_datetime(df_matches.apply(lambda row : Week(int(row["year"]),int(row["week_number"])).monday(),axis=1))


    #The selected columns from the competitive dataframe 
    selected_columns = [
        'week_date', 
        'match_time',
        'team1_coach_id', 
        'team2_coach_id',
        'team1_win', 
        'team2_win', 
        'team1_race_name', 
        'team2_race_name'
    ]

    #final dataframe 
    new_df = df_matches[selected_columns].copy()
    new_df['team1_race_name'] = new_df['team1_race_name'].astype('category')
    new_df['team2_race_name'] = new_df['team2_race_name'].astype('category')


    #Store the cleaned data in your local directory 
    new_df.to_csv(os.path.join(path, 'df_matches_clean.csv'), index=False)