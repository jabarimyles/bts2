#-- Base packages
import os
from tqdm import tqdm
import copy
import time
import pdb 

#-- Pypi packages
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100)

#-- Custom packages

# Takes in a dictionary of particular 
def get_matchups(prod=False, table_dict={}):
    if table_dict == {}:
        data = pd.read_csv('./data/statcast.csv')
    else:
        data = table_dict['statcast']

    # Keeps and transforms variables that indentify batter/pitcher matchup
    game_date_df = data[['game_date', 'game_pk', 'batter', 'pitcher', 'events_grouped']]
    events_grouped_dummy = pd.get_dummies(game_date_df['events_grouped'])
    game_date_df = pd.concat([game_date_df, events_grouped_dummy], axis=1)
    rm_cols = game_date_df.drop(['events_grouped'], axis=1)
    game_lvl=pd.pivot_table(rm_cols, index=['game_date','batter', 'pitcher', 'game_pk'], values=['hit', 'non_ab', 'out'],aggfunc=np.sum).reset_index()
    game_lvl = game_lvl.sort_values(by=['batter', 'pitcher', 'game_date'], ascending=True)
    game_lvl['year'] = game_lvl['game_date'].str[0:4]
    game_lvl['game_ind'] = 1
    
    # Creates stats for pitch and batters matchups for the last year
    game_lvl['year_hits'] = game_lvl.groupby(['batter', 'pitcher', 'year'])['hit'].transform(lambda s: s.shift(1).rolling(300, min_periods=1).sum())
    game_lvl['year_outs'] = game_lvl.groupby(['batter', 'pitcher', 'year'])['out'].transform(lambda s: s.shift(1).rolling(300, min_periods=1).sum())
    game_lvl['year_non_abs'] = game_lvl.groupby(['batter', 'pitcher', 'year'])['non_ab'].transform(lambda s: s.shift(1).rolling(300, min_periods=1).sum())
    game_lvl['year_games_played'] = game_lvl.groupby(['batter', 'pitcher', 'year'])['game_ind'].transform(lambda s: s.shift(1).rolling(300, min_periods=1).sum())

    # Creates careers stats for matchup
    game_lvl['career_hits'] = game_lvl.groupby(['batter', 'pitcher'])['hit'].transform(lambda s: s.shift(1).rolling(365*20, min_periods=1).sum())
    game_lvl['career_outs'] = game_lvl.groupby(['batter', 'pitcher'])['out'].transform(lambda s: s.shift(1).rolling(365*20, min_periods=1).sum())
    game_lvl['career_non_abs'] = game_lvl.groupby(['batter', 'pitcher'])['non_ab'].transform(lambda s: s.shift(1).rolling(365*20, min_periods=1).sum())
    game_lvl['career_games_played'] = game_lvl.groupby(['batter', 'pitcher'])['game_ind'].transform(lambda s: s.shift(1).rolling(365*20, min_periods=1).sum())

    if prod==True:
        return game_lvl
    game_lvl.to_csv(os.path.join('./data/', 'matchups.csv'), index=False)
    return None
