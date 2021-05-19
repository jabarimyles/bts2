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


def get_rpplayer(rp_period, file_name_out, pitch_or_bat, prod=False, table_dict={}):
    if prod == False:
        data = pd.read_csv('./data/statcast.csv')
    else:
        data = table_dict['statcast']
    
    # Grabs either pitcher or batter specified and what the events for each game for each at bat
    game_date_df = data[['game_date', 'game_pk', pitch_or_bat, 'events_grouped']]
    events_grouped_dummy = pd.get_dummies(game_date_df['events_grouped'])
    game_date_df = pd.concat([game_date_df, events_grouped_dummy], axis=1)

    # Drops events_grouped
    rm_cols = game_date_df.drop(['events_grouped'], axis=1)

    # Sums performance metrics for each player for each game
    game_lvl=pd.pivot_table(rm_cols, index=['game_date', pitch_or_bat, 'game_pk'], values=['hit', 'non_ab', 'out'],aggfunc=np.sum).reset_index()


    '''
        fill in missing dates for each player
        dummy variables for events_grouped
        rolling sum for each
    '''
    
    in_season_dates = game_date_df['game_date'].drop_duplicates()
    batter_rp = pd.DataFrame(columns=['game_date', pitch_or_bat])
    for bat in tqdm(game_date_df[pitch_or_bat].unique()):
        _ = pd.DataFrame(columns=['game_date', pitch_or_bat])
        _['game_date'] = in_season_dates #returns most recent dates
        _[pitch_or_bat] = bat #brings in batter/pit id
        batter_rp = pd.concat([batter_rp, _], ignore_index=True) #concats id and dates
    full_dates = pd.merge(batter_rp, game_lvl, how='left', on=['game_date', pitch_or_bat])
    full_dates = full_dates.sort_values(by=[pitch_or_bat, 'game_date'], ascending=True)
    #full_dates['hit'] = full_dates['hit'].fillna(0)
    #full_dates['out'] = full_dates['out'].fillna(0)
    #full_dates['non_ab'] = full_dates['non_ab'].fillna(0)
    full_dates['games_played'] = np.where(full_dates['game_pk'].isna(), 0, 1)
    full_dates['year'] = full_dates['game_date'].str[0:4]
    #full_dates['cur_rp_hits'] = full_dates.groupby([pitch_or_bat, 'year'])['hit'].transform(lambda s: s.rolling(rp_period, min_periods=1).sum())
    #full_dates['cur_rp_outs'] = full_dates.groupby([pitch_or_bat, 'year'])['out'].transform(lambda s: s.rolling(rp_period, min_periods=1).sum())
    #full_dates['cur_rp_non_abs'] = full_dates.groupby([pitch_or_bat, 'year'])['non_ab'].transform(lambda s: s.rolling(rp_period, min_periods=1).sum())
    #full_dates['cur_rp_games_played'] = full_dates.groupby([pitch_or_bat, 'year'])['games_played'].transform(lambda s: s.rolling(rp_period, min_periods=1).sum())
    #full_dates['rp_hits'] = full_dates['cur_rp_hits'] - full_dates['hit']
    #full_dates['rp_outs'] = full_dates['cur_rp_outs'] - full_dates['out']
    #full_dates['rp_non_abs'] = full_dates['cur_rp_non_abs'] - full_dates['non_ab']
    #full_dates['rp_games_played'] = full_dates['cur_rp_games_played'] - full_dates['games_played']
    full_dates['rp_outs'] = full_dates.groupby([pitch_or_bat, 'year'])['out'].transform(lambda s: s.shift(1).rolling(rp_period, min_periods=1).sum())
    full_dates['rp_non_abs'] = full_dates.groupby([pitch_or_bat, 'year'])['non_ab'].transform(lambda s: s.shift(1).rolling(rp_period, min_periods=1).sum())
    full_dates['rp_games_played'] = full_dates.groupby([pitch_or_bat, 'year'])['games_played'].transform(lambda s: s.shift(1).rolling(rp_period, min_periods=1).sum())
    full_dates['rp_hits'] = full_dates.groupby([pitch_or_bat, 'year'])['hit'].transform(lambda s: s.shift(1).rolling(rp_period, min_periods=1).sum())
    full_dates['rp_hits_var'] = full_dates.groupby([pitch_or_bat, 'year'])['hit'].transform(lambda s: s.shift(1).rolling(rp_period, min_periods=1).var())

    if prod==False:
        full_dates.to_csv(os.path.join('./data/', file_name_out), index=False)
    else:
        return full_dates
    return None
    ''' -- To delete
        batter_rp = game_date_df[['game_date', 'batter', 'rp_start_date']].drop_duplicates()
        print('dropping duplicates...')
        batter_rp['hits'] = 0
        batter_rp['outs'] = 0
        batter_rp['non_abs'] = 0
        batter_rp['games_played'] = 0
        for i, r in tqdm(batter_rp.iterrows(), total=batter_rp.shape[0]):
            d = game_date_df.loc[(game_date_df['game_date']<=r['game_date']) & (game_date_df['game_date']>=r['rp_start_date']) & (game_date_df['batter'] == r['batter']) ]
            counts = d['events_grouped'].value_counts()
            batter_rp.loc[i, 'hits'] = counts['hit'] if 'hit' in counts.keys() else 0
            batter_rp.loc[i, 'outs'] = counts['out'] if 'out' in counts.keys() else 0
            batter_rp.loc[i, 'non_abs'] = counts['non_ab'] if 'non_ab' in counts.keys() else 0
            batter_rp.loc[i, 'games_played'] = len(d['game_pk'].unique())
        batter_rp.to_csv('./data/RPBatter.csv', index=False)
        return None
    '''

