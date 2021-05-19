
#-- Base packages
import os
import pdb

#-- Pypi packages
import pandas as pd
pd.set_option('display.max_columns', 100)
from baseball_scraper import statcast


def get_game_lvl(prod=False, table_dict={}):

    if prod == False:
        orig_data = pd.read_csv('./data/statcast.csv')
    else:
        orig_data = table_dict['statcast']
    
    #Keeps only regular season games
    orig_data = orig_data.loc[orig_data['game_type']=='R'] 

    #List of variables we are interested in for game level stats
    keep_cols = ['index', 'game_date', 'player_name', 'batter',
            'events', 'inning_topbot',
            'description',  'des',
            'game_type', 'stand','home_team', 'away_team', 'type',
            'game_year',  'game_pk', 'events_grouped']

    # List of columns we want for SP info
    keep_cols_p = ['pitcher', 'game_pk', 'inning', 'inning_topbot', 'pitch_number']
    sp_info = orig_data[keep_cols_p]

    # Defines criteria for SP each game
    sp_info = sp_info.loc[sp_info['inning']==1.0].drop_duplicates()
    sp_info = sp_info.sort_values(by=['game_pk', 'pitch_number'], ascending=True)
    sp_info = sp_info.drop_duplicates(subset=['game_pk', 'inning_topbot'], keep='first')
    piv_cols= ['game_pk', 'pitcher', 'inning_topbot']
    sp_info = sp_info[piv_cols].rename(columns={'pitcher':'starting_pitcher'})

    # Creates dataset and dummies for starting pitcher and their events given up
    data = orig_data[keep_cols]
    data = pd.get_dummies(data, columns=['events_grouped'], prefix ='', prefix_sep = '')
    print('--- data, game_date ---')
    print(data['game_date'].max())
    #data['at_bats'] = data.groupby(['batter','game_pk'])['at_bat_number'].transform(max)
    group_cols = ['game_year', 'batter','inning_topbot', 'game_pk','game_date','stand','home_team','away_team']
    game_data = data.groupby(group_cols).sum().reset_index()
    print('---- game_data ----')
    print(game_data['game_date'].max())

    game_data = game_data.rename(columns={'hit':'hits', 'non_ab':'non_abs', 'non_play':'non_plays', 'out':'outs'})
    game_data = pd.merge(game_data, sp_info, how='left', on=['game_pk', 'inning_topbot'])
    print('---- game_data ----')
    print(game_data['game_date'].max())
    if prod==True:
        return game_data
    else:
        game_data.to_csv('./data/GameLvl.csv', index=False)
        return None


'''
def get_season_bat(start_date, end_date){

    data = batting_stats_range(start_date, end_date)
}
'''
