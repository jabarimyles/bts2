#-- Base packages
import os
import sys
from datetime import date
import pickle
import math

#-- Pypi packages
import pandas as pd
import pdb
pd.set_option('display.max_columns', 100)
from baseball_scraper import statcast

#-- Custom packages
from get_data import get_statcast
from createTableRPPlayer import get_rpplayer
from createTableGameLvl import get_game_lvl
from createModelingData import get_modeling_data
from createTablePlayerMeta import get_player_meta
from createTableMatchups import get_matchups
from createTodaysMatchups import get_todays_matchups
from train_model import logistic, tune_lgbm, lgbm 
from enterDailyPreds import enter


if __name__ == '__main__':

    try:
        train_or_prod = sys.argv[1]
    except IndexError:
        train_or_prod = 'train'
    if train_or_prod == 'train':
        start_date = '2012-01-01'
        end_date   = '2021-01-01'

        print('querying statcast data....')
        # Returns play level data from baseball_scraper
        get_statcast(start_date, end_date) 
        print('creating rp_player data...')

        # Returns recent player performance
        get_rpplayer(16, 'RPBatter.csv', 'batter')
        get_rpplayer(300, 'YTDBatter.csv', 'batter')
        get_rpplayer(20, 'RPPitcher.csv', 'pitcher')
        get_rpplayer(300, 'YTDPitcher.csv', 'pitcher')
        print('Creating player meta data...')
        get_player_meta()
        print('creating game level data...')
        get_game_lvl()
        print('get matchups...')
        get_matchups()

    elif train_or_prod == 'prod':
        start_date = '2021-01-01'
        #start_date needs to query current db to get max date + 1
        today = date.today()
        end_date   = '{}-{}-{}'.format(today.year, today.month, today.day)

        table_dict = {}

        print('getting todays matchups... from {} to {}'.format(start_date, end_date))
        # Returns pitcher/batter matchups for today's games
        table_dict['todays_players'] = get_todays_matchups()
        print('querying statcast data....')
        table_dict['statcast'] = get_statcast(sd=start_date, ed=end_date, prod=True, table_dict=table_dict)
        ##append todays records to statcast...

        # Creates modeling data needed for today's matchups
        print('creating rp_player data...')
        table_dict['rpbatter'] = get_rpplayer(16, 'RPBatter.csv', 'batter', prod=True, table_dict=table_dict)
        table_dict['ytdbatter'] = get_rpplayer(300, 'YTDBatter.csv', 'batter', prod=True, table_dict=table_dict)
        table_dict['rppitcher'] = get_rpplayer(20, 'RPPitcher.csv', 'pitcher', prod=True, table_dict=table_dict)
        table_dict['ytdpitcher'] = get_rpplayer(300, 'YTDPitcher.csv', 'pitcher', prod=True, table_dict=table_dict)
        print('Creating player meta data...')
        table_dict['player_meta'] = get_player_meta(table_dict=table_dict)
        print('creating game level data...')
        table_dict['gamelvl'] = get_game_lvl(prod=True, table_dict=table_dict)
        print('get matchups...')
        table_dict['matchups'] = get_matchups(prod=True, table_dict=table_dict)

        print('get prod data...')
        table_dict['prod_data'] = get_modeling_data(prod=True, table_dict=table_dict)

        # Loads model from prod to make predictions
        model_fp = './prod_model/model.pickle'
        f = open(model_fp, 'rb')
        model = pickle.load(f)

        # Filters to rows we want to need for prediction
        #table_dict['todays_preds'] = table_dict['prod_data'].loc[table_dict['prod_data']['game_date']==table_dict['prod_data']['game_date'].max()]
        table_dict['todays_preds'] = table_dict['prod_data']
        id_vars = ['game_date', 'game_pk', 'batter', 'starting_pitcher', 'ABs', 'hits', 'hit_ind']
        print('--- checking cols ---')
        print(table_dict['todays_preds'].drop(id_vars, axis=1).columns)
        
        # Calculates predicted probabilities
        preds = model.predict_proba(table_dict['todays_preds'].drop(id_vars, axis=1))
        print(preds[0:10])
        proba_hit = [i[1] for i in preds]
        #print(proba_hit.head())
        #print(proba_hit.shape)
        #print(table_dict['todays_preds'].shape)
        table_dict['todays_preds']['proba'] = proba_hit

        coef_df = pd.DataFrame({'coef_': model.coef_[0].tolist()})
        #coef_df['input'] = list(table_dict['todays_preds'].drop(id_vars, axis=1).columns)
        #coef_df = coef_df.append({'coef_':model.intercept_[0], 'input': 'intercept'})
        #coef_df['prob_'] = coef_df['coef_'].apply(lambda x: math.exp(x) / (1+math.exp(x)))
        #


        # Gets player names mapped to ID to display
        keep_cols = ['MLBID', 'MLBNAME']
        meta = pd.read_csv('./data/player_id_mapping.csv')
        meta = meta[keep_cols]
        table_dict['todays_preds'] = pd.merge(table_dict['todays_preds'], meta.rename(columns={'MLBNAME':'pitcher_name'}), how='left', left_on='starting_pitcher', right_on='MLBID')
        table_dict['todays_preds'] = pd.merge(table_dict['todays_preds'], meta.rename(columns={'MLBNAME':'batter_name'}), how='left', left_on='batter', right_on='MLBID')

        # Saves table of today's predictions
        print(table_dict['todays_preds']['proba'].head())
        name_file = './data/table_dict' + '.pickle'
        pickle.dump(table_dict, open(name_file, 'wb'))
        print(table_dict['todays_preds']['game_date'].value_counts())
        table_dict['todays_preds'] = table_dict['todays_preds'].loc[~table_dict['todays_preds']['game_date'].isna()]
        
        # Assures that we are just using today and not a day we didn't predict previously
        max_date = table_dict['todays_preds']['game_date'].max()
        print(max_date)
        table_dict['todays_preds'] = table_dict['todays_preds'].loc[table_dict['todays_preds']['game_date']==max_date]
        print(table_dict['todays_preds'].head())
        print(table_dict['todays_preds'].shape)

        # Sorts batters by probability and gets the highest one
        table_dict['todays_preds'] = table_dict['todays_preds'].sort_values('proba', ascending=False)
        print(table_dict['todays_preds'].shape)
        print(table_dict['todays_preds'].shape)
        print(table_dict['todays_preds'][['batter', 'proba']].head())
        table_dict['todays_preds']['batter'] = table_dict['todays_preds']['batter'].astype(int).astype(str).head()
        pick1 = table_dict['todays_preds']['batter'].iloc[0]
        pick2 = table_dict['todays_preds']['batter'].iloc[1]
        print(pick1)
        print(pick2)
        
        # Submits picks
        #enter(pick1, pick2)


        #preditc_data_comb.to_sql()
