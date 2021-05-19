
#-- Base packages
import os
import sys
from datetime import date
import requests
import re
import json

#-- Pypi packages
import pandas as pd
from bs4 import BeautifulSoup

#-- Custom packages

def get_todays_matchups():
    url = 'https://baseballsavant.mlb.com/daily_matchups'
    html = requests.get(url).content
    soup = BeautifulSoup(html, "html.parser")
    reg = 'var matchups_data = (.*)}];'
    matchups = re.findall(reg, str(soup))[0] + '}]'
    matchups_df = pd.read_json(matchups, orient='records')

    keep_cols = ['player_id', 'pitcher_id', 'batter_team', 'pitcher_team']
    matchups_df = matchups_df[keep_cols]
    today = date.today()
    matchups_df['game_date'] = '{:04d}-{:02d}-{:02d} 00:00:00'.format(today.year, today.month, today.day)

    matchups_df['game'] = matchups_df.apply(lambda x: ", ".join(sorted([x['batter_team'], x['pitcher_team']])), axis=1)
    start_id = -111111.0
    game_pk_ph = pd.DataFrame(columns=['game', 'game_pk'])
    for game in matchups_df['game'].unique():
        game_pk_ph = game_pk_ph.append({'game':game, 'game_pk': start_id}, ignore_index=True)
        start_id -= 1
    matchups_df = pd.merge(matchups_df, game_pk_ph, how='left', on='game')
    matchups_df['events'] = '_placeholder_'
    matchups_df['inning_topbot'] = 'Bot' # temporary
    rename_cols = {
        'player_id': 'batter',
        'pitcher_id': 'pitcher',
    }
    matchups_df = matchups_df.rename(columns=rename_cols)

    return matchups_df
