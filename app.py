#-- Base packages
import os
import sys
import pickle
import pdb


#-- Pypi paackages
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from datetime import datetime as dt
from bs4 import BeautifulSoup
#import urllib.request
#import pandas.io.sql as psql
#import sqlalchemy
#from sqlalchemy.types import INTEGER, TEXT

#-- Custom packages


app = Flask(__name__) # References this file

#app.config['img_dir'] = 'static/images'

#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///coffee.db'
#app.secret_key = 'java'
#db = SQLAlchemy(app)


@app.route('/bts2') #URL string of our route. Below main() is the function for this route
def main():
    # Loads pickled model
    model_fp = './data/table_dict.pickle'
    f = open(model_fp, 'rb')
    table_dict = pickle.load(f)
    
    # Creates matchup table with probs, hit outcomes and indicators
    preds = table_dict['todays_preds'].sort_values('proba', ascending=False)
    preds['proba'] = preds['proba'].apply(lambda x: "{0:.1f}%".format(x*100))
    preds['batter_img'] = preds['batter'].apply(lambda x: '<img class="food-img" src=./static/images/{}.jpg  onError="this.onerror=null;this.src=./static/images/placeholder.jpg;"></img>'.format(x))
    preds['pitcher_img'] = preds['starting_pitcher'].apply(lambda x: '<img class="food-img" src=./static/images/{}.jpg onError="this.onerror=null;this.src=./static/images/placeholder.jpg;"></img>'.format(x))
    preds['batter_disp'] = "<a class='player-name'>"+preds['batter_name']+"</a>"+preds['batter_img']
    preds['pitcher_disp'] = "<a class='player-name'>"+preds['pitcher_name']+"</a>"+preds['pitcher_img']
    preds['hit_outcomes'] = preds['hits'].astype(str) +' / '+ preds['ABs'].astype(str)


    dates_arr = table_dict['todays_preds'][['game_date']].drop_duplicates().sort_values(by='game_date', ascending=True).reset_index()
    dates_arr['month'] = pd.to_datetime(dates_arr['game_date']).dt.month_name()
    dates_arr['day'] = pd.to_datetime(dates_arr['game_date']).dt.day.astype(str)
    dates_arr['month_num'] = pd.to_datetime(dates_arr['game_date']).dt.month.astype(str)
    dates_json = dates_arr[['month', 'day', 'month_num', 'game_date']].to_json(orient="records")

    current_date = preds['game_date'].max()
    print('--- printing requests ---')
    print(current_date)
    #preds = preds.loc[preds['game_date']==current_date]
    print(preds['game_date'].unique())
    print(preds.shape)
    month = str(dates_arr.loc[dates_arr['game_date']==current_date, 'month'].unique()[0])
    day = str(dates_arr.loc[dates_arr['game_date']==current_date, 'day'].unique()[0])
    index = dates_arr.loc[dates_arr['game_date']==current_date, 'month'].index[0]
    current_date = preds['game_date'].max()
    keep_cols = ['game_date', 'batter_disp', 'pitcher_disp', 'proba', 'hit_outcomes', 'hit_ind']
    rename_cols = {
        'proba': 'Estimated Hit Probability',
        'batter_disp': 'Player',
        'pitcher_disp': 'Starting Pitcher',
    }
    preds = preds[keep_cols]
    preds = preds.sort_values(by=['game_date', 'proba'], ascending=(False, False)).rename(columns=rename_cols)
    table = preds.to_html(index=False, escape=False)
    soup = BeautifulSoup(table, "html.parser")
    rows = soup.find_all('tr')
    rows[0]['class'] = 'header'
    for i, gd in enumerate(preds['game_date']):
        rows[i+1]['class'] = preds['game_date'].iloc[i]
    print(preds.shape)

    return render_template("index.html", table=soup, dates_arr=dates_json, month=month, day=day,index=index)


if __name__ == "__main__":
    app.run(debug=True)
